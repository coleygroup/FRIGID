# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spec2Mol Unified Model for end-to-end spectra-to-molecule generation.

This module provides:
1. Spec2MolModel: Unified model combining MIST encoder and DLM decoder
2. Loading utilities for unified checkpoints

Usage:
    from dlm.utils.spec2mol import Spec2MolModel, load_spec2mol_model

    model = load_spec2mol_model('checkpoints/spec2mol_united.ckpt')
    predictions = model.predict(spectra_batch, n_samples=16)
"""

import os
import itertools
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from mist.models import SpectraEncoderGrowing
from dlm.model import DLM

# Suppress RDKit warnings
warnings.filterwarnings('ignore')


class Spec2MolSampler:
    """
    Sampler wrapper for Spec2MolModel.
    
    Provides the same interface as dlm.sampler.Sampler for
    compatibility with existing generation code.
    """

    def __init__(self, model: 'Spec2MolModel'):
        """
        Initialize sampler wrapper.
        
        Args:
            model: Parent Spec2MolModel instance
        """
        self.model = model
        self.decoder = model.decoder
        self.device = model.device

        # Copy attributes from underlying model
        self.tokenizer = self.decoder.tokenizer
        self.mask_index = self.decoder.mask_index
        self.bos_index = self.decoder.bos_index
        self.eos_index = self.decoder.eos_index
        self.pad_index = self.tokenizer.pad_token_id
        self.dot_index = self.tokenizer('.')['input_ids'][1]
        self.mdlm = self.decoder.mdlm
        self.mdlm.to_device(self.device)
        self.fingerprint_bits = self.decoder.config.model.get('fingerprint_bits', 4096)
        self.max_position_embeddings = self.decoder.config.model.get('max_position_embeddings', 256)

    @property
    def use_fingerprint_conditioning(self) -> bool:
        """Check if fingerprint conditioning is enabled."""
        return self.decoder.use_fingerprint_conditioning
    
    @property
    def use_formula_conditioning(self) -> bool:
        """Check if formula conditioning is enabled."""
        return self.decoder.use_formula_conditioning

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        softmax_temp: float = 1.2,
        randomness: float = 2,
        fix: bool = True,
        gamma: float = 0,
        w: float = 2,
        formula: Optional[Union[str, List[str]]] = None,
        fingerprint: Optional[np.ndarray] = None,
    ) -> List[str]:
        """Generate molecules from masked input."""
        import random
        from dlm.utils.utils_chem import safe_to_smiles
        from dlm.utils.bracket_safe_converter import bracketsafe2safe

        x = x.to(self.device)
        num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_index

        if formula is not None and isinstance(formula, str):
            formula = [formula] * x.size(0)

        fingerprint_tensor = None
        fingerprint_mask = None
        if fingerprint is not None:
            if not self.use_fingerprint_conditioning:
                raise RuntimeError("Fingerprint conditioning not enabled in this model")
            fingerprint_tensor = self._prepare_fingerprint_batch(fingerprint, x.size(0))
            fingerprint_mask = torch.ones(
                fingerprint_tensor.size(0),
                device=fingerprint_tensor.device,
                dtype=torch.float32
            )

        for i in range(num_steps):
            logits = self.decoder(
                x,
                attention_mask,
                formula=formula,
                fingerprint=fingerprint_tensor,
                fingerprint_mask=fingerprint_mask
            )

            if gamma and w:
                x_poor = x.clone()
                context_tokens = (x_poor[0] != self.bos_index).to(int) * \
                    (x_poor[0] != self.eos_index).to(int) * \
                    (x_poor[0] != self.mask_index).to(int) * \
                    (x_poor[0] != self.pad_index).to(int)
                context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                num_mask_poor = int(context_tokens.sum() * gamma)
                mask_idx_poor = random.sample(context_token_ids, num_mask_poor)
                x_poor[:, mask_idx_poor] = self.mask_index
                logits_poor = self.decoder(
                    x_poor,
                    attention_mask=attention_mask,
                    formula=formula,
                    fingerprint=fingerprint_tensor,
                    fingerprint_mask=fingerprint_mask
                )
                logits = w * logits + (1 - w) * logits_poor

            x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)

        samples = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        if self.decoder.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _insert_mask(
        self,
        x: torch.Tensor,
        num_samples: int,
        min_add_len: int = 18,
        target_lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Insert mask tokens into input sequence."""
        import pickle
        import random
        
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

        if target_lengths is None:
            try:
                with open(os.path.join(ROOT_DIR, 'data/len.pk'), 'rb') as f:
                    seq_len_list = pickle.load(f)
            except FileNotFoundError:
                seq_len_list = list(range(20, 100))
            target_lengths = [
                max(random.choice(seq_len_list), len(x[0]) + min_add_len)
                for _ in range(num_samples)
            ]
        elif len(target_lengths) == 1 and num_samples > 1:
            target_lengths = target_lengths * num_samples

        x = x[0]
        current_len = len(x)
        x_new = []

        for i in range(num_samples):
            target_len = target_lengths[i] if i < len(target_lengths) else target_lengths[-1]
            add_seq_len = max(target_len - current_len, min_add_len)
            add_seq_len = min(add_seq_len, self.max_position_embeddings - current_len)
            add_seq_len = max(add_seq_len, min_add_len)

            x_new.append(torch.hstack([
                x[:-1],
                torch.full((add_seq_len,), self.mask_index),
                x[-1:],
            ]))

        pad_len = max([len(xx) for xx in x_new])
        x_new = [
            torch.hstack([xx, torch.full((pad_len - len(xx),), self.pad_index)])
            for xx in x_new
        ]
        return torch.stack(x_new)

    @torch.no_grad()
    def fingerprint_conditioned_generation(
        self,
        fingerprint: np.ndarray,
        num_samples: int = 1,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        target_lengths: Optional[List[int]] = None,
    ) -> List[str]:
        """Generate molecules conditioned on fingerprint."""
        original_training = self.decoder.training
        self.decoder.eval()

        x = torch.hstack([
            torch.full((1, 1), self.bos_index),
            torch.full((1, 1), self.eos_index)
        ])
        x = self._insert_mask(x, num_samples, min_add_len=40, target_lengths=target_lengths)
        x = x.to(self.device)

        samples = self.generate(
            x,
            softmax_temp,
            randomness,
            fingerprint=fingerprint
        )

        if original_training:
            self.decoder.train()
        return samples
    
    @torch.no_grad()
    def unified_conditioned_generation(
        self,
        formula: Optional[str] = None,
        fingerprint: Optional[np.ndarray] = None,
        num_samples: int = 1,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        target_lengths: Optional[List[int]] = None,
        min_add_len: int = 40,
    ) -> List[str]:
        """Generate molecules with unified conditioning."""
        original_training = self.decoder.training
        self.decoder.eval()
        
        x = torch.hstack([
            torch.full((1, 1), self.bos_index),
            torch.full((1, 1), self.eos_index)
        ])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len, target_lengths=target_lengths)
        x = x.to(self.device)
        
        samples = self.generate(
            x,
            softmax_temp,
            randomness,
            formula=formula,
            fingerprint=fingerprint
        )
        
        if original_training:
            self.decoder.train()
        
        return samples

    def _prepare_fingerprint_batch(self, fingerprint: np.ndarray, batch_size: int) -> torch.Tensor:
        """Prepare fingerprint tensor for batch generation."""
        if torch.is_tensor(fingerprint):
            fp = fingerprint.detach().clone()
        else:
            fp = torch.as_tensor(fingerprint, dtype=torch.float32)
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        if fp.size(-1) != self.fingerprint_bits:
            raise ValueError(f"Expected FP dim {self.fingerprint_bits}, got {fp.size(-1)}")
        if fp.size(0) == 1 and batch_size > 1:
            fp = fp.expand(batch_size, -1)
        elif fp.size(0) != batch_size:
            raise ValueError(f"FP batch mismatch: got {fp.size(0)}, expected {batch_size}")
        return fp.to(self.device)


class Spec2MolModel(nn.Module):
    """
    Unified Spec2Mol model for end-to-end spectra-to-molecule generation.
    
    Combines:
    - MIST Encoder: Predicts molecular fingerprints from mass spectra
    - DLM Decoder: Generates molecules conditioned on fingerprints
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: DLM,
        fingerprint_config: Dict[str, Any],
        device: torch.device = None
    ):
        """
        Initialize Spec2MolModel.
        
        Args:
            encoder: MIST encoder module
            decoder: DLM model
            fingerprint_config: Dict with 'bits', 'radius', 'threshold' keys
            device: Target device
        """
        super().__init__()
        
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        
        self.decoder = decoder.to(self.device)
        self.decoder.backbone.eval()
        
        self.fingerprint_config = fingerprint_config
        self.fp_bits = fingerprint_config.get('bits', 4096)
        self.fp_radius = fingerprint_config.get('radius', 2)
        self.fp_threshold = fingerprint_config.get('threshold', 0.172)
        
        self._sampler = None

    @property
    def sampler(self) -> Spec2MolSampler:
        """Lazy-load sampler wrapper."""
        if self._sampler is None:
            self._sampler = Spec2MolSampler(self)
        return self._sampler

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        strict: bool = False
    ) -> 'Spec2MolModel':
        """
        Load Spec2MolModel from a unified checkpoint.
        
        Args:
            checkpoint_path: Path to unified checkpoint file
            device: Target device
            strict: Whether to use strict state dict loading
        
        Returns:
            Initialized Spec2MolModel instance
        """
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading Spec2Mol model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if checkpoint.get('checkpoint_type') != 'spec2mol_united':
            raise ValueError(f"Invalid checkpoint type: {checkpoint.get('checkpoint_type')}")

        # Load encoder
        encoder_config = checkpoint['encoder_config']
        encoder = SpectraEncoderGrowing(
            form_embedder=encoder_config.get('form_embedder', 'pos-cos'),
            output_size=encoder_config.get('output_size', 4096),
            hidden_size=encoder_config.get('hidden_size', 512),
            spectra_dropout=encoder_config.get('spectra_dropout', 0.1),
            peak_attn_layers=encoder_config.get('peak_attn_layers', 2),
            num_heads=encoder_config.get('num_heads', 8),
            set_pooling=encoder_config.get('set_pooling', 'cls'),
            refine_layers=encoder_config.get('refine_layers', 4),
            pairwise_featurization=encoder_config.get('pairwise_featurization', True),
            embed_instrument=encoder_config.get('embed_instrument', False),
            inten_transform=encoder_config.get('inten_transform', 'float'),
            magma_modulo=encoder_config.get('magma_modulo', 2048),
            inten_prob=encoder_config.get('inten_prob', 0.1),
            remove_prob=encoder_config.get('remove_prob', 0.5),
            cls_type=encoder_config.get('cls_type', 'ms1'),
            spec_features=encoder_config.get('spec_features', 'peakformula'),
            mol_features=encoder_config.get('mol_features', 'fingerprint'),
            top_layers=encoder_config.get('top_layers', 1),
        )
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=strict)
        print("✓ Loaded MIST encoder")

        # Load decoder
        decoder_checkpoint = checkpoint['decoder_checkpoint']
        decoder = _load_dlm_from_dict(decoder_checkpoint, device)
        print("✓ Loaded DLM decoder")

        # Get fingerprint config
        fingerprint_config = checkpoint.get('fingerprint_config', {
            'bits': 4096, 'radius': 2, 'threshold': 0.172,
        })

        model = cls(
            encoder=encoder,
            decoder=decoder,
            fingerprint_config=fingerprint_config,
            device=device
        )

        print(f"✓ Spec2Mol model loaded on {device}")
        return model

    @torch.no_grad()
    def encode_spectra(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode spectra batch to fingerprint predictions.
        
        Args:
            batch: Dictionary containing spectra data
        
        Returns:
            Tuple of (fingerprint_probs, fingerprint_binary)
        """
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }

        fp_probs, _ = self.encoder(batch)
        fp_probs = fp_probs.cpu().numpy()
        fp_binary = (fp_probs >= self.fp_threshold).astype(np.float32)

        return fp_probs, fp_binary

    @torch.no_grad()
    def generate_from_fingerprint(
        self,
        fingerprint: np.ndarray,
        num_samples: int = 16,
        softmax_temp: float = 1.0,
        randomness: float = 0.1,
    ) -> List[str]:
        """Generate molecules conditioned on a fingerprint."""
        return self.sampler.fingerprint_conditioned_generation(
            fingerprint=fingerprint,
            num_samples=num_samples,
            softmax_temp=softmax_temp,
            randomness=randomness,
        )

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 16,
        softmax_temp: float = 1.0,
        randomness: float = 0.1,
        return_fingerprints: bool = False,
    ) -> Union[List[str], Tuple[List[str], np.ndarray, np.ndarray]]:
        """
        End-to-end prediction: spectra -> fingerprint -> molecules.
        """
        fp_probs, fp_binary = self.encode_spectra(batch)

        molecules = self.generate_from_fingerprint(
            fingerprint=fp_binary[0] if len(fp_binary.shape) > 1 else fp_binary,
            num_samples=num_samples,
            softmax_temp=softmax_temp,
            randomness=randomness,
        )

        if return_fingerprints:
            return molecules, fp_probs, fp_binary
        return molecules

    def eval(self):
        """Set model to evaluation mode."""
        self.encoder.eval()
        self.decoder.backbone.eval()
        return self

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return self


def _load_dlm_from_dict(checkpoint_dict: Dict[str, Any], device: torch.device) -> DLM:
    """Load DLM from a checkpoint dictionary."""
    hparams = checkpoint_dict.get('hyper_parameters', {})
    config = hparams.get('config', None)

    if config is None:
        raise ValueError("Cannot load DLM: missing 'config' in hyper_parameters")

    model = DLM(config)
    state_dict = checkpoint_dict.get('state_dict', {})
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.backbone.eval()

    if model.ema and 'ema' in checkpoint_dict:
        model.ema.load_state_dict(checkpoint_dict['ema'])
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))

    return model


def load_spec2mol_model(checkpoint_path: str, device: Optional[torch.device] = None) -> Spec2MolModel:
    """
    Convenience function to load Spec2MolModel from checkpoint.
    
    Args:
        checkpoint_path: Path to unified checkpoint
        device: Target device
    
    Returns:
        Loaded Spec2MolModel instance
    """
    return Spec2MolModel.from_checkpoint(checkpoint_path, device=device)
