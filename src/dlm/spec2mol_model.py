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

"""Spec2Mol Model: End-to-end training for MIST encoder + DLM decoder.

Architecture:
    Spectra --[MIST Encoder]--> Predicted FP --+
                                               |
    GT Formula -------------------------------+|
                                               v
                                   [DLM Decoder] --> SAFE tokens

Loss:
    MDLM cross-entropy loss on SAFE token prediction
"""

import itertools
import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from bionemo.moco.interpolants import MDLM
from bionemo.moco.distributions.time import UniformTimeDistribution
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

from mist.models.spectra_encoder import SpectraEncoderGrowing
from dlm.utils.ema import ExponentialMovingAverage
from dlm.utils.utils_data import get_tokenizer
from dlm.utils.utils_save import clean_checkpoint, fast_forward_info
from dlm.utils.utils_moco import AntitheticUniformTimeDistribution
from dlm.utils.formula_encoder import FormulaEncoder
from dlm.model_components import (
    FormulaConditioner,
    AdaptiveFormulaConditioner,
    CrossAttentionFormulaConditioner,
    FingerprintConditioner,
    AdaptiveFingerprintConditioner,
    CrossAttentionFingerprintConditioner,
)
from dlm.bert_with_cross_attention import BertForMaskedLMWithCrossAttention


# Keys to skip when preparing encoder inputs (batch keys not for encoder)
ENCODER_SKIP_KEYS = {"gt_fingerprint", "formula", "input_ids", "attention_mask"}


def _prepare_encoder_inputs(batch: Dict, device: torch.device) -> Dict:
    """Prepare inputs for the MIST encoder from batch.
    
    Args:
        batch: Batch dictionary from dataloader
        device: Target device
        
    Returns:
        Dictionary with encoder inputs (spectra features)
    """
    encoder_inputs = {}
    for key, value in batch.items():
        if key in ENCODER_SKIP_KEYS:
            continue
        if torch.is_tensor(value):
            encoder_inputs[key] = value.to(device)
        else:
            encoder_inputs[key] = value
    return encoder_inputs


class Spec2MolModel(L.LightningModule):
    """Spec2Mol Model for end-to-end spectra-to-molecule generation.
    
    Combines:
    - MIST encoder: Encodes mass spectra into fingerprint representations
    - DLM decoder: Generates SAFE tokens conditioned on fingerprints and formulas
    
    Loss: MDLM cross-entropy on masked SAFE token prediction
    
    Args:
        cfg: Configuration object with model and training settings
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Set up tokenizer
        self.tokenizer = get_tokenizer(cfg.data.get('hf_cache_dir', None))
        self.mask_index = self.tokenizer.mask_token_id
        self.pad_index = self.tokenizer.pad_token_id
        
        # Get configurations
        model_cfg = cfg.model
        encoder_cfg = cfg.encoder
        
        # ==================== MIST Encoder ====================
        self.encoder = SpectraEncoderGrowing(
            inten_transform=encoder_cfg.get('inten_transform', 'float'),
            inten_prob=encoder_cfg.get('inten_prob', 0.1),
            remove_prob=encoder_cfg.get('remove_prob', 0.5),
            peak_attn_layers=encoder_cfg.get('peak_attn_layers', 2),
            num_heads=encoder_cfg.get('num_heads', 8),
            pairwise_featurization=encoder_cfg.get('pairwise_featurization', True),
            embed_instrument=encoder_cfg.get('embed_instrument', False),
            cls_type=encoder_cfg.get('cls_type', 'ms1'),
            set_pooling=encoder_cfg.get('set_pooling', 'cls'),
            spec_features=encoder_cfg.get('spec_features', 'peakformula'),
            mol_features=encoder_cfg.get('mol_features', 'fingerprint'),
            form_embedder=encoder_cfg.get('form_embedder', 'pos-cos'),
            output_size=encoder_cfg.get('output_size', 4096),
            hidden_size=encoder_cfg.get('hidden_size', 256),
            spectra_dropout=encoder_cfg.get('spectra_dropout', 0.1),
            top_layers=encoder_cfg.get('top_layers', 1),
            refine_layers=encoder_cfg.get('refine_layers', 4),
            magma_modulo=encoder_cfg.get('magma_modulo', 512),
        )
        
        # Load pretrained encoder weights if provided
        if encoder_cfg.get('pretrained_path'):
            self._load_encoder_weights(encoder_cfg.pretrained_path)
        
        # ==================== Fingerprint Projection ====================
        # Project encoder output to decoder fingerprint dimension
        self.encoder_output_size = encoder_cfg.get('output_size', 4096)
        self.fingerprint_bits = model_cfg.get('fingerprint_bits', 4096)
        
        # Merge function: encoder output -> fingerprint for conditioning
        merge_type = cfg.get('merge', 'none')
        if merge_type == 'downproject':
            self.merge_function = nn.Linear(self.encoder_output_size, self.fingerprint_bits)
        elif merge_type == 'mlp':
            self.merge_function = nn.Sequential(
                nn.Linear(self.encoder_output_size, 1024),
                nn.SiLU(),
                nn.Linear(1024, self.fingerprint_bits)
            )
        else:
            # 'none' or 'mist_fp' - use encoder output directly
            self.merge_function = None
            # Ensure dimensions match
            assert self.encoder_output_size == self.fingerprint_bits, \
                f"Encoder output ({self.encoder_output_size}) must match fingerprint bits ({self.fingerprint_bits}) when merge='none'"
        
        # ==================== Formula Conditioning ====================
        self.use_formula_conditioning = model_cfg.get('use_formula_conditioning', True)
        self.formula_conditioner_type = model_cfg.get('formula_conditioner_type', 'cross_attention')
        
        if self.use_formula_conditioning:
            self.formula_encoder = FormulaEncoder(
                normalize=model_cfg.get('formula_normalize', 'none')
            )
            
            if self.formula_conditioner_type == 'cross_attention':
                self.formula_conditioner = CrossAttentionFormulaConditioner(
                    num_atom_types=self.formula_encoder.vocab_size,
                    hidden_size=model_cfg.get('hidden_size', 768),
                    num_attention_heads=model_cfg.get('num_attention_heads', 12),
                    num_layers=model_cfg.get('num_hidden_layers', 12),
                    dropout=model_cfg.get('hidden_dropout_prob', 0.1),
                    use_count_embedding=model_cfg.get('formula_use_count_embedding', True)
                )
            elif self.formula_conditioner_type == 'adaptive':
                self.formula_conditioner = AdaptiveFormulaConditioner(
                    input_dim=self.formula_encoder.vocab_size,
                    hidden_dim=model_cfg.get('formula_hidden_dim', 256),
                    output_dim=model_cfg.get('hidden_size', 768),
                    dropout=model_cfg.get('hidden_dropout_prob', 0.1),
                    num_layers=model_cfg.get('formula_num_layers', 3),
                    learnable_scale=model_cfg.get('formula_learnable_scale', True),
                    initial_scale=model_cfg.get('formula_initial_scale', 1.0)
                )
            else:  # 'basic'
                self.formula_conditioner = FormulaConditioner(
                    input_dim=self.formula_encoder.vocab_size,
                    hidden_dim=model_cfg.get('formula_hidden_dim', 256),
                    output_dim=model_cfg.get('hidden_size', 768),
                    dropout=model_cfg.get('hidden_dropout_prob', 0.1),
                    num_layers=model_cfg.get('formula_num_layers', 3)
                )
            
            self.formula_dropout_prob = model_cfg.get('formula_dropout_prob', 0.0)
        else:
            # Create a formula encoder even when formula conditioning is disabled
            # (needed for token atom counts in differentiable formula loss)
            self.formula_encoder = FormulaEncoder(normalize='none')
            self.formula_conditioner = None
            self.formula_dropout_prob = 0.0
        
        # Set up token atom counts buffer for differentiable formula loss
        self._setup_token_atom_counts()
        
        # ==================== Fingerprint Conditioning ====================
        self.use_fingerprint_conditioning = model_cfg.get('use_fingerprint_conditioning', True)
        self.fingerprint_conditioner_type = model_cfg.get('fingerprint_conditioner_type', 'cross_attention')
        
        # Shared cross-attention mode: when True, formula and fingerprint embeddings are concatenated
        # and passed through a single set of cross-attention layers
        self.use_shared_cross_attention = model_cfg.get('use_shared_cross_attention', False)
        
        if self.use_fingerprint_conditioning:
            if self.fingerprint_conditioner_type == 'cross_attention':
                # When using shared cross-attention, fingerprint conditioner doesn't create its own
                # cross-attention layers - they'll be shared with formula conditioner
                create_fp_cross_attn = not self.use_shared_cross_attention
                self.fingerprint_conditioner = CrossAttentionFingerprintConditioner(
                    num_bits=self.fingerprint_bits,
                    hidden_size=model_cfg.get('hidden_size', 768),
                    num_attention_heads=model_cfg.get('num_attention_heads', 12),
                    num_layers=model_cfg.get('num_hidden_layers', 12),
                    max_seq_len=model_cfg.get('fingerprint_seq_max_len', 256),
                    activation_threshold=model_cfg.get('fingerprint_activation_threshold', 0.0),
                    dropout=model_cfg.get('hidden_dropout_prob', 0.1),
                    num_self_attention_layers=model_cfg.get('fingerprint_num_self_attention_layers', 2),
                    create_cross_attention_layers=create_fp_cross_attn,
                )
            elif self.fingerprint_conditioner_type == 'adaptive':
                self.fingerprint_conditioner = AdaptiveFingerprintConditioner(
                    input_dim=self.fingerprint_bits,
                    hidden_dim=model_cfg.get('fingerprint_hidden_dim', 512),
                    output_dim=model_cfg.get('hidden_size', 768),
                    dropout=model_cfg.get('hidden_dropout_prob', 0.1),
                    num_layers=model_cfg.get('fingerprint_num_layers', 3),
                    learnable_scale=model_cfg.get('fingerprint_learnable_scale', True),
                    initial_scale=model_cfg.get('fingerprint_initial_scale', 0.1),
                )
            else:  # 'basic'
                self.fingerprint_conditioner = FingerprintConditioner(
                    input_dim=self.fingerprint_bits,
                    hidden_dim=model_cfg.get('fingerprint_hidden_dim', 512),
                    output_dim=model_cfg.get('hidden_size', 768),
                    dropout=model_cfg.get('hidden_dropout_prob', 0.1),
                    num_layers=model_cfg.get('fingerprint_num_layers', 3),
                )
            
            self.fingerprint_dropout_prob = model_cfg.get('fingerprint_dropout_prob', 0.1)
        else:
            self.fingerprint_conditioner = None
            self.fingerprint_dropout_prob = 0.0
        
        # ==================== BERT Backbone ====================
        bert_config_dict = {
            'attention_probs_dropout_prob': model_cfg.get('attention_probs_dropout_prob', 0.1),
            'hidden_act': model_cfg.get('hidden_act', 'gelu'),
            'hidden_dropout_prob': model_cfg.get('hidden_dropout_prob', 0.1),
            'hidden_size': model_cfg.get('hidden_size', 768),
            'initializer_range': model_cfg.get('initializer_range', 0.02),
            'intermediate_size': model_cfg.get('intermediate_size', 3072),
            'layer_norm_eps': model_cfg.get('layer_norm_eps', 1e-12),
            'max_position_embeddings': model_cfg.get('max_position_embeddings', 256),
            'model_type': 'bert',
            'num_attention_heads': model_cfg.get('num_attention_heads', 12),
            'num_hidden_layers': model_cfg.get('num_hidden_layers', 12),
            'pad_token_id': self.pad_index,
            'position_embedding_type': 'absolute',
            'type_vocab_size': 2,
            'use_cache': True,
            'vocab_size': model_cfg.get('vocab_size', 1880),
        }
        
        # Get cross-attention layers
        formula_cross_attn = None
        fp_cross_attn = None
        
        if self.use_formula_conditioning and self.formula_conditioner_type == 'cross_attention':
            formula_cross_attn = self.formula_conditioner.cross_attention_layers
        
        # In shared mode, fingerprint uses the same cross-attention layers as formula
        # In independent mode, fingerprint has its own cross-attention layers
        if self.use_fingerprint_conditioning and self.fingerprint_conditioner_type == 'cross_attention':
            if not self.use_shared_cross_attention:
                fp_cross_attn = self.fingerprint_conditioner.cross_attention_layers
        
        # Create backbone
        if formula_cross_attn is not None or fp_cross_attn is not None:
            self.backbone = BertForMaskedLMWithCrossAttention(
                BertConfig.from_dict(bert_config_dict),
                cross_attention_layers=formula_cross_attn,
                fingerprint_cross_attention_layers=fp_cross_attn,
                use_shared_cross_attention=self.use_shared_cross_attention,
            )
        else:
            self.backbone = BertForMaskedLM(BertConfig.from_dict(bert_config_dict))
        
        # Load pretrained decoder weights if provided
        if model_cfg.get('pretrained_path'):
            self._load_decoder_weights(model_cfg.pretrained_path)
        
        # ==================== MDLM Setup ====================
        if cfg.training.get('antithetic_sampling', True):
            time_distribution = AntitheticUniformTimeDistribution(
                sampling_eps=cfg.training.get('sampling_eps', 1e-3)
            )
        else:
            time_distribution = UniformTimeDistribution()
        
        prior = DiscreteMaskedPrior(
            num_classes=self.tokenizer.vocab_size,
            mask_dim=self.mask_index
        )
        noise_schedule = LogLinearExpNoiseTransform()
        
        self.mdlm = MDLM(
            time_distribution=time_distribution,
            prior_distribution=prior,
            noise_schedule=noise_schedule,
        )
        
        # ==================== EMA ====================
        if cfg.training.get('ema', 0) > 0:
            self.ema = ExponentialMovingAverage(
                itertools.chain(
                    self.backbone.parameters(),
                    self.encoder.parameters(),
                ),
                decay=cfg.training.ema
            )
        else:
            self.ema = None

    def _load_encoder_weights(self, path: str):
        """Load pretrained encoder weights."""
        logging.info(f"Loading encoder weights from {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter for encoder keys and strip 'encoder.' prefix if present
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_key = k.replace('encoder.', '', 1)
            elif not any(k.startswith(p) for p in ['backbone.', 'conditioner.', 'formula_conditioner.', 'fingerprint_conditioner.']):
                new_key = k
            else:
                continue
            encoder_state[new_key] = v
        
        # Handle key naming differences between checkpoint formats
        remapped_state = {}
        for k, v in encoder_state.items():
            new_key = k
            # Remap spectra_X to spectra_encoder.X
            if k.startswith('spectra_0.'):
                new_key = k.replace('spectra_0.', 'spectra_encoder.0.', 1)
            elif k.startswith('spectra_1.'):
                new_key = k.replace('spectra_1.', 'spectra_encoder.1.', 1)
            elif k.startswith('spectra_2.'):
                new_key = k.replace('spectra_2.', 'spectra_encoder.2.', 1)
            remapped_state[new_key] = v
        self.encoder.load_state_dict(remapped_state, strict=True)

    
    def _load_decoder_weights(self, path: str):
        """Load pretrained decoder weights."""
        logging.info(f"Loading decoder weights from {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Load backbone weights
        backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                        if k.startswith('backbone.')}
        if backbone_state:
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            logging.info(f"Backbone loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Load conditioner weights
        if self.use_formula_conditioning:
            formula_state = {k.replace('formula_conditioner.', ''): v for k, v in state_dict.items()
                           if k.startswith('formula_conditioner.')}
            if formula_state:
                missing, unexpected = self.formula_conditioner.load_state_dict(formula_state, strict=False)
                logging.info(f"Formula conditioner loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        if self.use_fingerprint_conditioning:
            fp_state = {k.replace('fingerprint_conditioner.', ''): v for k, v in state_dict.items()
                       if k.startswith('fingerprint_conditioner.')}
            if fp_state:
                missing, unexpected = self.fingerprint_conditioner.load_state_dict(fp_state, strict=False)
                logging.info(f"FP conditioner loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def _setup_token_atom_counts(self):
        """
        Pre-compute a static matrix mapping every token in the vocabulary to its 
        constituent atom counts. This is used for the differentiable formula loss.
        
        Creates a buffer 'token_atom_counts' of shape [vocab_size, num_atom_types]
        where each row contains the raw (unnormalized) atom counts for that token.
        """
        # Use the formula encoder for parsing tokens
        encoder = self.formula_encoder
        
        num_atom_types = encoder.vocab_size
        vocab_size = self.tokenizer.vocab_size
        
        # Build token atom counts matrix
        token_atom_counts = torch.zeros(vocab_size, num_atom_types, dtype=torch.float32)
        
        for token_id in range(vocab_size):
            # Decode token to string
            token_str = self.tokenizer.decode([token_id])
            
            # Parse atoms from token string
            try:
                counts = encoder.formula_to_counts(token_str)
                # Convert to vector with NO normalization (raw integers)
                count_vector = encoder.counts_to_vector(counts, normalize='none')
                token_atom_counts[token_id] = count_vector
            except Exception:
                # If parsing fails, leave as zeros (special tokens, etc.)
                pass
        
        # Register as persistent buffer so it moves to correct device automatically
        self.register_buffer('token_atom_counts', token_atom_counts)
        
        # Store num_atom_types for reference
        self._num_atom_types = num_atom_types

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override load_state_dict to handle missing non-trainable buffers gracefully.
        
        This allows loading checkpoints from before certain buffers were added
        (e.g., token_atom_counts) without failing. The buffers are already
        initialized in __init__, so missing keys for non-trainable buffers are safe.
        """
        # List of buffers that can be safely missing (they're computed, not learned)
        safe_missing_buffers = {'token_atom_counts'}
        
        # First try strict loading
        if strict:
            # Check for missing keys that are safe to skip
            model_keys = set(self.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            missing_keys = model_keys - loaded_keys
            
            # Filter out safe missing buffers
            unsafe_missing = missing_keys - safe_missing_buffers
            
            if unsafe_missing:
                # There are genuinely missing keys, use strict mode
                return super().load_state_dict(state_dict, strict=True, assign=assign)
            else:
                # Only safe buffers are missing, use non-strict mode
                if missing_keys & safe_missing_buffers:
                    print(f"[Spec2MolModel] Loading checkpoint with missing buffers (safe to skip): {missing_keys & safe_missing_buffers}")
                return super().load_state_dict(state_dict, strict=False, assign=assign)
        else:
            return super().load_state_dict(state_dict, strict=False, assign=assign)

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint.get('ema', {}))
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        self._checkpoint_random_state = checkpoint.get('sampler', {}).get('random_state', None)

    def on_save_checkpoint(self, checkpoint):
        import random
        import numpy as np
        
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        
        if 'sampler' not in checkpoint:
            checkpoint['sampler'] = {}
        checkpoint['sampler']['random_state'] = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }

    def configure_optimizers(self):
        """Configure optimizer with all trainable parameters."""
        params = list(self.backbone.parameters())
        params.extend(list(self.encoder.parameters()))
        
        if self.use_formula_conditioning and self.formula_conditioner is not None:
            params.extend(list(self.formula_conditioner.parameters()))
        
        if self.use_fingerprint_conditioning and self.fingerprint_conditioner is not None:
            params.extend(list(self.fingerprint_conditioner.parameters()))
        
        if self.merge_function is not None:
            params.extend(list(self.merge_function.parameters()))
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.optim.lr,
            betas=(self.cfg.optim.get('beta1', 0.9), self.cfg.optim.get('beta2', 0.999)),
            eps=self.cfg.optim.get('eps', 1e-8),
            weight_decay=self.cfg.optim.get('weight_decay', 0),
        )
        
        # Learning rate scheduler
        if hasattr(self.cfg, 'scheduler') and self.cfg.scheduler is not None:
            scheduler_cfg = self.cfg.scheduler
            
            # Auto-calculate total steps from trainer if available
            # This uses Lightning's estimated_stepping_batches which accounts for
            # epochs, dataset size, accumulation steps, and distributed training
            total_steps = scheduler_cfg.get('T_max', None)
            if total_steps is None or total_steps == 'auto':
                if self.trainer is not None:
                    total_steps = self.trainer.estimated_stepping_batches
                    logging.info(f"Auto-calculated total_steps: {total_steps}")
                else:
                    # Fallback if trainer not available
                    total_steps = 100000
                    logging.warning(f"Trainer not available, using default total_steps: {total_steps}")
            
            # Calculate warmup_steps from warmup_frac (fraction of total steps)
            warmup_frac = scheduler_cfg.get('warmup_frac', 0.0)
            warmup_steps = 0
            
            if warmup_frac > 0:
                warmup_steps = int(warmup_frac * total_steps)
                logging.info(f"Warmup: {warmup_frac*100:.1f}% of {total_steps} = {warmup_steps} steps")
            
            eta_min = scheduler_cfg.get('eta_min', 1e-5)
            
            if warmup_steps > 0:
                from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
                
                warmup_scheduler = LinearLR(
                    optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
                )
                # T_max for main scheduler is total_steps - warmup_steps
                main_scheduler = CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
                )
                scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_steps])
            else:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=total_steps, eta_min=eta_min
                )
            
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'lr'}]
        
        return optimizer

    def on_after_backward(self):
        """Log gradient norm every 20 steps."""
        if self.global_step % 20 == 0:
            total_norm = torch.linalg.vector_norm(
                torch.stack([
                    torch.linalg.vector_norm(p.grad.detach())
                    for p in self.parameters()
                    if p.grad is not None
                ])
            )
            self.log('grad_norm', total_norm, prog_bar=False, on_step=True, sync_dist=True)

    def on_train_start(self):
        self.backbone.train()
        self.encoder.train()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
        # Restore random state if resuming from checkpoint
        if hasattr(self, '_checkpoint_random_state') and self._checkpoint_random_state is not None:
            import random
            import numpy as np
            if isinstance(self._checkpoint_random_state, tuple):
                random.setstate(self._checkpoint_random_state)
            elif isinstance(self._checkpoint_random_state, dict):
                if 'python' in self._checkpoint_random_state:
                    random.setstate(self._checkpoint_random_state['python'])
                if 'numpy' in self._checkpoint_random_state:
                    np.random.set_state(self._checkpoint_random_state['numpy'])
            del self._checkpoint_random_state

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(
                self.backbone.parameters(),
                self.encoder.parameters()
            ))

    def _encode_spectra(self, batch: Dict) -> torch.Tensor:
        """Encode spectra through MIST encoder.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Tuple of (fingerprint tensor, auxiliary outputs)
        """
        encoder_inputs = _prepare_encoder_inputs(batch, self.device)
        encoder_output, aux = self.encoder(encoder_inputs)
        
        # Apply merge function if needed
        if self.merge_function is not None:
            fingerprint = self.merge_function(encoder_output)
        else:
            fingerprint = encoder_output
        
        return fingerprint, aux

    def _prepare_formula_conditioning(self, formula: List[str], batch_size: int):
        """Prepare formula conditioning embeddings."""
        if not self.use_formula_conditioning or formula is None:
            return None, None
        
        formula_vectors = self.formula_encoder.encode_batch(formula).to(self.device)
        
        if self.formula_conditioner_type == 'cross_attention':
            embeddings, mask = self.formula_conditioner.encode_formula(formula_vectors)
            # Apply dropout
            if self.training and torch.rand(1).item() < self.formula_dropout_prob:
                embeddings = embeddings * 0.0
            return embeddings, mask
        else:
            embeddings = self.formula_conditioner(formula_vectors)
            if self.training and torch.rand(1).item() < self.formula_dropout_prob:
                embeddings = embeddings * 0.0
            return embeddings, None

    def _prepare_fingerprint_conditioning(self, fingerprint: torch.Tensor):
        """Prepare fingerprint conditioning embeddings."""
        if not self.use_fingerprint_conditioning:
            return None, None
        
        if self.fingerprint_conditioner_type == 'cross_attention':
            embeddings, mask = self.fingerprint_conditioner.encode_fingerprint(fingerprint)
            # Apply dropout
            if self.training and torch.rand(1).item() < self.fingerprint_dropout_prob:
                embeddings = embeddings * 0.0
            return embeddings, mask
        else:
            embeddings = self.fingerprint_conditioner(fingerprint)
            if self.training and torch.rand(1).item() < self.fingerprint_dropout_prob:
                embeddings = embeddings * 0.0
            return embeddings, None

    def _build_token_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Build token embeddings for decoder."""
        token_embeddings = self.backbone.bert.embeddings.word_embeddings(x)
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        position_embeddings = self.backbone.bert.embeddings.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(x)
        token_type_embeddings = self.backbone.bert.embeddings.token_type_embeddings(token_type_ids)
        return token_embeddings + position_embeddings + token_type_embeddings

    def _encode_with_backbone(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        formula_embeddings: Optional[torch.Tensor] = None,
        formula_mask: Optional[torch.Tensor] = None,
        fingerprint_embeddings: Optional[torch.Tensor] = None,
        fingerprint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through BERT backbone with conditioning."""
        embeddings = self.backbone.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.backbone.bert.embeddings.dropout(embeddings)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        extended_attention_mask = self.backbone.get_extended_attention_mask(
            attention_mask, x.shape, x.device
        )
        
        encoder_kwargs = {}
        if formula_embeddings is not None and formula_mask is not None:
            encoder_kwargs['condition_embeddings'] = formula_embeddings
            encoder_kwargs['condition_mask'] = formula_mask
        if fingerprint_embeddings is not None and fingerprint_mask is not None:
            encoder_kwargs['fingerprint_embeddings'] = fingerprint_embeddings
            encoder_kwargs['fingerprint_mask'] = fingerprint_mask
        
        encoder_outputs = self.backbone.bert.encoder(
            embeddings, attention_mask=extended_attention_mask, **encoder_kwargs
        )
        sequence_output = encoder_outputs[0]
        logits = self.backbone.cls(sequence_output)
        return logits

    def compute_formula_loss(self, logits, gt_formula_vectors, attention_mask):
        """
        Compute differentiable loss between expected atom counts and ground truth formula.
        
        This loss penalizes the model based on the difference between the expected 
        number of atoms in the generated sequence and the ground truth atom counts.
        
        Args:
            logits: Model output logits of shape [batch, seq_len, vocab_size]
            gt_formula_vectors: Ground truth formula vectors of shape [batch, num_atom_types]
                               (unnormalized raw atom counts)
            attention_mask: Attention mask of shape [batch, seq_len]
        
        Returns:
            Scalar MSE loss between predicted and ground truth atom counts
        """
        # Compute token probabilities from logits
        probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Compute expected atom counts per position via matrix multiplication
        # probs: [batch, seq_len, vocab_size]
        # token_atom_counts: [vocab_size, num_atom_types]
        # Result: [batch, seq_len, num_atom_types]
        expected_counts_per_pos = torch.matmul(probs, self.token_atom_counts)
        
        # Apply attention mask to zero out padding token contributions
        # attention_mask: [batch, seq_len] -> [batch, seq_len, 1]
        if attention_mask is not None:
            masked_counts = expected_counts_per_pos * attention_mask.unsqueeze(-1)
        else:
            masked_counts = expected_counts_per_pos
        
        # Sum over sequence dimension to get total predicted atom counts
        # Result: [batch, num_atom_types]
        total_predicted_counts = masked_counts.sum(dim=1)
        
        # Compute MSE loss between predicted and ground truth
        loss = torch.nn.functional.mse_loss(total_predicted_counts, gt_formula_vectors)
        
        return loss

    def forward(
        self,
        x: torch.Tensor,
        fingerprint: torch.Tensor,
        formula: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through decoder with conditioning.
        
        Supports two cross-attention modes:
        1. Independent (default): Formula and fingerprint have separate cross-attention layers
        2. Shared (use_shared_cross_attention=True): Formula and fingerprint embeddings are
           concatenated and passed through a single set of cross-attention layers
        
        Args:
            x: Input token IDs [batch, seq_len]
            fingerprint: Predicted fingerprint from encoder [batch, fp_bits]
            formula: List of formula strings
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size = x.size(0)
        
        # Prepare conditioning
        formula_emb, formula_mask = self._prepare_formula_conditioning(formula, batch_size)
        fp_emb, fp_mask = self._prepare_fingerprint_conditioning(fingerprint)
        
        # Build token embeddings
        embeddings = self._build_token_embeddings(x)
        
        # Add global conditioning (for non-cross-attention types)
        if self.use_formula_conditioning and self.formula_conditioner_type != 'cross_attention' and formula_emb is not None:
            embeddings = embeddings + formula_emb.unsqueeze(1)
        
        if self.use_fingerprint_conditioning and self.fingerprint_conditioner_type != 'cross_attention' and fp_emb is not None:
            embeddings = embeddings + fp_emb.unsqueeze(1)
        
        # Prepare cross-attention inputs
        cross_formula_emb = formula_emb if self.formula_conditioner_type == 'cross_attention' else None
        cross_formula_mask = formula_mask if self.formula_conditioner_type == 'cross_attention' else None
        cross_fp_emb = fp_emb if self.fingerprint_conditioner_type == 'cross_attention' else None
        cross_fp_mask = fp_mask if self.fingerprint_conditioner_type == 'cross_attention' else None
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # Handle shared vs independent cross-attention modes
            if self.use_shared_cross_attention:
                # SHARED MODE: Concatenate formula and fingerprint embeddings
                condition_embeddings = cross_formula_emb
                condition_mask = cross_formula_mask
                
                if cross_fp_emb is not None:
                    if condition_embeddings is None:
                        condition_embeddings = cross_fp_emb
                        condition_mask = cross_fp_mask
                    else:
                        # Concatenate formula and fingerprint for joint cross-attention
                        condition_embeddings = torch.cat([condition_embeddings, cross_fp_emb], dim=1)
                        condition_mask = torch.cat([condition_mask, cross_fp_mask], dim=1)
                
                logits = self._encode_with_backbone(
                    embeddings, attention_mask, x,
                    formula_embeddings=condition_embeddings,
                    formula_mask=condition_mask,
                    fingerprint_embeddings=None,  # Not used in shared mode
                    fingerprint_mask=None,
                )
            else:
                # INDEPENDENT MODE: Pass formula and fingerprint separately
                logits = self._encode_with_backbone(
                    embeddings, attention_mask, x,
                    formula_embeddings=cross_formula_emb,
                    formula_mask=cross_formula_mask,
                    fingerprint_embeddings=cross_fp_emb,
                    fingerprint_mask=cross_fp_mask,
                )
        
        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Loss = MDLM cross-entropy loss on SAFE token prediction
        
        Args:
            batch: Batch containing spectra, formulas, and SAFE tokens
            batch_idx: Batch index
            
        Returns:
            MDLM loss
        """
        # Extract batch data
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        formula = batch['formula']
        
        # Encode spectra -> predicted fingerprint
        pred_fingerprint, _ = self._encode_spectra(batch)
        
        # Sample time for MDLM
        t = self.mdlm.sample_time(input_ids.shape[0])
        t = t.to(self.device)
        
        # Forward process: mask tokens
        xt = self.mdlm.forward_process(input_ids, t)
        
        # Forward through decoder with conditioning
        logits = self.forward(xt, pred_fingerprint, formula, attention_mask)
        
        # MDLM Loss
        if self.cfg.training.get('global_mean_loss', True):
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
        else:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()

        self.log('train_reconstruction_loss', loss.item(),
                on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        # Compute formula loss (differentiable expected atom count loss)
        formula_loss_weight = self.cfg.training.get('formula_loss_weight', 0.0)
        if formula is not None and formula_loss_weight > 0:
            encoder = self.formula_encoder
 
            # Encode ground truth formulas to vectors (unnormalized)
            gt_formula_vectors = encoder.encode_batch(
                formula, normalize='none'
            ).to(self.device)
            
            # Compute formula loss
            formula_loss = self.compute_formula_loss(logits, gt_formula_vectors, attention_mask)
            
            # Add weighted formula loss to total loss
            loss = loss + formula_loss_weight * formula_loss
            
            # Log formula loss
            self.log('train_formula_loss', formula_loss.item(),
                     on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        self.log(name='train_total_loss',
                 value=loss.item(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=False, sync_dist=True)
        
        # Log adaptive scale if using adaptive conditioner
        if self.use_formula_conditioning and hasattr(self.formula_conditioner, 'get_scale'):
            scale = self.formula_conditioner.get_scale()
            self.log('formula_scale', scale, prog_bar=False, sync_dist=True)
        if self.use_fingerprint_conditioning and hasattr(self.fingerprint_conditioner, 'get_scale'):
            fp_scale = self.fingerprint_conditioner.get_scale()
            self.log('fingerprint_scale', fp_scale, prog_bar=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        formula = batch['formula']
        
        pred_fingerprint, _ = self._encode_spectra(batch)
        
        t = self.mdlm.sample_time(input_ids.shape[0])
        t = t.to(self.device)
        
        xt = self.mdlm.forward_process(input_ids, t)
        logits = self.forward(xt, pred_fingerprint, formula, attention_mask)
        
        if self.cfg.training.get('global_mean_loss', True):
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
        else:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()
        
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self.validation_step(batch, batch_idx)


# ==================== Fine-tuning Utilities ====================

def apply_encoder_finetuning(model: Spec2MolModel, strategy: Optional[str]):
    """Apply fine-tuning strategy to encoder.
    
    Strategies:
    - freeze: Freeze all encoder parameters
    - ft-unfold: Only train unfold layers
    - freeze-unfold: Freeze unfold layers
    - ft-transformer: Only train transformer
    - freeze-transformer: Freeze transformer
    """
    if strategy is None:
        return
    
    if strategy == 'freeze':
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == 'ft-unfold':
        for name, param in model.encoder.named_parameters():
            layer = name.split('.')[1] if len(name.split('.')) > 1 else ''
            if layer != '2':
                param.requires_grad = False
    elif strategy == 'freeze-unfold':
        for name, param in model.encoder.named_parameters():
            layer = name.split('.')[1] if len(name.split('.')) > 1 else ''
            if layer == '2':
                param.requires_grad = False
    elif strategy == 'ft-transformer':
        for name, param in model.encoder.named_parameters():
            layer = name.split('.')[1] if len(name.split('.')) > 1 else ''
            if layer != '0':
                param.requires_grad = False
    elif strategy == 'freeze-transformer':
        for name, param in model.encoder.named_parameters():
            layer = name.split('.')[1] if len(name.split('.')) > 1 else ''
            if layer == '0':
                param.requires_grad = False
    else:
        raise NotImplementedError(f'Unknown encoder strategy: {strategy}')


def apply_decoder_finetuning(model: Spec2MolModel, strategy: Optional[str]):
    """Apply fine-tuning strategy to decoder.
    
    Strategies:
    - freeze: Freeze all decoder parameters
    - ft-input: Only train input embeddings
    - freeze-input: Freeze input embeddings
    - ft-transformer: Only train transformer layers
    - freeze-transformer: Freeze transformer layers
    """
    if strategy is None:
        return
    
    if strategy == 'freeze':
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif strategy == 'ft-input':
        for name, param in model.backbone.named_parameters():
            if 'embeddings' not in name:
                param.requires_grad = False
    elif strategy == 'freeze-input':
        for name, param in model.backbone.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
    elif strategy == 'ft-transformer':
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.bert.encoder.parameters():
            param.requires_grad = True
    elif strategy == 'freeze-transformer':
        for param in model.backbone.bert.encoder.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError(f'Unknown decoder strategy: {strategy}')


def load_weights(model: Spec2MolModel, path: str) -> Spec2MolModel:
    """Load weights from checkpoint into model."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
    
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logging.info(f"Loaded weights from {path}")
    logging.info(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    return model
