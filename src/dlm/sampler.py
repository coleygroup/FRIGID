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
Sampler module for DLM molecule generation.

Supports:
- De novo generation
- Formula-conditioned generation  
- Fingerprint-conditioned generation
- Unified (formula + fingerprint) conditioned generation
- Length prediction using Gradient Boosting model
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import re
import json
import itertools
import pickle
import torch
import random
import numpy as np
from typing import List, Optional, Union, Dict, Any
from collections import defaultdict
import safe as sf
from rdkit import Chem
from dlm.utils.utils_chem import safe_to_smiles, filter_by_substructure, mix_sequences, Slicer
from dlm.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from dlm.model import DLM


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Element pattern for formula parsing
ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')


def _load_checkpoint_with_fallback(path: str, map_location, **kwargs) -> 'DLM':
    """
    Load a DLM checkpoint, trying strict=True first and falling back to strict=False.
    
    Args:
        path: Path to the checkpoint file
        map_location: Device to load the checkpoint to
        **kwargs: Additional arguments to pass to load_from_checkpoint
        
    Returns:
        Loaded DLM model
    """
    try:
        return DLM.load_from_checkpoint(path, map_location=map_location, strict=True, **kwargs)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e) or "Unexpected key(s) in state_dict" in str(e):
            print(f"[load_checkpoint] Strict loading failed: {e}")
            print("[load_checkpoint] Retrying with strict=False...")
            return DLM.load_from_checkpoint(path, map_location=map_location, strict=False, **kwargs)
        else:
            raise


def load_model_from_path(path, config_overrides: Optional[Dict[str, Any]] = None):
    """
    Load DLM checkpoint and place it on the appropriate device.
    
    Args:
        path: Path to the checkpoint file
        config_overrides: Optional dict of config overrides to apply before loading.
                         Useful for loading checkpoints from different branches
                         (e.g., use_shared_cross_attention=True)
    
    Returns:
        Loaded DLM model
    """
    target_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load checkpoint to inspect/modify hyperparameters if needed
    if config_overrides:
        # weights_only=False needed for OmegaConf configs in checkpoints (PyTorch 2.6+)
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            config = checkpoint['hyper_parameters']['config']
            # Apply model config overrides using OmegaConf to handle struct mode
            from omegaconf import OmegaConf, open_dict
            with open_dict(config.model):
                for key, value in config_overrides.items():
                    config.model[key] = value
            checkpoint['hyper_parameters']['config'] = config
        
        # Handle state_dict key remapping for shared cross-attention mode
        # In unified branch with shared mode: cross-attention layers stored in backbone only
        if config_overrides.get('use_shared_cross_attention', False):
            state_dict = checkpoint.get('state_dict', {})
            keys_to_remove = []
            
            # Find and remove fingerprint_conditioner.cross_attention_layers keys
            # (they're duplicates of backbone cross-attention in shared mode)
            for key in list(state_dict.keys()):
                if 'fingerprint_conditioner.cross_attention_layers' in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del state_dict[key]
            
            if keys_to_remove:
                print(f"[load_model] Removed {len(keys_to_remove)} fingerprint_conditioner.cross_attention_layers keys for shared mode")
            
            checkpoint['state_dict'] = state_dict
            
        # Save modified checkpoint to temp file for loading
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        model = _load_checkpoint_with_fallback(temp_path, map_location=target_device)
        os.unlink(temp_path)
    else:
        model = _load_checkpoint_with_fallback(path, map_location=target_device)
    
    model = model.to(target_device)
    model.backbone.eval()
    if model.ema:
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))
    return model


class Sampler:
    """
    DLM Sampler for molecule generation with various conditioning modes.
    
    Supports:
    - De novo (unconditional) generation
    - Formula-conditioned generation
    - Fingerprint-conditioned generation
    - Unified (formula + fingerprint) conditioned generation
    """
    
    def __init__(self, path, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the sampler.
        
        Args:
            path: Path to the DLM checkpoint
            config_overrides: Optional dict of config overrides. Useful for loading
                            checkpoints from different branches. Example:
                            {'use_shared_cross_attention': True}
        """
        self.model = load_model_from_path(path, config_overrides=config_overrides)
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer('.')['input_ids'][1]
        self.pad_index = self.model.tokenizer.pad_token_id
        self.mdlm = self.model.mdlm
        self.mdlm.to_device(self.model.device)
        self.fingerprint_bits = self.model.config.model.get('fingerprint_bits', 4096)
        self.max_position_embeddings = self.model.config.model.get('max_position_embeddings', 256)

        # Fallback length distribution
        try:
            with open(os.path.join(ROOT_DIR, 'data/len.pk'), 'rb') as f:
                self.seq_len_list = pickle.load(f)
        except FileNotFoundError:
            print("[Sampler] Warning: data/len.pk not found. Using default length range.")
            self.seq_len_list = list(range(20, 100))
        
        # Length predictor (Gradient Boosting model)
        self.token_count_model = None
        self.token_count_features = None
        self._try_load_default_length_predictor()
    
    def _try_load_default_length_predictor(self):
        """Try to load the default GB length predictor if available."""
        default_model_path = os.path.join(ROOT_DIR, 'gb_token_full_model.joblib')
        default_features_path = os.path.join(ROOT_DIR, 'gb_token_full_model_features.json')
        
        if os.path.exists(default_model_path) and os.path.exists(default_features_path):
            try:
                self.load_length_predictor(default_model_path, default_features_path)
            except Exception as e:
                print(f"[Sampler] Warning: Failed to load default length predictor: {e}")
    
    def load_length_predictor(self, model_path: str, features_path: str = None):
        """
        Load Gradient Boosting length predictor model.
        
        Args:
            model_path: Path to .joblib model file
            features_path: Path to features JSON file (auto-detected if None)
        """
        try:
            import joblib
        except ImportError:
            print("[Sampler] Warning: joblib not installed. Length predictor disabled.")
            return
        
        if not os.path.exists(model_path):
            print(f"[Sampler] Warning: Model file not found: {model_path}")
            return
        
        # Auto-detect features path
        if features_path is None:
            features_path = model_path.replace('.joblib', '_features.json')
        
        self.token_count_model = joblib.load(model_path)
        print(f"[Sampler] Loaded length predictor from {model_path}")
        
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.token_count_features = json.load(f)
            print(f"[Sampler] Loaded {len(self.token_count_features)} features")
        else:
            # Try to get features from model
            self.token_count_features = getattr(self.token_count_model, 'feature_names_in_', None)
            if self.token_count_features is not None:
                self.token_count_features = list(self.token_count_features)
                print(f"[Sampler] Using model's feature_names_in_")
            else:
                print(f"[Sampler] Warning: Features file not found: {features_path}")
    
    def predict_token_length(self, formula: str) -> Optional[int]:
        """
        Predict token sequence length from molecular formula.
        
        Args:
            formula: Molecular formula string (e.g., "C6H12O6")
            
        Returns:
            Predicted token length or None if prediction fails
        """
        if self.token_count_model is None or self.token_count_features is None:
            return None
        
        # Parse formula into atom counts
        atoms = defaultdict(int)
        try:
            for element, count in ELEMENT_PATTERN.findall(formula):
                if element:
                    count = int(count) if count else 1
                    atoms[element] += count
        except Exception:
            return None
        
        # Build feature vector
        vector = [atoms.get(f, 0) for f in self.token_count_features]
        
        try:
            pred = self.token_count_model.predict([vector])[0]
            return int(round(pred))
        except Exception:
            return None
        
    @torch.no_grad()
    def generate(self, x, softmax_temp=1.2, randomness=2, fix=True, gamma=0, w=2, formula=None, fingerprint=None):
        """
        Generate molecules from masked input.
        
        Args:
            x: Input token IDs
            softmax_temp: Temperature for sampling
            randomness: Randomness factor for confidence sampling
            fix: Whether to fix invalid SMILES
            gamma: Gamma parameter for molecular context guidance
            w: Weight parameter for molecular context guidance
            formula: Optional molecular formula for conditioning (string or list of strings)
            fingerprint: Optional fingerprint vector(s) aligned with batch
        
        Returns:
            List of generated SMILES strings
        """
        x = x.to(self.model.device)
        num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_index
        
        # Convert single formula string to list if needed
        if formula is not None and isinstance(formula, str):
            formula = [formula] * x.size(0)

        fingerprint_tensor = None
        fingerprint_mask = None
        if fingerprint is not None:
            if not self.model.use_fingerprint_conditioning:
                raise RuntimeError(
                    "Fingerprint conditioning requested but the loaded checkpoint "
                    "was trained without fingerprint conditioning."
                )
            fingerprint_tensor = self._prepare_fingerprint_batch(fingerprint, x.size(0))
            fingerprint_mask = torch.ones(
                fingerprint_tensor.size(0),
                device=fingerprint_tensor.device,
                dtype=torch.float32
            )
        
        for i in range(num_steps):
            # Use model's forward method which supports formula conditioning
            logits = self.model(
                x,
                attention_mask,
                formula=formula,
                fingerprint=fingerprint_tensor,
                fingerprint_mask=fingerprint_mask
            )

            if gamma and w:
                x_poor = x.clone()
                context_tokens = (x_poor[0] != self.model.bos_index).to(int) * \
                    (x_poor[0] != self.model.eos_index).to(int) * \
                    (x_poor[0] != self.model.mask_index).to(int) * \
                    (x_poor[0] != self.pad_index).to(int)
                context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                # mask 100 * gamma % of the context (given fragments) tokens
                num_mask_poor = int(context_tokens.sum() * gamma)
                mask_idx_poor = random.sample(context_token_ids, num_mask_poor)
                x_poor[:, mask_idx_poor] = self.model.mask_index
                logits_poor = self.model(
                    x_poor,
                    attention_mask=attention_mask,
                    formula=formula,
                    fingerprint=fingerprint_tensor,
                    fingerprint_mask=fingerprint_mask
                )
                logits = w * logits + (1 - w) * logits_poor

            x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)
            
        # decode to SAFE strings
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        # convert to SMILES strings
        if self.model.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        # remove None and take the largest
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _insert_mask(
        self,
        x,
        num_samples,
        min_add_len=18,
        target_lengths: Optional[List[int]] = None,
        sequence_lengths: Optional[Union[List[int], int]] = None, # Alias for backward compatibility
    ):
        """
        Insert mask tokens into the input sequence.

        Args:
            x: Input tensor of shape (1, seq_len) with [BOS ... EOS]
            num_samples: Number of samples to generate
            min_add_len: Minimum number of mask tokens to add
            target_lengths: Optional list of target total lengths for each sample.
                           If provided, masks are inserted to achieve these lengths.
                           If None, lengths are sampled from len.pk distribution.
            sequence_lengths: Alias for target_lengths

        Returns:
            Batched tensor of shape (num_samples, max_len) with masks inserted
        """
        # Handle alias
        if target_lengths is None and sequence_lengths is not None:
            if isinstance(sequence_lengths, int):
                target_lengths = [sequence_lengths] * num_samples
            else:
                target_lengths = sequence_lengths

        if target_lengths is None:
            # Fallback to random sampling from len.pk
            # Use self.seq_len_list which was loaded in __init__
            target_lengths = [
                max(random.choice(self.seq_len_list), len(x[0]) + min_add_len)
                for _ in range(num_samples)
            ]
        elif len(target_lengths) == 1 and num_samples > 1:
            # Broadcast single target to all samples
            target_lengths = target_lengths * num_samples

        x = x[0]  # Remove batch dimension
        current_len = len(x)  # Includes BOS and EOS
        x_new = []

        for i in range(num_samples):
            target_len = target_lengths[i] if i < len(target_lengths) else target_lengths[-1]
            # Number of masks = target_length - current_length (BOS + EOS are already counted)
            add_seq_len = max(target_len - current_len, min_add_len)
            # Clamp to max position embeddings
            add_seq_len = min(add_seq_len, self.max_position_embeddings - current_len)
            add_seq_len = max(add_seq_len, min_add_len)

            x_new.append(torch.hstack([
                x[:-1],  # Everything except EOS
                torch.full((add_seq_len,), self.model.mask_index),
                x[-1:],  # EOS
            ]))

        pad_len = max([len(xx) for xx in x_new])
        x_new = [
            torch.hstack([xx, torch.full((pad_len - len(xx),), self.pad_index)])
            for xx in x_new
        ]
        return torch.stack(x_new)
    
    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5):
        # Prepare Fully Masked Inputs
        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        x = self._insert_mask(x, num_samples, min_add_len=40)
        x = x.to(self.model.device)
        return self.generate(x, softmax_temp, randomness)
    
    @torch.no_grad()
    def formula_conditioned_generation(self, formula, num_samples=1, softmax_temp=0.8, randomness=0.5, sequence_lengths=None, min_add_len=40):
        """
        Generate molecules conditioned on a molecular formula.
        
        Args:
            formula: Molecular formula string (e.g., "C6H12O6")
            num_samples: Number of molecules to generate
            softmax_temp: Temperature for sampling
            randomness: Randomness factor
            sequence_lengths: Optional list of sequence lengths for each sample
        
        Returns:
            List of generated SMILES strings
        """
        # Temporarily enable formula conditioning during inference
        original_training = self.model.training
        self.model.eval()
        
        # Prepare fully masked inputs
        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        # Pass sequence_lengths to _insert_mask
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len, sequence_lengths=sequence_lengths)
        x = x.to(self.model.device)
        
        # Generate with formula conditioning
        # We need to modify the forward method to work during inference
        samples = self.generate(x, softmax_temp, randomness, formula=formula)
        
        # Restore training state
        if original_training:
            self.model.train()
        
        return samples

    @torch.no_grad()
    def fingerprint_conditioned_generation(
        self,
        fingerprint,
        num_samples=1,
        softmax_temp=0.8,
        randomness=0.5,
        target_lengths: Optional[List[int]] = None,
    ):
        """
        Generate molecules conditioned on a target fingerprint vector.

        Args:
            fingerprint: Target fingerprint array
            num_samples: Number of molecules to generate
            softmax_temp: Temperature for sampling
            randomness: Randomness factor
            target_lengths: Optional explicit target lengths.

        Returns:
            List of generated SMILES strings
        """
        original_training = self.model.training
        self.model.eval()

        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        x = self._insert_mask(x, num_samples, min_add_len=40, target_lengths=target_lengths)
        x = x.to(self.model.device)

        samples = self.generate(
            x,
            softmax_temp,
            randomness,
            fingerprint=fingerprint
        )

        if original_training:
            self.model.train()
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
        """
        Generate molecules with unified conditioning (formula + fingerprint).
        
        This method supports flexible conditioning:
        - Both formula and fingerprint provided: joint conditioning
        - Only formula: formula-conditioned generation
        - Only fingerprint: fingerprint-conditioned generation
        - Neither: de novo generation
        
        Args:
            formula: Molecular formula string (e.g., "C6H12O6")
            fingerprint: Target fingerprint array
            num_samples: Number of molecules to generate
            softmax_temp: Temperature for sampling
            randomness: Randomness factor
            target_lengths: Optional explicit target lengths per sample.
                           If None and formula is provided, uses length predictor.
            min_add_len: Minimum number of mask tokens to add
            
        Returns:
            List of generated SMILES strings
        """
        original_training = self.model.training
        self.model.eval()
        
        # Determine target lengths using length predictor if available
        if target_lengths is None and formula is not None:
            predicted_len = self.predict_token_length(formula)
            if predicted_len is not None:
                # Add some variance around predicted length
                target_lengths = [
                    max(min_add_len + 2, predicted_len + random.randint(-3, 3))
                    for _ in range(num_samples)
                ]
        
        # Prepare masked input
        x = torch.hstack([
            torch.full((1, 1), self.model.bos_index),
            torch.full((1, 1), self.model.eos_index)
        ])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len, target_lengths=target_lengths)
        x = x.to(self.model.device)
        
        # Generate with both conditioning signals
        samples = self.generate(
            x,
            softmax_temp,
            randomness,
            formula=formula,
            fingerprint=fingerprint
        )
        
        if original_training:
            self.model.train()
        
        return samples
    
    @torch.no_grad()
    def formula_and_fingerprint_conditioned_generation(
        self,
        formula: str,
        fingerprint: np.ndarray,
        num_samples: int = 1,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        sequence_lengths: Optional[List[int]] = None,
        min_add_len: int = 40,
    ) -> List[str]:
        """
        Generate molecules conditioned on both formula AND fingerprint.
        
        Args:
            formula: Molecular formula string (e.g., "C6H12O6")
            fingerprint: Target fingerprint array
            num_samples: Number of molecules to generate
            softmax_temp: Temperature for sampling
            randomness: Randomness factor
            sequence_lengths: Optional explicit target lengths per sample (alias for target_lengths)
            min_add_len: Minimum number of mask tokens to add
            
        Returns:
            List of generated SMILES strings
        """
        return self.unified_conditioned_generation(
            formula=formula,
            fingerprint=fingerprint,
            num_samples=num_samples,
            softmax_temp=softmax_temp,
            randomness=randomness,
            target_lengths=sequence_lengths,
            min_add_len=min_add_len,
        )
    
    def fragment_linking_onestep(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0):
        if self.model.config.training.get('use_bracket_safe'):
            encoded_fragment = BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        else:
            encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        
        x = self.model.tokenizer([encoded_fragment + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=30)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma)
        samples = filter_by_substructure(samples, fragment)
        return samples
    
    def fragment_linking(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0):
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        prefix, suffix = encoded_fragment.split('.')

        x = self.model.tokenizer([prefix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=30)
        prefix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma)

        x = self.model.tokenizer([suffix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=30)
        suffix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma)
        
        samples = filter_by_substructure(mix_sequences(prefix_samples, suffix_samples,
                                                      *fragment.split('.'), num_samples), fragment)
        return samples
        
    def fragment_completion(self, fragment, num_samples=1, apply_filter=True, softmax_temp=1.2, randomness=2, gamma=0):
        if '*' not in fragment:     # superstructure generation
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)
            
        encoded_fragment = sf.SAFEConverter(ignore_stereo=True).encoder(fragment, allow_empty=True) + '.'
        x = self.model.tokenizer([encoded_fragment],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma)

        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    def mask_modification(self, smiles, min_len=30, **kwargs):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
        x = self.model.tokenizer([encoded_smiles],
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.model.config.model.max_position_embeddings)['input_ids']
        if x.shape[-1] < min_len:
            return self.addmask(smiles, num_edit=min_len-x.shape[-1]+1, **kwargs)
        return self.remask(smiles, input_ids=x, **kwargs)

    def addmask(self, smiles, num_edit=3, **kwargs):
        try:
            samples = self.fragment_completion(smiles, mask_len=num_edit, apply_filter=False, **kwargs)
        except:
            return smiles
        if samples:
            return samples[0]
        return smiles
    
    def remask(self, smiles, input_ids=None, **kwargs):
        x = input_ids
        if x is None:
            encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
            x = self.model.tokenizer([encoded_smiles],
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=self.model.config.model.max_position_embeddings)['input_ids']
        
        # fragment mask replacement
        special_token_idx = [0] + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist() + [len(x[0]) - 1]
        frag_idx = random.randint(0, len(special_token_idx) - 2)
        mask_start_idx = special_token_idx[frag_idx] + 1
        mask_end_idx = special_token_idx[frag_idx + 1]
        num_insert_mask = random.randint(5, 15)
        num_insert_mask = min(num_insert_mask,
                              self.model.config.model.max_position_embeddings - x.shape[-1] + mask_end_idx - mask_start_idx)
        x = torch.hstack([x[:, :mask_start_idx],
                          torch.full((1, num_insert_mask), self.model.mask_index),
                          x[:, mask_end_idx:]])
        samples = self.generate(x, **kwargs)
        if samples:
            return samples[0]
        return smiles

    def _prepare_fingerprint_batch(self, fingerprint, batch_size):
        if torch.is_tensor(fingerprint):
            fp = fingerprint.detach().clone()
        else:
            fp = torch.as_tensor(fingerprint, dtype=torch.float32)
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        if fp.size(-1) != self.fingerprint_bits:
            raise ValueError(f"Expected fingerprint dim {self.fingerprint_bits}, got {fp.size(-1)}")
        if fp.size(0) == 1 and batch_size > 1:
            fp = fp.expand(batch_size, -1)
        elif fp.size(0) != batch_size:
            raise ValueError(f"Fingerprint batch mismatch: got {fp.size(0)}, expected {batch_size}")
        return fp.to(self.model.device)
