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


import itertools
import hydra.utils
import lightning as L
import torch
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from bionemo.moco.interpolants import MDLM
from bionemo.moco.distributions.time import UniformTimeDistribution
from dlm.utils.utils_moco import AntitheticUniformTimeDistribution
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

from dlm.utils.ema import ExponentialMovingAverage
from dlm.utils.utils_data import get_tokenizer
from dlm.utils.utils_save import clean_checkpoint, fast_forward_info
from dlm.utils.formula_encoder import FormulaEncoder
from dlm.model_components import (
    FormulaConditioner, 
    AdaptiveFormulaConditioner,
    FingerprintConditioner,
    AdaptiveFingerprintConditioner,
    CrossAttentionFormulaConditioner,
    CrossAttentionFingerprintConditioner
)
from dlm.bert_with_cross_attention import BertForMaskedLMWithCrossAttention

class DLM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # set up tokenizer
        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        
        # set up formula conditioning (if enabled)
        self.use_formula_conditioning = self.config.model.get('use_formula_conditioning', False)
        self.conditioner_type = self.config.model.get('formula_conditioner_type', 'basic')
        
        # Extract BERT config (exclude lr_scheduler which is not a BERT parameter)
        bert_config_dict = dict(self.config.model)
        bert_config_dict.pop('lr_scheduler', None)  # Remove lr_scheduler if present
        
        self._formula_uses_cross_attention = (
            self.use_formula_conditioning and self.conditioner_type == 'cross_attention'
        )

        if self.use_formula_conditioning:
            self.formula_encoder = FormulaEncoder(
                normalize=self.config.model.get('formula_normalize', 'none')
            )
            
            # Select conditioner type based on config
            if self.conditioner_type == 'cross_attention':
                # Create cross-attention conditioner
                self.formula_conditioner = CrossAttentionFormulaConditioner(
                    num_atom_types=self.formula_encoder.vocab_size,
                    hidden_size=self.config.model.hidden_size,
                    num_attention_heads=self.config.model.num_attention_heads,
                    num_layers=self.config.model.num_hidden_layers,
                    dropout=self.config.model.hidden_dropout_prob,
                    use_count_embedding=self.config.model.get('formula_use_count_embedding', True)
                )
            elif self.conditioner_type == 'adaptive':
                self.formula_conditioner = AdaptiveFormulaConditioner(
                    input_dim=self.formula_encoder.vocab_size,
                    hidden_dim=self.config.model.get('formula_hidden_dim', 256),
                    output_dim=self.config.model.hidden_size,
                    dropout=self.config.model.hidden_dropout_prob,
                    num_layers=self.config.model.get('formula_num_layers', 3),
                    learnable_scale=self.config.model.get('formula_learnable_scale', True),
                    initial_scale=self.config.model.get('formula_initial_scale', 1.0)
                )
            else:  # 'basic' or default
                self.formula_conditioner = FormulaConditioner(
                    input_dim=self.formula_encoder.vocab_size,
                    hidden_dim=self.config.model.get('formula_hidden_dim', 256),
                    output_dim=self.config.model.hidden_size,
                    dropout=self.config.model.hidden_dropout_prob,
                    num_layers=self.config.model.get('formula_num_layers', 3)
                )
            
            # Conditional dropout probability (for training robustness)
            self.formula_dropout_prob = self.config.model.get('formula_dropout_prob', 0.0)
        else:
            self.formula_encoder = FormulaEncoder(normalize='none')
            self.formula_conditioner = None
            self.formula_dropout_prob = 0.0
        
        # Set up token atom counts buffer for differentiable formula loss
        # This maps each token in vocabulary to its constituent atom counts
        self._setup_token_atom_counts()
        
        # set up fingerprint conditioning (if enabled)
        self.use_fingerprint_conditioning = self.config.model.get('use_fingerprint_conditioning', False)
        self.fingerprint_conditioner_type = self.config.model.get('fingerprint_conditioner_type', 'basic')
        self.fingerprint_bits = self.config.model.get('fingerprint_bits', 4096)
        self.fingerprint_seq_max_len = self.config.model.get('fingerprint_seq_max_len', 256)
        self.fingerprint_activation_threshold = self.config.model.get('fingerprint_activation_threshold', 0.0)
        self.fingerprint_use_value_projection = self.config.model.get('fingerprint_use_value_projection', True)
        self.use_shared_cross_attention = self.config.model.get('use_shared_cross_attention', True)

        if self.use_fingerprint_conditioning:
            fp_hidden_dim = self.config.model.get('fingerprint_hidden_dim', 512)
            fp_num_layers = self.config.model.get('fingerprint_num_layers', 3)
            fp_dropout = self.config.model.get('fingerprint_dropout_prob', 0.1)
            fp_learnable_scale = self.config.model.get('fingerprint_learnable_scale', True)
            fp_initial_scale = self.config.model.get('fingerprint_initial_scale', 1.0)

            if self.fingerprint_conditioner_type == 'adaptive':
                self.fingerprint_conditioner = AdaptiveFingerprintConditioner(
                    input_dim=self.fingerprint_bits,
                    hidden_dim=fp_hidden_dim,
                    output_dim=self.config.model.hidden_size,
                    dropout=self.config.model.hidden_dropout_prob,
                    num_layers=fp_num_layers,
                    learnable_scale=fp_learnable_scale,
                    initial_scale=fp_initial_scale
                )
            elif self.fingerprint_conditioner_type == 'cross_attention':
                fp_num_self_attn_layers = self.config.model.get('fingerprint_num_self_attention_layers', 2)
                # When using shared cross-attention, fingerprint conditioner doesn't create its own
                # cross-attention layers - they'll be shared with formula conditioner
                create_fp_cross_attn = not self.use_shared_cross_attention
                self.fingerprint_conditioner = CrossAttentionFingerprintConditioner(
                    num_bits=self.fingerprint_bits,
                    hidden_size=self.config.model.hidden_size,
                    num_attention_heads=self.config.model.num_attention_heads,
                    num_layers=self.config.model.num_hidden_layers,
                    max_seq_len=self.fingerprint_seq_max_len,
                    activation_threshold=self.fingerprint_activation_threshold,
                    dropout=self.config.model.hidden_dropout_prob,
                    num_self_attention_layers=fp_num_self_attn_layers,
                    create_cross_attention_layers=create_fp_cross_attn
                )
                self._fingerprint_uses_cross_attention = True
            else:
                self.fingerprint_conditioner = FingerprintConditioner(
                    input_dim=self.fingerprint_bits,
                    hidden_dim=fp_hidden_dim,
                    output_dim=self.config.model.hidden_size,
                    dropout=self.config.model.hidden_dropout_prob,
                    num_layers=fp_num_layers
                )

            self.fingerprint_dropout_prob = fp_dropout
        else:
            self.fingerprint_conditioner = None
            self.fingerprint_dropout_prob = 0.0
        
        self.fingerprint_flip_prob = self.config.model.get('fingerprint_flip_prob', 0.0)
        
        # Create unified backbone with cross-attention support
        # When both conditioners use cross-attention, we use shared cross-attention layers
        # and concatenate their embeddings at inference time
        self._setup_unified_backbone(bert_config_dict)
        
    def _setup_unified_backbone(self, bert_config_dict):
        """
        Set up the unified BERT backbone with cross-attention support.
        
        Supports two modes:
        1. Independent cross-attention (default): Formula and fingerprint have separate 
           cross-attention layers, allowing each modality to learn its own patterns.
        2. Shared cross-attention (use_shared_cross_attention=True): Formula and fingerprint
           embeddings are concatenated and passed through a single set of cross-attention
           layers.
        """
        use_formula_cross_attn = (
            self.use_formula_conditioning and 
            self.conditioner_type == 'cross_attention'
        )
        use_fp_cross_attn = (
            self.use_fingerprint_conditioning and 
            self.fingerprint_conditioner_type == 'cross_attention'
        )
        
        # Get cross-attention layers from conditioners (if they exist)
        formula_cross_attn_layers = None
        fp_cross_attn_layers = None
        
        if use_formula_cross_attn:
            formula_cross_attn_layers = self.formula_conditioner.cross_attention_layers
        
        # In shared mode, fingerprint uses the same cross-attention layers as formula
        # In independent mode, fingerprint has its own cross-attention layers
        if use_fp_cross_attn and not self.use_shared_cross_attention:
            fp_cross_attn_layers = self.fingerprint_conditioner.cross_attention_layers
        
        # Create backbone with cross-attention layers
        if formula_cross_attn_layers is not None or fp_cross_attn_layers is not None:
            self.backbone = BertForMaskedLMWithCrossAttention(
                BertConfig.from_dict(bert_config_dict),
                cross_attention_layers=formula_cross_attn_layers,
                fingerprint_cross_attention_layers=fp_cross_attn_layers,
                use_shared_cross_attention=self.use_shared_cross_attention
            )
        elif not hasattr(self, 'backbone'):
            # Neither uses cross-attention (standard BERT)
            self.backbone = BertForMaskedLM(BertConfig.from_dict(bert_config_dict))

        # set up mdlm
        if self.config.training.antithetic_sampling:
            time_distribution = AntitheticUniformTimeDistribution(sampling_eps = self.config.training.sampling_eps)
        else:
            time_distribution = UniformTimeDistribution()
        prior = DiscreteMaskedPrior(num_classes = self.tokenizer.vocab_size, mask_dim = self.mask_index)
        noise_schedule = LogLinearExpNoiseTransform()
        self.mdlm = MDLM(time_distribution=time_distribution,
                          prior_distribution=prior,
                          noise_schedule = noise_schedule)
        # set up ema
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        else:
            self.ema = None
    
    def _setup_token_atom_counts(self):
        """
        Pre-compute a static matrix mapping every token in the vocabulary to its 
        constituent atom counts. This is used for the differentiable formula loss.
        
        Creates a buffer 'token_atom_counts' of shape [vocab_size, num_atom_types]
        where each row contains the raw (unnormalized) atom counts for that token.
        """
        # Ensure we have a formula encoder for parsing tokens
        if self.formula_encoder is not None:
            encoder = self.formula_encoder
        else:
            # Create a temporary encoder if formula conditioning is disabled
            encoder = FormulaEncoder(normalize='none')
        
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
                    print(f"[DLM] Loading checkpoint with missing buffers (safe to skip): {missing_keys & safe_missing_buffers}")
                return super().load_state_dict(state_dict, strict=False, assign=assign)
        else:
            return super().load_state_dict(state_dict, strict=False, assign=assign)

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        # Store random state for later restoration in on_train_start
        self._checkpoint_random_state = checkpoint.get('sampler', {}).get('random_state', None)
        
    def on_save_checkpoint(self, checkpoint):
        import random
        import numpy as np
        
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        
        # Save comprehensive random state for reproducibility
        checkpoint['sampler']['random_state'] = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        
        # Also save sampler state if available
        if hasattr(self.trainer.train_dataloader, 'sampler') and hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['sampler_state'] = sampler_state_dict

    def on_after_backward(self):
        if self.global_step % 20 == 0:
            total_norm = torch.linalg.vector_norm(
                torch.stack([
                    torch.linalg.vector_norm(p.grad.detach())
                    for p in self.parameters()
                    if p.grad is not None
                ])
            )
            self.log('grad_norm', total_norm, prog_bar=False, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        # Collect all parameters to optimize
        params = list(self.backbone.parameters())
        if self.use_formula_conditioning and self.formula_conditioner is not None:
            params.extend(list(self.formula_conditioner.parameters()))
        if self.use_fingerprint_conditioning and self.fingerprint_conditioner is not None:
            params.extend(list(self.fingerprint_conditioner.parameters()))
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        # Check if lr_scheduler is configured in model config
        if hasattr(self.config.model, 'lr_scheduler'):
            scheduler_config = self.config.model.lr_scheduler
            
            # Check if warmup is requested
            warmup_steps = getattr(scheduler_config, 'warmup_steps', None)
            if warmup_steps is not None:
                # Use SequentialLR: Linear warmup + CosineAnnealingLR
                from torch.optim.lr_scheduler import SequentialLR, LinearLR
                
                # Create the main scheduler (cosine annealing) without warmup_steps
                main_scheduler_config = {
                    '_target_': scheduler_config._target_,
                    'T_max': scheduler_config.T_max,
                    'eta_min': scheduler_config.eta_min,
                    'optimizer': optimizer
                }
                main_scheduler = hydra.utils.instantiate(main_scheduler_config)
                
                # Create warmup scheduler (linear from 0 to initial LR)
                warmup_scheduler = LinearLR(
                    optimizer, 
                    start_factor=1e-6,  # Start very small to avoid division by zero
                    end_factor=1.0, 
                    total_iters=warmup_steps
                )
                
                # Combine them sequentially
                scheduler = SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, main_scheduler], 
                    milestones=[warmup_steps]
                )
            else:
                # No warmup, use scheduler directly
                scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
            
            scheduler_dict = {
                'scheduler': scheduler,
                'interval': 'step',
                'name': 'lr'
            }
            return [optimizer], [scheduler_dict]
        else:
            # Default warmup scheduler
            scheduler = hydra.utils.instantiate(
                {'_target_': 'transformers.get_constant_schedule_with_warmup',
                 'num_warmup_steps': 2500},
                 optimizer=optimizer)
            scheduler_dict = {
                'scheduler': scheduler,
                'interval': 'step',
                'name': 'lr'}
            return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

        # Restore optimizer state if doing partial resume (weights + optimizer, no callbacks)
        if hasattr(self, '_resume_optimizer_state') and self._resume_optimizer_state is not None:
            print("[DLM] Restoring optimizer state from partial resume...")
            optimizers = self.trainer.optimizers
            if optimizers and len(self._resume_optimizer_state) > 0:
                for opt_idx, opt_state in enumerate(self._resume_optimizer_state):
                    if opt_idx < len(optimizers):
                        try:
                            optimizers[opt_idx].load_state_dict(opt_state)
                            print(f"[DLM] Restored optimizer {opt_idx} state")
                        except Exception as e:
                            print(f"[DLM] Warning: Could not restore optimizer {opt_idx} state: {e}")
            del self._resume_optimizer_state
        
        # Restore global step if doing partial resume
        if hasattr(self, '_resume_global_step') and self._resume_global_step is not None:
            # Note: This sets the step counter for logging purposes
            # The dataloader skip handles actual data position
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed = self._resume_global_step
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_step_progress.total.completed = self._resume_global_step
            print(f"[DLM] Set global step to {self._resume_global_step}")
            del self._resume_global_step

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
            self.ema.update(itertools.chain(self.backbone.parameters()))
        
    def _build_token_embeddings(self, x):
        token_embeddings = self.backbone.bert.embeddings.word_embeddings(x)
        position_embeddings = self.backbone.bert.embeddings.position_embeddings(
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        )
        token_type_ids = torch.zeros_like(x)
        token_type_embeddings = self.backbone.bert.embeddings.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        return embeddings

    def _encode_with_backbone(self, embeddings, attention_mask, x,
                              condition_embeddings=None, condition_mask=None,
                              fingerprint_embeddings=None, fingerprint_mask=None):
        """
        Encode through backbone with optional formula and fingerprint conditioning.
        
        Args:
            embeddings: Token embeddings
            attention_mask: Self-attention mask
            x: Input token IDs (for shape reference)
            condition_embeddings: Formula conditioning embeddings
            condition_mask: Formula conditioning mask
            fingerprint_embeddings: Fingerprint conditioning embeddings (independent)
            fingerprint_mask: Fingerprint conditioning mask (independent)
        """
        embeddings = self.backbone.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.backbone.bert.embeddings.dropout(embeddings)
        if attention_mask is None:
            attention_mask = torch.ones_like(x)
        extended_attention_mask = self.backbone.get_extended_attention_mask(
            attention_mask, x.shape, x.device
        )
        encoder_kwargs = {}
        # Pass formula conditioning if provided
        if condition_embeddings is not None and condition_mask is not None:
            encoder_kwargs['condition_embeddings'] = condition_embeddings
            encoder_kwargs['condition_mask'] = condition_mask
        # Pass fingerprint conditioning if provided (independent from formula)
        if fingerprint_embeddings is not None and fingerprint_mask is not None:
            encoder_kwargs['fingerprint_embeddings'] = fingerprint_embeddings
            encoder_kwargs['fingerprint_mask'] = fingerprint_mask
        encoder_outputs = self.backbone.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            **encoder_kwargs,
        )
        sequence_output = encoder_outputs[0]
        logits = self.backbone.cls(sequence_output)
        return logits

    def _prepare_formula_global_embeddings(self, formula, x):
        if not self.use_formula_conditioning or self.conditioner_type == 'cross_attention':
            return None
        if formula is not None:
            formula_vectors = self.formula_encoder.encode_batch(formula).to(x.device)
            use_conditioning = True
        else:
            dummy_formula = ["C"] * x.size(0)
            formula_vectors = self.formula_encoder.encode_batch(dummy_formula).to(x.device)
            use_conditioning = False
        formula_embeddings = self.formula_conditioner(formula_vectors)
        if self.training and formula is not None:
            if torch.rand(1).item() < self.formula_dropout_prob:
                use_conditioning = False
        if not use_conditioning:
            formula_embeddings = formula_embeddings * 0.0
        return formula_embeddings

    def _prepare_formula_sequence_embeddings(self, formula, x):
        if not self.use_formula_conditioning or self.conditioner_type != 'cross_attention':
            return None, None
        if formula is not None:
            formula_vectors = self.formula_encoder.encode_batch(formula).to(x.device)
            use_conditioning = True
        else:
            dummy_formula = ["C"] * x.size(0)
            formula_vectors = self.formula_encoder.encode_batch(dummy_formula).to(x.device)
            use_conditioning = False
        formula_embeddings, formula_mask = self.formula_conditioner.encode_formula(formula_vectors)
        if self.training and formula is not None:
            if torch.rand(1).item() < self.formula_dropout_prob:
                use_conditioning = False
        if not use_conditioning:
            formula_embeddings = formula_embeddings * 0.0
        return formula_embeddings, formula_mask

    def _prepare_fingerprint_sequence_embeddings(self, fingerprint, x):
        if not self.use_fingerprint_conditioning or self.fingerprint_conditioner_type != 'cross_attention':
            return None, None

        if fingerprint is not None:
            fp_vectors = self._standardize_fingerprint_input(fingerprint, x.size(0), x.device)
            use_conditioning = True
        else:
            fp_vectors = torch.zeros(x.size(0), self.fingerprint_bits, device=x.device, dtype=torch.float32)
            use_conditioning = False

        embeddings, mask = self.fingerprint_conditioner.encode_fingerprint(fp_vectors)

        if self.training and use_conditioning:
            if torch.rand(1).item() < self.fingerprint_dropout_prob:
                use_conditioning = False

        if not use_conditioning:
            embeddings = embeddings * 0.0

        return embeddings, mask

    def _standardize_fingerprint_input(self, fingerprint, batch_size, device):
        if fingerprint is None:
            return torch.zeros(batch_size, self.fingerprint_bits, device=device, dtype=torch.float32)
        if torch.is_tensor(fingerprint):
            fp = fingerprint.to(device=device, dtype=torch.float32)
        else:
            fp = torch.as_tensor(fingerprint, device=device, dtype=torch.float32)
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        if fp.size(-1) != self.fingerprint_bits:
            raise ValueError(f"Expected fingerprint dim {self.fingerprint_bits}, got {fp.size(-1)}")
        if fp.size(0) == 1 and batch_size > 1:
            fp = fp.expand(batch_size, -1)
        elif fp.size(0) != batch_size:
            raise ValueError(f"Fingerprint batch mismatch: {fp.size(0)} vs {batch_size}")
        return fp

    def _standardize_fingerprint_mask(self, fingerprint_mask, batch_size, device):
        if fingerprint_mask is None:
            return torch.ones(batch_size, device=device, dtype=torch.float32)
        if torch.is_tensor(fingerprint_mask):
            mask = fingerprint_mask.to(device=device, dtype=torch.float32)
        else:
            mask = torch.as_tensor(fingerprint_mask, device=device, dtype=torch.float32)
        if mask.dim() == 0:
            mask = mask.repeat(batch_size)
        if mask.dim() != 1:
            raise ValueError("Fingerprint mask must be 1D.")
        if mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size)
        elif mask.size(0) != batch_size:
            raise ValueError(f"Fingerprint mask batch mismatch: {mask.size(0)} vs {batch_size}")
        return mask.clamp(0.0, 1.0)

    def _prepare_fingerprint_embeddings(self, fingerprint, fingerprint_mask, x):
        if not self.use_fingerprint_conditioning:
            return None
        fp_vectors = self._standardize_fingerprint_input(fingerprint, x.size(0), x.device)
        embeddings = self.fingerprint_conditioner(fp_vectors)
        if fingerprint is None:
            mask = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
        else:
            mask = self._standardize_fingerprint_mask(fingerprint_mask, x.size(0), x.device)
        if self.training and torch.any(mask > 0):
            if torch.rand(1).item() < self.fingerprint_dropout_prob:
                mask = torch.zeros_like(mask)
        embeddings = embeddings * mask.unsqueeze(1)
        return embeddings
    
    def _apply_fingerprint_flip(self, fingerprint, device):
        if fingerprint is None or self.fingerprint_flip_prob <= 0:
            return fingerprint
        if torch.is_tensor(fingerprint):
            fp_tensor = fingerprint.to(device=device, dtype=torch.float32)
        else:
            fp_tensor = torch.as_tensor(fingerprint, device=device, dtype=torch.float32)
        flip_mask = torch.rand_like(fp_tensor) < self.fingerprint_flip_prob
        flipped = torch.where(flip_mask, 1.0 - fp_tensor, fp_tensor)
        return flipped

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

    def forward(self, x, attention_mask=None, formula=None, fingerprint=None, fingerprint_mask=None):
        """
        Forward pass through the model with optional formula and fingerprint conditioning.
        
        Supports two cross-attention modes:
        1. Independent (default): Formula and fingerprint have separate cross-attention layers
        2. Shared (use_shared_cross_attention=True): Formula and fingerprint embeddings are
           concatenated and passed through a single set of cross-attention layers
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            formula: Optional list of molecular formula strings or None
                    If provided, conditions the generation on these formulas
            fingerprint: Optional fingerprint tensors or arrays
            fingerprint_mask: Optional mask for fingerprint conditioning
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # Prepare formula conditioning (if using cross-attention)
            formula_cond_embeddings = None
            formula_cond_mask = None
            if self.use_formula_conditioning and self.conditioner_type == 'cross_attention':
                formula_cond_embeddings, formula_cond_mask = self._prepare_formula_sequence_embeddings(formula, x)

            # Prepare fingerprint conditioning (if using cross-attention) - INDEPENDENT from formula
            fp_cond_embeddings = None
            fp_cond_mask = None
            if self.use_fingerprint_conditioning and self.fingerprint_conditioner_type == 'cross_attention':
                fp_cond_embeddings, fp_cond_mask = self._prepare_fingerprint_sequence_embeddings(fingerprint, x)

            # If any cross-attention conditioning is used
            if formula_cond_embeddings is not None or fp_cond_embeddings is not None:
                embeddings = self._build_token_embeddings(x)

                # Add global formula embeddings (if not using cross-attention for formula)
                if self.use_formula_conditioning and self.conditioner_type != 'cross_attention':
                    formula_global_embeddings = self._prepare_formula_global_embeddings(formula, x)
                    if formula_global_embeddings is not None:
                        embeddings = embeddings + formula_global_embeddings.unsqueeze(1)

                # Add global fingerprint embeddings (if not using cross-attention for fingerprint)
                if self.use_fingerprint_conditioning and self.fingerprint_conditioner_type != 'cross_attention':
                    fingerprint_global_embeddings = self._prepare_fingerprint_embeddings(fingerprint, fingerprint_mask, x)
                    if fingerprint_global_embeddings is not None:
                        embeddings = embeddings + fingerprint_global_embeddings.unsqueeze(1)

                # Handle shared vs independent cross-attention modes
                if self.use_shared_cross_attention:
                    # SHARED MODE: Concatenate formula and fingerprint embeddings
                    condition_embeddings = formula_cond_embeddings
                    condition_mask = formula_cond_mask
                    
                    if fp_cond_embeddings is not None:
                        if condition_embeddings is None:
                            condition_embeddings = fp_cond_embeddings
                            condition_mask = fp_cond_mask
                        else:
                            # Concatenate formula and fingerprint for joint cross-attention
                            condition_embeddings = torch.cat([condition_embeddings, fp_cond_embeddings], dim=1)
                            condition_mask = torch.cat([condition_mask, fp_cond_mask], dim=1)
                    
                    logits = self._encode_with_backbone(
                        embeddings,
                        attention_mask,
                        x,
                        condition_embeddings=condition_embeddings,
                        condition_mask=condition_mask,
                        fingerprint_embeddings=None,  # Not used in shared mode
                        fingerprint_mask=None
                    )
                else:
                    # INDEPENDENT MODE: Pass formula and fingerprint separately
                    logits = self._encode_with_backbone(
                        embeddings,
                        attention_mask,
                        x,
                        condition_embeddings=formula_cond_embeddings,
                        condition_mask=formula_cond_mask,
                        fingerprint_embeddings=fp_cond_embeddings,
                        fingerprint_mask=fp_cond_mask
                    )
                return logits

            # Non-cross-attention conditioning path
            if self.use_formula_conditioning or self.use_fingerprint_conditioning:
                embeddings = self._build_token_embeddings(x)

                if self.use_formula_conditioning:
                    formula_global_embeddings = self._prepare_formula_global_embeddings(formula, x)
                    if formula_global_embeddings is not None:
                        embeddings = embeddings + formula_global_embeddings.unsqueeze(1)

                fingerprint_global_embeddings = self._prepare_fingerprint_embeddings(fingerprint, fingerprint_mask, x)
                if fingerprint_global_embeddings is not None:
                    embeddings = embeddings + fingerprint_global_embeddings.unsqueeze(1)

                logits = self._encode_with_backbone(embeddings, attention_mask, x)
                return logits

            # Standard forward pass without conditioning
            return self.backbone(x, attention_mask)['logits']
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Extract conditioning inputs
        formula = batch.get('formula', None) if self.use_formula_conditioning else None
        fingerprint = batch.get('fingerprint', None) if self.use_fingerprint_conditioning else None
        fingerprint_mask = batch.get('fingerprint_mask', None) if self.use_fingerprint_conditioning else None
        
        # Handle sample exclusion (e.g., test set molecules)
        # If exclude_mask is present, zero out attention_mask for excluded samples
        exclude_mask = batch.get('exclude_mask', None)
        if exclude_mask is not None:
            exclude_mask = exclude_mask.to(input_ids.device)
            # exclude_mask: [batch_size], 0.0 = exclude, 1.0 = keep
            # Apply to attention_mask: [batch_size, seq_len]
            attention_mask = attention_mask * exclude_mask.unsqueeze(1)
        
        if self.training and self.use_fingerprint_conditioning and self.fingerprint_flip_prob > 0:
            fingerprint = self._apply_fingerprint_flip(fingerprint, input_ids.device)
        
        # Sample time for diffusion
        t = self.mdlm.sample_time(input_ids.shape[0])
        t = t.to(input_ids.device)

        # Forward process to add mask tokens
        xt = self.mdlm.forward_process(input_ids, t)
        
        # Forward model pass with conditioning
        with torch.amp.autocast('cuda', dtype=torch.float32):
            logits = self.forward(xt, attention_mask, formula=formula, fingerprint=fingerprint, fingerprint_mask=fingerprint_mask)
        
        # Compute loss
        if self.config.training.global_mean_loss:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
        else:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()

        self.log('train_reconstruction_loss', loss.item(),
                on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        # Compute formula loss (differentiable expected atom count loss)
        formula_loss_weight = self.config.training.get('formula_loss_weight', 0.0)
        if batch.get('formula', None) is not None:
            encoder = self.formula_encoder
 
            # Encode ground truth formulas to vectors (unnormalized)
            gt_formula_vectors = encoder.encode_batch(
                batch['formula'], normalize='none'
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

        # Log current learning rate
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
