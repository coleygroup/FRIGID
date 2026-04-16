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


import torch
import torch.nn as nn
import math


class FormulaConditioner(nn.Module):
    """
    MLP that projects molecular formula vectors into conditioning embeddings.
    
    This module takes atom count vectors and projects them into the same
    dimensionality as the BERT hidden states, allowing them to be added
    to token embeddings for global conditioning.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 768,
                 dropout: float = 0.1,
                 num_layers: int = 3):
        """
        Initialize the FormulaConditioner.
        
        Args:
            input_dim: Size of input formula vector (number of atom types)
            hidden_dim: Size of hidden layers
            output_dim: Size of output embedding (should match BERT hidden_size)
            dropout: Dropout probability
            num_layers: Number of MLP layers (minimum 2)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values to start conditioning gently."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize with small weights
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, formula_vectors: torch.Tensor) -> torch.Tensor:
        """
        Project formula vectors to conditioning embeddings.
        
        Args:
            formula_vectors: Tensor of shape (batch_size, input_dim)
                            or (input_dim,) for single example
            
        Returns:
            Conditioning embeddings of shape (batch_size, output_dim)
            or (output_dim,) for single example
        """
        # Handle single example (no batch dimension)
        single_example = formula_vectors.dim() == 1
        if single_example:
            formula_vectors = formula_vectors.unsqueeze(0)
        
        # Project through MLP
        embeddings = self.mlp(formula_vectors)
        
        # Remove batch dimension if input was single example
        if single_example:
            embeddings = embeddings.squeeze(0)
        
        return embeddings


class FingerprintConditioner(FormulaConditioner):
    """
    Conditioner for molecular fingerprint vectors.
    
    Reuses the MLP structure from FormulaConditioner but defaults to
    higher hidden dimensions suitable for 4096-bit fingerprints.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 output_dim: int = 768,
                 dropout: float = 0.1,
                 num_layers: int = 3):
        super().__init__(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         dropout=dropout,
                         num_layers=num_layers)

    def _init_weights(self):
        """
        Initialize weights with special care for high-dimensional fingerprint inputs.
        Using zero initialization for the final layer ensures the conditioning 
        starts as a null operation, preventing shock to the backbone model.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module == self.mlp[-1]:
                    # Last layer: zero init to start with no effect
                    nn.init.zeros_(module.weight)
                else:
                    # Internal layers: Kaiming init for ReLUs
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    # Or keep small normal if preferred, but fan-in=4096 needs care
                    # nn.init.normal_(module.weight, mean=0.0, std=0.02) 
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class AdaptiveFormulaConditioner(nn.Module):
    """
    Advanced conditioner with learnable scaling and layer-specific conditioning.
    
    This variant allows the model to learn how much to weight the conditioning
    and can provide different conditioning vectors for different layers.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 768,
                 dropout: float = 0.1,
                 num_layers: int = 3,
                 learnable_scale: bool = True,
                 initial_scale: float = 1.0):
        """
        Initialize the AdaptiveFormulaConditioner.
        
        Args:
            input_dim: Size of input formula vector
            hidden_dim: Size of hidden layers
            output_dim: Size of output embedding
            dropout: Dropout probability
            num_layers: Number of MLP layers
            learnable_scale: If True, learn a scaling parameter
            initial_scale: Initial value for scaling parameter
        """
        super().__init__()
        
        # Base conditioner
        self.conditioner = FormulaConditioner(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers
        )
        
        # Learnable scaling parameter
        self.learnable_scale = learnable_scale
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer('scale', torch.tensor(initial_scale))
    
    def forward(self, formula_vectors: torch.Tensor) -> torch.Tensor:
        """
        Project and scale formula vectors.
        
        Args:
            formula_vectors: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Scaled conditioning embeddings of shape (batch_size, output_dim)
        """
        embeddings = self.conditioner(formula_vectors)
        
        # Apply scaling
        embeddings = embeddings * self.scale
        
        return embeddings
    
    def get_scale(self) -> float:
        """Get current value of the scaling parameter."""
        return self.scale.item()


class AdaptiveFingerprintConditioner(AdaptiveFormulaConditioner):
    """
    Adaptive variant for fingerprint conditioning, mirroring the formula path.
    It uses FingerprintConditioner internally to ensure correct initialization (zero-init last layer).
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 output_dim: int = 768,
                 dropout: float = 0.1,
                 num_layers: int = 3,
                 learnable_scale: bool = True,
                 initial_scale: float = 1.0):
        super().__init__(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         dropout=dropout,
                         num_layers=num_layers,
                         learnable_scale=learnable_scale,
                         initial_scale=initial_scale)
        
        # Override the conditioner with FingerprintConditioner
        # This ensures we get the specialized initialization (zero last layer, kaiming others)
        # rather than the generic FormulaConditioner initialization
        self.conditioner = FingerprintConditioner(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers
        )


class LayerwiseFormulaConditioner(nn.Module):
    """
    Conditioner that produces layer-specific conditioning embeddings.
    
    This allows different BERT layers to receive different conditioning signals,
    potentially learning to use formula information differently at different depths.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 768,
                 num_bert_layers: int = 12,
                 dropout: float = 0.1,
                 shared_base: bool = True):
        """
        Initialize the LayerwiseFormulaConditioner.
        
        Args:
            input_dim: Size of input formula vector
            hidden_dim: Size of hidden layers
            output_dim: Size of output embedding
            num_bert_layers: Number of BERT layers to condition
            dropout: Dropout probability
            shared_base: If True, share base MLP across layers
        """
        super().__init__()
        
        self.num_bert_layers = num_bert_layers
        self.shared_base = shared_base
        
        if shared_base:
            # Shared base MLP
            self.base_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Layer-specific output projections
            self.layer_projections = nn.ModuleList([
                nn.Linear(hidden_dim, output_dim)
                for _ in range(num_bert_layers)
            ])
        else:
            # Separate MLP for each layer
            self.layer_conditioners = nn.ModuleList([
                FormulaConditioner(input_dim, hidden_dim, output_dim, dropout)
                for _ in range(num_bert_layers)
            ])
    
    def forward(self, formula_vectors: torch.Tensor, 
                layer_idx: int = 0) -> torch.Tensor:
        """
        Project formula vectors with layer-specific conditioning.
        
        Args:
            formula_vectors: Tensor of shape (batch_size, input_dim)
            layer_idx: Which BERT layer to generate conditioning for
            
        Returns:
            Layer-specific conditioning embeddings
        """
        if self.shared_base:
            # Shared transformation
            hidden = self.base_mlp(formula_vectors)
            # Layer-specific projection
            embeddings = self.layer_projections[layer_idx](hidden)
        else:
            # Layer-specific conditioner
            embeddings = self.layer_conditioners[layer_idx](formula_vectors)
        
        return embeddings
    
    def forward_all_layers(self, formula_vectors: torch.Tensor) -> list:
        """
        Generate conditioning embeddings for all layers at once.
        
        Args:
            formula_vectors: Tensor of shape (batch_size, input_dim)
            
        Returns:
            List of conditioning embeddings, one per layer
        """
        if self.shared_base:
            hidden = self.base_mlp(formula_vectors)
            embeddings = [proj(hidden) for proj in self.layer_projections]
        else:
            embeddings = [cond(formula_vectors) for cond in self.layer_conditioners]
        
        return embeddings


class FormulaSequenceEncoder(nn.Module):
    """
    Encodes molecular formula as a sequence of atom type embeddings.
    
    This module treats the formula as a sequence where each position represents
    an atom type (C, H, O, N, etc.) and its count. This sequence can then be
    used for cross-attention with the SAFE token sequence.
    """
    
    def __init__(self,
                 num_atom_types: int = 30,
                 embedding_dim: int = 768,
                 max_count: int = 200,
                 use_count_embedding: bool = True):
        """
        Initialize the FormulaSequenceEncoder.
        
        Args:
            num_atom_types: Number of different atom types to support
            embedding_dim: Dimension of output embeddings
            max_count: Maximum count value for count embeddings
            use_count_embedding: If True, use separate embeddings for counts
        """
        super().__init__()
        
        self.num_atom_types = num_atom_types
        self.embedding_dim = embedding_dim
        self.use_count_embedding = use_count_embedding
        
        # Atom type embeddings (C, H, O, N, ...)
        self.atom_embeddings = nn.Embedding(num_atom_types, embedding_dim)
        
        # Count embeddings (1, 2, 3, ... counts)
        if use_count_embedding:
            self.count_embeddings = nn.Embedding(max_count + 1, embedding_dim)
        
        # Position embeddings for the formula sequence
        self.position_embeddings = nn.Embedding(num_atom_types, embedding_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.atom_embeddings.weight, mean=0.0, std=0.02)
        if self.use_count_embedding:
            nn.init.normal_(self.count_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, formula_vectors: torch.Tensor) -> tuple:
        """
        Encode formula vectors as a sequence of embeddings.
        
        Args:
            formula_vectors: Tensor of shape (batch_size, num_atom_types)
                           Contains count for each atom type
        
        Returns:
            Tuple of (embeddings, mask):
                embeddings: (batch_size, seq_len, embedding_dim)
                mask: (batch_size, seq_len) - 1 for valid positions, 0 for padding
        """
        batch_size = formula_vectors.shape[0]
        device = formula_vectors.device
        
        # Create sequence by selecting non-zero atoms
        # For simplicity, we'll create a fixed-length sequence with all atom types
        # and mask out the ones with zero count
        
        seq_len = self.num_atom_types
        
        # Atom type indices (0, 1, 2, ..., num_atom_types-1)
        atom_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get atom embeddings
        atom_emb = self.atom_embeddings(atom_indices)  # (batch_size, seq_len, embedding_dim)
        
        # Get count values and create count embeddings
        counts = formula_vectors  # (batch_size, num_atom_types)
        
        if self.use_count_embedding:
            # Clip counts to max range and convert to long
            counts_clipped = counts.clamp(0, 200).long()
            count_emb = self.count_embeddings(counts_clipped)  # (batch_size, seq_len, embedding_dim)
        else:
            # Use counts as scaling factors
            count_emb = counts.unsqueeze(-1) * atom_emb
        
        # Position embeddings
        pos_emb = self.position_embeddings(atom_indices)
        
        # Combine embeddings
        embeddings = atom_emb + count_emb + pos_emb
        
        # Apply layer norm
        embeddings = self.layer_norm(embeddings)
        
        # Create attention mask (1 for non-zero atoms, 0 for zero atoms)
        mask = (counts > 0).float()  # (batch_size, seq_len)
        
        return embeddings, mask


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for conditioning SAFE tokens on formula sequence.
    
    This layer allows SAFE token representations (queries) to attend to
    formula atom embeddings (keys/values).
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1):
        """
        Initialize CrossAttentionLayer.
        
        Args:
            hidden_size: Dimension of hidden representations
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query projection (from SAFE tokens)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        
        # Key and Value projections (from formula sequence)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_size)
    
    def forward(self,
                hidden_states: torch.Tensor,
                condition_embeddings: torch.Tensor,
                condition_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Args:
            hidden_states: SAFE token hidden states (batch, safe_len, hidden_size)
            condition_embeddings: Conditioning sequence embeddings (batch, cond_len, hidden_size)
            condition_mask: Attention mask for conditioning tokens (batch, cond_len)
        
        Returns:
            Output hidden states (batch, safe_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        
        # Project queries from SAFE tokens
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Project keys and values from formula
        key_layer = self.transpose_for_scores(self.key(condition_embeddings))
        value_layer = self.transpose_for_scores(self.value(condition_embeddings))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply conditioning mask if provided
        if condition_mask is not None:
            # Reshape mask for broadcasting: (batch, 1, 1, cond_len)
            extended_mask = condition_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask to attention mask (1 -> 0, 0 -> -10000)
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask
        
        # Normalize attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Output projection
        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        
        # Residual connection and layer norm
        output = self.output_layer_norm(hidden_states + output)
        
        return output


class CrossAttentionFormulaConditioner(nn.Module):
    """
    Complete cross-attention based formula conditioner.
    
    This module:
    1. Encodes formula as a sequence of atom embeddings
    2. Provides cross-attention layers to inject into BERT blocks
    """
    
    def __init__(self,
                 num_atom_types: int = 30,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 use_count_embedding: bool = True):
        """
        Initialize CrossAttentionFormulaConditioner.
        
        Args:
            num_atom_types: Number of atom types in formula vocabulary
            hidden_size: BERT hidden size
            num_attention_heads: Number of attention heads per layer
            num_layers: Number of BERT layers (create one cross-attn per layer)
            dropout: Dropout probability
            use_count_embedding: Use separate embeddings for atom counts
        """
        super().__init__()
        
        self.num_atom_types = num_atom_types
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Formula sequence encoder
        self.formula_encoder = FormulaSequenceEncoder(
            num_atom_types=num_atom_types,
            embedding_dim=hidden_size,
            use_count_embedding=use_count_embedding
        )
        
        # One cross-attention layer per BERT layer
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def encode_formula(self, formula_vectors: torch.Tensor) -> tuple:
        """
        Encode formula vectors as sequence.
        
        Args:
            formula_vectors: (batch_size, num_atom_types)
        
        Returns:
            Tuple of (embeddings, mask)
        """
        return self.formula_encoder(formula_vectors)
    
    def apply_cross_attention(self,
                             hidden_states: torch.Tensor,
                             layer_idx: int,
                             formula_embeddings: torch.Tensor,
                             formula_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention at a specific layer.
        
        Args:
            hidden_states: Current hidden states (batch, seq_len, hidden_size)
            layer_idx: Which BERT layer (0 to num_layers-1)
            formula_embeddings: Formula sequence embeddings
            formula_mask: Formula attention mask
        
        Returns:
            Updated hidden states after cross-attention
        """
        return self.cross_attention_layers[layer_idx](
            hidden_states, formula_embeddings, formula_mask
        )


class SetSelfAttention(nn.Module):
    """
    Permutation-equivariant self-attention for set-structured data.
    No position embeddings - treats input as an unordered set.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            Output hidden states (batch, seq_len, hidden_size)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is not None:
            # Create attention mask: (batch, 1, 1, seq_len) for keys
            extended_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        output = self.output_layer_norm(hidden_states + output)

        return output


class FingerprintSequenceEncoder(nn.Module):
    """
    Set-aware encoder for Morgan fingerprints.

    Treats fingerprint as an unordered set of active bits:
    - Each bit index has a learnable embedding (semantic meaning of substructure)
    - NO position embeddings (set has no order)
    - Self-attention learns combinatorial relationships between substructures
    """

    def __init__(self,
                 num_bits: int = 4096,
                 embedding_dim: int = 768,
                 max_seq_len: int = 256,
                 activation_threshold: float = 0.0,
                 dropout: float = 0.1,
                 num_self_attention_layers: int = 2,
                 num_attention_heads: int = 12,
                 use_value_projection: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.activation_threshold = activation_threshold
        self.num_self_attention_layers = num_self_attention_layers

        # Bit embeddings: each bit index maps to a unique embedding
        # This encodes "which substructure" semantically
        self.bit_embeddings = nn.Embedding(num_bits, embedding_dim)

        # Layer norm applied after embedding lookup
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Self-attention layers for learning combinatorial relationships
        self.self_attention_layers = nn.ModuleList([
            SetSelfAttention(
                hidden_size=embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_self_attention_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.bit_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, fingerprint: torch.Tensor) -> tuple:
        """
        Encode fingerprint as a set of active bit embeddings.

        Args:
            fingerprint: Tensor of shape (batch_size, num_bits) - binary Morgan FP

        Returns:
            embeddings: (batch_size, seq_len, embedding_dim) - set embeddings
            mask: (batch_size, seq_len) - 1 for valid bits, 0 for padding
        """
        if fingerprint is None:
            raise ValueError("Fingerprint tensor is required for sequence encoding.")

        if not torch.is_tensor(fingerprint):
            fingerprint = torch.as_tensor(fingerprint, dtype=torch.float32)

        fingerprint = fingerprint.to(dtype=torch.float32)

        single_example = fingerprint.dim() == 1
        if single_example:
            fingerprint = fingerprint.unsqueeze(0)

        batch_size, num_bits = fingerprint.shape
        device = fingerprint.device

        # Find active bits (value > threshold) for each sample
        active_mask = fingerprint > self.activation_threshold  # (batch, num_bits)
        
        # Count active bits per sample
        num_active_per_sample = active_mask.sum(dim=1)  # (batch,)
        max_active = min(self.max_seq_len, int(num_active_per_sample.max().item()))
        max_active = max(max_active, 1)  # At least 1 to avoid empty sequences

        # Vectorized approach: use topk on the mask to get indices
        # For binary mask, topk will give us the indices of 1s
        # We add small noise to break ties consistently
        scores = active_mask.float()
        if self.training:
            # During training, add noise for random sampling when truncating
            scores = scores + torch.rand_like(scores) * 0.5
        
        # Get top-k indices (these are the active bit indices we want)
        _, topk_indices = torch.topk(scores, k=max_active, dim=1, sorted=False)
        
        # Create mask based on actual number of active bits
        # topk_indices contains indices, but some might be from padding (0 values)
        # We need to mask based on original active counts
        position_indices = torch.arange(max_active, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (position_indices < num_active_per_sample.unsqueeze(1)).float()
        
        # For samples with no active bits, ensure at least one position for stability
        # but keep it masked
        no_active = num_active_per_sample == 0
        if no_active.any():
            topk_indices[no_active, 0] = 0  # Use bit 0 as dummy
            mask[no_active] = 0.0

        # Lookup bit embeddings using the actual bit indices
        embeddings = self.bit_embeddings(topk_indices)  # (batch, max_active, embedding_dim)

        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply self-attention layers to learn combinatorial relationships
        for self_attn in self.self_attention_layers:
            embeddings = self_attn(embeddings, mask)

        # Zero out padding positions
        embeddings = embeddings * mask.unsqueeze(-1)

        if single_example:
            embeddings = embeddings.squeeze(0)
            mask = mask.squeeze(0)

        return embeddings, mask


class CrossAttentionFingerprintConditioner(nn.Module):
    """
    Cross-attention conditioner for fingerprints using set-aware encoding.

    The fingerprint encoder treats the fingerprint as an unordered set of active bits,
    learns combinatorial relationships via self-attention, then provides embeddings
    for the decoder to query via cross-attention.
    """

    def __init__(self,
                 num_bits: int = 4096,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_layers: int = 12,
                 max_seq_len: int = 256,
                 activation_threshold: float = 0.0,
                 dropout: float = 0.1,
                 num_self_attention_layers: int = 2,
                 use_value_projection: bool = True,  # kept for backward compat, ignored
                 create_cross_attention_layers: bool = True):
        super().__init__()

        self.fingerprint_encoder = FingerprintSequenceEncoder(
            num_bits=num_bits,
            embedding_dim=hidden_size,
            max_seq_len=max_seq_len,
            activation_threshold=activation_threshold,
            dropout=dropout,
            num_self_attention_layers=num_self_attention_layers,
            num_attention_heads=num_attention_heads
        )

        if create_cross_attention_layers:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
        else:
            # Sharing another module's cross-attention layers (e.g., formula path).
            self.cross_attention_layers = None

    def encode_fingerprint(self, fingerprint_vectors: torch.Tensor) -> tuple:
        """
        Encode fingerprint vectors into set-aware sequence embeddings.

        Args:
            fingerprint_vectors: Tensor (batch_size, num_bits) - binary Morgan FP

        Returns:
            embeddings: (batch_size, seq_len, hidden_size) - set embeddings after self-attention
            mask: (batch_size, seq_len) - 1 for valid bits, 0 for padding
        """
        return self.fingerprint_encoder(fingerprint_vectors)

def test_conditioners():
    """Test the conditioner modules."""
    print("Testing Formula Conditioners")
    print("=" * 60)
    
    # Test basic FormulaConditioner
    print("\n1. Basic FormulaConditioner:")
    print("-" * 60)
    
    input_dim = 30  # Number of atom types
    output_dim = 768  # BERT hidden size
    batch_size = 4
    
    conditioner = FormulaConditioner(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        dropout=0.1
    )
    
    # Test with batch
    formula_vecs = torch.randn(batch_size, input_dim)
    embeddings = conditioner(formula_vecs)
    print(f"Input shape: {formula_vecs.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: ({batch_size}, {output_dim})")
    
    # Test with single example
    single_vec = torch.randn(input_dim)
    single_emb = conditioner(single_vec)
    print(f"\nSingle input shape: {single_vec.shape}")
    print(f"Single output shape: {single_emb.shape}")
    print(f"Expected: ({output_dim},)")
    
    # Test AdaptiveFormulaConditioner
    print("\n2. AdaptiveFormulaConditioner:")
    print("-" * 60)
    
    adaptive = AdaptiveFormulaConditioner(
        input_dim=input_dim,
        output_dim=output_dim,
        learnable_scale=True,
        initial_scale=0.5
    )
    
    embeddings = adaptive(formula_vecs)
    print(f"Output shape: {embeddings.shape}")
    print(f"Initial scale: {adaptive.get_scale():.4f}")
    
    # Test LayerwiseFormulaConditioner
    print("\n3. LayerwiseFormulaConditioner:")
    print("-" * 60)
    
    layerwise = LayerwiseFormulaConditioner(
        input_dim=input_dim,
        output_dim=output_dim,
        num_bert_layers=12,
        shared_base=True
    )
    
    # Test single layer
    layer_0_emb = layerwise(formula_vecs, layer_idx=0)
    print(f"Layer 0 output shape: {layer_0_emb.shape}")
    
    # Test all layers
    all_embeddings = layerwise.forward_all_layers(formula_vecs)
    print(f"Number of layer embeddings: {len(all_embeddings)}")
    print(f"Each embedding shape: {all_embeddings[0].shape}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_conditioners()
