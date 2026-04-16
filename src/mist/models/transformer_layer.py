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

"""Transformer layers with pairwise attention support for MIST encoder."""

import math
from typing import Optional, Union, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, LayerNorm, Linear, Dropout, Parameter
from torch.nn.init import xavier_uniform_, constant_

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer with optional pairwise featurization.

    Based on "Attention Is All You Need" by Vaswani et al., with extensions
    for pairwise attention mechanisms used in mass spectrometry analysis.

    Args:
        d_model: Number of expected features in the input
        nhead: Number of heads in the multiheadattention models
        dim_feedforward: Dimension of the feedforward network model (default=2048)
        dropout: Dropout value (default=0.1)
        activation: Activation function of intermediate layer
        layer_norm_eps: Eps value in layer normalization components (default=1e-5)
        batch_first: If True, input/output tensors are (batch, seq, feature)
        norm_first: If True, layer norm is done prior to attention
        additive_attn: If True, use additive attention instead of scaled dot product
        pairwise_featurization: If True, include pairwise features in attention
    """
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        additive_attn: bool = False,
        pairwise_featurization: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.pairwise_featurization = pairwise_featurization
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            additive_attn=additive_attn,
            pairwise_featurization=self.pairwise_featurization,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        pairwise_features: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: The sequence to the encoder layer (required)
            pairwise_features: Optional pairwise features tensor
            src_key_padding_mask: Mask for the src keys per batch (optional)

        Returns:
            Output tensor and pairwise features
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), pairwise_features, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, pairwise_features, src_key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))

        return x, pairwise_features

    def _sa_block(
        self,
        x: Tensor,
        pairwise_features: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Self-attention block."""
        x = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            pairwise_features=pairwise_features,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed forward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MultiheadAttention(Module):
    """Multi-Head Attention with optional pairwise featurization.

    Allows the model to jointly attend to information from different
    representation subspaces as described in "Attention Is All You Need".

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        additive_attn: If true, use additive attention
        dropout: Dropout probability on attention weights (default=0.0)
        batch_first: If True, input/output tensors are (batch, seq, feature)
        pairwise_featurization: If True, use pairwise featurization
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        additive_attn=False,
        pairwise_featurization: bool = False,
        dropout=0.0,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True
        self.additive_attn = additive_attn
        self.pairwise_featurization = pairwise_featurization

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        if self.additive_attn:
            head_1_input = (
                self.head_dim * 3 if self.pairwise_featurization else self.head_dim * 2
            )
            self.attn_weight_1_weight = Parameter(
                torch.empty(
                    (self.num_heads, head_1_input, self.head_dim), **factory_kwargs
                ),
            )
            self.attn_weight_1_bias = Parameter(
                torch.empty((self.num_heads, self.head_dim), **factory_kwargs),
            )

            self.attn_weight_2_weight = Parameter(
                torch.empty((self.num_heads, self.head_dim, 1), **factory_kwargs),
            )
            self.attn_weight_2_bias = Parameter(
                torch.empty((self.num_heads, 1), **factory_kwargs),
            )
        else:
            if self.pairwise_featurization:
                self.bias_u = Parameter(
                    torch.empty((self.num_heads, self.head_dim), **factory_kwargs),
                )
                self.bias_v = Parameter(
                    torch.empty((self.num_heads, self.head_dim), **factory_kwargs),
                )

        self.in_proj_weight = Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=True, **factory_kwargs
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)
        if self.additive_attn:
            xavier_uniform_(self.attn_weight_1_weight)
            xavier_uniform_(self.attn_weight_2_weight)
            constant_(self.attn_weight_1_bias, 0.0)
            constant_(self.attn_weight_2_bias, 0.0)
        else:
            if self.pairwise_featurization:
                constant_(self.bias_u, 0.0)
                constant_(self.bias_v, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass for multi-head attention.

        Args:
            query: Query embeddings
            key: Key embeddings
            value: Value embeddings
            key_padding_mask: Mask for key elements to ignore
            pairwise_features: Optional pairwise features

        Returns:
            Tuple of (attention output, attention weights)
        """
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = self.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            pairwise_features=pairwise_features,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Multi-head attention forward computation.

        Args:
            query, key, value: Input tensors
            embed_dim_to_check: Expected embedding dimension
            num_heads: Number of attention heads
            in_proj_weight, in_proj_bias: Input projection parameters
            dropout_p: Dropout probability
            out_proj_weight, out_proj_bias: Output projection parameters
            training: Whether in training mode
            key_padding_mask: Optional key padding mask
            pairwise_features: Optional pairwise features

        Returns:
            Tuple of (attention output, attention weights)
        """
        # Set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert (
            embed_dim == embed_dim_to_check
        ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        
        if isinstance(embed_dim, torch.Tensor):
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        # Reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        if pairwise_features is not None:
            # Expand pairwise features
            # B x L x L x H  => L x L x (B*Nh) x (H/nh)
            pairwise_features = pairwise_features.permute(1, 2, 0, 3).contiguous()
            pairwise_features = pairwise_features.view(
                tgt_len, tgt_len, bsz * num_heads, head_dim
            )

            # L x L x (B*Nh) x (H/nh)  => (B*Nh) x L x L x (H / Nh)
            pairwise_features = pairwise_features.permute(2, 0, 1, 3)

        # Update source sequence length after adjustments
        src_len = k.size(1)

        # Merge key padding and attention masks
        attn_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            attn_mask = key_padding_mask
            assert attn_mask.dtype == torch.bool

        # Adjust dropout probability
        if not training:
            dropout_p = 0.0

        # Calculate attention and out projection
        if self.additive_attn:
            attn_output, attn_output_weights = self._additive_attn(
                q, k, v, attn_mask, dropout_p, pairwise_features=pairwise_features
            )
        else:
            attn_output, attn_output_weights = self._scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, pairwise_features=pairwise_features
            )
        
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights

    def _additive_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Additive attention mechanism.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            attn_mask: Optional attention mask
            dropout_p: Dropout probability
            pairwise_features: Optional pairwise features

        Returns:
            Tuple of (attention output, attention weights)
        """
        B, Nt, E = q.shape
        # B x Nt x E => B x Nt x Nt x E
        q_expand = q[:, :, None, :].expand(B, Nt, Nt, E)
        v_expand = v[:, None, :, :].expand(B, Nt, Nt, E)
        # B x Nt x Nt x E => B x Nt x Nt x 2E
        cat_ar = [q_expand, v_expand]
        if pairwise_features is not None:
            cat_ar.append(pairwise_features)

        output = torch.cat(cat_ar, -1)
        E_long = E * len(cat_ar)

        output = output.view(-1, self.num_heads, Nt, Nt, E_long)

        # B x Nt x Nt x len(cat_ar)*E => B x Nt x Nt x E
        output = torch.einsum("bnlwe,neh->bnlwh", output, self.attn_weight_1_weight)

        output = output + self.attn_weight_1_bias[None, :, None, None, :]

        output = F.leaky_relu(output)

        # B x Nt x Nt x len(cat_ar)*E => B x Nt x Nt
        attn = torch.einsum("bnlwh,nhi->bnlwi", output, self.attn_weight_2_weight)
        attn = attn + self.attn_weight_2_bias[None, :, None, None, :]
        attn = attn.contiguous().view(-1, Nt, Nt)
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += new_attn_mask
        attn = F.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Scaled dot product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            attn_mask: Optional attention mask
            dropout_p: Dropout probability
            pairwise_features: Optional pairwise features

        Returns:
            Tuple of (attention output, attention weights)
        """
        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if self.pairwise_featurization:
            # Inspired by Graph2Smiles and TransformerXL
            if pairwise_features is None:
                raise ValueError()

            # B*Nh x Nt x E => B x Nh x Nt x E
            q = q.view(-1, self.num_heads, Nt, E)
            q_1 = q + self.bias_u[None, :, None, :]
            q_2 = q + self.bias_v[None, :, None, :]

            # B x Nh x Nt x E => B*Nh x Nt x E
            q_1 = q_1.view(-1, Nt, E)
            q_2 = q_2.view(-1, Nt, E)

            # B x Nh x Nt x E => B x Nh x Nt x Nt
            a_c = torch.einsum("ble,bwe->blw", q_1, k)

            # pairwise: B*Nh x Nt x Nt x E
            # q_2: B*Nh x Nt x E
            b_d = torch.einsum("ble,blwe->blw", q_2, pairwise_features)

            attn = a_c + b_d
        else:
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q, k.transpose(-2, -1))

        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += new_attn_mask

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn
