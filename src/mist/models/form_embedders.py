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

"""Formula embedders for encoding chemical formula counts in MIST."""

import torch
import torch.nn as nn
import numpy as np

from ..utils.chem_utils import NORM_VEC


class IntFeaturizer(nn.Module):
    """Base class for mapping integers to vector representations.

    Creates embeddings for integer values (typically atom counts in chemical formulas).
    Subclasses define `int_to_feat_matrix` where each row is the vector for that integer.

    Includes extra learnable embeddings for special tokens (e.g., padding).
    """

    MAX_COUNT_INT = 255  # Maximum integer count to embed (0 to MAX_COUNT_INT-1)
    NUM_EXTRA_EMBEDDINGS = 1  # Extra embeddings for special tokens

    def __init__(self, embedding_dim):
        super().__init__()
        weights = torch.zeros(self.NUM_EXTRA_EMBEDDINGS, embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
        self.embedding_dim = embedding_dim

    def forward(self, tensor):
        """Convert integer tensor to embedding representation."""
        orig_shape = tensor.shape
        out_tensor = torch.empty(
            (*orig_shape, self.embedding_dim), device=tensor.device
        )
        extra_embed = tensor >= self.MAX_COUNT_INT

        tensor = tensor.long()
        norm_embeds = self.int_to_feat_matrix[tensor[~extra_embed]]
        extra_embeds = self._extra_embeddings[tensor[extra_embed] - self.MAX_COUNT_INT]

        out_tensor[~extra_embed] = norm_embeds
        out_tensor[extra_embed] = extra_embeds

        temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
        return temp_out

    @property
    def num_dim(self):
        return self.int_to_feat_matrix.shape[1]

    @property
    def full_dim(self):
        return self.num_dim * NORM_VEC.shape[0]


class FourierFeaturizer(IntFeaturizer):
    """Fourier feature embeddings using sin and cos functions.

    Based on "Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains" by Tancik et al.
    Uses frequencies at powers of 1/2.
    """

    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2

        freqs = 0.5 ** torch.arange(num_freqs, dtype=torch.float32)
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(embedding_dim=2 * freqs_time_2pi.shape[0])  # cos and sin

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        all_features = torch.cat(
            [torch.cos(combo_of_sinusoid_args), torch.sin(combo_of_sinusoid_args)],
            dim=1,
        )

        self.int_to_feat_matrix = nn.Parameter(all_features.float())
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerSines(IntFeaturizer):
    """Fourier features using only sine functions."""

    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2

        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            torch.sin(combo_of_sinusoid_args).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
    """Fourier features using absolute value of sine functions."""

    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2

        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            torch.abs(torch.sin(combo_of_sinusoid_args)).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerPosCos(IntFeaturizer):
    """Fourier features using positive cosine transform.

    Uses (-cos(x) + 1) to create features that are 0 at x=0 and increase smoothly.
    """

    def __init__(self, num_funcs=9):
        self.num_funcs = num_funcs

        # Define frequencies that smoothly increase from 0 to max
        max_freq = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 1
        freqs = 0.5 ** np.linspace(1, max_freq, num_funcs)
        freqs_time_2pi = 2 * np.pi * freqs
        freqs_time_2pi = torch.from_numpy(freqs_time_2pi).float()
        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            (-torch.cos(combo_of_sinusoid_args) + 1).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class RBFFeaturizer(IntFeaturizer):
    """Radial basis function embeddings.

    Places RBF centers evenly between 0 and max_count-1 with widths
    chosen so functions decay to ~0.6 at neighboring centers.
    """

    def __init__(self, num_funcs=32):
        """
        Args:
            num_funcs: Number of radial basis functions to use
        """
        super().__init__(embedding_dim=num_funcs)
        width = (self.MAX_COUNT_INT - 1) / num_funcs
        centers = torch.linspace(0, self.MAX_COUNT_INT - 1, num_funcs)

        pre_exponential_terms = (
            -0.5
            * ((torch.arange(self.MAX_COUNT_INT)[:, None] - centers[None, :]) / width)
            ** 2
        )
        feats = torch.exp(pre_exponential_terms)

        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class OneHotFeaturizer(IntFeaturizer):
    """One-hot encoding for integers."""

    def __init__(self):
        super().__init__(embedding_dim=self.MAX_COUNT_INT)
        feats = torch.eye(self.MAX_COUNT_INT)
        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class LearnedFeaturizer(IntFeaturizer):
    """Learned embeddings for integers (similar to nn.Embedding)."""

    def __init__(self, feature_dim=32):
        super().__init__(embedding_dim=feature_dim)
        self.nn_embedder = nn.Embedding(
            self.MAX_COUNT_INT + self.NUM_EXTRA_EMBEDDINGS, feature_dim
        )
        self.int_to_feat_matrix = list(self.nn_embedder.parameters())[0]

    def forward(self, tensor):
        """Convert integer tensor to learned embedding."""
        orig_shape = tensor.shape
        out_tensor = self.nn_embedder(tensor.long())
        temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
        return temp_out


class FloatFeaturizer(IntFeaturizer):
    """Simple normalized float features."""

    def __init__(self):
        super().__init__(embedding_dim=1)
        self.norm_vec = torch.from_numpy(NORM_VEC).float()
        self.norm_vec = nn.Parameter(self.norm_vec)
        self.norm_vec.requires_grad = False

    def forward(self, tensor):
        """Normalize tensor by element-specific max counts."""
        tens_shape = tensor.shape
        out_shape = [1] * (len(tens_shape) - 1) + [-1]
        return tensor / self.norm_vec.reshape(*out_shape)

    @property
    def num_dim(self):
        return 1


def get_embedder(embedder):
    """Get embedder instance by name.

    Args:
        embedder: String name of embedder type

    Returns:
        Instantiated embedder module
    """
    if embedder == "fourier":
        embedder = FourierFeaturizer()
    elif embedder == "rbf":
        embedder = RBFFeaturizer()
    elif embedder == "one-hot":
        embedder = OneHotFeaturizer()
    elif embedder == "learnt":
        embedder = LearnedFeaturizer()
    elif embedder == "float":
        embedder = FloatFeaturizer()
    elif embedder == "fourier-sines":
        embedder = FourierFeaturizerSines()
    elif embedder == "abs-sines":
        embedder = FourierFeaturizerAbsoluteSines()
    elif embedder == "pos-cos":
        embedder = FourierFeaturizerPosCos()
    else:
        raise NotImplementedError(f"Unknown embedder type: {embedder}")
    return embedder
