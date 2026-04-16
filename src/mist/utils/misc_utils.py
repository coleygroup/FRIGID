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

"""Miscellaneous utility functions for MIST encoder."""

from typing import List, Iterable, Iterator
from itertools import islice

import numpy as np
import torch


def unravel_index(index, shape):
    """Unravel flat index to multi-dimensional index."""
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode="trunc")
    return tuple(reversed(out))


def np_clamp(x, _min=-100):
    """Clamp numpy array to minimum value."""
    x = np.ones_like(x) * x
    x[x <= _min] = _min
    return x


def clamped_log_np(x, _min=-100):
    """Compute clamped logarithm."""
    res = np.log(x)
    return np_clamp(res, _min=_min)


def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size."""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


def pad_packed_tensor(input, lengths, value):
    """Pad a packed tensor to uniform length.

    Args:
        input: Packed tensor
        lengths: Length of each sequence
        value: Padding value

    Returns:
        Padded tensor of shape (batch_size, max_len, ...)
    """
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    # Initialize a tensor with an index for every value in the array
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    # Row shifts
    row_shifts = torch.cumsum(max_len - lengths, 0)

    # Calculate shifts for second row, third row... nth row (not the n+1th row)
    # Expand this out to match the shape of all entries after the first row
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    # Add this to the list of inds _after_ the first row
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0]:] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def reverse_packed_tensor(packed_tensor, lengths):
    """Reverse a padded tensor to packed format.

    Args:
        packed_tensor: Batch x length x feat_dim tensor
        lengths: Length of each sequence

    Returns:
        Packed tensor of shape [sum(lengths), feat_dim]
    """
    device = packed_tensor.device
    batch_size, batch_len, feat_dim = packed_tensor.shape
    max_length = torch.arange(batch_len).to(device)
    indices = max_length.unsqueeze(0).expand(batch_size, batch_len)
    bool_mask = indices < lengths.unsqueeze(1)
    output = packed_tensor[bool_mask]
    return output


def unpack_bits(vec, num_bits):
    """Unpack bit vector to full representation."""
    return np.unpackbits(vec, axis=-1)[..., -num_bits:]
