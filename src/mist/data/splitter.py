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

"""Dataset splitting utilities for train/val/test partitioning."""

from pathlib import Path
from typing import List, Tuple, Iterator
import pandas as pd
import numpy as np

from .data import Spectra, Mol

DATASET = List[Tuple[Spectra, Mol]]


def get_splitter(**kwargs):
    """Get splitter instance by type."""
    return {"preset": PresetSpectraSplitter}["preset"](**kwargs)


class SpectraSplitter(object):
    """Base class for dataset splitting."""

    def __init__(self, **kwargs):
        pass

    def split_from_indices(
        self,
        full_dataset: DATASET,
        train_inds: np.ndarray,
        val_inds: np.ndarray,
        test_inds: np.ndarray,
    ) -> Tuple[DATASET]:
        """Split dataset by index arrays.

        Args:
            full_dataset: Complete dataset
            train_inds: Indices for training set
            val_inds: Indices for validation set
            test_inds: Indices for test set

        Returns:
            Tuple of (train, val, test) datasets
        """
        full_dataset = np.array(full_dataset)
        train_sub = full_dataset[train_inds].tolist()
        val_sub = full_dataset[val_inds].tolist()
        test_sub = full_dataset[test_inds].tolist()
        return (train_sub, val_sub, test_sub)


class PresetSpectraSplitter(SpectraSplitter):
    """Splitter using preset split assignments from a TSV file.

    Reads split assignments from a TSV file with columns:
        - name: Spectrum identifier
        - split: Split assignment ("train", "val", or "test")

    Args:
        split_file: Path to TSV file with split assignments
    """

    def __init__(self, split_file: str = None, **kwargs):
        super().__init__(**kwargs)
        if split_file is None:
            raise ValueError("Preset splitter requires split_file arg.")

        self.split_file = split_file
        self.split_name = Path(split_file).stem
        self.split_df = pd.read_csv(self.split_file, sep="\t")
        self.name_to_fold = dict(zip(self.split_df["name"], self.split_df["split"]))

    def get_splits(self, full_dataset: DATASET) -> Iterator[Tuple[str, Tuple[DATASET]]]:
        """Get train/val/test splits from dataset.

        Args:
            full_dataset: Complete dataset of (Spectra, Mol) pairs

        Returns:
            Tuple of (split_name, (train, val, test) datasets)
        """
        # Map name to index
        spec_names = [i.get_spec_name() for i, j in full_dataset]
        train_inds = [
            i for i, j in enumerate(spec_names) if self.name_to_fold.get(j) == "train"
        ]
        val_inds = [
            i for i, j in enumerate(spec_names) if self.name_to_fold.get(j) == "val"
        ]
        test_inds = [
            i for i, j in enumerate(spec_names) if self.name_to_fold.get(j) == "test"
        ]
        new_split = self.split_from_indices(
            full_dataset, train_inds, val_inds, test_inds
        )
        return (self.split_name, new_split)
