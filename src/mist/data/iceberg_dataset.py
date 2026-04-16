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

"""ICEBERG multi-part dataset for training with synthetic spectra data.

This module provides a DataModule that cycles through multiple HDF5 part files,
using one part per epoch. This avoids the need to merge all parts into a single
large file while still enabling training on the full dataset.

Key features:
- Each epoch uses a different part file (part_000, part_001, ..., part_009)
- Part cycling is automatic and wraps around
- Validation/test sets use a fixed part (first by default)
- Fully compatible with existing HDF5SpectrumDataset
- Does not affect other training pipelines
"""

import os
import logging
from functools import partial
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from .hdf5_dataset import (
    HDF5SpectrumDataset, 
    _collate_hdf5_spec2mol,
)
from dlm.utils.utils_data import get_tokenizer


class IcebergMultiPartDataModule(pl.LightningDataModule):
    """DataModule for training with multiple ICEBERG HDF5 part files.
    
    This module cycles through multiple part files, using one part per epoch
    for training. This is useful when:
    - Merging all parts into one file is too time/space consuming
    - You want to train on the full ICEBERG dataset (10 parts)
    - Each part represents ~20 msg_X folders worth of synthetic spectra
    
    Design:
    - Training: Cycles through parts (part_000 -> part_001 -> ... -> part_009 -> part_000)
    - Validation: Uses first part by default (configurable)
    - Test: Uses first part by default (configurable)
    
    Args:
        cfg: Configuration object with:
            - dataset.parts_dir: Directory containing part_000/, part_001/, etc.
            - dataset.num_parts: Number of parts (default: auto-detect)
            - dataset.val_part_idx: Part index for validation (default: 0)
            - dataset.test_part_idx: Part index for test (default: 0)
            - Other HDF5 dataset settings (augment_data, cls_type, etc.)
    
    Example config:
        dataset:
          use_iceberg_parts: True
          parts_dir: '/path/to/ICEBERG/parts'
          num_parts: 10
          val_part_idx: 0
          test_part_idx: 0
          reorder_elements: 'auto'
          augment_data: True
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.get('batch_size_per_device', cfg.train.batch_size)
        self.eval_batch_size = cfg.train.get(
            'eval_batch_size_per_device',
            cfg.train.get('eval_batch_size', self.batch_size),
        )
        self.num_workers = cfg.train.get('num_workers', 1)
        self.pin_memory = cfg.train.get('pin_memory', True)
        self.max_length = cfg.model.get('max_position_embeddings', 256)
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(cfg.data.get('hf_cache_dir', None))
        
        # Get dataset configuration
        dataset_cfg = cfg.dataset
        self.parts_dir = Path(dataset_cfg.parts_dir)
        
        # Discover available parts
        self.part_paths = self._discover_parts()
        self.num_parts = len(self.part_paths)
        
        if self.num_parts == 0:
            raise ValueError(f"No part directories found in {self.parts_dir}")
        
        logging.info(f"IcebergMultiPartDataModule: Found {self.num_parts} parts in {self.parts_dir}")
        
        # Part indices for val/test (default to first part)
        self.val_part_idx = dataset_cfg.get('val_part_idx', 0)
        self.test_part_idx = dataset_cfg.get('test_part_idx', 0)
        
        # Current training part index (will be updated each epoch)
        self._current_train_part_idx = 0
        
        # Dataset settings
        self.augment_data = dataset_cfg.get('augment_data', False)
        self.remove_prob = dataset_cfg.get('remove_prob', 0.1)
        self.inten_prob = dataset_cfg.get('inten_prob', 0.1)
        self.cls_type = dataset_cfg.get('cls_type', 'ms1')
        self.reorder_elements = dataset_cfg.get('reorder_elements', 'auto')
        
        # Initialize datasets (training dataset will be recreated each epoch)
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
        # Pre-create val and test datasets (they don't change)
        self._create_val_test_datasets()
        
        logging.info(
            f"IcebergMultiPartDataModule: val_part={self.val_part_idx}, "
            f"test_part={self.test_part_idx}, augment={self.augment_data}"
        )
    
    def _discover_parts(self) -> List[Path]:
        """Discover available part directories."""
        if not self.parts_dir.exists():
            raise ValueError(f"Parts directory does not exist: {self.parts_dir}")
        
        part_dirs = sorted([
            d for d in self.parts_dir.iterdir()
            if d.is_dir() and d.name.startswith('part_')
        ])
        
        # Filter to those with preds.hdf5
        valid_parts = []
        for part_dir in part_dirs:
            hdf5_path = part_dir / 'preds.hdf5'
            if hdf5_path.exists():
                valid_parts.append(hdf5_path)
            else:
                logging.warning(f"Part directory missing preds.hdf5: {part_dir}")
        
        return valid_parts
    
    def _create_dataset(self, part_idx: int, augment: bool = False) -> HDF5SpectrumDataset:
        """Create a dataset for the specified part."""
        hdf5_path = self.part_paths[part_idx]
        
        return HDF5SpectrumDataset(
            hdf5_path=str(hdf5_path),
            split_indices=None,  # Use all data in the part
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment_data=augment,
            remove_prob=self.remove_prob,
            inten_prob=self.inten_prob,
            cls_type=self.cls_type,
            reorder_elements=self.reorder_elements,
        )
    
    def _create_val_test_datasets(self):
        """Create validation and test datasets (fixed parts)."""
        self._val_dataset = self._create_dataset(self.val_part_idx, augment=False)
        self._val_dataset.set_train_mode(False)
        
        self._test_dataset = self._create_dataset(self.test_part_idx, augment=False)
        self._test_dataset.set_train_mode(False)
        
        logging.info(f"IcebergMultiPartDataModule: val={len(self._val_dataset)}, test={len(self._test_dataset)}")
    
    def _create_train_dataset(self, part_idx: int):
        """Create training dataset for the specified part."""
        # Close previous dataset if exists
        if self._train_dataset is not None:
            try:
                if hasattr(self._train_dataset, 'h5_file') and self._train_dataset.h5_file:
                    self._train_dataset.h5_file.close()
            except Exception:
                pass
        
        self._train_dataset = self._create_dataset(part_idx, augment=self.augment_data)
        self._train_dataset.set_train_mode(True)
        
        logging.info(
            f"IcebergMultiPartDataModule: Created train dataset from part_{part_idx:03d} "
            f"with {len(self._train_dataset)} samples"
        )
    
    def set_epoch(self, epoch: int):
        """Set the current epoch to determine which part to use for training.
        
        This method should be called at the start of each training epoch.
        Part index cycles: epoch 0 -> part_000, epoch 1 -> part_001, etc.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        new_part_idx = epoch % self.num_parts
        
        if new_part_idx != self._current_train_part_idx or self._train_dataset is None:
            self._current_train_part_idx = new_part_idx
            self._create_train_dataset(new_part_idx)
            logging.info(f"IcebergMultiPartDataModule: Epoch {epoch} using part_{new_part_idx:03d}")
    
    @property
    def train_dataset(self):
        """Get current training dataset."""
        if self._train_dataset is None:
            self._create_train_dataset(0)
        return self._train_dataset
    
    @property
    def val_dataset(self):
        """Get validation dataset."""
        return self._val_dataset
    
    @property
    def test_dataset(self):
        """Get test dataset."""
        return self._test_dataset
    
    def _get_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Create a DataLoader for the given dataset."""
        collate_fn = partial(
            _collate_hdf5_spec2mol,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size if shuffle else self.eval_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def train_dataloader(self):
        """Create training dataloader for current part."""
        return self._get_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return self._get_dataloader(self._val_dataset, shuffle=False)
    
    def test_dataloader(self):
        """Create test dataloader."""
        return self._get_dataloader(self._test_dataset, shuffle=False)


class IcebergEpochCallback(pl.Callback):
    """Lightning callback to update ICEBERG part at each epoch.
    
    This callback should be added to the trainer when using IcebergMultiPartDataModule.
    It automatically calls set_epoch() at the start of each training epoch.
    
    Usage:
        callbacks = [IcebergEpochCallback()]
        trainer = L.Trainer(callbacks=callbacks, ...)
    """
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the start of each training epoch."""
        datamodule = trainer.datamodule
        
        if isinstance(datamodule, IcebergMultiPartDataModule):
            current_epoch = trainer.current_epoch
            datamodule.set_epoch(current_epoch)
