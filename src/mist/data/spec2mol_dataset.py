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

"""Spec2Mol DataModule for end-to-end training with MIST encoder and DLM decoder.

This module loads:
- Spectra data (for MIST encoder input)
- Ground truth formulas (for DLM conditioning)
- SAFE tokens (for DLM decoder target)
"""

import os
import random
import logging
from functools import partial
from typing import List, Dict, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from rdkit import Chem
import safe as sf

from . import datasets, featurizers, splitter
from dlm.utils.utils_data import get_tokenizer


# Create SAFE converter once (reused for all conversions)
_SAFE_CONVERTER = sf.SAFEConverter(slicer=None, ignore_stereo=True)


def smiles_to_safe(smiles: str) -> Optional[str]:
    """Convert SMILES to SAFE representation.
    
    Uses SAFEConverter.encoder which handles molecules without cuttable bonds
    (like benzene) by returning the original SMILES as a valid SAFE string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        SAFE string or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Use SAFEConverter.encoder which is more robust
        safe_str = _SAFE_CONVERTER.encoder(smiles, allow_empty=True)
        return safe_str
    except Exception as e:
        logging.debug(f"Failed to convert SMILES to SAFE: {smiles}, error: {e}")
        return None


class Spec2MolDataset(Dataset):
    """Dataset for end-to-end spectra-to-molecule training.
    
    Each item contains:
    - Spectra features (for MIST encoder)
    - Ground truth formula string (for DLM conditioning)
    - SAFE tokens (for DLM decoder target)
    
    Args:
        spectra_mol_list: List of (Spectra, Mol) pairs
        featurizer: PairedFeaturizer for spectra
        tokenizer: SAFE tokenizer
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        spectra_mol_list: List,
        featurizer: featurizers.PairedFeaturizer,
        tokenizer,
        max_length: int = 256,
        **kwargs,
    ):
        super().__init__()
        spectra_list, mol_list = list(zip(*spectra_mol_list))
        self.spectra_list = np.array(spectra_list)
        self.mol_list = np.array(mol_list)
        
        # Pre-compute SMILES, formulas, and SAFE strings
        self.smiles_list = []
        self.formula_list = []
        self.safe_list = []
        
        valid_indices = []
        for i, (spec, mol) in enumerate(zip(spectra_list, mol_list)):
            smiles = mol.get_smiles()
            formula = spec.get_spectra_formula()
            safe_str = smiles_to_safe(smiles)
            
            if safe_str is not None:
                self.smiles_list.append(smiles)
                self.formula_list.append(formula)
                self.safe_list.append(safe_str)
                valid_indices.append(i)
        
        # Filter to valid samples
        self.spectra_list = self.spectra_list[valid_indices]
        self.mol_list = self.mol_list[valid_indices]
        self.smiles_list = np.array(self.smiles_list)
        self.formula_list = np.array(self.formula_list)
        self.safe_list = np.array(self.safe_list)
        
        logging.info(f"Spec2MolDataset: {len(valid_indices)}/{len(spectra_list)} samples valid after SAFE conversion")
        
        self.featurizer = featurizer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_mode = False
    
    def set_train_mode(self, train_mode: bool):
        """Set training mode (for data augmentation)."""
        self.train_mode = train_mode
    
    def __len__(self) -> int:
        return len(self.spectra_list)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.
        
        Returns:
            Dictionary with:
            - spec: Spectra features for MIST encoder
            - formula: Formula string
            - safe: SAFE string
        """
        spec = self.spectra_list[idx]
        formula = self.formula_list[idx]
        safe_str = self.safe_list[idx]
        
        spec_features = self.featurizer.featurize_spec(spec, train_mode=self.train_mode)
        
        return {
            "spec": spec_features,
            "formula": formula,
            "safe": safe_str,
        }


def _collate_spec2mol(
    input_batch: List[Dict],
    spec_collate_fn: Callable,
    tokenizer,
    max_length: int,
) -> Dict:
    """Collate function for Spec2Mol dataset.
    
    Args:
        input_batch: List of sample dicts
        spec_collate_fn: Spectra collate function
        tokenizer: SAFE tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Batched dictionary with:
        - Spectra features (for MIST encoder)
        - formula: List of formula strings
        - input_ids: Tokenized SAFE sequences
        - attention_mask: Attention mask for SAFE tokens
    """
    # Collate spectra features
    spec_dict = spec_collate_fn([item["spec"] for item in input_batch])
    
    # Extract formulas
    formulas = [item["formula"] for item in input_batch]
    
    # Tokenize SAFE strings
    safe_strings = [item["safe"] for item in input_batch]
    tokenized = tokenizer(
        safe_strings,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    
    # Build output batch
    output = {}
    output.update(spec_dict)
    output["formula"] = formulas
    output["input_ids"] = tokenized["input_ids"]
    output["attention_mask"] = tokenized["attention_mask"]
    
    return output


def get_spec2mol_loader(
    dataset: Spec2MolDataset,
    shuffle: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for Spec2Mol dataset.
    
    Args:
        dataset: Spec2MolDataset instance
        shuffle: Whether to shuffle
        batch_size: Batch size
        num_workers: Number of workers
        persistent_workers: Keep workers alive
        
    Returns:
        DataLoader instance
    """
    spec_collate_fn = dataset.featurizer.get_spec_collate()
    
    collate_fn = partial(
        _collate_spec2mol,
        spec_collate_fn=spec_collate_fn,
        tokenizer=dataset.tokenizer,
        max_length=dataset.max_length,
    )
    
    _persistent_workers = num_workers > 0 and persistent_workers
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=_persistent_workers,
    )


class Spec2MolDataModule(pl.LightningDataModule):
    """DataModule for Spec2Mol end-to-end training.
    
    Loads:
    - Spectra features (PeakFormula) for MIST encoder
    - Ground truth formulas for DLM conditioning
    - SAFE tokens as decoder target
    
    Args:
        cfg: Configuration object
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
        
        # Create data splitter
        data_splitter = splitter.PresetSpectraSplitter(split_file=dataset_cfg.split_file)
        
        # Create spectra featurizer (only need spec featurizer, not mol featurizer)
        spec_featurizer = featurizers.PeakFormula(
            subform_folder=dataset_cfg.subform_folder,
            augment_data=dataset_cfg.get('augment_data', False),
            remove_prob=dataset_cfg.get('remove_prob', 0.1),
            remove_weights=dataset_cfg.get('remove_weights', 'exp'),
            inten_prob=dataset_cfg.get('inten_prob', 0.1),
            inten_transform=dataset_cfg.get('inten_transform', 'float'),
            cls_type=dataset_cfg.get('cls_type', 'ms1'),
            magma_modulo=dataset_cfg.get('magma_modulo', 512),
            cache_featurizers=dataset_cfg.get('cache_featurizers', True),
        )
        
        paired_featurizer = featurizers.PairedFeaturizer(
            spec_featurizer=spec_featurizer,
            mol_featurizer=None,  # Not needed - we only need spectra features
        )
        
        # Load spectra-molecule pairs
        spectra_mol_pairs = datasets.get_paired_spectra(
            labels_file=dataset_cfg.labels_file,
            spec_folder=dataset_cfg.spec_folder,
            max_count=dataset_cfg.get('max_count', None),
        )
        spectra_mol_pairs = list(zip(*spectra_mol_pairs))
        
        # Split the data
        split_name, (train, val, test) = data_splitter.get_splits(spectra_mol_pairs)
        
        # Shuffle test set with fixed seed
        random.seed(42)
        random.shuffle(test)
        
        # Create datasets
        self.train_dataset = Spec2MolDataset(
            spectra_mol_list=train,
            featurizer=paired_featurizer,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.val_dataset = Spec2MolDataset(
            spectra_mol_list=val,
            featurizer=paired_featurizer,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        self.test_dataset = Spec2MolDataset(
            spectra_mol_list=test,
            featurizer=paired_featurizer,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        # Set train mode
        self.train_dataset.set_train_mode(True)
        self.val_dataset.set_train_mode(False)
        self.test_dataset.set_train_mode(False)
        
        logging.info(f"Spec2MolDataModule: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def train_dataloader(self):
        """Create training dataloader."""
        return get_spec2mol_loader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return get_spec2mol_loader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return get_spec2mol_loader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
