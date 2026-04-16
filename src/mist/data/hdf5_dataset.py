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

"""HDF5-based dataset for efficient loading of merged spectral data.

This module provides a DataModule that loads spectra from a single merged HDF5 file,
which is more efficient than loading many small JSON files.

The HDF5 file should have the following structure (created by merge_hdf5.py):
  /spectra/{idx}/f  - float32 array [N, 2]: [masses_no_adduct, intens]
  /spectra/{idx}/u  - uint8 array [N, ELEMENT_DIM]: frag_form_vecs
  
  With attributes on each spectrum group: inchikey, collision_energy, smiles, formula, adduct, instrument
"""

import os
import random
import logging
from functools import partial
from typing import List, Dict, Callable, Optional, Tuple
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from rdkit import Chem
import safe as sf

from .. import utils
from dlm.utils.utils_data import get_tokenizer


# Create SAFE converter once (reused for all conversions)
_SAFE_CONVERTER = sf.SAFEConverter(slicer=None, ignore_stereo=True)


# Element ordering conversion for HDF5 files created with ms_pred.common ordering
# ms_pred.common: ['C', 'N', 'P', 'O', 'S', 'Si', 'I', 'H', 'Cl', 'F', 'Br', 'B', 'Se', 'Fe', 'Co', 'As', 'Na', 'K']
# mist.utils:     ['C', 'H', 'As', 'B', 'Br', 'Cl', 'Co', 'F', 'Fe', 'I', 'K', 'N', 'Na', 'O', 'P', 'S', 'Se', 'Si']
# This mapping converts from ms_pred.common order to mist.utils order
# Element at position i in ms_pred.common goes to position ELEMENT_REORDER_IDX[i] in mist.utils
ELEMENT_REORDER_IDX = np.array([0, 11, 14, 13, 15, 17, 9, 1, 5, 7, 4, 3, 16, 8, 6, 2, 12, 10], dtype=np.int64)


def reorder_element_vectors(form_vecs: np.ndarray) -> np.ndarray:
    """Reorder element vectors from ms_pred.common order to mist.utils order.
    
    Args:
        form_vecs: Array of shape (N, 18) with element counts in ms_pred.common order
        
    Returns:
        Array of shape (N, 18) with element counts in mist.utils order
    """
    if form_vecs.ndim == 1:
        # Single vector
        new_vec = np.zeros_like(form_vecs)
        for i, target_idx in enumerate(ELEMENT_REORDER_IDX):
            new_vec[target_idx] = form_vecs[i]
        return new_vec
    else:
        # Batch of vectors (N, 18)
        new_vecs = np.zeros_like(form_vecs)
        for i, target_idx in enumerate(ELEMENT_REORDER_IDX):
            new_vecs[:, target_idx] = form_vecs[:, i]
        return new_vecs


def smiles_to_safe(smiles: str) -> Optional[str]:
    """Convert SMILES to SAFE representation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        safe_str = _SAFE_CONVERTER.encoder(smiles, allow_empty=True)
        return safe_str
    except Exception as e:
        logging.debug(f"Failed to convert SMILES to SAFE: {smiles}, error: {e}")
        return None


class HDF5SpectrumDataset(Dataset):
    """Dataset that loads spectra from a merged HDF5 file.
    
    This is more efficient than loading many small JSON files since all data
    is stored sequentially in a single file.
    
    Args:
        hdf5_path: Path to the merged HDF5 file
        split_indices: Optional list of indices to use (for train/val/test splits)
        tokenizer: SAFE tokenizer for molecule tokenization
        max_length: Maximum sequence length for tokenization
        augment_data: Whether to apply data augmentation
        remove_prob: Probability of removing peaks during augmentation
        inten_prob: Probability of rescaling intensities
        cls_type: Type of CLS token ("ms1" or "zeros")
        reorder_elements: Whether to convert element vectors from ms_pred.common to mist.utils order.
                         Set to True for HDF5 files created with older merge_hdf5.py versions.
                         Set to 'auto' to auto-detect based on mass verification.
    """
    
    cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}
    cls_type_idx = cat_types.get("cls")
    
    def __init__(
        self,
        hdf5_path: str,
        split_indices: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 256,
        augment_data: bool = False,
        remove_prob: float = 0.1,
        inten_prob: float = 0.1,
        cls_type: str = "ms1",
        reorder_elements: str = "auto",
        **kwargs,
    ):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_data = augment_data
        self.remove_prob = remove_prob
        self.inten_prob = inten_prob
        self.cls_type = cls_type
        self.train_mode = False
        
        # Open HDF5 file
        self.h5_file = h5py.File(self.hdf5_path, 'r')
        self.spectra_grp = self.h5_file['spectra']
        self.num_spectra = self.h5_file.attrs['num_spectra']
        self.element_dim = self.h5_file.attrs.get('element_dim', 18)
        
        # Check if element reordering is needed
        self.reorder_elements = self._detect_element_order(reorder_elements)
        if self.reorder_elements:
            logging.info(f"HDF5SpectrumDataset: Element reordering enabled (ms_pred.common -> mist.utils)")
        
        # Determine indices to use
        if split_indices is not None:
            self.indices = split_indices
        else:
            self.indices = list(range(self.num_spectra))
        
        # Pre-load metadata and filter valid samples
        self._load_metadata()
    
    def _detect_element_order(self, reorder_setting: str) -> bool:
        """Detect if element reordering is needed.
        
        Args:
            reorder_setting: 'auto', True, or False
            
        Returns:
            True if reordering is needed, False otherwise
        """
        if reorder_setting is True or reorder_setting == "true":
            return True
        if reorder_setting is False or reorder_setting == "false":
            return False
        
        # Auto-detect by checking if stored element_order attr matches mist.utils
        stored_order = self.h5_file.attrs.get('element_order', None)
        if stored_order is not None:
            mist_order = utils.VALID_ELEMENTS
            if list(stored_order) == list(mist_order):
                return False  # Already in correct order
            else:
                return True  # Needs reordering
        
        # If no element_order attr, check by mass verification on first sample
        try:
            spec_grp = self.spectra_grp['0']
            f_data = spec_grp['f'][:]
            u_data = spec_grp['u'][:]
            
            if len(f_data) == 0 or len(u_data) == 0:
                return False  # Can't verify, assume correct
            
            recorded_mass = f_data[0, 0]
            computed_mass_mist = (u_data[0].astype(float) * utils.VALID_MONO_MASSES).sum()
            
            # If computed mass with mist.utils order matches recorded mass, no reorder needed
            if np.isclose(recorded_mass, computed_mass_mist, rtol=1e-4):
                return False
            
            # Check if reordering fixes it
            reordered_vec = reorder_element_vectors(u_data[0])
            computed_mass_reordered = (reordered_vec.astype(float) * utils.VALID_MONO_MASSES).sum()
            
            if np.isclose(recorded_mass, computed_mass_reordered, rtol=1e-4):
                logging.warning(
                    f"HDF5SpectrumDataset: Detected element order mismatch. "
                    f"Recorded mass={recorded_mass:.2f}, direct computation={computed_mass_mist:.2f}, "
                    f"after reorder={computed_mass_reordered:.2f}. Enabling element reordering."
                )
                return True
            
            # Neither matches well, log warning and don't reorder
            logging.warning(
                f"HDF5SpectrumDataset: Could not verify element order. "
                f"Recorded mass={recorded_mass:.2f}, computed={computed_mass_mist:.2f}"
            )
            return False
            
        except Exception as e:
            logging.warning(f"HDF5SpectrumDataset: Error detecting element order: {e}")
            return False
    
    def _load_metadata(self):
        """Pre-load metadata for all spectra and filter invalid samples."""
        valid_indices = []
        self.smiles_list = []
        self.formula_list = []
        self.safe_list = []
        self.inchikey_list = []
        
        for idx in self.indices:
            spec_grp = self.spectra_grp[str(idx)]
            smiles = spec_grp.attrs['smiles']
            formula = spec_grp.attrs['formula']
            inchikey = spec_grp.attrs['inchikey']
            
            # Try to convert to SAFE
            safe_str = smiles_to_safe(smiles)
            if safe_str is not None:
                valid_indices.append(idx)
                self.smiles_list.append(smiles)
                self.formula_list.append(formula)
                self.safe_list.append(safe_str)
                self.inchikey_list.append(inchikey)
        
        self.indices = valid_indices
        logging.info(f"HDF5SpectrumDataset: {len(valid_indices)}/{len(self.indices)} samples valid after SAFE conversion")
    
    def set_train_mode(self, train_mode: bool):
        """Set training mode (for data augmentation)."""
        self.train_mode = train_mode
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _augment_spectrum(self, form_vecs: np.ndarray, intens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to spectrum."""
        if len(form_vecs) == 0:
            return form_vecs, intens
        
        # Random peak removal
        keep_prob = 1 - self.remove_prob
        num_to_keep = np.random.binomial(n=len(form_vecs), p=keep_prob)
        num_to_keep = max(1, num_to_keep)  # Keep at least one peak
        
        # Weight by intensity for selection
        keep_probs = np.exp(intens + 1e-5)
        keep_probs = keep_probs / keep_probs.sum()
        
        keep_inds = np.random.choice(
            len(form_vecs), size=num_to_keep, replace=False, p=keep_probs
        )
        
        form_vecs = form_vecs[keep_inds]
        intens = intens[keep_inds]
        
        # Random intensity rescaling
        rescale_mask = np.random.random(len(intens)) < self.inten_prob
        scale_factors = np.random.normal(loc=1, size=len(intens))
        scale_factors = np.clip(scale_factors, 0, None)
        scale_factors[~rescale_mask] = 1.0
        
        intens = intens * scale_factors
        intens = intens / (intens.max() + 1e-12)
        
        return form_vecs, intens
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        actual_idx = self.indices[idx]
        spec_grp = self.spectra_grp[str(actual_idx)]
        
        # Load data
        f_data = spec_grp['f'][:]  # [masses_no_adduct, intens]
        u_data = spec_grp['u'][:]  # frag_form_vecs
        
        # Extract components
        masses_no_adduct = f_data[:, 0]
        intens = f_data[:, 1]
        form_vecs = u_data.astype(np.float32)
        
        # Apply element reordering if needed (ms_pred.common -> mist.utils order)
        if self.reorder_elements:
            form_vecs = reorder_element_vectors(form_vecs)
        
        # Get metadata
        formula = self.formula_list[idx]
        safe_str = self.safe_list[idx]
        adduct = spec_grp.attrs['adduct']
        instrument = spec_grp.attrs.get('instrument', 'Orbitrap')
        
        # Apply augmentation if in training mode
        if self.train_mode and self.augment_data:
            form_vecs, intens = self._augment_spectrum(form_vecs, intens)
        
        # Get root formula info
        root_form_vec = utils.formula_to_dense(formula)
        root_ion_idx = utils.get_ion_idx(adduct)
        root_mass = (root_form_vec * utils.VALID_MONO_MASSES).sum()
        
        # Build peak type vector (all frags)
        type_vec = np.array([self.cat_types["frags"]] * len(form_vecs))
        ion_vec = [root_ion_idx] * len(form_vecs)
        instr_idx = utils.get_instr_idx(instrument)
        
        # Add CLS token
        if self.cls_type == "ms1":
            cls_ind = self.cat_types.get("cls")
            intens = np.append(intens, 1.0)
            type_vec = np.append(type_vec, cls_ind)
            form_vecs = np.vstack([form_vecs, root_form_vec])
            ion_vec.append(root_ion_idx)
        elif self.cls_type == "zeros":
            cls_ind = self.cat_types.get("cls")
            intens = np.append(intens, 0.0)
            type_vec = np.append(type_vec, cls_ind)
            form_vecs = np.vstack([form_vecs, np.zeros_like(root_form_vec)])
            ion_vec.append(root_ion_idx)
        
        # Build output
        spec_features = {
            "peak_type": type_vec.astype(np.int64),
            "form_vec": form_vecs.astype(np.float32),
            "ion_vec": ion_vec,
            "frag_intens": intens.astype(np.float32),
            "name": str(actual_idx),
            "instrument": instr_idx,
        }
        
        return {
            "spec": spec_features,
            "formula": formula,
            "safe": safe_str,
        }
    
    def __del__(self):
        """Close HDF5 file on deletion."""
        try:
            if hasattr(self, 'h5_file') and self.h5_file and self.h5_file.id.valid:
                self.h5_file.close()
        except Exception:
            pass  # Ignore errors during cleanup


def _collate_hdf5_spec2mol(
    input_batch: List[Dict],
    tokenizer,
    max_length: int,
) -> Dict:
    """Collate function for HDF5 Spec2Mol dataset."""
    # Collate spectra features
    names = [item["spec"]["name"] for item in input_batch]
    peak_form_tensors = [torch.from_numpy(item["spec"]["form_vec"]) for item in input_batch]
    inten_tensors = [torch.from_numpy(item["spec"]["frag_intens"]) for item in input_batch]
    type_tensors = [torch.from_numpy(item["spec"]["peak_type"]) for item in input_batch]
    instrument_tensors = torch.FloatTensor([item["spec"]["instrument"] for item in input_batch])
    ion_tensors = [torch.FloatTensor(item["spec"]["ion_vec"]) for item in input_batch]
    
    # Pad sequences
    peak_form_lens = np.array([i.shape[0] for i in peak_form_tensors])
    max_len = np.max(peak_form_lens)
    padding_amts = max_len - peak_form_lens
    
    type_tensors = [
        torch.nn.functional.pad(i, (0, pad_len))
        for i, pad_len in zip(type_tensors, padding_amts)
    ]
    ion_tensors = [
        torch.nn.functional.pad(i, (0, pad_len))
        for i, pad_len in zip(ion_tensors, padding_amts)
    ]
    inten_tensors = [
        torch.nn.functional.pad(i, (0, pad_len))
        for i, pad_len in zip(inten_tensors, padding_amts)
    ]
    peak_form_tensors = [
        torch.nn.functional.pad(i, (0, 0, 0, pad_len))
        for i, pad_len in zip(peak_form_tensors, padding_amts)
    ]
    
    # Stack
    type_tensors = torch.stack(type_tensors, dim=0).long()
    peak_form_tensors = torch.stack(peak_form_tensors, dim=0).float()
    ion_tensors = torch.stack(ion_tensors, dim=0).float()
    inten_tensors = torch.stack(inten_tensors, dim=0).float()
    num_peaks = torch.from_numpy(peak_form_lens).long()
    
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
    
    return {
        "types": type_tensors,
        "form_vec": peak_form_tensors,
        "ion_vec": ion_tensors,
        "intens": inten_tensors,
        "names": names,
        "num_peaks": num_peaks,
        "instruments": instrument_tensors,
        "formula": formulas,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


class HDF5Spec2MolDataModule(pl.LightningDataModule):
    """DataModule for Spec2Mol training with HDF5 data.
    
    Loads spectra from a single merged HDF5 file, which is more efficient
    than loading many small JSON files.
    
    Args:
        cfg: Configuration object with:
            - dataset.hdf5_path: Path to merged HDF5 file
            - dataset.split_file: Path to split TSV (or None for random split)
            - train/eval batch sizes, num_workers, etc.
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
        self.hdf5_path = dataset_cfg.hdf5_path
        self.split_file = dataset_cfg.get('split_file', None)
        
        # Augmentation settings
        self.augment_data = dataset_cfg.get('augment_data', False)
        self.remove_prob = dataset_cfg.get('remove_prob', 0.1)
        self.inten_prob = dataset_cfg.get('inten_prob', 0.1)
        self.cls_type = dataset_cfg.get('cls_type', 'ms1')
        
        # Element reordering setting (for HDF5 files created with ms_pred.common order)
        # Options: 'auto' (default), True, False
        self.reorder_elements = dataset_cfg.get('reorder_elements', 'auto')
        
        # Load splits
        train_indices, val_indices, test_indices = self._get_split_indices()
        
        # Create datasets
        self.train_dataset = HDF5SpectrumDataset(
            hdf5_path=self.hdf5_path,
            split_indices=train_indices,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment_data=self.augment_data,
            remove_prob=self.remove_prob,
            inten_prob=self.inten_prob,
            cls_type=self.cls_type,
            reorder_elements=self.reorder_elements,
        )
        self.val_dataset = HDF5SpectrumDataset(
            hdf5_path=self.hdf5_path,
            split_indices=val_indices,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment_data=False,
            cls_type=self.cls_type,
            reorder_elements=self.reorder_elements,
        )
        self.test_dataset = HDF5SpectrumDataset(
            hdf5_path=self.hdf5_path,
            split_indices=test_indices,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            augment_data=False,
            cls_type=self.cls_type,
            reorder_elements=self.reorder_elements,
        )
        
        # Set train mode
        self.train_dataset.set_train_mode(True)
        self.val_dataset.set_train_mode(False)
        self.test_dataset.set_train_mode(False)
        
        logging.info(f"HDF5Spec2MolDataModule: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
    
    def _get_split_indices(self) -> Tuple[List[int], List[int], List[int]]:
        """Get train/val/test split indices."""
        # Open HDF5 to get total count and inchikeys
        with h5py.File(self.hdf5_path, 'r') as f:
            num_spectra = f.attrs['num_spectra']
            spectra_grp = f['spectra']
            
            # Build inchikey to indices mapping
            inchikey_to_indices = {}
            for idx in range(num_spectra):
                spec_grp = spectra_grp[str(idx)]
                inchikey = spec_grp.attrs['inchikey']
                if inchikey not in inchikey_to_indices:
                    inchikey_to_indices[inchikey] = []
                inchikey_to_indices[inchikey].append(idx)
        
        if self.split_file and os.path.exists(self.split_file):
            # Use provided split file
            import pandas as pd
            split_df = pd.read_csv(self.split_file, sep='\t')
            
            train_indices = []
            val_indices = []
            test_indices = []
            
            for _, row in split_df.iterrows():
                # The split file should have 'inchikey' and 'split' columns
                # Or 'spec' (which we treat as formula) and 'split'
                if 'inchikey' in split_df.columns:
                    key = row['inchikey']
                elif 'spec' in split_df.columns:
                    # Need to map spec to inchikey - just use all indices
                    key = row.get('inchikey', None)
                else:
                    continue
                
                if key is None or key not in inchikey_to_indices:
                    continue
                
                split = row['split']
                indices = inchikey_to_indices[key]
                
                if split == 'train':
                    train_indices.extend(indices)
                elif split == 'val':
                    val_indices.extend(indices)
                elif split == 'test':
                    test_indices.extend(indices)
            
            return train_indices, val_indices, test_indices
        else:
            # Random split by inchikey (80/10/10)
            all_inchikeys = list(inchikey_to_indices.keys())
            random.seed(42)
            random.shuffle(all_inchikeys)
            
            n = len(all_inchikeys)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            
            train_keys = all_inchikeys[:n_train]
            val_keys = all_inchikeys[n_train:n_train + n_val]
            test_keys = all_inchikeys[n_train + n_val:]
            
            train_indices = [idx for k in train_keys for idx in inchikey_to_indices[k]]
            val_indices = [idx for k in val_keys for idx in inchikey_to_indices[k]]
            test_indices = [idx for k in test_keys for idx in inchikey_to_indices[k]]
            
            return train_indices, val_indices, test_indices
    
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
        """Create training dataloader."""
        return self._get_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return self._get_dataloader(self.val_dataset, shuffle=False)
    
    def test_dataloader(self):
        """Create test dataloader."""
        return self._get_dataloader(self.test_dataset, shuffle=False)


