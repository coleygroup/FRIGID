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

"""Dataset classes for mass spectrometry data."""

from pathlib import Path
import logging
from functools import partial
from typing import Optional, List, Tuple, Set, Callable
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import featurizers
from .data import Spectra, Mol


import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging
from tqdm import tqdm

def get_paired_spectra(
    labels_file: str,
    spec_folder: str = None,
    max_count: Optional[int] = None,
    allow_none_smiles: bool = False,
    prog_bars: bool = True,
    split_file: Optional[str] = None,
    split_val: Optional[str] = None,
    **kwargs,
) -> Tuple[List, List]:
    """Load paired spectra and molecule data with optional split filtering.

    Args:
        labels_file: Path to TSV file with labels
        spec_folder: Path to folder with spectrum files
        max_count: Maximum number of spectra to load
        allow_none_smiles: Whether to allow entries without SMILES
        prog_bars: Whether to show progress bars
        split_file: Path to TSV file containing 'name' and 'split' columns
        split_val: The specific split to load (e.g., 'test', 'train', 'val')

    Returns:
        Tuple of (spectra_list, mol_list)
    """
    # 1. Load labels
    compound_id_file = pd.read_csv(labels_file, sep="\t").astype(str)

    # 2. OPTIMIZATION: Filter labels by split immediately
    if split_file is not None and split_val is not None:
        logging.info(f"Filtering for split: {split_val}")
        split_df = pd.read_csv(split_file, sep="\t").astype(str)
        
        # specific names belonging to the requested split
        split_names = set(split_df[split_df["split"] == split_val]["name"])
        
        # Filter the main labels file to only include these rows
        compound_id_file = compound_id_file[compound_id_file["spec"].isin(split_names)]

    # Create mappings based on the (potentially filtered) dataframe
    name_to_formula = dict(compound_id_file[["spec", "formula"]].values)

    name_to_smiles = {}
    if "smiles" in compound_id_file.keys():
        name_to_smiles = dict(compound_id_file[["spec", "smiles"]].values)

    name_to_inchikey = {}
    if "inchikey" in compound_id_file.keys():
        name_to_inchikey = dict(compound_id_file[["spec", "inchikey"]].values)

    name_to_instrument = {}
    if "instrument" in compound_id_file.keys():
        name_to_instrument = dict(compound_id_file[["spec", "instrument"]].values)

    logging.info("Loading paired specs")
    spec_folder = Path(spec_folder) if spec_folder is not None else None

    spectra_files = []
    
    # 3. OPTIMIZATION: Construct paths directly instead of globbing
    if spec_folder is not None and spec_folder.exists():
        # Iterate only over the filtered keys
        for name in tqdm(name_to_formula.keys(), desc="Finding spectra files", total=len(name_to_formula), leave=False):
            # Construct the expected path
            file_path = spec_folder / f"{name}.ms"
            if file_path.exists():
                spectra_files.append(file_path.resolve())
    else:
        logging.info(
            f"Unable to find spec folder {str(spec_folder)}, adding placeholders"
        )
        spectra_files = [Path(f"{i}.ms") for i in name_to_formula]

    if max_count is not None:
        spectra_files = spectra_files[:max_count]

    # Get file name from Path obj
    get_name = lambda x: x.name.split(".")[0]

    spectra_names = [get_name(spectra_file) for spectra_file in spectra_files]
    spectra_formulas = [name_to_formula[spectra_name] for spectra_name in spectra_names]
    spectra_instruments = [
        name_to_instrument.get(spectra_name, "") for spectra_name in spectra_names
    ]

    logging.info(f"Converting {len(spectra_files)} paired samples into Spectra objects")

    tq = tqdm if prog_bars else lambda x: x

    spectra_list = [
        Spectra(
            spectra_name=spectra_name,
            spectra_file=str(spectra_file),
            spectra_formula=spectra_formula,
            instrument=instrument,
            **kwargs,
        )
        for spectra_name, spectra_file, spectra_formula, instrument in tq(
            zip(spectra_names, spectra_files, spectra_formulas, spectra_instruments)
        )
    ]

    # Create molecules
    spectra_smiles = [name_to_smiles.get(j, None) for j in spectra_names]
    spectra_inchikey = [name_to_inchikey.get(j, None) for j in spectra_names]
    if not allow_none_smiles:
        mol_list = [
            Mol.MolFromSmiles(smiles, inchikey=inchikey)
            for smiles, inchikey in tq(zip(spectra_smiles, spectra_inchikey))
            if smiles is not None
        ]
        spectra_list = [
            spec
            for spec, smi in tq(zip(spectra_list, spectra_smiles))
            if smi is not None
        ]
    else:
        mol_list = [
            Mol.MolFromSmiles(smiles, inchikey=inchikey)
            if smiles is not None
            else Mol.MolFromSmiles("")
            for smiles, inchikey in tq(zip(spectra_smiles, spectra_inchikey))
        ]
        spectra_list = [spec for spec, smi in tq(zip(spectra_list, spectra_smiles))]

    # Remove any samples that contain atoms other than supported set
    VALID_ATOMS = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
    updated_spectra_list = []
    updated_mol_list = []
    for spec, mol in zip(spectra_list, mol_list):
        if mol is not None:
            valid = True
            for atom in mol.get_rdkit_mol().GetAtoms():
                if atom.GetSymbol() not in VALID_ATOMS:
                    valid = False
                    break
            if valid:
                updated_spectra_list.append(spec)
                updated_mol_list.append(mol)

    logging.info("Done creating spectra objects")
    return (updated_spectra_list, updated_mol_list)


class SpectraMolDataset(Dataset):
    """Dataset for paired spectra and molecules.

    Args:
        spectra_mol_list: List of (Spectra, Mol) pairs
        featurizer: PairedFeaturizer instance
        include_graph: Whether to include graph features
    """

    def __init__(
        self,
        spectra_mol_list: List[Tuple[Spectra, Mol]],
        featurizer: featurizers.PairedFeaturizer,
        include_graph: bool = False,
        **kwargs,
    ):
        super().__init__()
        spectra_list, mol_list = list(zip(*spectra_mol_list))
        self.spectra_list = np.array(spectra_list)
        self.mol_list = np.array(mol_list)
        self.smi_list = np.array([mol.get_smiles() for mol in self.mol_list])
        self.inchikey_list = np.array([mol.get_inchikey() for mol in self.mol_list])
        self.orig_len = len(self.mol_list)
        self.len = len(self.mol_list)

        # Extract all chem formulas
        self.chem_formulas = set()
        for spec in spectra_list:
            formula = spec.get_spectra_formula()
            self.chem_formulas.add(formula)

        # Verify same length
        assert len(self.spectra_list) == len(self.mol_list)

        # Store paired featurizer
        self.featurizer = featurizer
        self.train_mode = False
        self.include_graph = include_graph

        # Save for subsetting
        self.kwargs = kwargs

    def set_train_mode(self, train_mode: bool):
        """Set whether dataset is in training mode."""
        self.train_mode = train_mode

    def get_spectra_list(self) -> List[Spectra]:
        """Get list of spectra."""
        return self.spectra_list

    def get_featurizer(self) -> featurizers.PairedFeaturizer:
        """Get featurizer."""
        return self.featurizer

    def set_featurizer(self, featurizer: featurizers.PairedFeaturizer):
        """Set featurizer."""
        self.featurizer = featurizer

    def get_spectra_names(self) -> List[str]:
        """Get spectrum names."""
        return [i.spectra_name for i in self.spectra_list]

    def get_mol_list(self) -> List[Mol]:
        """Get molecule list."""
        return self.mol_list

    def get_smi_list(self) -> List[str]:
        """Get SMILES list."""
        return self.smi_list

    def get_inchikey_list(self) -> List[str]:
        """Get InChIKey list."""
        return self.inchikey_list

    def get_all_formulas(self) -> Set[str]:
        """Get set of all chemical formulas."""
        return self.chem_formulas

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get item by index."""
        mol = self.mol_list[idx]
        spec = self.spectra_list[idx]

        mol_features = self.featurizer.featurize_mol(mol, train_mode=self.train_mode)
        spec_features = self.featurizer.featurize_spec(spec, train_mode=self.train_mode)

        output = {
            "spec": [spec_features],
            "mol": [mol_features],
            "spec_indices": [0],
            "mol_indices": [0],
            "matched": [True],
        }

        if self.include_graph:
            graph_features = self.featurizer.featurize_graph(mol, train_mode=self.train_mode)
            output["graph"] = [graph_features]

        return output


def _collate_pairs(
    input_batch: List[dict], mol_collate_fn: Callable, spec_collate_fn: Callable
) -> dict:
    """Collate pairs of spectra and molecules.

    Args:
        input_batch: List of dataset outputs
        mol_collate_fn: Molecule collate function
        spec_collate_fn: Spectra collate function

    Returns:
        Batched dictionary
    """
    # Spectra loading
    spec_dict = spec_collate_fn([j for jj in input_batch for j in jj["spec"]])

    # Mol loading
    mol_dict = mol_collate_fn([j for jj in input_batch for j in jj["mol"]])

    # Get the paired molecule and paired spectra indices
    mol_indices = torch.tensor([j for jj in input_batch for j in jj["mol_indices"]])
    spec_indices = torch.tensor([j for jj in input_batch for j in jj["spec_indices"]])

    # Calculate number of unique molecules and spec in each entry
    len_mols = torch.tensor([len(j["mol"]) for j in input_batch])
    len_specs = torch.tensor([len(j["spec"]) for j in input_batch])

    # Number of pairs
    num_pairs = torch.tensor([len(j["mol_indices"]) for j in input_batch])

    # Modify mol_indices pairs s.t. they are consistent with batch
    expanded_indices = torch.arange(len(len_mols)).repeat_interleave(num_pairs)
    addition_factor = torch.cumsum(len_mols, 0)
    addition_factor = torch.nn.functional.pad(addition_factor[:-1], (1, 0))

    mol_indices = mol_indices + addition_factor[expanded_indices]

    # Modify spec_indices pairs s.t. they are consistent with batch
    expanded_indices = torch.arange(len(len_specs)).repeat_interleave(num_pairs)
    addition_factor = torch.cumsum(len_specs, 0)
    addition_factor = torch.nn.functional.pad(addition_factor[:-1], (1, 0))

    spec_indices = spec_indices + addition_factor[expanded_indices]

    # Matched loading
    matched = torch.tensor([j for jj in input_batch for j in jj["matched"]])

    # Create output
    base_dict = {
        "matched": matched,
        "spec_indices": spec_indices,
        "mol_indices": mol_indices,
    }
    base_dict.update(spec_dict)
    base_dict.update(mol_dict)

    return base_dict


def get_paired_loader(
    dataset: SpectraMolDataset,
    shuffle: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for paired dataset.

    Args:
        dataset: SpectraMolDataset instance
        shuffle: Whether to shuffle
        batch_size: Batch size
        num_workers: Number of data loading workers
        persistent_workers: Whether to keep workers alive

    Returns:
        DataLoader instance
    """
    mol_collate_fn = dataset.get_featurizer().get_mol_collate()
    spec_collate_fn = dataset.get_featurizer().get_spec_collate()
    collate_pairs = partial(
        _collate_pairs,
        mol_collate_fn=mol_collate_fn,
        spec_collate_fn=spec_collate_fn,
    )

    _persistent_workers = False
    if num_workers > 0 and persistent_workers:
        _persistent_workers = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        persistent_workers=_persistent_workers,
    )


def _collate_pairs_graph(
    input_batch: List[dict],
    mol_collate_fn: Callable,
    spec_collate_fn: Callable,
    graph_collate_fn: Callable,
) -> dict:
    """Collate pairs of spectra, molecules, and graphs.

    Args:
        input_batch: List of dataset outputs
        mol_collate_fn: Molecule collate function
        spec_collate_fn: Spectra collate function
        graph_collate_fn: Graph collate function

    Returns:
        Batched dictionary with graph data
    """
    # Spectra loading
    spec_dict = spec_collate_fn([j for jj in input_batch for j in jj["spec"]])

    # Mol loading
    mol_dict = mol_collate_fn([j for jj in input_batch for j in jj["mol"]])

    # Graph loading
    graph_batch = graph_collate_fn([j for jj in input_batch for j in jj["graph"]])

    # Get the paired molecule and paired spectra indices
    mol_indices = torch.tensor([j for jj in input_batch for j in jj["mol_indices"]])
    spec_indices = torch.tensor([j for jj in input_batch for j in jj["spec_indices"]])

    # Calculate number of unique molecules and spec in each entry
    len_mols = torch.tensor([len(j["mol"]) for j in input_batch])
    len_specs = torch.tensor([len(j["spec"]) for j in input_batch])

    # Number of pairs
    num_pairs = torch.tensor([len(j["mol_indices"]) for j in input_batch])

    # Modify mol_indices pairs s.t. they are consistent with batch
    expanded_indices = torch.arange(len(len_mols)).repeat_interleave(num_pairs)
    addition_factor = torch.cumsum(len_mols, 0)
    addition_factor = torch.nn.functional.pad(addition_factor[:-1], (1, 0))

    mol_indices = mol_indices + addition_factor[expanded_indices]

    # Modify spec_indices pairs s.t. they are consistent with batch
    expanded_indices = torch.arange(len(len_specs)).repeat_interleave(num_pairs)
    addition_factor = torch.cumsum(len_specs, 0)
    addition_factor = torch.nn.functional.pad(addition_factor[:-1], (1, 0))

    spec_indices = spec_indices + addition_factor[expanded_indices]

    # Matched loading
    matched = torch.tensor([j for jj in input_batch for j in jj["matched"]])

    # Create output
    base_dict = {
        "matched": matched,
        "spec_indices": spec_indices,
        "mol_indices": mol_indices,
    }
    base_dict.update(spec_dict)
    base_dict.update(mol_dict)
    base_dict["graph"] = graph_batch

    return base_dict


def get_paired_loader_graph(
    dataset: SpectraMolDataset,
    shuffle: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for paired dataset with graph data.

    Args:
        dataset: SpectraMolDataset instance with graph features enabled
        shuffle: Whether to shuffle
        batch_size: Batch size
        num_workers: Number of data loading workers
        persistent_workers: Whether to keep workers alive

    Returns:
        DataLoader instance with graph batching
    """
    mol_collate_fn = dataset.get_featurizer().get_mol_collate()
    spec_collate_fn = dataset.get_featurizer().get_spec_collate()
    graph_collate_fn = dataset.get_featurizer().get_graph_collate()
    
    collate_pairs = partial(
        _collate_pairs_graph,
        mol_collate_fn=mol_collate_fn,
        spec_collate_fn=spec_collate_fn,
        graph_collate_fn=graph_collate_fn,
    )

    _persistent_workers = False
    if num_workers > 0 and persistent_workers:
        _persistent_workers = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        persistent_workers=_persistent_workers,
    )
