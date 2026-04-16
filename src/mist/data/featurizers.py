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

"""Featurizers for spectra and molecules in MIST encoder."""

from pathlib import Path
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
import json

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdchem import BondType as BT

from .. import utils
from . import data

# Atom and bond type mappings for graph featurization
ATOM_DECODER = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
ATOM_TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def get_mol_featurizer(mol_features, **kwargs):
    """Get molecule featurizer by name."""
    return {
        "none": NoneFeaturizer,
        "fingerprint": FingerprintFeaturizer,
    }[mol_features](**kwargs)


def get_spec_featurizer(spec_features, **kwargs):
    """Get spectra featurizer by name."""
    return {
        "none": NoneFeaturizer,
        "peakformula": PeakFormula,
    }[spec_features](**kwargs)


def get_paired_featurizer(spec_features, mol_features, **kwargs):
    """Create paired featurizer for spectra and molecules.

    Args:
        spec_features: Spectra featurizer type
        mol_features: Molecule featurizer type
        **kwargs: Additional arguments

    Returns:
        PairedFeaturizer instance
    """
    mol_featurizer = get_mol_featurizer(mol_features, **kwargs)
    spec_featurizer = get_spec_featurizer(spec_features, **kwargs)
    paired_featurizer = PairedFeaturizer(spec_featurizer, mol_featurizer, **kwargs)
    return paired_featurizer


class PairedFeaturizer(object):
    """Featurizer for paired spectra and molecule data."""

    def __init__(self, spec_featurizer, mol_featurizer, graph_featurizer=None, **kwarg):
        self.spec_featurizer = spec_featurizer
        self.mol_featurizer = mol_featurizer
        self.graph_featurizer = graph_featurizer

    def featurize_mol(self, mol: data.Mol, **kwargs) -> Dict:
        """Featurize a molecule."""
        return self.mol_featurizer.featurize(mol, **kwargs)

    def featurize_spec(self, mol: data.Mol, **kwargs) -> Dict:
        """Featurize a spectrum."""
        return self.spec_featurizer.featurize(mol, **kwargs)

    def featurize_graph(self, mol: data.Mol, **kwargs):
        """Featurize a molecule as a graph."""
        if self.graph_featurizer is not None:
            return self.graph_featurizer.featurize(mol, **kwargs)
        return None

    def get_mol_collate(self) -> Callable:
        """Get molecule collate function."""
        return self.mol_featurizer.collate_fn

    def get_spec_collate(self) -> Callable:
        """Get spectra collate function."""
        return self.spec_featurizer.collate_fn

    def get_graph_collate(self) -> Callable:
        """Get graph collate function."""
        if self.graph_featurizer is not None:
            return self.graph_featurizer.collate_fn
        return None

    def set_spec_featurizer(self, spec_featurizer):
        """Set spectra featurizer."""
        self.spec_featurizer = spec_featurizer

    def set_mol_featurizer(self, mol_featurizer):
        """Set molecule featurizer."""
        self.mol_featurizer = mol_featurizer

    def set_graph_featurizer(self, graph_featurizer):
        """Set graph featurizer."""
        self.graph_featurizer = graph_featurizer


class Featurizer(ABC):
    """Abstract base class for featurizers."""

    def __init__(self, cache_featurizers: bool = False, **kwargs):
        super().__init__()
        self.cache_featurizers = cache_featurizers
        self.cache = {}

    @abstractmethod
    def _encode(self, obj: object) -> str:
        """Encode object into a string representation."""
        raise NotImplementedError()

    def _featurize(self, obj: object) -> Dict:
        """Internal featurize class that does not utilize the cache."""
        return {}

    def featurize(self, obj: object, train_mode=False, **kwargs) -> Dict:
        """Featurize a single object."""
        encoded_obj = self._encode(obj)

        if self.cache_featurizers:
            if encoded_obj in self.cache:
                featurized = self.cache[encoded_obj]
            else:
                featurized = self._featurize(obj)
                self.cache[encoded_obj] = featurized
        else:
            featurized = self._featurize(obj)

        return featurized


class NoneFeaturizer(Featurizer):
    """Null featurizer that returns empty dict."""

    def _encode(self, obj) -> str:
        return ""

    @staticmethod
    def collate_fn(objs) -> Dict:
        return {}

    def featurize(self, *args, **kwargs) -> Dict:
        return {}


class MolFeaturizer(Featurizer):
    """Base class for molecule featurizers."""

    def _encode(self, mol: data.Mol) -> str:
        """Encode mol into SMILES repr."""
        smi = mol.get_smiles()
        return smi


class SpecFeaturizer(Featurizer):
    """Base class for spectra featurizers."""

    def _encode(self, spec: data.Spectra) -> str:
        """Encode spectra into name."""
        return spec.get_spec_name()


class FingerprintFeaturizer(MolFeaturizer):
    """Featurizer for molecular fingerprints.

    Computes various molecular fingerprints including Morgan fingerprints.

    Args:
        fp_names: List of fingerprint types to compute
        fp_file: Optional file with precomputed fingerprints
    """

    def __init__(self, fp_names: List[str], fp_file: str = None, **kwargs):
        super().__init__(**kwargs)
        self._fp_cache = {}
        self._morgan_projection = np.random.randn(50, 2048)
        self.fp_names = fp_names
        self.fp_file = fp_file

    @staticmethod
    def collate_fn(mols: List[dict]) -> dict:
        """Collate fingerprints into batch."""
        fp_ar = torch.tensor(np.array(mols))
        return {"mols": fp_ar}

    def featurize_smiles(self, smiles: str, **kwargs) -> np.ndarray:
        """Featurize a SMILES string."""
        mol_obj = data.Mol.MolFromSmiles(smiles)
        return self._featurize(mol_obj)

    def _featurize(self, mol: data.Mol, **kwargs) -> Dict:
        """Compute fingerprint for molecule."""
        fp_list = []
        for fp_name in self.fp_names:
            fingerprint = self._get_fingerprint(mol, fp_name)
            fp_list.append(fingerprint)

        fp = np.concatenate(fp_list)
        return fp

    def _get_morgan_fp_base(self, mol: data.Mol, nbits: int = 2048, radius=2):
        """Get Morgan fingerprint."""
        def fp_fn(m):
            return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

        mol = mol.get_rdkit_mol()
        fingerprint = fp_fn(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _get_morgan_2048(self, mol: data.Mol):
        return self._get_morgan_fp_base(mol, nbits=2048)

    def _get_morgan_4096(self, mol: data.Mol):
        return self._get_morgan_fp_base(mol, nbits=4096)

    @classmethod
    def get_fingerprint_size(cls, fp_names: list = [], **kwargs):
        """Get total fingerprint size for given types."""
        fp_name_to_bits = {
            "morgan256": 256,
            "morgan512": 512,
            "morgan1024": 1024,
            "morgan2048": 2048,
            "morgan4096": 4096,
        }
        num_bits = 0
        for fp_name in fp_names:
            num_bits += fp_name_to_bits.get(fp_name)
        return num_bits

    def _get_fingerprint(self, mol: data.Mol, fp_name: str):
        """Get fingerprint by name."""
        return {
            "morgan2048": self._get_morgan_2048,
            "morgan4096": self._get_morgan_4096,
        }[fp_name](mol)


class PeakFormula(SpecFeaturizer):
    """Featurizer for peak formula annotations.

    Extracts formula assignments for MS/MS peaks from JSON annotation files.

    Args:
        subform_folder: Path to folder with subformula JSON files
        augment_data: Whether to apply data augmentation
        remove_prob: Probability of removing peaks during augmentation
        inten_prob: Probability of rescaling intensities
        cls_type: Type of CLS token ("ms1" or "zeros")
        inten_transform: Intensity transformation ("float", "log", etc.)
        magma_modulo: Dimension for fragment fingerprints (2048 for MSG version)
        max_peaks: Maximum number of peaks to featurize (None for no limit)
    """

    cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}
    num_inten_bins = 10
    num_types = len(cat_types)
    cls_type_idx = cat_types.get("cls")

    num_adducts = len(utils.ION_LST)

    def __init__(
        self,
        subform_folder: str,
        augment_data: bool = False,
        augment_prob: float = 1,
        remove_prob: float = 0.1,
        remove_weights: float = "uniform",
        inten_prob: float = 0.1,
        cls_type: str = "ms1",
        inten_transform: str = "float",
        magma_modulo: int = 2048,
        max_peaks: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cls_type = cls_type
        self.augment_data = augment_data
        self.remove_prob = remove_prob
        self.augment_prob = augment_prob
        self.remove_weights = remove_weights
        self.inten_prob = inten_prob
        self.inten_transform = inten_transform
        self.aug_nbits = magma_modulo
        self.max_peaks = max_peaks

        subform_files = list(Path(subform_folder).glob("*.json"))
        self.spec_name_to_subform_file = {i.stem: i for i in subform_files}

    def _get_peak_dict(self, spec: data.Spectra) -> dict:
        """Load peak annotations from JSON file."""
        spec_name = spec.get_spec_name()

        subform_file = Path(self.spec_name_to_subform_file[spec_name])

        if not subform_file.exists():
            return {}

        with open(subform_file, "r") as fp:
            tree = json.load(fp)

        root_form = tree["cand_form"]
        root_ion = tree["cand_ion"]
        output_tbl = tree["output_tbl"]

        if output_tbl is None:
            frags = []
            intens = []
            ions = []
        else:
            frags = output_tbl["formula"]
            intens = output_tbl["ms2_inten"]
            ions = output_tbl["ions"]

        out_dict = {
            "frags": frags,
            "intens": intens,
            "ions": ions,
            "root_form": root_form,
            "root_ion": root_ion,
        }

        if self.max_peaks is not None:

            # Sort by intensity
            inten_list = list(out_dict["intens"])

            new_order = np.argsort(inten_list)[::-1]
            cutoff_ind = min(len(inten_list) - 1, self.max_peaks)
            new_inds = new_order[:cutoff_ind]

            # Get new frags, intens, ions and assign to outdict
            inten_list = np.array(inten_list)[new_inds].tolist()
            frag_list = np.array(out_dict["frags"])[new_inds].tolist()
            ion_list = np.array(out_dict["ions"])[new_inds].tolist()

            out_dict["frags"] = frag_list
            out_dict["intens"] = inten_list
            out_dict["ions"] = ion_list

        return out_dict

    def augment_peak_dict(self, peak_dict: dict, **kwargs):
        """Apply data augmentation to peak dictionary."""
        frags = np.array(peak_dict["frags"])
        intens = np.array(peak_dict["intens"])
        ions = np.array(peak_dict["ions"])

        # Compute removal probability
        num_modify_peaks = len(frags)
        keep_prob = 1 - self.remove_prob
        num_to_keep = np.random.binomial(n=num_modify_peaks, p=keep_prob)

        if len(frags) == 0:
            return peak_dict

        keep_inds = np.arange(0, num_modify_peaks)

        # Probability weighting
        if self.remove_weights == "quadratic":
            keep_probs = intens[0:].reshape(-1) ** 2 + 1e-9
            keep_probs = keep_probs / keep_probs.sum()
        elif self.remove_weights == "uniform":
            keep_probs = intens[0:] + 1e-9
            keep_probs = np.ones(len(keep_probs)) / len(keep_probs)
        elif self.remove_weights == "exp":
            keep_probs = np.exp(intens[0:].reshape(-1) + 1e-5)
            keep_probs = keep_probs / keep_probs.sum()
        else:
            raise NotImplementedError()

        # Keep indices
        ind_samples = np.random.choice(
            keep_inds, size=num_to_keep, replace=False, p=keep_probs
        )
        frags, intens, ions = frags[ind_samples], intens[ind_samples], ions[ind_samples]

        rescale_prob = np.random.random(len(intens))
        inten_scalar_factor = np.random.normal(loc=1, size=len(intens))
        inten_scalar_factor[inten_scalar_factor <= 0] = 0

        # Where rescale prob is >= self.inten_prob set inten rescale to 1
        inten_scalar_factor[rescale_prob >= self.inten_prob] = 1

        # Rescale intens
        intens = intens * inten_scalar_factor
        new_max = intens.max() + 1e-12 if len(intens) > 0 else 1
        intens /= new_max

        # Replace peak dict with new values
        peak_dict["intens"] = intens
        peak_dict["frags"] = frags
        peak_dict["ions"] = ions

        return peak_dict

    def _featurize(
        self, spec: data.Spectra, train_mode: bool = False, **kwargs
    ) -> Dict:
        """Featurize spectrum with formula annotations."""
        spec_name = spec.get_spec_name()

        # Return get_peak_formulas output
        peak_dict = self._get_peak_dict(spec)

        # Augment peak dict with chem formulae
        if train_mode and self.augment_data:
            augment_peak = np.random.random() < self.augment_prob
            if augment_peak:
                peak_dict = self.augment_peak_dict(peak_dict)

        # Add in chemical formulae
        root = peak_dict["root_form"]

        forms_vec = [utils.formula_to_dense(i) for i in peak_dict["frags"]]
        if len(forms_vec) == 0:
            mz_vec = []
        else:
            mz_vec = (np.array(forms_vec) * utils.VALID_MONO_MASSES).sum(-1).tolist()
        root_vec = utils.formula_to_dense(root)
        root_ion = utils.get_ion_idx(peak_dict["root_ion"])
        root_mass = (root_vec * utils.VALID_MONO_MASSES).sum()
        inten_vec = list(peak_dict["intens"])
        ion_vec = [utils.get_ion_idx(i) for i in peak_dict["ions"]]
        type_vec = len(forms_vec) * [self.cat_types["frags"]]
        instrument = utils.get_instr_idx(spec.get_instrument())

        if self.cls_type == "ms1":
            cls_ind = self.cat_types.get("cls")
            inten_vec.append(1.0)
            type_vec.append(cls_ind)
            forms_vec.append(root_vec)
            mz_vec.append(root_mass)
            ion_vec.append(root_ion)

        elif self.cls_type == "zeros":
            cls_ind = self.cat_types.get("cls")
            inten_vec.append(0.0)
            type_vec.append(cls_ind)
            forms_vec.append(np.zeros_like(root_vec))
            mz_vec.append(0)
            ion_vec.append(root_ion)
        else:
            raise NotImplementedError()

        # Featurize all formulae
        inten_vec = np.array(inten_vec)
        if self.inten_transform == "float":
            self.inten_feats = 1
        elif self.inten_transform == "zero":
            self.inten_feats = 1
            inten_vec = np.zeros_like(inten_vec)
        elif self.inten_transform == "log":
            self.inten_feats = 1
            inten_vec = np.log(inten_vec + 1e-5)
        elif self.inten_transform == "cat":
            self.inten_feats = self.num_inten_bins
            bins = np.linspace(0, 1, self.num_inten_bins)
            inten_vec = np.digitize(inten_vec, bins)
        else:
            raise NotImplementedError()

        forms_vec = np.array(forms_vec)

        # Build output dict
        out_dict = {
            "peak_type": np.array(type_vec),
            "form_vec": forms_vec,
            "ion_vec": ion_vec,
            "frag_intens": inten_vec,
            "name": spec_name,
            "instrument": instrument,
        }
        return out_dict

    def featurize(self, spec: data.Spectra, train_mode=False, **kwargs) -> Dict:
        """Featurize a single spectrum."""
        encoded_obj = self._encode(spec)
        if train_mode:
            featurized = self._featurize(spec, train_mode=train_mode)
        else:
            if self.cache_featurizers:
                if encoded_obj in self.cache:
                    featurized = self.cache[encoded_obj]
                else:
                    featurized = self._featurize(spec)
                    self.cache[encoded_obj] = featurized
            else:
                featurized = self._featurize(spec)

        return featurized

    @staticmethod
    def collate_fn(input_list: List[dict]) -> Dict:
        """Collate peak formula features into batch."""
        names = [j["name"] for j in input_list]
        peak_form_tensors = [torch.from_numpy(j["form_vec"]) for j in input_list]
        inten_tensors = [torch.from_numpy(j["frag_intens"]) for j in input_list]
        type_tensors = [torch.from_numpy(j["peak_type"]) for j in input_list]
        instrument_tensors = torch.FloatTensor([j["instrument"] for j in input_list])
        ion_tensors = [torch.FloatTensor(j["ion_vec"]) for j in input_list]

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

        # Stack everything
        type_tensors = torch.stack(type_tensors, dim=0).long()
        peak_form_tensors = torch.stack(peak_form_tensors, dim=0).float()
        ion_tensors = torch.stack(ion_tensors, dim=0).float()

        inten_tensors = torch.stack(inten_tensors, dim=0).float()
        num_peaks = torch.from_numpy(peak_form_lens).long()

        return_dict = {
            "types": type_tensors,
            "form_vec": peak_form_tensors,
            "ion_vec": ion_tensors,
            "intens": inten_tensors,
            "names": names,
            "num_peaks": num_peaks,
            "instruments": instrument_tensors,
        }

        return return_dict
