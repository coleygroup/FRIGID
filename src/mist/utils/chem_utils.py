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

"""Chemistry utility functions for MIST encoder."""

import re
import numpy as np
import pandas as pd
from functools import reduce
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]

CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))

# Convert to onehot
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6])

NORM_VEC_MASS = np.array(NORM_VEC.tolist() + [1471])

# Assume 64 is the highest repeat of any 1 atom
MAX_ELEMENT_NUM = 64

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

ion_remap = dict(zip(ION_LST, ION_LST))
ion_remap.update(
    {
        "[M+NH4]+": "[M+H3N+H]+",
        "M+H": "[M+H]+",
        "M+Na": "[M+Na]+",
        "M+H-H2O": "[M-H2O+H]+",
        "M-H2O+H": "[M-H2O+H]+",
        "M+NH4": "[M+H3N+H]+",
        "M-2H2O+H": "[M-H4O2+H]+",
        "[M-2H2O+H]+": "[M-H4O2+H]+",
    }
)

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST))))

ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
}

ion_to_add_vec = {
    "[M+H]+": element_to_position["H"],
    "[M+Na]+": element_to_position["Na"],
    "[M+K]+": element_to_position["K"],
    "[M-H2O+H]+": -element_to_position["O"] - element_to_position["H"],
    "[M+H3N+H]+": element_to_position["N"] + element_to_position["H"] * 4,
    "[M]+": np.zeros_like(element_to_position["H"]),
    "[M-H4O2+H]+": -element_to_position["O"] * 2 - element_to_position["H"] * 3,
}

instrument_to_type = defaultdict(lambda: "unknown")
instrument_to_type.update({
    "Thermo Finnigan Velos Orbitrap": "orbitrap",
    "Thermo Finnigan Elite Orbitrap": "orbitrap",
    "Orbitrap Fusion Lumos": "orbitrap",
    "Q-ToF (LCMS)": "qtof",
    "Unknown (LCMS)": "unknown",
    "ion trap": "iontrap",
    "FTICR (LCMS)": "fticr",
    "Bruker Q-ToF (LCMS)": "qtof",
    "Orbitrap (LCMS)": "orbitrap",
})

instruments = sorted(list(set(instrument_to_type.values())))
max_instr_idx = len(instruments) + 1
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments))))


# Define rdbe mult
rdbe_mult = np.zeros_like(ELEMENT_VECTORS[0])
els = ["C", "N", "P", "H", "Cl", "Br", "I", "F"]
weights = [2, 1, 1, -1, -1, -1, -1, -1]
for k, v in zip(els, weights):
    rdbe_mult[element_to_ind[k]] = v


def get_ion_idx(ionization: str) -> int:
    """Map ionization to its index in one hot encoding."""
    return ion_to_idx[ionization]


def get_instr_idx(instrument: str) -> int:
    """Map instrument to its index in one hot encoding."""
    inst = instrument_to_type.get(instrument, "unknown")
    return instrument_to_idx[inst]


def has_valid_els(chem_formula: str) -> bool:
    """Check if chemical formula contains only valid elements."""
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        if chem_symbol not in VALID_ELEMENTS:
            return False
    return True


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """Convert chemical formula to dense vector representation.

    Args:
        chem_formula: Input chemical formula string

    Returns:
        np.ndarray of element counts
    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def cross_sum(x, y):
    """Compute cross sum of two arrays."""
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def get_all_subsets_dense(
    dense_formula: str, element_vectors
) -> tuple:
    """Get all subsets of a dense formula vector.

    Args:
        dense_formula: Dense formula vector
        element_vectors: Element basis vectors

    Returns:
        Tuple of (cross_prod, all_masses)
    """
    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = []
    for nonzero_ind in non_zero:
        temp = element_vectors[nonzero_ind] * np.arange(
            0, dense_formula[nonzero_ind] + 1
        ).reshape(-1, 1)
        vectorized_formula.append(temp)

    zero_vec = np.zeros((1, element_vectors.shape[-1]))
    cross_prod = reduce(cross_sum, vectorized_formula, zero_vec)

    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]
    all_masses = cross_prod.dot(VALID_MONO_MASSES)
    return cross_prod, all_masses


def get_all_subsets(chem_formula: str):
    """Get all valid subsets of a chemical formula."""
    dense_formula = formula_to_dense(chem_formula)
    return get_all_subsets_dense(dense_formula, element_vectors=ELEMENT_VECTORS)


def rdbe_filter(cross_prod):
    """Filter by ring and double bond equivalent.

    Args:
        cross_prod: Cross product array to filter

    Returns:
        Indices of valid entries
    """
    rdbe_total = 1 + 0.5 * cross_prod.dot(rdbe_mult)
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds


def formula_to_dense_mass(chem_formula: str) -> np.ndarray:
    """Convert formula to dense representation including mass.

    Args:
        chem_formula: Input chemical formula

    Returns:
        np.ndarray vector including mass dimension
    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position_mass[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position_mass["H"]))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def formula_to_dense_mass_norm(chem_formula: str) -> np.ndarray:
    """Convert formula to normalized dense representation with mass.

    Args:
        chem_formula: Input chemical formula

    Returns:
        Normalized np.ndarray vector
    """
    dense_vec = formula_to_dense_mass(chem_formula)
    dense_vec = dense_vec / NORM_VEC_MASS
    return dense_vec


def formula_mass(chem_formula: str) -> float:
    """Calculate mass from chemical formula."""
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def electron_correct(mass: float) -> float:
    """Subtract the rest mass of an electron."""
    return mass - ELECTRON_MASS


def formula_difference(formula_1, formula_2):
    """Compute formula_1 - formula_2."""
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }

    for k, v in form_2.items():
        form_1[k] = form_1[k] - form_2[k]
    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v > 0])
    return out_formula


def get_mol_from_structure_string(structure_string, structure_type):
    """Get RDKit mol from structure string."""
    if structure_type == "InChI":
        mol = Chem.MolFromInchi(structure_string)
    else:
        mol = Chem.MolFromSmiles(structure_string)
    return mol


def vec_to_formula(form_vec):
    """Convert dense vector back to formula string."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def standardize_form(i):
    """Standardize chemical formula."""
    return vec_to_formula(formula_to_dense(i))


def standardize_adduct(adduct):
    """Standardize adduct notation."""
    adduct = adduct.replace(" ", "")
    adduct = ion_remap.get(adduct, adduct)
    if adduct not in ION_LST:
        raise ValueError(f"Adduct {adduct} not in ION_LST")
    return adduct


def calc_structure_string_type(structure_string):
    """Determine the type of structure string (InChI or SMILES)."""
    structure_type = None
    if pd.isna(structure_string):
        structure_type = "empty"
    elif structure_string.startswith("InChI="):
        structure_type = "InChI"
    elif Chem.MolFromSmiles(structure_string) is not None:
        structure_type = "Smiles"
    return structure_type


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula from molecule."""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def form_from_smi(smi: str) -> str:
    """Get formula from SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return CalcMolFormula(mol)


def inchikey_from_smiles(smi: str) -> str:
    """Get InChIKey from SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)


def contains_metals(formula: str) -> bool:
    """Check if formula contains metals."""
    METAL_RE = "(Fe|Co|Zn|Rh|Pt|Li)"
    return len(re.findall(METAL_RE, formula)) > 0


class SmilesStandardizer(object):
    """Standardize SMILES strings."""

    def __init__(self, *args, **kwargs):
        self.fragment_standardizer = rdMolStandardize.LargestFragmentChooser()
        self.charge_standardizer = rdMolStandardize.Uncharger()

    def standardize_smiles(self, smi):
        """Standardize SMILES string."""
        mol = Chem.MolFromSmiles(smi)
        out_smi = self.standardize_mol(mol)
        return out_smi

    def standardize_mol(self, mol) -> str:
        """Standardize molecule."""
        mol = self.fragment_standardizer.choose(mol)
        mol = self.charge_standardizer.uncharge(mol)

        # Round trip to and from inchi to tautomer correct
        # Also standardize tautomer in the middle
        output_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        return output_smi


def mass_from_smi(smi: str) -> float:
    """Get exact mass from SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return ExactMolWt(mol)


def min_formal_from_smi(smi: str):
    """Get minimum formal charge from SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.min()


def max_formal_from_smi(smi: str):
    """Get maximum formal charge from SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        formal = np.array([j.GetFormalCharge() for j in mol.GetAtoms()])
        return formal.max()


def atoms_from_smi(smi: str) -> int:
    """Get number of atoms from SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    else:
        return mol.GetNumAtoms()


def add_ion(form: str, ion: str):
    """Add ion to formula."""
    ion_vec = ion_to_add_vec[ion]
    form_vec = formula_to_dense(form)
    return vec_to_formula(form_vec + ion_vec)


def achiral_smi(smi: str) -> str:
    """Convert to achiral SMILES (remove stereochemistry)."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            return smi
        else:
            return ""
    except Exception:
        return ""


def clipped_ppm(mass_diff: np.ndarray, parentmass: np.ndarray) -> np.ndarray:
    """Calculate clipped ppm mass difference.

    Args:
        mass_diff: Mass difference array
        parentmass: Parent mass array

    Returns:
        PPM values clipped to minimum of 200 Da
    """
    parentmass_copy = parentmass * 1
    parentmass_copy[parentmass < 200] = 200
    ppm = mass_diff / parentmass_copy * 1e6
    return ppm


def clipped_ppm_single(cls_mass_diff: float, parentmass: float):
    """Calculate clipped ppm for single value."""
    div_factor = 200 if parentmass < 200 else parentmass
    cls_ppm = cls_mass_diff / div_factor * 1e6
    return cls_ppm
