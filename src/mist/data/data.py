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

"""Data structures for mass spectrometry spectra and molecules."""

import logging
from typing import Optional
import re

from rdkit import Chem
from rdkit.Chem import Descriptors

from .. import utils


class Spectra(object):
    """Container for mass spectrometry spectrum data.

    Holds spectrum metadata and lazily loads spectral peak data.

    Args:
        spectra_name: Unique identifier for the spectrum
        spectra_file: Path to the spectrum file
        spectra_formula: Molecular formula of the compound
        instrument: Instrument type used for acquisition
    """

    def __init__(
        self,
        spectra_name: str = "",
        spectra_file: str = "",
        spectra_formula: str = "",
        instrument: str = "",
        **kwargs,
    ):
        self.spectra_name = spectra_name
        self.spectra_file = spectra_file
        self.formula = spectra_formula
        self.instrument = instrument

        # Lazy loading attributes
        self._is_loaded = False
        self.parentmass = None
        self.num_spectra = None
        self.meta = None
        self.spectrum_names = None
        self.spectra = None

    def get_instrument(self):
        """Get instrument type."""
        return self.instrument

    def _load_spectra(self):
        """Load the spectra from file."""
        meta, spectrum_tuples = utils.parse_spectra(self.spectra_file)

        self.meta = meta
        self.parentmass = None
        for parent_kw in ["parentmass", "PEPMASS"]:
            self.parentmass = self.meta.get(parent_kw, None)
            if self.parentmass is not None:
                break

        if self.parentmass is None:
            logging.debug(f"Unable to find precursor mass for {self.spectra_name}")
            self.parentmass = 0
        else:
            self.parentmass = float(self.parentmass)

        # Store all the spectrum names and spectra arrays
        self.spectrum_names, self.spectra = zip(*spectrum_tuples)
        self.num_spectra = len(self.spectra)
        self._is_loaded = True

    def get_spec_name(self, **kwargs):
        """Get spectrum name."""
        return self.spectra_name

    def get_spec(self, **kwargs):
        """Get spectrum peak data, loading if necessary."""
        if not self._is_loaded:
            self._load_spectra()
        return self.spectra

    def get_meta(self, **kwargs):
        """Get spectrum metadata."""
        if not self._is_loaded:
            self._load_spectra()
        return self.meta

    def get_spectra_formula(self):
        """Get chemical formula."""
        return self.formula


class Mol(object):
    """Container for molecular data.

    Stores an RDKit molecule with associated SMILES, InChIKey, and formula.

    Args:
        mol: RDKit Mol object
        smiles: SMILES string (computed if not provided)
        inchikey: InChIKey (computed if not provided)
        mol_formula: Molecular formula (computed if not provided)
    """

    def __init__(
        self,
        mol: Chem.Mol,
        smiles: Optional[str] = None,
        inchikey: Optional[str] = None,
        mol_formula: Optional[str] = None,
    ):
        self.mol = mol

        self.smiles = smiles
        if self.smiles is None:
            self.smiles = Chem.MolToSmiles(mol)

        self.inchikey = inchikey
        if self.inchikey is None and self.smiles != "":
            self.inchikey = Chem.MolToInchiKey(mol)

        self.mol_formula = mol_formula
        if self.mol_formula is None:
            self.mol_formula = utils.uncharged_formula(self.mol, mol_type="mol")
        self.num_hs = None

    @classmethod
    def MolFromInchi(cls, inchi: str, **kwargs):
        """Create Mol from InChI string."""
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return cls(mol=mol, smiles=None, **kwargs)

    @classmethod
    def MolFromSmiles(cls, smiles: str, **kwargs):
        """Create Mol from SMILES string."""
        if not smiles or isinstance(smiles, float):
            smiles = ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return cls(mol=mol, smiles=smiles, **kwargs)

    @classmethod
    def MolFromFormula(cls, formula: str, **kwargs):
        """Create Mol from chemical formula (atoms only, no bonds)."""
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, formula)

        mol = Chem.RWMol()
        for element, count in matches:
            count = int(count) if count else 1
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
            for _ in range(count):
                atom = Chem.Atom(atomic_num)
                mol.AddAtom(atom)

        return cls(mol=mol, mol_formula=formula, **kwargs)

    def get_smiles(self) -> str:
        """Get SMILES string."""
        return self.smiles

    def get_inchikey(self) -> str:
        """Get InChIKey."""
        return self.inchikey

    def get_molform(self) -> str:
        """Get molecular formula."""
        return self.mol_formula

    def get_num_hs(self):
        """Get number of hydrogen atoms from formula."""
        if self.num_hs is None:
            num = re.findall("H([0-9]*)", self.mol_formula)
            if num is None:
                out_num_hs = 0
            else:
                if len(num) == 0:
                    out_num_hs = 0
                elif len(num) == 1:
                    num = num[0]
                    out_num_hs = 1 if num == "" else int(num)
                else:
                    raise ValueError()
            self.num_hs = out_num_hs
        else:
            out_num_hs = self.num_hs
        return out_num_hs

    def get_mol_mass(self):
        """Get molecular weight."""
        return Descriptors.MolWt(self.mol)

    def get_rdkit_mol(self) -> Chem.Mol:
        """Get RDKit Mol object."""
        return self.mol
