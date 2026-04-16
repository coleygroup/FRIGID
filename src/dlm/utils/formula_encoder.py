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


import re
import torch
from typing import Dict, Optional, List


class FormulaEncoder:
    """
    Encoder for molecular formulas to fixed-size numerical vectors.
    
    Converts molecular formula strings (e.g., "C9H10N2O2") into vectors of atom counts
    that can be used for conditioning molecular generation models.
    """
    
    # Define common atoms in organic molecules
    # Order matters - this defines the fixed positions in the output vector
    ATOM_VOCAB = [
        'C', 'H', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br', 'I',
        'B', 'Si', 'Se', 'As', 'Al', 'Sn', 'Li', 'Na', 'K', 'Mg',
        'Ca', 'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Pt', 'Pd', 'Au'
    ]
    
    def __init__(self, normalize: str = 'none'):
        """
        Initialize the FormulaEncoder.
        
        Args:
            normalize: Normalization strategy for atom counts.
                - 'none': No normalization
                - 'sum': Divide by total number of atoms
                - 'max': Divide by maximum count in formula
                - 'log': Apply log(count + 1) transformation
        """
        self.atom_vocab = self.ATOM_VOCAB
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_vocab)}
        self.normalize = normalize
        
    def formula_to_counts(self, formula_str: str) -> Dict[str, int]:
        """
        Parse a molecular formula string into a dictionary of atom counts.
        
        Args:
            formula_str: Molecular formula string (e.g., "C9H10N2O2")
            
        Returns:
            Dictionary mapping atom symbols to their counts
            
        Examples:
            >>> encoder = FormulaEncoder()
            >>> encoder.formula_to_counts("C9H10N2O2")
            {'C': 9, 'H': 10, 'N': 2, 'O': 2}
            >>> encoder.formula_to_counts("H2O")
            {'H': 2, 'O': 1}
        """
        if not formula_str or formula_str == "":
            return {}
        
        counts = {}
        
        # Regular expression to match element symbols and their counts
        # Matches: Element (1-2 letters) followed by optional number
        # E.g., "C9", "Cl2", "H", etc.
        pattern = r'([A-Z][a-z]?)(\d*)'
        
        matches = re.findall(pattern, formula_str)
        
        for element, count in matches:
            if element:  # Skip empty matches
                # If no count specified, default to 1
                count_val = int(count) if count else 1
                
                # Accumulate counts (in case element appears multiple times)
                if element in counts:
                    counts[element] += count_val
                else:
                    counts[element] = count_val
        
        return counts
    
    def counts_to_vector(self, counts: Dict[str, int], 
                        normalize: Optional[str] = None) -> torch.Tensor:
        """
        Convert atom counts dictionary to a fixed-size vector.
        
        Args:
            counts: Dictionary mapping atom symbols to counts
            normalize: Override normalization strategy (uses self.normalize if None)
            
        Returns:
            Tensor of shape (len(atom_vocab),) with counts for each atom type
        """
        # Use instance normalization if not overridden
        if normalize is None:
            normalize = self.normalize
        
        # Initialize vector with zeros
        vector = torch.zeros(len(self.atom_vocab), dtype=torch.float32)
        
        # Fill in counts for known atoms
        for atom, count in counts.items():
            if atom in self.atom_to_idx:
                idx = self.atom_to_idx[atom]
                vector[idx] = float(count)
            # Note: Unknown atoms are silently ignored
            # Could log warning if needed
        
        # Apply normalization
        if normalize == 'sum':
            total = vector.sum()
            if total > 0:
                vector = vector / total
        elif normalize == 'max':
            max_val = vector.max()
            if max_val > 0:
                vector = vector / max_val
        elif normalize == 'log':
            vector = torch.log(vector + 1.0)
        elif normalize == 'none':
            pass  # No normalization
        else:
            raise ValueError(f"Unknown normalization strategy: {normalize}")
        
        return vector
    
    def encode(self, formula_str: str, normalize: Optional[str] = None) -> torch.Tensor:
        """
        Encode a molecular formula string into a fixed-size vector.
        
        This is the main method that combines parsing and vectorization.
        
        Args:
            formula_str: Molecular formula string (e.g., "C9H10N2O2")
            normalize: Override normalization strategy
            
        Returns:
            Tensor of shape (len(atom_vocab),) representing the formula
            
        Examples:
            >>> encoder = FormulaEncoder(normalize='sum')
            >>> vec = encoder.encode("C6H12O6")
            >>> vec.shape
            torch.Size([30])
        """
        if formula_str is None or formula_str == "":
            # Return zero vector for empty/invalid formula
            return torch.zeros(len(self.atom_vocab), dtype=torch.float32)
        
        try:
            counts = self.formula_to_counts(formula_str)
            vector = self.counts_to_vector(counts, normalize)
            return vector
        except Exception as e:
            # If parsing fails, return zero vector
            # Could log warning if needed
            return torch.zeros(len(self.atom_vocab), dtype=torch.float32)
    
    def encode_batch(self, formulas: List[str], 
                     normalize: Optional[str] = None) -> torch.Tensor:
        """
        Encode a batch of molecular formulas.
        
        Args:
            formulas: List of molecular formula strings
            normalize: Override normalization strategy
            
        Returns:
            Tensor of shape (batch_size, len(atom_vocab))
        """
        vectors = [self.encode(f, normalize) for f in formulas]
        return torch.stack(vectors)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the atom vocabulary (dimensionality of output vectors)."""
        return len(self.atom_vocab)
    
    def decode(self, vector: torch.Tensor, threshold: float = 0.5) -> str:
        """
        Decode a vector back into a molecular formula string (approximate).
        
        This is useful for visualization/debugging but may not be exact due to normalization.
        
        Args:
            vector: Tensor of shape (len(atom_vocab),)
            threshold: Minimum count to include atom in formula
            
        Returns:
            Approximate molecular formula string
        """
        formula_parts = []
        
        for idx, count in enumerate(vector.tolist()):
            if count > threshold:
                atom = self.atom_vocab[idx]
                # Round to nearest integer
                count_int = round(count)
                if count_int > 1:
                    formula_parts.append(f"{atom}{count_int}")
                elif count_int == 1:
                    formula_parts.append(atom)
        
        return ''.join(formula_parts)