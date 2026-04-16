"""
Masking utilities for ICEBERG-guided inference-time scaling.

This module provides classes for:
1. Identifying hallucinated peaks between predicted and real spectra
2. Mapping hallucinated peaks to atoms via fragmentation graphs
3. Mapping atoms to SAFE tokens
4. Creating masked inputs for refinement generation
"""

import re
from typing import List, Dict, Set, Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

import ms_pred.common as common
from ms_pred.common import chem_utils
import ms_pred.magma.fragmentation as fragmentation


class BaseMaskingStrategy(ABC):
    """
    Abstract base class for masking strategies.
    
    Subclasses implement different logic for:
    1. Identifying hallucinated peaks
    2. Mapping peaks to "bad" atoms
    3. Creating masked token sequences
    """
    
    @abstractmethod
    def get_hallucinated_peaks(
        self,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        **kwargs,
    ) -> List[Dict]:
        """
        Identify hallucinated peaks between predicted and real spectra.

        Args:
            real_specs: Real (experimental) composite mass spectrum
            pred_specs: Predicted composite mass spectrum
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            List[Dict]: A list of dictionaries representing hallucinated peaks.
                       Each dict should contain at minimum:
                       - 'mz': m/z value
                       - 'intensity': intensity value
                       - 'norm_intensity': normalized intensity
                       - 'distance': distance to nearest experimental peak
                       - 'fragment': fragment index (for atom mapping)
        """
        raise NotImplementedError("Subclasses must implement get_hallucinated_peaks")
    
    @abstractmethod
    def get_bad_atoms(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        **kwargs,
    ) -> Set[int]:
        """
        Map hallucinated peaks to atom indices.
        
        Args:
            smiles: SMILES string of the molecule
            hallucinated_peaks: List of hallucinated peaks from get_hallucinated_peaks
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            Set of atom indices that are "bad" (contribute to hallucinated peaks)
        """
        raise NotImplementedError("Subclasses must implement get_bad_atoms")
    
    @abstractmethod
    def create_masked_inputs(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        tokenizer: Any,
        num_masks: int = 1,
        mask_prob: float = 0.5,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Create masked input token sequences for refinement.
        
        Args:
            smiles: SMILES string of the molecule
            hallucinated_peaks: List of hallucinated peaks
            tokenizer: HuggingFace tokenizer
            num_masks: Number of masked versions to create
            mask_prob: Probability of masking each bad token
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            List of masked input_ids tensors
        """
        raise NotImplementedError("Subclasses must implement create_masked_inputs")


class SimpleMaskingStrategy(BaseMaskingStrategy):
    """
    Simple masking strategy based on peak distance.
    
    This strategy:
    1. Identifies hallucinated peaks as predicted peaks that are far from any experimental peak
    2. Maps these peaks to atoms using MAGMA fragmentation
    3. Creates stochastically masked token sequences
    """
    
    def __init__(
        self,
        top_k_peaks: int = 3,
        ppm: int = 20,
        real_inten_threshold: float = 0.05,
        sim_inten_threshold: float = 0.01,
    ):
        """
        Initialize the simple masking strategy.
        
        Args:
            top_k_peaks: Number of top hallucinated peaks to return
            ppm: Parts-per-million tolerance for mass comparison
            real_inten_threshold: Intensity threshold for real peaks (normalized)
            sim_inten_threshold: Intensity threshold for simulated peaks (normalized)
        """
        self.top_k_peaks = top_k_peaks
        self.ppm = ppm
        self.real_inten_threshold = real_inten_threshold
        self.sim_inten_threshold = sim_inten_threshold
    
    def get_hallucinated_peaks(
        self,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        real_smi: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Identify hallucinated peaks between predicted and real spectra.
        
        Handles 4 cases:
        1. Single real spectrum with known CE
        2. Single real spectrum with unknown CE  
        3. Multiple real spectra with known CEs
        4. Multiple real spectra with unknown CEs (not fully supported)
        
        Args:
            real_specs: Real (experimental) composite mass spectrum
            pred_specs: Predicted composite mass spectrum
            real_smi: Real SMILES (optional, for precursor mass calculation)
            
        Returns:
            List of hallucinated peak dicts sorted by distance (most hallucinated first)
        """
        try:
            real_CEs = sorted([ce for ce in real_specs.keys() if ce is not None])
        except Exception as e:
            print(f"Warning [get_hallucinated_peaks]: Failed to get real spectrum CEs: {e}")
            return []
        
        # Handle None CE case
        if len(real_CEs) == 0:
            # Check for None key
            if None in real_specs.keys():
                real_CEs = [None]
            else:
                return []
        
        try:
            if len(real_CEs) == 1:
                real_spec = real_specs[real_CEs[0]]
                real_ce = real_CEs[0]
                
                if real_ce is None:
                    # Case 2 and 4: Unknown CE - find most similar spectrum among pred_specs
                    try:
                        if real_smi is not None:
                            real_precursor_mz = chem_utils.mass_from_smi(real_smi)
                            best_sim, best_ce = pred_specs.similarity(
                                real_spec, merge_method='unknown', 
                                ignore_mass=real_precursor_mz, return_ce=True
                            )
                        else:
                            best_sim, best_ce = pred_specs.similarity(
                                real_spec, merge_method='unknown', return_ce=True
                            )
                    except Exception as e:
                        print(f"Warning [get_hallucinated_peaks]: Failed to compute similarity: {e}")
                        return []
                    if best_ce is None:
                        return []
                    pred_spec = pred_specs[best_ce]
                else:
                    # Case 1: Use spectrum with known CE
                    if real_ce not in pred_specs.keys():
                        # Try to find closest CE
                        pred_CEs = [float(ce) for ce in pred_specs.keys() if ce is not None]
                        if not pred_CEs:
                            return []
                        closest_ce = min(pred_CEs, key=lambda x: abs(x - float(real_ce)))
                        pred_spec = pred_specs[closest_ce]
                    else:
                        pred_spec = pred_specs[real_ce]
            else:
                # Case 3: Multiple known real spectra
                # Instead of merging (which can fail with inhomogeneous fragment arrays),
                # process each CE pair separately and aggregate hallucinated peaks
                return self._get_hallucinated_peaks_multi_ce(real_specs, pred_specs, real_CEs)
        except Exception as e:
            print(f"Warning [get_hallucinated_peaks]: Failed to get spectra for comparison: {e}")
            return []
        
        # Single spectrum case - compute hallucination scores directly
        return self._compute_halluc_peaks_single(real_spec, pred_spec)
    
    def _get_hallucinated_peaks_multi_ce(
        self,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        real_CEs: List,
    ) -> List[Dict]:
        """
        Handle multiple collision energies by processing each CE separately
        and aggregating the results. This avoids the merge_spectra call which
        can fail when MassSpec objects have inconsistent fragment data.
        """
        all_halluc_peaks = []
        seen_mz_frag = set()  # Deduplicate by (mz, fragment) pairs
        
        for real_ce in real_CEs:
            try:
                real_spec = real_specs[real_ce]
            except (KeyError, Exception):
                continue
            
            # Find matching or closest predicted CE
            pred_ce = None
            if real_ce in pred_specs.keys():
                pred_ce = real_ce
            else:
                pred_CEs = [ce for ce in pred_specs.keys() if ce is not None]
                if pred_CEs:
                    target = float(real_ce) if real_ce is not None else 0
                    pred_ce = min(pred_CEs, key=lambda x: abs(float(x) - target))
            
            if pred_ce is None:
                continue
            
            try:
                pred_spec = pred_specs[pred_ce]
            except (KeyError, Exception):
                continue
            
            # Get hallucinated peaks for this CE pair
            ce_halluc_peaks = self._compute_halluc_peaks_single(real_spec, pred_spec)
            
            # Add unique peaks (deduplicate by mz and fragment)
            for peak in ce_halluc_peaks:
                key = (round(peak['mz'], 4), peak.get('fragment'))
                if key not in seen_mz_frag:
                    seen_mz_frag.add(key)
                    peak['collision_energy'] = real_ce  # Track which CE this came from
                    all_halluc_peaks.append(peak)
        
        # Sort by distance and return top K
        all_halluc_peaks = sorted(all_halluc_peaks, key=lambda x: x['distance'], reverse=True)
        return all_halluc_peaks[:self.top_k_peaks]
    
    def _compute_halluc_peaks_single(
        self,
        real_spec: common.MassSpec,
        pred_spec: common.MassSpec,
    ) -> List[Dict]:
        """Compute hallucinated peaks for a single real/pred spectrum pair."""
        # Validate real spectrum
        if real_spec.intens is None or len(real_spec.intens) == 0:
            return []
        
        real_max_inten = np.max(real_spec.intens)
        if real_max_inten == 0:
            return []
        
        real_norm_intens = real_spec.intens / real_max_inten
        
        # Select m/z values where real intensity >= threshold
        exp_mzs = real_spec.masses[real_norm_intens >= self.real_inten_threshold]
        
        if len(exp_mzs) == 0:
            exp_mzs = real_spec.masses
        
        # Check each predicted peak
        halluc_peaks = []
        
        if pred_spec.intens is None or len(pred_spec.intens) == 0:
            return []
        
        max_inten = np.max(pred_spec.intens)
        if max_inten == 0:
            return []
        
        # Get precursor m/z for exclusion
        precursor_mz = getattr(pred_spec, 'precursor_mz', None)
        
        # Safely get spec and int_frags
        try:
            pred_spec_data = pred_spec.spec
            pred_int_frags = pred_spec.int_frags
            if pred_spec_data is None or pred_int_frags is None:
                print(f"Warning [get_hallucinated_peaks]: pred_spec.spec or pred_spec.int_frags is None")
                return []
        except Exception as e:
            print(f"Warning [get_hallucinated_peaks]: Failed to access pred_spec.spec or int_frags: {e}")
            return []
        
        for i, ((mz, inten), frag) in enumerate(zip(pred_spec_data, pred_int_frags)):
            try:
                # Ignore precursor peak
                if precursor_mz is not None:
                    if np.abs(mz - precursor_mz) < precursor_mz * self.ppm * 1e-6:
                        continue
                
                norm_inten = inten / max_inten
                
                if norm_inten < self.sim_inten_threshold:
                    continue
                
                # Compute minimum distance to closest experimental peaks
                min_dist = np.min(np.abs(exp_mzs - mz))

                halluc_peaks.append({
                    'mz': mz,
                    'intensity': inten,
                    'norm_intensity': norm_inten,
                    'distance': min_dist,
                    'fragment': frag,
                    'peak_idx': i,
                })
            except Exception as e:
                print(f"Warning [get_hallucinated_peaks]: Error processing peak {i}: {e}")
                continue
        
        # Sort by distance (most hallucinated first)
        halluc_peaks = sorted(halluc_peaks, key=lambda x: x['distance'], reverse=True)
        
        return halluc_peaks[:self.top_k_peaks]
    
    def get_bad_atoms(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        mol_str_canonicalized: bool = True,
        **kwargs,
    ) -> Set[int]:
        """
        Map hallucinated peaks to atom indices using MAGMA fragmentation.
        
        Args:
            smiles: SMILES string (should be canonical if mol_str_canonicalized=True)
            hallucinated_peaks: List of hallucinated peaks with 'fragment' indices
            mol_str_canonicalized: Whether the SMILES is already canonicalized
            
        Returns:
            Set of atom indices that appear in hallucinated fragments
        """
        if not hallucinated_peaks:
            return set()
        
        try:
            engine = fragmentation.FragmentEngine(
                smiles, 
                mol_str_type='smiles', 
                mol_str_canonicalized=mol_str_canonicalized
            )
        except Exception as e:
            print(f"Warning [get_bad_atoms]: Failed to create FragmentEngine for '{smiles[:50]}...': {e}")
            return set()
        
        bad_atoms = set()
        
        for peak in hallucinated_peaks:
            frag_idx = peak.get('fragment')
            if frag_idx is None:
                continue
            
            # Get atoms for this fragment DAG node
            try:
                present_atoms, _ = engine.get_present_atoms(frag_idx)
                bad_atoms.update(present_atoms)
            except Exception as e:
                print(f"Warning [get_bad_atoms]: Failed to get atoms for fragment {frag_idx}: {e}")
                continue
        
        return bad_atoms
    
    def create_masked_inputs(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        tokenizer: Any,
        num_masks: int = 1,
        mask_prob: float = 0.5,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Create masked input token sequences for refinement.
        
        Args:
            smiles: SMILES/SAFE string of the molecule
            hallucinated_peaks: List of hallucinated peaks
            tokenizer: HuggingFace tokenizer with mask_token_id
            num_masks: Number of masked versions to create
            mask_prob: Probability of masking each bad token
            
        Returns:
            List of masked input_ids tensors (each shape [1, seq_len])
        """
        # Get bad atoms
        try:
            bad_atoms = self.get_bad_atoms(smiles, hallucinated_peaks, mol_str_canonicalized=True)
        except Exception as e:
            print(f"Warning [create_masked_inputs]: Failed to get bad atoms for '{smiles[:50]}...': {e}")
            bad_atoms = set()
        
        # Tokenize with offset mapping
        try:
            encoding = tokenizer(smiles, return_offsets_mapping=True, return_tensors='pt')
            base_input_ids = encoding['input_ids'][0]
            offsets = encoding['offset_mapping'][0]
            mask_token_id = tokenizer.mask_token_id
        except Exception as e:
            print(f"Warning [create_masked_inputs]: Failed to tokenize '{smiles[:50]}...': {e}")
            return []
        
        if not bad_atoms:
            # No bad atoms identified - mask the entire sequence
            masked_inputs = []
            for _ in range(num_masks):
                input_ids = base_input_ids.clone()
                # Mask every token
                input_ids[:] = mask_token_id
                masked_inputs.append(input_ids.unsqueeze(0))
            return masked_inputs
        
        # Map SMILES characters to atom indices
        try:
            char_to_atom = _map_smiles_chars_to_atoms(smiles)
        except Exception as e:
            print(f"Warning [create_masked_inputs]: Failed to map SMILES chars to atoms for '{smiles[:50]}...': {e}")
            # Fall back to masking everything
            masked_inputs = []
            for _ in range(num_masks):
                input_ids = base_input_ids.clone()
                input_ids[:] = mask_token_id
                masked_inputs.append(input_ids.unsqueeze(0))
            return masked_inputs
        
        # Find bad token indices
        bad_token_indices = []
        
        for token_idx, (start_char, end_char) in enumerate(offsets):
            # Skip special tokens
            if start_char == end_char:
                continue
            
            # Check if this token covers any bad atoms
            covered_atoms = set()
            for char_idx in range(start_char, end_char):
                if char_idx < len(char_to_atom):
                    atom_idx = char_to_atom[char_idx]
                    if atom_idx != -1:
                        covered_atoms.add(atom_idx)
            
            # If token overlaps with any bad atom, mark it
            if not covered_atoms.isdisjoint(bad_atoms):
                bad_token_indices.append(token_idx)
        
        # Create masked versions
        mask_token_id = tokenizer.mask_token_id
        masked_inputs = []
        
        for _ in range(num_masks):
            input_ids = base_input_ids.clone()
            
            for idx in bad_token_indices:
                if np.random.random() < mask_prob:
                    input_ids[idx] = mask_token_id
            
            masked_inputs.append(input_ids.unsqueeze(0))
        
        return masked_inputs


class IntensityWeightedMaskingStrategy(BaseMaskingStrategy):
    """
    Intensity-weighted masking strategy with fragment-aware atom scoring.
    
    This strategy improves upon SimpleMaskingStrategy by:
    
    1. **Intensity-Weighted Hallucination Scoring**: Instead of just using distance
       to the nearest experimental peak, this strategy computes a composite score
       that considers:
       - Distance to nearest experimental peak (higher = more hallucinated)
       - Normalized intensity of the predicted peak (higher intensity hallucinations
         are more problematic)
       - Whether the peak is in a "crowded" vs "sparse" region of the spectrum
    
    2. **Fragment-Aware Atom Scoring**: Rather than treating all atoms in a 
       hallucinated fragment equally, this strategy:
       - Computes per-atom hallucination scores based on fragment membership
       - Weighs atoms by how many hallucinated fragments they appear in
       - Considers fragment size (smaller fragments = more specific attribution)
    
    3. **Probabilistic Masking with Atom Scores**: Instead of uniform mask_prob
       for all bad atoms, masking probability is proportional to the atom's
       hallucination score, allowing the model to focus on the most problematic
       regions while preserving likely-correct structure.
    
    4. **Neighbor-Aware Masking**: Optionally extends masking to neighboring atoms
       of highly-scored atoms, since errors often affect local structure.
    """
    
    def __init__(
        self,
        top_k_peaks: int = 5,
        ppm: int = 20,
        real_inten_threshold: float = 0.05,
        sim_inten_threshold: float = 0.01,
        distance_weight: float = 0.6,
        intensity_weight: float = 0.4,
        fragment_size_penalty: bool = True,
        neighbor_extension: bool = True,
        neighbor_weight: float = 0.3,
        min_mask_prob: float = 0.1,
        max_mask_prob: float = 0.9,
    ):
        """
        Initialize the intensity-weighted masking strategy.
        
        Args:
            top_k_peaks: Number of top hallucinated peaks to consider
            ppm: Parts-per-million tolerance for mass comparison
            real_inten_threshold: Intensity threshold for real peaks (normalized)
            sim_inten_threshold: Intensity threshold for simulated peaks (normalized)
            distance_weight: Weight for distance component in hallucination score
            intensity_weight: Weight for intensity component in hallucination score
            fragment_size_penalty: If True, smaller fragments get higher scores
                                   (more specific attribution)
            neighbor_extension: If True, extend scores to neighboring atoms
            neighbor_weight: Weight factor for neighboring atom scores
            min_mask_prob: Minimum masking probability for any bad atom
            max_mask_prob: Maximum masking probability for any bad atom
        """
        self.top_k_peaks = top_k_peaks
        self.ppm = ppm
        self.real_inten_threshold = real_inten_threshold
        self.sim_inten_threshold = sim_inten_threshold
        self.distance_weight = distance_weight
        self.intensity_weight = intensity_weight
        self.fragment_size_penalty = fragment_size_penalty
        self.neighbor_extension = neighbor_extension
        self.neighbor_weight = neighbor_weight
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob
    
    def get_hallucinated_peaks(
        self,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        real_smi: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Identify hallucinated peaks with intensity-weighted scoring.
        
        Returns peaks sorted by a composite hallucination score that combines
        distance and intensity information.
        """
        try:
            real_CEs = sorted([ce for ce in real_specs.keys() if ce is not None])
        except Exception as e:
            print(f"Warning [IntensityWeighted.get_hallucinated_peaks]: Failed to get real spectrum CEs: {e}")
            return []
        
        if len(real_CEs) == 0:
            if None in real_specs.keys():
                real_CEs = [None]
            else:
                return []
        
        try:
            if len(real_CEs) == 1:
                real_spec = real_specs[real_CEs[0]]
                real_ce = real_CEs[0]
                
                if real_ce is None:
                    try:
                        if real_smi is not None:
                            real_precursor_mz = chem_utils.mass_from_smi(real_smi)
                            best_sim, best_ce = pred_specs.similarity(
                                real_spec, merge_method='unknown', 
                                ignore_mass=real_precursor_mz, return_ce=True
                            )
                        else:
                            best_sim, best_ce = pred_specs.similarity(
                                real_spec, merge_method='unknown', return_ce=True
                            )
                    except Exception as e:
                        print(f"Warning [IntensityWeighted.get_hallucinated_peaks]: Failed to compute similarity: {e}")
                        return []
                    if best_ce is None:
                        return []
                    pred_spec = pred_specs[best_ce]
                else:
                    if real_ce not in pred_specs.keys():
                        pred_CEs = [float(ce) for ce in pred_specs.keys() if ce is not None]
                        if not pred_CEs:
                            return []
                        closest_ce = min(pred_CEs, key=lambda x: abs(x - float(real_ce)))
                        pred_spec = pred_specs[closest_ce]
                    else:
                        pred_spec = pred_specs[real_ce]
            else:
                # Case 3: Multiple known real spectra
                # Instead of merging (which can fail with inhomogeneous fragment arrays),
                # process each CE pair separately and aggregate hallucinated peaks
                return self._get_hallucinated_peaks_multi_ce(real_specs, pred_specs, real_CEs)
        except Exception as e:
            print(f"Warning [IntensityWeighted.get_hallucinated_peaks]: Failed to get spectra: {e}")
            return []
        
        # Single spectrum case - compute hallucination scores directly
        return self._compute_halluc_peaks_single(real_spec, pred_spec)
    
    def _get_hallucinated_peaks_multi_ce(
        self,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        real_CEs: List,
    ) -> List[Dict]:
        """
        Handle multiple collision energies by processing each CE separately
        and aggregating the results. This avoids the merge_spectra call which
        can fail when MassSpec objects have inconsistent fragment data.
        """
        all_halluc_peaks = []
        seen_mz_frag = set()  # Deduplicate by (mz, fragment) pairs
        
        for real_ce in real_CEs:
            try:
                real_spec = real_specs[real_ce]
            except (KeyError, Exception):
                continue
            
            # Find matching or closest predicted CE
            pred_ce = None
            if real_ce in pred_specs.keys():
                pred_ce = real_ce
            else:
                pred_CEs = [ce for ce in pred_specs.keys() if ce is not None]
                if pred_CEs:
                    target = float(real_ce) if real_ce is not None else 0
                    pred_ce = min(pred_CEs, key=lambda x: abs(float(x) - target))
            
            if pred_ce is None:
                continue
            
            try:
                pred_spec = pred_specs[pred_ce]
            except (KeyError, Exception):
                continue
            
            # Get hallucinated peaks for this CE pair (no top_k limit yet)
            ce_halluc_peaks = self._compute_halluc_peaks_single(real_spec, pred_spec, apply_top_k=False)
            
            # Add unique peaks (deduplicate by mz and fragment)
            for peak in ce_halluc_peaks:
                key = (round(peak['mz'], 4), peak.get('fragment'))
                if key not in seen_mz_frag:
                    seen_mz_frag.add(key)
                    peak['collision_energy'] = real_ce  # Track which CE this came from
                    all_halluc_peaks.append(peak)
        
        # Sort by hallucination score and return top K
        all_halluc_peaks = sorted(all_halluc_peaks, key=lambda x: x.get('halluc_score', x['distance']), reverse=True)
        return all_halluc_peaks[:self.top_k_peaks]
    
    def _compute_halluc_peaks_single(
        self,
        real_spec: common.MassSpec,
        pred_spec: common.MassSpec,
        apply_top_k: bool = True,
    ) -> List[Dict]:
        """Compute hallucinated peaks for a single real/pred spectrum pair."""
        # Validate spectra
        if real_spec.intens is None or len(real_spec.intens) == 0:
            return []
        
        real_max_inten = np.max(real_spec.intens)
        if real_max_inten == 0:
            return []
        
        real_norm_intens = real_spec.intens / real_max_inten
        
        # Get experimental m/z values above threshold
        exp_mask = real_norm_intens >= self.real_inten_threshold
        exp_mzs = real_spec.masses[exp_mask]
        exp_intens = real_norm_intens[exp_mask]
        
        if len(exp_mzs) == 0:
            exp_mzs = real_spec.masses
            exp_intens = real_norm_intens
        
        # Compute local density for crowdedness estimation
        exp_density = self._compute_local_density(exp_mzs, self.ppm)
        
        # Validate predicted spectrum
        if pred_spec.intens is None or len(pred_spec.intens) == 0:
            return []
        
        max_inten = np.max(pred_spec.intens)
        if max_inten == 0:
            return []
        
        precursor_mz = getattr(pred_spec, 'precursor_mz', None)
        
        try:
            pred_spec_data = pred_spec.spec
            pred_int_frags = pred_spec.int_frags
            if pred_spec_data is None or pred_int_frags is None:
                print(f"Warning [IntensityWeighted._compute_halluc_peaks_single]: pred_spec.spec or int_frags is None")
                return []
        except Exception as e:
            print(f"Warning [IntensityWeighted._compute_halluc_peaks_single]: Failed to access pred_spec: {e}")
            return []
        
        halluc_peaks = []
        
        # Compute maximum distance for normalization
        max_possible_dist = np.max(exp_mzs) * self.ppm * 1e-6 * 10  # 10x PPM tolerance
        
        for i, ((mz, inten), frag) in enumerate(zip(pred_spec_data, pred_int_frags)):
            try:
                # Skip precursor peak
                if precursor_mz is not None:
                    if np.abs(mz - precursor_mz) < precursor_mz * self.ppm * 1e-6:
                        continue
                
                norm_inten = inten / max_inten
                
                if norm_inten < self.sim_inten_threshold:
                    continue
                
                # Compute distance to nearest experimental peak
                distances = np.abs(exp_mzs - mz)
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # Check if within PPM tolerance (not a hallucination)
                ppm_tol = exp_mzs[min_dist_idx] * self.ppm * 1e-6
                if min_dist <= ppm_tol:
                    continue  # This is a real peak match
                
                # Normalized distance score (0-1, higher = more hallucinated)
                norm_dist = min(1.0, min_dist / max_possible_dist)
                
                # Local density at this m/z (lower density = more suspicious)
                local_density = self._get_local_density_at_mz(mz, exp_mzs, exp_density)
                density_factor = 1.0 - min(1.0, local_density)  # Invert: sparse = higher score
                
                # Composite hallucination score
                halluc_score = (
                    self.distance_weight * norm_dist +
                    self.intensity_weight * norm_inten
                ) * (0.5 + 0.5 * density_factor)  # Density modulates score
                
                halluc_peaks.append({
                    'mz': mz,
                    'intensity': inten,
                    'norm_intensity': norm_inten,
                    'distance': min_dist,
                    'norm_distance': norm_dist,
                    'halluc_score': halluc_score,
                    'local_density': local_density,
                    'fragment': frag,
                    'peak_idx': i,
                })
            except Exception as e:
                print(f"Warning [IntensityWeighted._compute_halluc_peaks_single]: Error processing peak {i}: {e}")
                continue
        
        # Sort by composite hallucination score
        halluc_peaks = sorted(halluc_peaks, key=lambda x: x['halluc_score'], reverse=True)
        
        if apply_top_k:
            return halluc_peaks[:self.top_k_peaks]
        return halluc_peaks
    
    def _compute_local_density(self, mzs: np.ndarray, ppm: int) -> np.ndarray:
        """Compute local density (number of nearby peaks) for each m/z."""
        if len(mzs) == 0:
            return np.array([])
        
        density = np.zeros(len(mzs))
        for i, mz in enumerate(mzs):
            tol = mz * ppm * 1e-6 * 5  # 5x PPM window
            density[i] = np.sum(np.abs(mzs - mz) < tol)
        
        # Normalize
        if density.max() > 0:
            density = density / density.max()
        
        return density
    
    def _get_local_density_at_mz(
        self, 
        target_mz: float, 
        exp_mzs: np.ndarray, 
        exp_density: np.ndarray
    ) -> float:
        """Get interpolated local density at a target m/z."""
        if len(exp_mzs) == 0:
            return 0.0
        
        # Find nearest experimental peak
        idx = np.argmin(np.abs(exp_mzs - target_mz))
        return exp_density[idx]
    
    def get_bad_atoms(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        mol_str_canonicalized: bool = True,
        **kwargs,
    ) -> Set[int]:
        """
        Map hallucinated peaks to atom indices with intensity-weighted scoring.
        
        Returns set of atom indices, but also stores per-atom scores in kwargs
        if 'atom_scores_out' dict is provided.
        """
        if not hallucinated_peaks:
            return set()
        
        try:
            engine = fragmentation.FragmentEngine(
                smiles, 
                mol_str_type='smiles', 
                mol_str_canonicalized=mol_str_canonicalized
            )
            num_atoms = engine.natoms
        except Exception as e:
            print(f"Warning [IntensityWeighted.get_bad_atoms]: Failed to create FragmentEngine: {e}")
            return set()
        
        # Compute per-atom hallucination scores
        atom_scores = np.zeros(num_atoms)
        atom_counts = np.zeros(num_atoms)  # Track how many fragments each atom appears in
        
        for peak in hallucinated_peaks:
            frag_idx = peak.get('fragment')
            if frag_idx is None:
                continue
            
            try:
                present_atoms, _ = engine.get_present_atoms(frag_idx)
                frag_size = len(present_atoms)
                
                # Get hallucination score (or fall back to norm_intensity)
                peak_score = peak.get('halluc_score', peak.get('norm_intensity', 0.5))
                
                # Apply fragment size penalty: smaller fragments = more specific
                if self.fragment_size_penalty and frag_size > 0:
                    size_factor = 1.0 / np.sqrt(frag_size)  # Inverse sqrt penalty
                else:
                    size_factor = 1.0
                
                weighted_score = peak_score * size_factor
                
                for atom_idx in present_atoms:
                    atom_scores[atom_idx] += weighted_score
                    atom_counts[atom_idx] += 1
                    
            except Exception as e:
                print(f"Warning [IntensityWeighted.get_bad_atoms]: Error processing fragment: {e}")
                continue
        
        # Normalize scores
        if atom_scores.max() > 0:
            atom_scores = atom_scores / atom_scores.max()
        
        # Extend scores to neighbors if enabled
        if self.neighbor_extension:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    extended_scores = atom_scores.copy()
                    for atom_idx in range(num_atoms):
                        if atom_scores[atom_idx] > 0:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            for neighbor in atom.GetNeighbors():
                                neighbor_idx = neighbor.GetIdx()
                                # Add fraction of score to neighbors
                                extended_scores[neighbor_idx] = max(
                                    extended_scores[neighbor_idx],
                                    atom_scores[atom_idx] * self.neighbor_weight
                                )
                    atom_scores = extended_scores
            except Exception as e:
                print(f"Warning [IntensityWeighted.get_bad_atoms]: Neighbor extension failed: {e}")
        
        # Store scores in output dict if provided
        atom_scores_out = kwargs.get('atom_scores_out')
        if atom_scores_out is not None and isinstance(atom_scores_out, dict):
            atom_scores_out['scores'] = atom_scores
            atom_scores_out['counts'] = atom_counts
        
        # Return atoms with non-zero scores
        bad_atoms = set(np.where(atom_scores > 0)[0].tolist())
        
        return bad_atoms
    
    def _get_atom_scores(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        mol_str_canonicalized: bool = True,
    ) -> np.ndarray:
        """Get per-atom hallucination scores."""
        atom_scores_out = {}
        self.get_bad_atoms(
            smiles, 
            hallucinated_peaks, 
            mol_str_canonicalized, 
            atom_scores_out=atom_scores_out
        )
        return atom_scores_out.get('scores', np.array([]))
    
    def create_masked_inputs(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        tokenizer: Any,
        num_masks: int = 1,
        mask_prob: float = 0.5,  # This becomes a scaling factor
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Create masked inputs with score-proportional masking probability.
        
        Unlike SimpleMaskingStrategy which uses uniform mask_prob, this method
        scales masking probability by each atom's hallucination score.
        """
        # Get per-atom scores
        try:
            atom_scores = self._get_atom_scores(smiles, hallucinated_peaks, mol_str_canonicalized=True)
        except Exception as e:
            print(f"Warning [IntensityWeighted.create_masked_inputs]: Failed to get atom scores: {e}")
            atom_scores = np.array([])
        
        # Tokenize
        try:
            encoding = tokenizer(smiles, return_offsets_mapping=True, return_tensors='pt')
            base_input_ids = encoding['input_ids'][0]
            offsets = encoding['offset_mapping'][0]
            mask_token_id = tokenizer.mask_token_id
        except Exception as e:
            print(f"Warning [IntensityWeighted.create_masked_inputs]: Tokenization failed: {e}")
            return []
        
        if len(atom_scores) == 0 or atom_scores.max() == 0:
            # No bad atoms - mask everything
            masked_inputs = []
            for _ in range(num_masks):
                input_ids = base_input_ids.clone()
                input_ids[:] = mask_token_id
                masked_inputs.append(input_ids.unsqueeze(0))
            return masked_inputs
        
        # Map SMILES chars to atoms
        try:
            char_to_atom = _map_smiles_chars_to_atoms(smiles)
        except Exception as e:
            print(f"Warning [IntensityWeighted.create_masked_inputs]: Char mapping failed: {e}")
            masked_inputs = []
            for _ in range(num_masks):
                input_ids = base_input_ids.clone()
                input_ids[:] = mask_token_id
                masked_inputs.append(input_ids.unsqueeze(0))
            return masked_inputs
        
        # Compute per-token scores (max of covered atom scores)
        token_scores = []
        for token_idx, (start_char, end_char) in enumerate(offsets):
            if start_char == end_char:
                token_scores.append(0.0)
                continue
            
            max_score = 0.0
            for char_idx in range(start_char, end_char):
                if char_idx < len(char_to_atom):
                    atom_idx = char_to_atom[char_idx]
                    if atom_idx != -1 and atom_idx < len(atom_scores):
                        max_score = max(max_score, atom_scores[atom_idx])
            
            token_scores.append(max_score)
        
        token_scores = np.array(token_scores)
        
        # Convert scores to masking probabilities
        # Scale by mask_prob parameter and clamp to [min_mask_prob, max_mask_prob]
        token_mask_probs = token_scores * mask_prob
        token_mask_probs = np.clip(
            token_mask_probs,
            0.0,  # Only mask if score > 0
            self.max_mask_prob
        )
        # Apply minimum for tokens with any score
        token_mask_probs[token_scores > 0] = np.maximum(
            token_mask_probs[token_scores > 0],
            self.min_mask_prob
        )
        
        # Create masked versions
        masked_inputs = []
        
        for _ in range(num_masks):
            input_ids = base_input_ids.clone()
            
            for idx, prob in enumerate(token_mask_probs):
                if prob > 0 and np.random.random() < prob:
                    input_ids[idx] = mask_token_id
            
            masked_inputs.append(input_ids.unsqueeze(0))
        
        return masked_inputs


# =============================================================================
# Factory function for creating masking strategies
# =============================================================================


class ScoreBasedMaskingStrategy(BaseMaskingStrategy):
    """
    Score-based masking strategy that considers both matched and unmatched peaks.
    
    This strategy:
    1. Uses PPM tolerance to determine if a simulated peak matches a real peak
    2. For ALL peaks (matched and unmatched):
       - Matched peaks give POSITIVE scores to atoms in their fragment
       - Unmatched peaks give NEGATIVE scores to atoms in their fragment
    3. Atoms with lower (more negative) scores have higher probability of being masked
    4. Uses ICEBERG fragments directly (via int_frags) instead of MAGMA
    
    The key insight is that atoms appearing only in matched peaks are likely correct,
    while atoms appearing predominantly in unmatched (hallucinated) peaks are likely
    wrong and should be masked for refinement.
    """
    
    def __init__(
        self,
        top_k_peaks: int = 10,  # Higher default since we consider all peaks
        ppm: int = 20,
        real_inten_threshold: float = 0.04,
        sim_inten_threshold: float = 0.04,
        match_reward: float = 1.0,
        unmatch_penalty: float = 1.0,
        intensity_weighting: bool = True,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 1.0,
        score_normalization: str = 'minmax',  # 'minmax', 'zscore', or 'none'
    ):
        """
        Initialize the score-based masking strategy.
        
        Args:
            top_k_peaks: Max number of peaks to consider (for efficiency)
            ppm: Parts-per-million tolerance for peak matching
            real_inten_threshold: Intensity threshold for real peaks (normalized)
            sim_inten_threshold: Intensity threshold for simulated peaks (normalized)
            match_reward: Score bonus for atoms in matched peaks
            unmatch_penalty: Score penalty for atoms in unmatched peaks
            intensity_weighting: If True, weight scores by peak intensity
            min_mask_prob: Minimum masking probability
            max_mask_prob: Maximum masking probability
            score_normalization: How to normalize final scores
        """
        self.top_k_peaks = top_k_peaks
        self.ppm = ppm
        self.real_inten_threshold = real_inten_threshold
        self.sim_inten_threshold = sim_inten_threshold
        self.match_reward = match_reward
        self.unmatch_penalty = unmatch_penalty
        self.intensity_weighting = intensity_weighting
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob
        self.score_normalization = score_normalization
    
    def _is_peak_matched(self, pred_mz: float, exp_mzs: np.ndarray) -> bool:
        """Check if a predicted peak matches any experimental peak within PPM tolerance."""
        if len(exp_mzs) == 0:
            return False
        
        # Compute PPM-based tolerance for this m/z
        tol = pred_mz * self.ppm * 1e-6
        
        # Check if any experimental peak is within tolerance
        min_dist = np.min(np.abs(exp_mzs - pred_mz))
        return min_dist <= tol
    
    def _get_atoms_from_fragment(self, frag_int: int, num_atoms: int) -> List[int]:
        """
        Extract atom indices from an ICEBERG fragment integer.
        
        The fragment integer is a bitmask where bit i is set if atom i is present.
        """
        atoms = []
        for i in range(num_atoms):
            if (frag_int >> i) & 1:
                atoms.append(i)
        return atoms
    
    def get_hallucinated_peaks(
        self,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        real_smi: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Identify ALL significant peaks (both matched and unmatched) with their match status.
        
        Returns peaks with a 'matched' boolean field indicating if within PPM tolerance.
        """
        try:
            real_CEs = sorted([ce for ce in real_specs.keys() if ce is not None])
        except Exception as e:
            print(f"Warning [ScoreBased.get_hallucinated_peaks]: Failed to get real CEs: {e}")
            return []
        
        if len(real_CEs) == 0:
            if None in real_specs.keys():
                real_CEs = [None]
            else:
                return []
        
        # Process each CE pair
        all_peaks = []
        
        for real_ce in real_CEs:
            try:
                real_spec = real_specs[real_ce]
            except (KeyError, Exception):
                continue
            
            # Find matching or closest predicted CE
            pred_ce = None
            if real_ce in pred_specs.keys():
                pred_ce = real_ce
            else:
                pred_CEs = [ce for ce in pred_specs.keys() if ce is not None]
                if pred_CEs:
                    target = float(real_ce) if real_ce is not None else 0
                    pred_ce = min(pred_CEs, key=lambda x: abs(float(x) - target))
            
            if pred_ce is None:
                continue
            
            try:
                pred_spec = pred_specs[pred_ce]
            except (KeyError, Exception):
                continue
            
            # Get peaks for this CE pair
            ce_peaks = self._compute_peaks_single(real_spec, pred_spec, real_ce)
            all_peaks.extend(ce_peaks)
        
        return all_peaks
    
    def _compute_peaks_single(
        self,
        real_spec: common.MassSpec,
        pred_spec: common.MassSpec,
        collision_energy: Optional[float] = None,
    ) -> List[Dict]:
        """
        Compute peak info for a single real/pred spectrum pair.
        
        Returns ALL peaks above threshold with matched/unmatched status.
        """
        # Validate real spectrum
        if real_spec.intens is None or len(real_spec.intens) == 0:
            return []
        
        real_max_inten = np.max(real_spec.intens)
        if real_max_inten == 0:
            return []
        
        real_norm_intens = real_spec.intens / real_max_inten
        
        # Get experimental m/z values above threshold
        exp_mask = real_norm_intens >= self.real_inten_threshold
        exp_mzs = real_spec.masses[exp_mask]
        
        if len(exp_mzs) == 0:
            exp_mzs = real_spec.masses
        
        # Validate predicted spectrum
        if pred_spec.intens is None or len(pred_spec.intens) == 0:
            return []
        
        pred_max_inten = np.max(pred_spec.intens)
        if pred_max_inten == 0:
            return []
        
        precursor_mz = getattr(pred_spec, 'precursor_mz', None)
        
        # Access spectrum data and fragments
        try:
            pred_spec_data = pred_spec.spec
            pred_int_frags = pred_spec.int_frags
            if pred_spec_data is None or pred_int_frags is None:
                return []
        except Exception as e:
            print(f"Warning [ScoreBased._compute_peaks_single]: Failed to access pred_spec: {e}")
            return []
        
        peaks = []
        
        for i, ((mz, inten), frag) in enumerate(zip(pred_spec_data, pred_int_frags)):
            try:
                # Skip precursor peak
                if precursor_mz is not None:
                    if np.abs(mz - precursor_mz) < precursor_mz * self.ppm * 1e-6:
                        continue
                
                norm_inten = inten / pred_max_inten
                
                # Skip low-intensity peaks
                if norm_inten < self.sim_inten_threshold:
                    continue
                
                # Check if peak is matched
                is_matched = self._is_peak_matched(mz, exp_mzs)
                
                # Compute distance to nearest experimental peak
                if len(exp_mzs) > 0:
                    min_dist = np.min(np.abs(exp_mzs - mz))
                else:
                    min_dist = float('inf')
                
                peaks.append({
                    'mz': mz,
                    'intensity': inten,
                    'norm_intensity': norm_inten,
                    'distance': min_dist,
                    'matched': is_matched,
                    'fragment': frag,
                    'peak_idx': i,
                    'collision_energy': collision_energy,
                })
                
            except Exception as e:
                continue
        
        # Sort by intensity (highest first) and limit
        peaks = sorted(peaks, key=lambda x: x['norm_intensity'], reverse=True)
        return peaks[:self.top_k_peaks]
    
    def get_bad_atoms(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        mol_str_canonicalized: bool = True,
        **kwargs,
    ) -> Set[int]:
        """
        Map peaks to atom scores and return atoms with negative scores.
        
        Matched peaks contribute positive scores, unmatched contribute negative.
        """
        atom_scores = self.compute_atom_scores(
            smiles, hallucinated_peaks, mol_str_canonicalized
        )
        
        if len(atom_scores) == 0:
            return set()
        
        # Return atoms with negative scores (more unmatched than matched)
        bad_atoms = set(np.where(atom_scores < 0)[0].tolist())
        
        return bad_atoms
    
    def compute_atom_scores(
        self,
        smiles: str,
        peaks: List[Dict],
        mol_str_canonicalized: bool = True,
    ) -> np.ndarray:
        """
        Compute per-atom scores based on matched/unmatched peak assignments.
        
        Args:
            smiles: SMILES string (should match the one used for ICEBERG prediction)
            peaks: List of peak dicts from get_hallucinated_peaks
            mol_str_canonicalized: Whether SMILES is already canonicalized
            
        Returns:
            Array of atom scores. Positive = good (matched), Negative = bad (unmatched)
        """
        if not peaks:
            return np.array([])
        
        # Get number of atoms from the molecule
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.array([])
            num_atoms = mol.GetNumAtoms()
        except Exception as e:
            print(f"Warning [ScoreBased.compute_atom_scores]: Failed to parse SMILES: {e}")
            return np.array([])
        
        atom_scores = np.zeros(num_atoms)
        
        for peak in peaks:
            frag_int = peak.get('fragment')
            if frag_int is None:
                continue
            
            # Extract atoms from the ICEBERG fragment integer
            present_atoms = self._get_atoms_from_fragment(frag_int, num_atoms)
            
            if not present_atoms:
                continue
            
            # Compute score contribution for this peak
            is_matched = peak.get('matched', False)
            norm_intensity = peak.get('norm_intensity', 0.5)
            
            # Weight by intensity if enabled
            weight = norm_intensity if self.intensity_weighting else 1.0
            
            if is_matched:
                # Matched peak: reward atoms
                score_delta = self.match_reward * weight
            else:
                # Unmatched peak: penalize atoms
                score_delta = -self.unmatch_penalty * weight
            
            # Apply to all atoms in fragment
            for atom_idx in present_atoms:
                if atom_idx < num_atoms:
                    atom_scores[atom_idx] += score_delta
        
        # Normalize scores
        if self.score_normalization == 'minmax':
            score_range = atom_scores.max() - atom_scores.min()
            if score_range > 0:
                # Normalize to [-1, 1] range
                atom_scores = 2 * (atom_scores - atom_scores.min()) / score_range - 1
        elif self.score_normalization == 'zscore':
            if atom_scores.std() > 0:
                atom_scores = (atom_scores - atom_scores.mean()) / atom_scores.std()
        # 'none': no normalization
        
        return atom_scores
    
    def create_masked_inputs(
        self,
        smiles: str,
        hallucinated_peaks: List[Dict],
        tokenizer: Any,
        num_masks: int = 1,
        mask_prob: float = 0.5,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Create masked inputs with score-proportional masking probability.
        
        Atoms with lower (more negative) scores have higher masking probability.
        """
        # Compute atom scores
        atom_scores = self.compute_atom_scores(
            smiles, hallucinated_peaks, mol_str_canonicalized=True
        )
        
        # Tokenize
        try:
            encoding = tokenizer(smiles, return_offsets_mapping=True, return_tensors='pt')
            base_input_ids = encoding['input_ids'][0]
            offsets = encoding['offset_mapping'][0]
            mask_token_id = tokenizer.mask_token_id
        except Exception as e:
            print(f"Warning [ScoreBased.create_masked_inputs]: Tokenization failed: {e}")
            return []
        
        if len(atom_scores) == 0:
            # No scores - mask everything
            masked_inputs = []
            for _ in range(num_masks):
                input_ids = base_input_ids.clone()
                input_ids[:] = mask_token_id
                masked_inputs.append(input_ids.unsqueeze(0))
            return masked_inputs
        
        # Map SMILES chars to atoms
        try:
            char_to_atom = _map_smiles_chars_to_atoms(smiles)
        except Exception as e:
            print(f"Warning [ScoreBased.create_masked_inputs]: Char mapping failed: {e}")
            masked_inputs = []
            for _ in range(num_masks):
                input_ids = base_input_ids.clone()
                input_ids[:] = mask_token_id
                masked_inputs.append(input_ids.unsqueeze(0))
            return masked_inputs
        
        # Compute per-token scores (minimum of covered atom scores - most negative wins)
        token_scores = []
        for token_idx, (start_char, end_char) in enumerate(offsets):
            if start_char == end_char:
                token_scores.append(0.0)  # Special token
                continue
            
            min_score = float('inf')
            has_atom = False
            for char_idx in range(start_char, end_char):
                if char_idx < len(char_to_atom):
                    atom_idx = char_to_atom[char_idx]
                    if atom_idx != -1 and atom_idx < len(atom_scores):
                        min_score = min(min_score, atom_scores[atom_idx])
                        has_atom = True
            
            if has_atom:
                token_scores.append(min_score)
            else:
                token_scores.append(0.0)
        
        token_scores = np.array(token_scores)
        
        # Convert scores to masking probabilities
        # Lower (more negative) scores -> higher mask probability
        # Score range is roughly [-1, 1] after normalization
        # We want: score=-1 -> max_mask_prob, score=+1 -> min_mask_prob (or 0)
        
        # Use the passed mask_prob as the effective maximum masking probability
        # This allows the CLI --mask-prob arg to control the upper bound
        effective_max_mask_prob = min(mask_prob, self.max_mask_prob)
        effective_min_mask_prob = min(self.min_mask_prob, effective_max_mask_prob)
        
        token_mask_probs = np.zeros(len(token_scores))
        
        for i, score in enumerate(token_scores):
            if token_scores[i] == 0 and (offsets[i][0] == offsets[i][1]):
                # Special token - don't mask
                token_mask_probs[i] = 0.0
            elif score <= 0:
                # Negative or zero score: mask with higher probability
                # score=0 -> min prob, score=-1 -> max prob
                prob = effective_min_mask_prob + (effective_max_mask_prob - effective_min_mask_prob) * (-score)
                token_mask_probs[i] = prob
            else:
                # Positive score: low or no masking
                # score=0 -> min_prob, score=+1 -> 0
                prob = effective_min_mask_prob * (1 - score)
                token_mask_probs[i] = max(0, prob)
        
        # Create masked versions
        masked_inputs = []
        
        for _ in range(num_masks):
            input_ids = base_input_ids.clone()
            
            for idx, prob in enumerate(token_mask_probs):
                if prob > 0 and np.random.random() < prob:
                    input_ids[idx] = mask_token_id
            
            masked_inputs.append(input_ids.unsqueeze(0))
        
        return masked_inputs

MASKING_STRATEGIES = {
    'simple': SimpleMaskingStrategy,
    'intensity_weighted': IntensityWeightedMaskingStrategy,
    'score_based': ScoreBasedMaskingStrategy,
}


def create_masking_strategy(
    strategy_name: str,
    top_k_peaks: int = 3,
    ppm: int = 20,
    real_inten_threshold: float = 0.05,
    sim_inten_threshold: float = 0.01,
    **kwargs,
) -> BaseMaskingStrategy:
    """
    Factory function to create a masking strategy by name.
    
    Args:
        strategy_name: Name of the strategy ('simple', 'intensity_weighted', or 'score_based')
        top_k_peaks: Number of top hallucinated peaks to consider
        ppm: Parts-per-million tolerance
        real_inten_threshold: Intensity threshold for real peaks
        sim_inten_threshold: Intensity threshold for simulated peaks
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        BaseMaskingStrategy instance
        
    Raises:
        ValueError: If strategy_name is not recognized
    """
    if strategy_name not in MASKING_STRATEGIES:
        available = ', '.join(MASKING_STRATEGIES.keys())
        raise ValueError(f"Unknown masking strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = MASKING_STRATEGIES[strategy_name]
    
    # Common arguments
    common_args = {
        'top_k_peaks': top_k_peaks,
        'ppm': ppm,
        'real_inten_threshold': real_inten_threshold,
        'sim_inten_threshold': sim_inten_threshold,
    }
    
    # Strategy-specific defaults
    if strategy_name == 'intensity_weighted':
        strategy_args = {
            **common_args,
            'distance_weight': kwargs.get('distance_weight', 0.6),
            'intensity_weight': kwargs.get('intensity_weight', 0.4),
            'fragment_size_penalty': kwargs.get('fragment_size_penalty', True),
            'neighbor_extension': kwargs.get('neighbor_extension', True),
            'neighbor_weight': kwargs.get('neighbor_weight', 0.3),
            'min_mask_prob': kwargs.get('min_mask_prob', 0.0),
            'max_mask_prob': kwargs.get('max_mask_prob', 1.0),
        }
    elif strategy_name == 'score_based':
        strategy_args = {
            **common_args,
            'top_k_peaks': kwargs.get('top_k_peaks', 10),  # Higher default for score_based
            'match_reward': kwargs.get('match_reward', 1.0),
            'unmatch_penalty': kwargs.get('unmatch_penalty', 1.0),
            'intensity_weighting': kwargs.get('intensity_weighting', True),
            'min_mask_prob': kwargs.get('min_mask_prob', 0.0),
            'max_mask_prob': kwargs.get('max_mask_prob', 1.0),
            'score_normalization': kwargs.get('score_normalization', 'minmax'),
        }
    else:
        strategy_args = common_args
    
    return strategy_class(**strategy_args)


def _map_smiles_chars_to_atoms(smiles: str) -> List[int]:
    """
    Map every character in a SMILES string to the Atom Index it belongs to.
    
    Heuristic:
    - Atom chars: belong to themselves
    - Bonds (-, =, #, etc.) and '(': belong to the *next* atom
    - Ring digits and ')': belong to the *previous* atom
    
    Args:
        smiles: SMILES string
        
    Returns:
        List mapping character index -> atom index (or -1 if not assigned)
    """
    # Regex for atoms (including bracketed and organic subset)
    atom_pattern = r"(\[[^]]+\]|Cl|Br|Si|Se|Na|Mg|Ca|Fe|Zn|Cu|Ag|Au|Hg|As|Co|Ni|Mn|Cr|V|Ti|Sc|Li|K|B|C|N|O|P|S|F|I|b|c|n|o|s|p)"
    
    char_to_atom = [-1] * len(smiles)
    
    current_atom_idx = 0
    for match in re.finditer(atom_pattern, smiles):
        start, end = match.span()
        for i in range(start, end):
            char_to_atom[i] = current_atom_idx
        current_atom_idx += 1
    
    # Assign non-atom characters
    last_atom_idx = -1
    
    for i in range(len(smiles)):
        # If this char is already assigned to an atom, update last_atom
        if char_to_atom[i] != -1:
            last_atom_idx = char_to_atom[i]
            continue
        
        char = smiles[i]
        
        # Backward association: Ring digits, Close Branch ')'
        if char.isdigit() or char in '%)':
            if last_atom_idx != -1:
                char_to_atom[i] = last_atom_idx
        
        # Forward association: Bonds, Open Branch '('
        elif char in '(=#$-:/\\':
            # Find the next atom char
            next_atom_idx = -1
            for j in range(i + 1, len(smiles)):
                if char_to_atom[j] != -1:
                    next_atom_idx = char_to_atom[j]
                    break
            
            if next_atom_idx != -1:
                char_to_atom[i] = next_atom_idx
            elif last_atom_idx != -1:
                # Fallback: if no next atom, assign to prev
                char_to_atom[i] = last_atom_idx
    
    return char_to_atom


class HallucationScorer:
    """
    Utility class for computing per-atom hallucination scores.
    """
    
    def __init__(
        self,
        masking_strategy: BaseMaskingStrategy,
        score_aggregation: str = 'sum',
    ):
        """
        Initialize the hallucination scorer.
        
        Args:
            masking_strategy: Masking strategy to use for hallucination detection
            score_aggregation: How to aggregate scores ('sum', 'max', 'mean')
        """
        self.masking_strategy = masking_strategy
        self.score_aggregation = score_aggregation
    
    def compute_atom_scores(
        self,
        smiles: str,
        real_specs: common.CompositeMassSpec,
        pred_specs: common.CompositeMassSpec,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute per-atom hallucination scores.
        
        Args:
            smiles: SMILES string
            real_specs: Real composite spectrum
            pred_specs: Predicted composite spectrum
            
        Returns:
            Array of shape [num_atoms] with hallucination scores
        """
        try:
            engine = fragmentation.FragmentEngine(smiles, mol_str_type='smiles')
            num_atoms = engine.natoms
        except Exception:
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms() if mol else 0
            if num_atoms == 0:
                return np.array([])
        
        atom_scores = np.zeros(num_atoms)
        
        halluc_peaks = self.masking_strategy.get_hallucinated_peaks(
            real_specs, pred_specs, **kwargs
        )
        
        for peak in halluc_peaks:
            frag_idx = peak.get('fragment')
            if frag_idx is None:
                continue
            
            try:
                present_atoms, _ = engine.get_present_atoms(frag_idx)
                
                if self.score_aggregation == 'sum':
                    atom_scores[present_atoms] += peak.get('norm_intensity', 0)
                elif self.score_aggregation == 'max':
                    atom_scores[present_atoms] = np.maximum(
                        atom_scores[present_atoms], 
                        peak.get('norm_intensity', 0)
                    )
                elif self.score_aggregation == 'mean':
                    # For mean, we'd need to track counts
                    atom_scores[present_atoms] += peak.get('norm_intensity', 0)
            except Exception:
                continue
        
        # Normalize
        if atom_scores.max() > 0:
            atom_scores = atom_scores / atom_scores.max()
        
        return atom_scores


class MaskBuilder:
    """
    Helper class to build masked inputs for batched refinement.
    """
    
    def __init__(
        self,
        masking_strategy: BaseMaskingStrategy,
        tokenizer: Any,
        mask_prob: float = 0.5,
    ):
        """
        Initialize the mask builder.
        
        Args:
            masking_strategy: Masking strategy to use
            tokenizer: HuggingFace tokenizer
            mask_prob: Probability of masking each bad token
        """
        self.masking_strategy = masking_strategy
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
    
    def build_masked_batch(
        self,
        molecules: List[Dict],
        num_masks_per_molecule: int = 1,
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Build a batch of masked inputs for multiple molecules.
        
        Args:
            molecules: List of dicts with keys:
                - 'smiles': SMILES/SAFE string
                - 'real_specs': Real composite spectrum
                - 'pred_specs': Predicted composite spectrum
            num_masks_per_molecule: Number of masked versions per molecule
            
        Returns:
            Tuple of:
            - List of masked input_ids tensors
            - List of metadata dicts (original smiles, etc.)
        """
        all_masked_inputs = []
        all_metadata = []
        
        for mol_data in molecules:
            smiles = mol_data['smiles']
            real_specs = mol_data['real_specs']
            pred_specs = mol_data['pred_specs']
            
            # Get hallucinated peaks
            halluc_peaks = self.masking_strategy.get_hallucinated_peaks(
                real_specs, pred_specs,
                real_smi=mol_data.get('real_smi'),
            )
            
            # Create masked inputs
            masked_inputs = self.masking_strategy.create_masked_inputs(
                smiles=smiles,
                hallucinated_peaks=halluc_peaks,
                tokenizer=self.tokenizer,
                num_masks=num_masks_per_molecule,
                mask_prob=self.mask_prob,
            )
            
            for masked_input in masked_inputs:
                all_masked_inputs.append(masked_input)
                all_metadata.append({
                    'original_smiles': smiles,
                    'halluc_peaks': halluc_peaks,
                })
        
        return all_masked_inputs, all_metadata