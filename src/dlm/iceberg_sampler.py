"""
ICEBERG-guided inference-time scaling sampler for DLM.

This module provides the IcebergSampler class that extends the standard FRIGID-base
sampler with ICEBERG-guided refinement for improved structure elucidation.

The algorithm ("Spectrum-Error Guided Refinement"):
1. Generate diverse candidate molecules (B samples per round)
2. Select top K unique candidates by fingerprint similarity
3. Simulate mass spectra with ICEBERG
4. Identify hallucinated peaks and map to atoms/tokens
5. Create M masked versions per molecule and refine with DLM
6. Repeat for R rounds
7. Rank all unique candidates by fingerprint similarity
"""

import os
import sys
import re
import uuid
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Callable, Union
from collections import defaultdict

import time

import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

# Ensure the repo-local ms-pred submodule is importable when not installed.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ms_pred_src_path = os.path.join(project_root, 'ms-pred', 'src')
if os.path.exists(ms_pred_src_path) and ms_pred_src_path not in sys.path:
    sys.path.insert(0, ms_pred_src_path)

import ms_pred.common as common
from ms_pred.dag_pred.iceberg_elucidation import iceberg_prediction, load_real_spec
from ms_pred.common.chem_utils import VALID_ELEMENTS

from dlm.sampler import Sampler
from dlm.utils.masking_utils import (
    BaseMaskingStrategy,
    SimpleMaskingStrategy,
    create_masking_strategy,
)
from dlm.utils.benchmark_utils import (
    normalize_formula,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    get_inchikey_first_block,
    predict_token_count,
)
from ms_pred.dag_pred.iceberg_elucidation import load_pred_spec


@dataclass
class IcebergConfig:
    """Configuration for ICEBERG spectrum prediction."""
    gen_ckpt: str = ''
    inten_ckpt: str = ''
    python_path: str = 'python'
    cuda_devices: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 8
    num_gpu_workers: int = 1
    num_cpu_workers: int = 8
    ppm: int = 20
    num_bins: int = 15000
    upper_limit: int = 1500
    adduct: str = '[M+H]+'
    sparse_k: int = 100
    max_nodes: int = 100
    threshold: float = 0.0


@dataclass
class ScalingConfig:
    """Configuration for inference-time scaling."""
    batch_size: int = 128  # B: total samples per round
    num_unique_to_refine: int = 8  # K: unique molecules to refine
    masks_per_molecule: int = 12  # M: masks per molecule
    num_rounds: int = 50  # R: number of refinement rounds
    mask_prob: float = 0.75  # Probability of masking each bad token
    top_k_halluc_peaks: int = 10  # Number of hallucinated peaks to consider
    halluc_inten_threshold: float = 0.05  # Intensity threshold for hallucination
    softmax_temp: float = 1.0  # Temperature for sampling
    randomness: float = 10.0  # Randomness factor
    collision_energies: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90])
    nce: bool = True  # Whether collision energies are normalized


def remove_stereo_from_smiles(smiles: str) -> Optional[str]:
    """Remove stereochemistry from a SMILES string using ms_pred utility."""
    try:
        return common.rm_stereo(smiles)
    except Exception:
        return None


def get_inchikey_no_stereo(smiles: str) -> Optional[str]:
    """Get InChI key first block (no stereochemistry) for a SMILES."""
    try:
        # Remove stereochemistry using ms_pred utility
        smi_no_stereo = common.rm_stereo(smiles)
        if smi_no_stereo is None:
            return None
        # Get full InChI key
        inchi_key = common.inchikey_from_smiles(smi_no_stereo)
        # Return first block (connectivity layer)
        return get_inchikey_first_block(inchi_key) if inchi_key else None
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES corresponds to a valid molecule."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        Chem.SanitizeMol(mol)
        
        # Check for fragmented molecules
        if len(Chem.GetMolFrags(mol)) > 1:
            return False
        
        # Check for chemistry problems
        if Chem.DetectChemistryProblems(mol):
            return False
        
        # Check for unsupported elements
        if not all(atom.GetSymbol() in VALID_ELEMENTS for atom in mol.GetAtoms()):
            return False
        
        # Check molecular weight
        if Descriptors.ExactMolWt(mol) > 1500:
            return False
        
        return True
    except Exception:
        return False


def normalize_instrument_type(instrument: Optional[str]) -> str:
    """
    Normalize instrument type to one of: 'Orbitrap', 'QTOF', or 'Unknown'.
    
    Handles cases where instrument type is not exactly 'QTOF' but contains 'qtof',
    or contains 'orbitrap', etc.
    
    Args:
        instrument: Raw instrument type string (e.g., 'QTOF (LCMS)', 'Orbitrap', 'Q-TOF', etc.)
        
    Returns:
        Normalized instrument type: 'Orbitrap', 'QTOF', or 'Unknown'
    """
    if instrument is None or (isinstance(instrument, float) and np.isnan(instrument)):
        return "Orbitrap" # for some reason some MSG data has NaN instruments, but MSG ICEBERG model doesn't support unknown
    
    if not isinstance(instrument, str):
        return 'Unknown'
    
    instrument_lower = instrument.lower().strip()
    
    # Check for empty string
    if not instrument_lower:
        return 'Unknown'
    
    # Check for QTOF variants (Q-TOF, qTOF, QTOF (LCMS), etc.)
    if 'qtof' in instrument_lower or 'q-tof' in instrument_lower:
        return 'QTOF'
    
    # Check for Orbitrap variants
    if 'orbitrap' in instrument_lower:
        return 'Orbitrap'
    
    # Unknown otherwise
    return 'Unknown'


class IcebergPredictionCache:
    """
    Cache for ICEBERG predictions to avoid redundant computation.

    Keys are (inchi_key_no_stereo, collision_energy, instrument, ionization) tuples.
    Values are CompositeMassSpec objects.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, int, str, str], common.CompositeMassSpec] = {}
        self._smiles_to_ikey: Dict[str, str] = {}  # Cache SMILES -> InChI key mapping
    
    def get_ikey(self, smiles: str) -> Optional[str]:
        """Get cached InChI key or compute it."""
        if smiles not in self._smiles_to_ikey:
            ikey = get_inchikey_no_stereo(smiles)
            if ikey:
                self._smiles_to_ikey[smiles] = ikey
        return self._smiles_to_ikey.get(smiles)
    
    def get(self, smiles: str, collision_energy: int, instrument: str, ionization: str = '[M+H]+') -> Optional[common.CompositeMassSpec]:
        """Get cached prediction if available."""
        ikey = self.get_ikey(smiles)
        if ikey is None:
            return None
        return self._cache.get((ikey, collision_energy, instrument, ionization))

    def has(self, smiles: str, collision_energy: int, instrument: str, ionization: str = '[M+H]+') -> bool:
        """Check if prediction is cached."""
        ikey = self.get_ikey(smiles)
        if ikey is None:
            return False
        return (ikey, collision_energy, instrument, ionization) in self._cache

    def put(self, smiles: str, collision_energy: int, instrument: str, pred_spec: common.CompositeMassSpec, ionization: str = '[M+H]+'):
        """Cache a prediction."""
        ikey = self.get_ikey(smiles)
        if ikey:
            self._cache[(ikey, collision_energy, instrument, ionization)] = pred_spec
    
    def get_uncached_pairs(
        self, 
        smiles_list: List[str], 
        collision_energies: List[int],
        instrument: str,
    ) -> Dict[int, List[str]]:
        """
        Get (collision_energy -> list of smiles) for uncached predictions.
        
        Returns a dict mapping collision energy to list of SMILES that need prediction
        for the given instrument type.
        """
        ce_to_smiles: Dict[int, List[str]] = defaultdict(list)
        seen_ikeys: Dict[int, Set[str]] = defaultdict(set)  # Avoid duplicate SMILES per CE
        
        for smiles in smiles_list:
            ikey = self.get_ikey(smiles)
            if ikey is None:
                continue
            
            for ce in collision_energies:
                if not self.has(smiles, ce, instrument) and ikey not in seen_ikeys[ce]:
                    ce_to_smiles[ce].append(smiles)
                    seen_ikeys[ce].add(ikey)
        
        return dict(ce_to_smiles)
    
    def get_uncached_pairs_grouped(
        self,
        smiles_instrument_ionization_triples: List[Tuple[str, str, str]],
        collision_energies: List[int],
    ) -> Dict[Tuple[int, str, str], List[str]]:
        """
        Get ((collision_energy, instrument, ionization) -> list of smiles) for uncached predictions.

        Args:
            smiles_instrument_ionization_triples: List of (SMILES, instrument, ionization) tuples
            collision_energies: List of collision energies to check

        Returns:
            Dict mapping (CE, instrument, ionization) tuple to list of SMILES needing prediction
        """
        ce_inst_ion_to_smiles: Dict[Tuple[int, str, str], List[str]] = defaultdict(list)
        seen_ikeys: Dict[Tuple[int, str, str], Set[str]] = defaultdict(set)

        for smiles, instrument, ionization in smiles_instrument_ionization_triples:
            ikey = self.get_ikey(smiles)
            if ikey is None:
                continue

            for ce in collision_energies:
                key = (ce, instrument, ionization)
                if not self.has(smiles, ce, instrument, ionization) and ikey not in seen_ikeys[key]:
                    ce_inst_ion_to_smiles[key].append(smiles)
                    seen_ikeys[key].add(ikey)

        return dict(ce_inst_ion_to_smiles)
    
    def __len__(self) -> int:
        return len(self._cache)


class SampleState:
    """
    Tracks the state of a single test sample across scaling rounds.

    Attributes:
        name: Spectrum name/identifier
        real_specs: Real composite mass spectrum
        target_fp: Target fingerprint (from MIST encoder)
        target_formula: Target molecular formula (or default formula when using candidates)
        instrument: Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
        ionization: Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+')
        candidate_formulas: Optional list of (formula, probability) tuples for sampling
        all_unique_smiles: Set of all unique SMILES generated (by InChI key)
        all_unique_ikeys: Set of InChI keys for uniqueness tracking
        smiles_to_similarity: Dict mapping SMILES -> Tanimoto similarity
    """

    def __init__(
        self,
        name: str,
        real_specs: common.CompositeMassSpec,
        target_fp: np.ndarray,
        target_formula: str,
        instrument: str = 'Unknown',
        ionization: str = '[M+H]+',
        candidate_formulas: Optional[List[Tuple[str, float]]] = None,
    ):
        self.name = name
        self.real_specs = real_specs
        self.target_fp = target_fp
        self.target_formula = target_formula
        self.instrument = normalize_instrument_type(instrument)
        self.ionization = ionization
        self.candidate_formulas = candidate_formulas
        
        # Tracking across rounds
        self.all_unique_smiles: Set[str] = set()
        self.all_unique_ikeys: Set[str] = set()
        self.smiles_to_similarity: Dict[str, float] = {}
    
    def sample_formula(self) -> str:
        """
        Sample a formula from candidate formulas if available, otherwise return target_formula.
        
        When candidate_formulas is provided, samples according to their probabilities.
        Otherwise, returns the fixed target_formula.
        
        Returns:
            A molecular formula string
        """
        if self.candidate_formulas is None or len(self.candidate_formulas) == 0:
            return self.target_formula
        
        formulas = [f for f, _ in self.candidate_formulas]
        probs = [p for _, p in self.candidate_formulas]
        
        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Fallback to uniform
            probs = [1.0 / len(formulas)] * len(formulas)
        
        return np.random.choice(formulas, p=probs)
    
    def sample_formulas(self, n: int) -> List[str]:
        """
        Sample n formulas from candidate formulas.
        
        Args:
            n: Number of formulas to sample
            
        Returns:
            List of molecular formula strings
        """
        return [self.sample_formula() for _ in range(n)]
    
    def add_smiles(self, smiles: str, fp_bits: int = 4096, fp_radius: int = 2) -> bool:
        """
        Add a SMILES to the collection if it's unique.
        
        Returns True if the SMILES was added (is unique), False otherwise.
        """
        if not is_valid_smiles(smiles):
            return False
        
        ikey = get_inchikey_no_stereo(smiles)
        if ikey is None or ikey in self.all_unique_ikeys:
            return False
        
        self.all_unique_ikeys.add(ikey)
        self.all_unique_smiles.add(smiles)
        
        # Compute and cache similarity
        fp = compute_morgan_fingerprint(smiles, fp_bits, fp_radius)
        if fp is not None:
            sim = compute_tanimoto_similarity(self.target_fp, fp)
            self.smiles_to_similarity[smiles] = sim
        else:
            self.smiles_to_similarity[smiles] = 0.0
        
        return True
    
    def add_smiles_batch(
        self, 
        smiles_list: List[str], 
        fp_bits: int = 4096, 
        fp_radius: int = 2
    ) -> int:
        """Add multiple SMILES, returns count of newly added unique SMILES."""
        count = 0
        for smiles in smiles_list:
            if self.add_smiles(smiles, fp_bits, fp_radius):
                count += 1
        return count
    
    def get_top_k_smiles(self, k: int) -> List[str]:
        """Get top K unique SMILES by similarity."""
        sorted_pairs = sorted(
            self.smiles_to_similarity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [smi for smi, _ in sorted_pairs[:k]]
    
    def get_ranked_smiles(self) -> List[str]:
        """Get all unique SMILES ranked by similarity (descending)."""
        sorted_pairs = sorted(
            self.smiles_to_similarity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [smi for smi, _ in sorted_pairs]


class IcebergSampler:
    """
    ICEBERG-guided inference-time scaling sampler for DLM.
    
    This class orchestrates the full inference-time scaling pipeline:
    1. Generate molecules from scratch (round 0)
    2. Run batched ICEBERG predictions
    3. Identify hallucinated peaks and create masked inputs
    4. Refine molecules using masked generation
    5. Repeat for R rounds
    
    For efficiency, ICEBERG predictions are batched across all test samples
    and cached to avoid redundant computation.
    """
    
    def __init__(
        self,
        sampler: Sampler,
        iceberg_config: IcebergConfig,
        scaling_config: ScalingConfig,
        results_dir: str = '/tmp/iceberg_scaling',
        token_model: Optional[Any] = None,
        token_features: Optional[List[str]] = None,
        is_ngboost: bool = False,
        sigma_lambda: float = 3.0,
        fp_bits: int = 4096,
        fp_radius: int = 2,
        masking_strategy: Optional[BaseMaskingStrategy] = None,
        masking_strategy_name: str = 'simple',
        incl_unknown_instrument: bool = True,
    ):
        """
        Initialize the ICEBERG sampler.
        
        Args:
            sampler: DLM Sampler instance
            iceberg_config: ICEBERG configuration
            scaling_config: Scaling configuration
            results_dir: Directory for intermediate ICEBERG results
            token_model: Optional token count prediction model
            token_features: Features for token model
            is_ngboost: Whether token model is NGBoost
            sigma_lambda: Variance multiplier for NGBoost sampling
            fp_bits: Fingerprint bits
            fp_radius: Fingerprint radius
            masking_strategy: Optional pre-configured masking strategy instance.
                              If None, creates one based on masking_strategy_name.
            masking_strategy_name: Name of masking strategy to use if masking_strategy
                                   is None. Options: 'simple', 'intensity_weighted'
            incl_unknown_instrument: Whether to include 'Unknown' instrument type in ICEBERG predictions
        """
        self.sampler = sampler
        self.iceberg_config = iceberg_config
        self.scaling_config = scaling_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.incl_unknown_instrument = incl_unknown_instrument
        
        # Validate configuration
        assert scaling_config.batch_size > 0, "batch_size must be positive"
        #assert scaling_config.num_unique_to_refine > 0, "num_unique_to_refine must be positive"
        #assert scaling_config.masks_per_molecule > 0, "masks_per_molecule must be positive"
        assert scaling_config.num_rounds > 0, "num_rounds must be positive"
        assert 0.0 <= scaling_config.mask_prob <= 1.0, "mask_prob must be in [0, 1]"
        
        # Ensure B >= M*K for rounds > 0
        M_times_K = scaling_config.masks_per_molecule * scaling_config.num_unique_to_refine
        assert scaling_config.batch_size >= M_times_K, (
            f"batch_size ({scaling_config.batch_size}) must be >= "
            f"masks_per_molecule * num_unique_to_refine ({M_times_K})"
        )
        
        # Token model for length prediction
        self.token_model = token_model
        self.token_features = token_features
        self.is_ngboost = is_ngboost
        self.sigma_lambda = sigma_lambda
        
        # Fingerprint settings
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        
        # ICEBERG prediction cache
        self._pred_cache = IcebergPredictionCache()
        
        # Masking strategy - use provided or create from name
        if masking_strategy is not None:
            self._masking_strategy = masking_strategy
        else:
            self._masking_strategy = create_masking_strategy(
                strategy_name=masking_strategy_name,
                top_k_peaks=scaling_config.top_k_halluc_peaks,
                ppm=iceberg_config.ppm,
                real_inten_threshold=scaling_config.halluc_inten_threshold,
                sim_inten_threshold=0.05,
            )
        
        print(f"Using masking strategy: {type(self._masking_strategy).__name__}")
        
        # Tokenizer for masking
        self._tokenizer = sampler.model.tokenizer
    
    def _predict_token_lengths(
        self,
        formula: str,
        num_samples: int,
    ) -> Optional[List[int]]:
        """Predict token lengths for a formula."""
        if self.token_model is None or self.token_features is None:
            return None
        
        predicted_len, predicted_sigma = predict_token_count(
            self.token_model,
            self.token_features,
            formula,
            self.is_ngboost,
            self.sigma_lambda,
        )
        
        if predicted_len is None:
            return None
        
        if self.is_ngboost and predicted_sigma is not None:
            # NGBoost: sample from normal distribution
            std_dev = np.sqrt(predicted_sigma * self.sigma_lambda)
            lengths = [
                max(1, int(round(np.random.normal(predicted_len, std_dev))))
                for _ in range(num_samples)
            ]
        else:
            # sklearn: sample from fixed range
            low = max(1, predicted_len - 3)
            high = predicted_len + 3
            lengths = [
                np.random.randint(low, high + 1)
                for _ in range(num_samples)
            ]
        
        return lengths
    
    def _generate_from_scratch(
        self,
        state: SampleState,
        num_samples: int,
    ) -> List[str]:
        """
        Generate molecules from scratch (fully masked).
        
        Args:
            state: Sample state with target fingerprint and formula
            num_samples: Number of samples to generate
            
        Returns:
            List of generated SMILES
        """
        if num_samples <= 0:
            return []
        
        # Check if we have candidate formulas to sample from
        if state.candidate_formulas is not None and len(state.candidate_formulas) > 0:
            # Sample formulas for each generation
            sampled_formulas = state.sample_formulas(num_samples)
            
            # Group samples by formula for efficient batching
            from collections import Counter
            formula_counts = Counter(sampled_formulas)
            
            all_samples = []
            for formula, count in formula_counts.items():
                # Predict target lengths for this formula
                target_lengths = self._predict_token_lengths(formula, count)
                
                # Generate using unified conditioning
                try:
                    samples = self.sampler.unified_conditioned_generation(
                        formula=formula,
                        fingerprint=state.target_fp,
                        num_samples=count,
                        softmax_temp=self.scaling_config.softmax_temp,
                        randomness=self.scaling_config.randomness,
                        target_lengths=target_lengths,
                        min_add_len=2,
                    )
                    all_samples.extend(samples)
                except Exception as e:
                    print(f"Warning: Generation failed for {state.name} with formula {formula}: {e}")
            
            return all_samples
        else:
            # Use fixed target formula (default behavior)
            target_lengths = self._predict_token_lengths(
                state.target_formula,
                num_samples,
            )
            
            # Generate using unified conditioning
            try:
                samples = self.sampler.unified_conditioned_generation(
                    formula=state.target_formula,
                    fingerprint=state.target_fp,
                    num_samples=num_samples,
                    softmax_temp=self.scaling_config.softmax_temp,
                    randomness=self.scaling_config.randomness,
                    target_lengths=target_lengths,
                    min_add_len=2,
                )
            except Exception as e:
                print(f"Warning: Generation failed for {state.name}: {e}")
                samples = []
            
            return samples
    
    def _generate_from_masked(
        self,
        masked_input_ids: torch.Tensor,
        formula: Union[str, List[str]],
        fingerprint: np.ndarray,
    ) -> List[str]:
        """
        Generate molecules from masked input.
        
        Args:
            masked_input_ids: Tensor of shape [batch, seq_len] with masked tokens
            formula: Target molecular formula (string) or list of formulas (one per batch item)
            fingerprint: Target fingerprint
            
        Returns:
            List of generated SMILES
        """
        if masked_input_ids.size(0) == 0:
            return []
        
        try:
            samples = self.sampler.generate(
                masked_input_ids,
                softmax_temp=self.scaling_config.softmax_temp,
                randomness=self.scaling_config.randomness,
                formula=formula,
                fingerprint=fingerprint,
            )
        except Exception as e:
            print(f"Warning: Masked generation failed: {e}")
            samples = []
        
        return samples
    
    def _run_iceberg_batch(
        self,
        smiles_list: List[str],
        collision_energies: List[int],
        instrument: str = 'Unknown',
        ionization: str = '[M+H]+',
    ) -> Dict[str, common.CompositeMassSpec]:
        """
        Run ICEBERG predictions for a batch of SMILES.

        Args:
            smiles_list: List of SMILES to predict
            collision_energies: List of collision energies
            instrument: Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
            ionization: Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+')

        Returns:
            Dict mapping SMILES -> CompositeMassSpec
        """
        if not smiles_list:
            return {}
        
        # Filter to valid SMILES
        valid_smiles = [s for s in smiles_list if is_valid_smiles(s)]
        if not valid_smiles:
            return {}
        
        # Normalize instrument type
        instrument = normalize_instrument_type(instrument)
        
        # Create unique experiment directory
        unique_id = uuid.uuid4().hex[:8]
        exp_name = f"scaling_{unique_id}"
        batch_results_dir = self.results_dir / f"batch_{unique_id}"
        batch_results_dir.mkdir(parents=True, exist_ok=True)
        
        results: Dict[str, common.CompositeMassSpec] = {}
        
        try:
            save_dir, _ = iceberg_prediction(
                candidate_smiles=valid_smiles,
                collision_energies=collision_energies,
                nce=self.scaling_config.nce,
                adduct=ionization,
                instrument=instrument,
                incl_unknown_instrument=self.incl_unknown_instrument,
                exp_name=exp_name,
                python_path=self.iceberg_config.python_path,
                gen_ckpt=self.iceberg_config.gen_ckpt,
                inten_ckpt=self.iceberg_config.inten_ckpt,
                cuda_devices=self.iceberg_config.cuda_devices,
                batch_size=self.iceberg_config.batch_size,
                num_cpu_workers=self.iceberg_config.num_cpu_workers,
                num_gpu_workers=self.iceberg_config.num_gpu_workers,
                sparse_k=self.iceberg_config.sparse_k,
                max_nodes=self.iceberg_config.max_nodes,
                threshold=self.iceberg_config.threshold,
                binned_out=False,
                ppm=self.iceberg_config.ppm,
                num_bins=self.iceberg_config.num_bins,
                results_dir=str(batch_results_dir),
            )
            
            # Load predicted spectra
            smiles_arr, pred_specs_list = load_pred_spec(save_dir)
            
            # Build results dict
            for smi, pred_spec in zip(smiles_arr, pred_specs_list):
                results[smi] = pred_spec
            
        except Exception as e:
            print(f"Warning: ICEBERG prediction failed: {e}")
        finally:
            # Clean up temporary directory
            try:
                if batch_results_dir.exists():
                    shutil.rmtree(batch_results_dir)
            except Exception:
                pass
        
        return results
    
    def _run_batched_iceberg_predictions(
        self,
        sample_states: List[SampleState],
        molecules_per_sample: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, common.CompositeMassSpec]]:
        """
        Run batched ICEBERG predictions across all samples.
        
        Groups predictions by (collision_energy, instrument) pairs for efficiency.
        
        Args:
            sample_states: List of sample states
            molecules_per_sample: Dict mapping sample name -> list of SMILES
            
        Returns:
            Dict mapping sample_name -> {smiles -> pred_spec}
        """
        # Collect all unique (SMILES, instrument, ionization) triples across all samples
        smiles_instrument_ionization_triples: List[Tuple[str, str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        for state in sample_states:
            smiles_list = molecules_per_sample.get(state.name, [])
            for smi in smiles_list:
                triple = (smi, state.instrument, state.ionization)
                if triple not in seen_triples:
                    smiles_instrument_ionization_triples.append(triple)
                    seen_triples.add(triple)

        # Get uncached (CE, instrument, ionization) -> smiles_list
        ce_inst_ion_to_smiles = self._pred_cache.get_uncached_pairs_grouped(
            smiles_instrument_ionization_triples,
            self.scaling_config.collision_energies,
        )

        # Run ICEBERG predictions for each (collision_energy, instrument, ionization) group
        for (ce, instrument, ionization), smiles_batch in ce_inst_ion_to_smiles.items():
            if not smiles_batch:
                continue

            print(f"  Running ICEBERG for {len(smiles_batch)} molecules at CE={ce}eV, instrument={instrument}, ionization={ionization}...")

            # Run predictions
            pred_results = self._run_iceberg_batch(
                smiles_batch,
                [ce],  # Single CE for efficiency
                instrument=instrument,
                ionization=ionization,
            )

            # Cache results with instrument and ionization
            for smi, pred_spec in pred_results.items():
                self._pred_cache.put(smi, ce, instrument, pred_spec, ionization=ionization)
        
        # Build per-sample results from cache
        results: Dict[str, Dict[str, common.CompositeMassSpec]] = {}
        
        for state in sample_states:
            sample_name = state.name
            smiles_list = molecules_per_sample.get(sample_name, [])
            instrument = state.instrument
            ionization = state.ionization

            results[sample_name] = {}
            for smi in smiles_list:
                # Combine predictions across all CEs into one CompositeMassSpec
                # Collect all MassSpec objects from different collision energies
                all_mass_specs = []
                for ce in self.scaling_config.collision_energies:
                    pred_spec = self._pred_cache.get(smi, ce, instrument, ionization)
                    if pred_spec is not None:
                        # Extract MassSpec objects from the CompositeMassSpec
                        for ms in pred_spec.values():
                            all_mass_specs.append(ms)
                
                if all_mass_specs:
                    # Normalize root_canonical_smiles to avoid assertion error in CompositeMassSpec
                    # Different ICEBERG runs might return slightly different SMILES representations
                    canonical_smi = all_mass_specs[0].root_canonical_smiles
                    for ms in all_mass_specs:
                        ms.root_canonical_smiles = canonical_smi
                    
                    # Create a new CompositeMassSpec from all collected MassSpec objects
                    combined_spec = common.CompositeMassSpec(all_mass_specs)
                    results[sample_name][smi] = combined_spec
        
        return results
    
    def _create_masked_inputs_for_sample(
        self,
        state: SampleState,
        smiles_to_refine: List[str],
        pred_specs: Dict[str, common.CompositeMassSpec],
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Create masked inputs for molecules to refine.
        
        Args:
            state: Sample state with real spectrum
            smiles_to_refine: List of SMILES to refine
            pred_specs: Dict mapping SMILES -> predicted spectrum
            
        Returns:
            Tuple of (list of masked input tensors, list of source SMILES)
        """
        masked_inputs: List[torch.Tensor] = []
        source_smiles: List[str] = []
        
        for smiles in smiles_to_refine:
            pred_spec = pred_specs.get(smiles)
            if pred_spec is None:
                continue
            
            # Get hallucinated peaks
            try:
                halluc_peaks = self._masking_strategy.get_hallucinated_peaks(
                    state.real_specs,
                    pred_spec,
                    real_smi=smiles,
                )
            except Exception as e:
                print(f"Warning: Failed to get hallucinated peaks for {smiles}: {e}")
                continue
            
            # Create M masked versions
            try:
                masks = self._masking_strategy.create_masked_inputs(
                    smiles=smiles,
                    hallucinated_peaks=halluc_peaks,
                    tokenizer=self._tokenizer,
                    num_masks=self.scaling_config.masks_per_molecule,
                    mask_prob=self.scaling_config.mask_prob,
                )
                
                for mask_tensor in masks:
                    masked_inputs.append(mask_tensor)
                    source_smiles.append(smiles)
                    
            except Exception as e:
                print(f"Warning: Failed to create masked inputs for {smiles}: {e}")
                continue
        
        return masked_inputs, source_smiles
    
    def _run_single_round(
        self,
        sample_states: List[SampleState],
        round_idx: int,
        pred_specs_cache: Dict[str, Dict[str, common.CompositeMassSpec]],
        verbose: bool = False,
    ) -> Dict[str, Dict[str, common.CompositeMassSpec]]:
        """
        Run a single scaling round for all samples.
        
        The round structure is:
        - Round 0: Generate B molecules from scratch for each sample
        - Round 1+: Generate (B - M*K) from scratch + M*K refined from top K
        
        At the end of each round, we run ICEBERG on the top K molecules
        to prepare for the next round's refinement.
        
        Args:
            sample_states: List of sample states
            round_idx: Current round index (0-indexed)
            pred_specs_cache: Cached ICEBERG predictions from previous round
                             Dict mapping sample_name -> {smiles -> pred_spec}
            verbose: Whether to print detailed progress
            
        Returns:
            Updated pred_specs_cache with new ICEBERG predictions for top K molecules
        """
        B = self.scaling_config.batch_size
        K = self.scaling_config.num_unique_to_refine
        M = self.scaling_config.masks_per_molecule
        
        is_first_round = (round_idx == 0)
        
        molecules_per_sample: Dict[str, List[str]] = {}
        
        if is_first_round:
            # =====================================================================
            # FIRST ROUND: All B generations from scratch
            # =====================================================================
            print(f"\n  Round {round_idx + 1}: Generating {B} molecules from scratch...")
            
            for state in tqdm(sample_states, desc=f"Round {round_idx + 1} Generation", disable=not verbose):
                scratch_samples = self._generate_from_scratch(state, B)
                state.add_smiles_batch(scratch_samples, self.fp_bits, self.fp_radius)
                molecules_per_sample[state.name] = scratch_samples
        else:
            # =====================================================================
            # LATER ROUNDS: (B - M*K) from scratch + M*K refined
            # =====================================================================
            num_scratch = max(0, B - M * K)
            num_refined_target = M * K
            
            print(f"\n  Round {round_idx + 1}: Generating {num_scratch} from scratch + up to {num_refined_target} refined...")
            
            for state in tqdm(sample_states, desc=f"Round {round_idx + 1} Generation", disable=not verbose):
                generated = []
                
                # Part 1: Generate from scratch
                if num_scratch > 0:
                    scratch_samples = self._generate_from_scratch(state, num_scratch)
                    generated.extend(scratch_samples)
                
                # Part 2: Refine top K molecules using cached ICEBERG predictions
                pred_specs = pred_specs_cache.get(state.name, {})
                top_k_smiles = state.get_top_k_smiles(K)
                
                # Create masked inputs for top K
                masked_inputs, source_smiles = self._create_masked_inputs_for_sample(
                    state,
                    top_k_smiles,
                    pred_specs,
                )
                
                if masked_inputs:
                    # Batch and pad masked inputs
                    max_len = max(t.size(-1) for t in masked_inputs)
                    padded = []
                    for t in masked_inputs:
                        if t.size(-1) < max_len:
                            pad_size = max_len - t.size(-1)
                            t = torch.nn.functional.pad(
                                t, (0, pad_size), 
                                value=self._tokenizer.pad_token_id
                            )
                        padded.append(t)
                    
                    batched = torch.cat(padded, dim=0)
                    
                    # Generate refinements
                    # When using candidate formulas, sample a formula for each masked input
                    if state.candidate_formulas is not None and len(state.candidate_formulas) > 0:
                        # Sample formulas for each refinement
                        refinement_formulas = state.sample_formulas(batched.size(0))
                        refined = self._generate_from_masked(
                            batched,
                            refinement_formulas,  # Pass list of formulas
                            state.target_fp,
                        )
                    else:
                        refined = self._generate_from_masked(
                            batched,
                            state.target_formula,
                            state.target_fp,
                        )
                    generated.extend(refined)
                
                # Add all generated to state
                state.add_smiles_batch(generated, self.fp_bits, self.fp_radius)
                molecules_per_sample[state.name] = generated
        
        # =====================================================================
        # Run ICEBERG predictions for top K molecules (preparation for next round)
        # =====================================================================
        print(f"  Round {round_idx + 1}: Running ICEBERG predictions for top {K} molecules...")
        
        # Collect top K molecules for each sample
        top_k_per_sample: Dict[str, List[str]] = {}
        for state in sample_states:
            top_k_per_sample[state.name] = state.get_top_k_smiles(K)
        
        # Run batched ICEBERG predictions
        new_pred_specs_cache = self._run_batched_iceberg_predictions(
            sample_states,
            top_k_per_sample,
        )
        
        return new_pred_specs_cache
    
    def run_scaling_all_samples(
        self,
        test_samples: List[Dict[str, Any]],
        verbose: bool = False,
        round_callback: Optional[Callable[[int, List[Dict], Dict[str, float]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the full scaling pipeline for all test samples.
        
        The algorithm follows "Spectrum-Error Guided Refinement":
        - Round 0: Generate B molecules from scratch
        - Round 1+: Generate (B - M*K) from scratch + M*K refined from top K
        
        ICEBERG predictions are run at the END of each round to prepare
        for the next round's refinement. This enables efficient batching.
        
        Args:
            test_samples: List of test sample dicts with keys:
                - 'name': Spectrum name
                - 'real_specs': Real CompositeMassSpec
                - 'target_fp': Target fingerprint (numpy array)
                - 'target_formula': Target molecular formula
                - 'instrument': (Optional) Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
                - 'ionization': (Optional) Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+'). Defaults to '[M+H]+'
                - 'candidate_formulas': (Optional) List of (formula, probability) tuples
            verbose: Whether to print detailed progress
            round_callback: Optional callback called after each round with
                           (round_idx, sample_states, timing_info) where:
                           - sample_states is List[Dict] with 'all_unique_smiles' and 'smiles_to_similarity' keys
                           - timing_info is Dict with 'round_time_seconds', 'cumulative_time_seconds',
                             and 'time_per_sample_seconds' keys
        
        Returns:
            List of result dicts with keys:
                - 'name': Spectrum name
                - 'generated_smiles': List of all unique SMILES (ranked by similarity)
                - 'num_unique': Number of unique molecules generated
        """
        # Initialize sample states
        sample_states: List[SampleState] = []
        
        for sample in test_samples:
            state = SampleState(
                name=sample['name'],
                real_specs=sample['real_specs'],
                target_fp=sample['target_fp'],
                target_formula=sample['target_formula'],
                instrument=sample.get('instrument', 'Unknown'),
                ionization=sample.get('ionization', '[M+H]+'),
                candidate_formulas=sample.get('candidate_formulas'),
            )
            sample_states.append(state)
        
        assert len(sample_states) > 0, "No valid test samples provided"
        
        # Run scaling rounds
        num_rounds = self.scaling_config.num_rounds
        num_samples = len(sample_states)
        
        # Cache for ICEBERG predictions (carried across rounds)
        pred_specs_cache: Dict[str, Dict[str, common.CompositeMassSpec]] = {}
        
        # Timing tracking
        cumulative_time = 0.0
        round_times: List[float] = []
        
        for round_idx in range(num_rounds):
            print(f"\n{'='*60}")
            print(f"SCALING ROUND {round_idx + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            # Start timing for this round
            round_start_time = time.time()
            
            # Synchronize CUDA before starting timer for accurate GPU timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Run the round and get updated cache
            pred_specs_cache = self._run_single_round(
                sample_states, 
                round_idx, 
                pred_specs_cache,
                verbose
            )
            
            # Synchronize CUDA and record timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            round_time = time.time() - round_start_time
            cumulative_time += round_time
            round_times.append(round_time)
            
            # Report progress
            total_unique = sum(len(s.all_unique_smiles) for s in sample_states)
            avg_unique = total_unique / len(sample_states)
            print(f"  Total unique molecules: {total_unique} (avg {avg_unique:.1f} per sample)")
            print(f"  ICEBERG cache size: {len(self._pred_cache)}")
            print(f"  Round time: {round_time:.2f}s ({round_time/num_samples:.4f}s per sample)")
            print(f"  Cumulative time: {cumulative_time:.2f}s ({cumulative_time/num_samples:.4f}s per sample)")
            
            # Call round callback if provided
            if round_callback is not None:
                # Convert to dict format for callback
                state_dicts = [
                    {
                        'all_unique_smiles': s.all_unique_smiles,
                        'smiles_to_similarity': s.smiles_to_similarity,
                    }
                    for s in sample_states
                ]
                # Build timing info dict
                timing_info = {
                    'round_time_seconds': round_time,
                    'cumulative_time_seconds': cumulative_time,
                    'time_per_sample_seconds': cumulative_time / num_samples,
                    'round_times': list(round_times),  # Copy of all round times so far
                }
                round_callback(round_idx, state_dicts, timing_info)
        
        # Build final results
        results: List[Dict[str, Any]] = []
        
        for state in sample_states:
            ranked_smiles = state.get_ranked_smiles()
            
            results.append({
                'name': state.name,
                'generated_smiles': ranked_smiles,
                'num_unique': len(ranked_smiles),
            })
        
        return results
    
    def run_scaling_single_sample(
        self,
        name: str,
        real_specs: common.CompositeMassSpec,
        target_fp: np.ndarray,
        target_formula: str,
        instrument: str = 'Unknown',
        ionization: str = '[M+H]+',
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run scaling for a single test sample.

        This is a convenience method for processing one sample at a time.
        For efficiency, prefer run_scaling_all_samples when processing multiple samples.

        Args:
            name: Spectrum name
            real_specs: Real composite mass spectrum
            target_fp: Target fingerprint
            target_formula: Target molecular formula
            instrument: Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
            ionization: Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+')
            verbose: Whether to print detailed progress

        Returns:
            Result dict with generated_smiles and num_unique
        """
        test_sample = {
            'name': name,
            'real_specs': real_specs,
            'target_fp': target_fp,
            'target_formula': target_formula,
            'instrument': instrument,
            'ionization': ionization,
        }
        
        results = self.run_scaling_all_samples([test_sample], verbose)
        
        assert len(results) == 1, "Expected single result"
        return results[0]
