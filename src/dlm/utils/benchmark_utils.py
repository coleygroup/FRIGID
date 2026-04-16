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

"""
Shared utilities for Spec2Mol and FP2Mol benchmarking.

This module provides common functions used across benchmark scripts:
- Formula normalization and matching
- Fingerprint computation
- Token model loading and prediction
- Molecule generation with formula filtering
- Evaluation metrics
"""

import os
import json
import re
import random
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from ngboost import NGBRegressor

# Element pattern for Hill notation
ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')


# =============================================================================
# Token Model Functions
# =============================================================================

def is_ngboost_model(model) -> bool:
    """Check if the model is an NGBoost model."""
    return isinstance(model, NGBRegressor)


def load_token_model(model_path: str) -> Tuple[Any, Optional[List[str]], bool]:
    """
    Load the token count prediction model and its features.
    
    Args:
        model_path: Path to the .joblib model file
        
    Returns:
        Tuple of (model, features, is_ngboost)
    """
    if not model_path or not os.path.exists(model_path):
        return None, None, False
        
    print(f"Loading token model from {model_path}")
    model = joblib.load(model_path)
    
    # Check if it's an NGBoost model
    is_ngb = is_ngboost_model(model)
    if is_ngb:
        print("Detected NGBoost model - will use sigma-based sampling")
    else:
        print("Detected sklearn model - will use fixed range sampling")
    
    # Try to load features from json
    features_path = model_path.replace('.joblib', '_features.json')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
        print(f"Loaded {len(features)} features from {features_path}")
    else:
        print(f"Warning: Features file {features_path} not found. Assuming model has feature_names_in_ or using default.")
        features = getattr(model, 'feature_names_in_', None)
        
    return model, features, is_ngb


def predict_token_count(
    model, 
    features: Optional[List[str]], 
    formula: str, 
    is_ngboost: bool = False, 
    sigma_lambda: float = 3.0
) -> Tuple[Optional[int], Optional[float]]:
    """
    Predict token count for a formula.
    
    Args:
        model: The prediction model (sklearn or NGBoost)
        features: List of feature names
        formula: Molecular formula string
        is_ngboost: Whether the model is an NGBoost model
        sigma_lambda: Variance multiplier for NGBoost normal distribution sampling.
                      The sampling variance is sigma * sigma_lambda.
        
    Returns:
        For sklearn: (mean, None) tuple
        For NGBoost: (mean, sigma) tuple where sigma is the model's predicted std dev
    """
    if model is None:
        return None, None
        
    # Parse formula
    atoms: Dict[str, int] = {}
    pattern = r"([A-Z][a-z]?)(\d*)"
    try:
        for element, count in re.findall(pattern, formula):
            count = int(count) if count else 1
            atoms[element] = atoms.get(element, 0) + count
    except Exception:
        return None, None
        
    # Create feature vector
    if features:
        vector = [atoms.get(f, 0) for f in features]
    else:
        # Fallback if no features known (risky)
        return None, None
    
    if is_ngboost:
        # NGBoost returns a distribution
        dist = model.pred_dist([vector])
        mean = dist.loc[0]
        sigma = dist.scale[0]
        return int(round(mean)), sigma
    else:
        # sklearn model returns point estimate
        pred = model.predict([vector])[0]
        return int(round(pred)), None


# =============================================================================
# Formula Functions
# =============================================================================

def normalize_formula(formula: Optional[str]) -> Optional[str]:
    """
    Normalize molecular formula into Hill notation for robust string comparison.
    
    Hill notation: C first, H second (if C present), then alphabetical.
    
    Args:
        formula: Molecular formula string
        
    Returns:
        Normalized formula string or None if invalid
    """
    if not formula:
        return None

    matches = ELEMENT_PATTERN.findall(str(formula))
    if not matches:
        return formula

    counts: Dict[str, int] = {}
    for element, count_str in matches:
        if element:
            count = int(count_str) if count_str else 1
            counts[element] = counts.get(element, 0) + count

    # Hill notation ordering
    ordered_elements: List[str] = []
    if 'C' in counts:
        ordered_elements.append('C')
        if 'H' in counts:
            ordered_elements.append('H')

    ordered_elements.extend(
        sorted(elem for elem in counts.keys() if elem not in ordered_elements)
    )

    normalized = ''.join(
        f"{elem}{counts[elem] if counts[elem] != 1 else ''}"
        for elem in ordered_elements
    )
    return normalized


def get_molecular_formula(smiles: str) -> Optional[str]:
    """Extract normalized molecular formula from SMILES."""
    try:
        if pd.isna(smiles) or not smiles:
            return None
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return normalize_formula(formula)
    except Exception:
        return None


def compute_morgan_fingerprint(
    smiles: str, 
    n_bits: int = 4096, 
    radius: int = 2
) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprint as numpy array.
    
    Args:
        smiles: SMILES string
        n_bits: Number of fingerprint bits
        radius: Morgan fingerprint radius
        
    Returns:
        Fingerprint as numpy array or None if failed
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprint arrays."""
    if fp1 is None or fp2 is None:
        return 0.0
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))
    return float(intersection / union) if union > 0 else 0.0


def get_inchikey_first_block(inchi_key: Optional[str]) -> Optional[str]:
    """Extract first block of InChI key (connectivity layer, no stereo)."""
    if not inchi_key:
        return None
    return inchi_key.strip().upper().split('-')[0]


def binarize_fingerprint(fp_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize fingerprint probabilities."""
    return (fp_probs >= threshold).astype(np.float32)


def generate_with_formula_filter(
    sampler,
    fingerprint_array: np.ndarray,
    target_formula: str,
    target_smiles: str,
    n_required: int,
    max_attempts: int,
    batch_size: int,
    softmax_temp: float,
    randomness: float,
    fp_radius: int = 2,
    fp_bits: int = 4096,
    token_model=None,
    token_features=None,
    is_ngboost: bool = False,
    sigma_lambda: float = 3.0,
) -> Tuple[List[str], List[float], int, int, int, Counter, Optional[str], float]:
    """
    Generate molecules with formula filtering.
    
    Generates molecules until either n_required UNIQUE formula matches are found
    or max_attempts generations have been made. Uniqueness is determined by
    InChI key first block (connectivity layer, ignoring stereochemistry).
    
    Args:
        sampler: DLM Sampler instance
        fingerprint_array: Target fingerprint as numpy array
        target_formula: Target molecular formula (normalized)
        target_smiles: Target SMILES (for length prediction)
        n_required: Required number of UNIQUE formula matches (by InChI key first block)
        max_attempts: Maximum generation attempts
        batch_size: Batch size for generation
        softmax_temp: Softmax temperature
        randomness: Randomness factor
        fp_radius: Fingerprint radius
        fp_bits: Fingerprint bits
        token_model: Optional token count prediction model
        token_features: Optional list of features for token model
        is_ngboost: Whether the token model is an NGBoost model
        sigma_lambda: Variance multiplier for NGBoost normal distribution sampling.
                      Lengths are sampled from N(mean, variance=sigma*sigma_lambda).
        
    Returns:
        Tuple of:
        - matched_smiles: List of UNIQUE formula-matched SMILES (padded with non-matched if needed)
        - matched_similarities: List of Tanimoto similarities for matched SMILES
        - total_generated: Total molecules generated
        - total_valid: Valid molecules (parseable SMILES)
        - total_formula_matched: Formula matched count (including duplicates)
        - global_counter: Counter of all valid SMILES
        - last_valid_smiles: Last valid SMILES (fallback)
        - generation_time: Total time spent in generation functions (seconds)
    """
    matched_smiles = []
    matched_similarities = []
    matched_inchi_keys: set = set()  # Track unique molecules by InChI key first block
    # Track non-formula-matched samples for padding
    non_matched_smiles = []
    non_matched_similarities = []
    non_matched_inchi_keys: set = set()  # Track unique non-matched molecules
    total_generated = 0
    total_valid = 0
    total_formula_matched = 0
    global_smiles_counter: Counter = Counter()
    last_valid_smiles: Optional[str] = None
    generation_time: float = 0.0
    
    # Convert fingerprint_array to RDKit DataStructs for Tanimoto calculation
    # This is the PREDICTED fingerprint from the encoder - NOT the ground truth molecule
    # Using the predicted fingerprint avoids data leakage (we don't know the true molecule)
    from rdkit import DataStructs as DS
    pred_fp_bitvect = DS.ExplicitBitVect(fp_bits)
    for i in range(fp_bits):
        if fingerprint_array[i] > 0.5:  # Threshold for binary fingerprint
            pred_fp_bitvect.SetBit(i)
    
    # Predict token count if model is available
    predicted_len = None
    predicted_sigma = None
    if token_model is not None and token_features is not None:
        predicted_len, predicted_sigma = predict_token_count(
            token_model, token_features, target_formula, is_ngboost, sigma_lambda
        )
    
    stop_generation = False
    while total_generated < max_attempts and not stop_generation:
        current_batch = min(batch_size, max_attempts - total_generated)
        
        if current_batch <= 0:
            break
            
        try:
            # Determine target lengths if token model is available
            target_lengths = None
            if predicted_len is not None:
                if is_ngboost and predicted_sigma is not None:
                    # NGBoost: sample from normal distribution N(mean, variance=sigma*lambda)
                    std_dev = np.sqrt(predicted_sigma * sigma_lambda)
                    target_lengths = [
                        max(1, int(round(np.random.normal(predicted_len, std_dev))))
                        for _ in range(current_batch)
                    ]
                else:
                    # sklearn: sample from fixed range (mean-3, mean+3)
                    low = max(1, predicted_len - 3)
                    high = predicted_len + 3
                    target_lengths = [
                        random.randint(low, high)
                        for _ in range(current_batch)
                    ]
            
            # Use unified generation with both formula and fingerprint
            if hasattr(sampler, 'unified_conditioned_generation'):
                gen_start = time.time()
                samples = sampler.unified_conditioned_generation(
                    formula=target_formula,
                    fingerprint=fingerprint_array,
                    num_samples=current_batch,
                    softmax_temp=softmax_temp,
                    randomness=randomness,
                    target_lengths=target_lengths,
                    min_add_len=2
                )
                generation_time += time.time() - gen_start
            elif sampler.model.use_fingerprint_conditioning:
                gen_start = time.time()
                samples = sampler.fingerprint_conditioned_generation(
                    fingerprint=fingerprint_array,
                    num_samples=current_batch,
                    softmax_temp=softmax_temp,
                    randomness=randomness,
                )
                generation_time += time.time() - gen_start
            else:
                gen_start = time.time()
                samples = sampler.de_novo_generation(
                    num_samples=current_batch,
                    softmax_temp=softmax_temp,
                    randomness=randomness
                )
                generation_time += time.time() - gen_start
        except Exception as exc:
            print(f"\nWarning: Batch generation failed: {exc}")
            samples = [None] * current_batch
        
        total_generated += len(samples)
        
        for smiles in samples:
            if not smiles or (isinstance(smiles, float) and pd.isna(smiles)):
                continue
            
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                
                total_valid += 1
                canonical = Chem.MolToSmiles(mol)
                last_valid_smiles = canonical
                global_smiles_counter[canonical] += 1

                # Compute Tanimoto similarity to PREDICTED fingerprint (not ground truth)
                # This avoids data leakage - we only use information available at inference time
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
                similarity = DataStructs.TanimotoSimilarity(pred_fp_bitvect, gen_fp)
                
                # Get InChI key first block for uniqueness check (connectivity, no stereochemistry)
                try:
                    inchi_key = Chem.MolToInchiKey(mol)
                    inchi_key_first_block = get_inchikey_first_block(inchi_key) if inchi_key else None
                except Exception:
                    inchi_key_first_block = None

                # Check formula match
                gen_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(mol))
                if gen_formula == target_formula:
                    total_formula_matched += 1
                    
                    # Only add if this is a unique molecule (by InChI key first block)
                    if inchi_key_first_block is not None and inchi_key_first_block not in matched_inchi_keys:
                        matched_inchi_keys.add(inchi_key_first_block)
                        matched_smiles.append(canonical)
                        matched_similarities.append(float(similarity))
                        
                        # Stop when we have enough UNIQUE formula-matched molecules
                        if len(matched_smiles) >= n_required:
                            stop_generation = True
                            break
                    elif inchi_key_first_block is None:
                        # If we can't compute InChI key, add the molecule anyway
                        matched_smiles.append(canonical)
                        matched_similarities.append(float(similarity))
                        
                        if len(matched_smiles) >= n_required:
                            stop_generation = True
                            break
                else:
                    # Track non-matched samples for potential padding (also unique)
                    if inchi_key_first_block is not None:
                        if inchi_key_first_block not in non_matched_inchi_keys:
                            non_matched_inchi_keys.add(inchi_key_first_block)
                            non_matched_smiles.append(canonical)
                            non_matched_similarities.append(float(similarity))
                    else:
                        # If we can't compute InChI key, add the molecule anyway
                        non_matched_smiles.append(canonical)
                        non_matched_similarities.append(float(similarity))
                        
            except Exception:
                continue
        
        if stop_generation:
            break

    # reorder the matched lists by similarity descending
    if matched_smiles:
        sim_smiles_pairs = sorted(
            zip(matched_similarities, matched_smiles),
            key=lambda x: x[0],
            reverse=True
        )
        matched_similarities, matched_smiles = zip(*sim_smiles_pairs)
        matched_similarities = list(matched_similarities)
        matched_smiles = list(matched_smiles)
    else:
        matched_similarities = []
        matched_smiles = []
    
    # Pad with non-formula-matched samples if we didn't reach n_required
    if len(matched_smiles) < n_required and non_matched_smiles:
        # Sort non-matched by similarity descending
        non_matched_pairs = sorted(
            zip(non_matched_similarities, non_matched_smiles),
            key=lambda x: x[0],
            reverse=True
        )
        
        # Add top non-matched samples to reach n_required
        num_to_pad = n_required - len(matched_smiles)
        for sim, smi in non_matched_pairs[:num_to_pad]:
            matched_smiles.append(smi)
            matched_similarities.append(sim)
    
    return (
        matched_smiles,
        matched_similarities,
        total_generated,
        total_valid,
        total_formula_matched,
        global_smiles_counter,
        last_valid_smiles,
        generation_time
    )


def build_prediction_entry(
    smiles: str,
    similarity: float,
    frequency: int,
    source: str,
    fp_bits: int = 4096,
    fp_radius: int = 2,
) -> Optional[Dict[str, Any]]:
    """Build a prediction entry dictionary with computed fingerprint."""
    if not smiles:
        return None
    mol_fp = compute_morgan_fingerprint(smiles, fp_bits, fp_radius)
    if mol_fp is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol) if mol else smiles
        inchi_key = get_inchikey_first_block(Chem.MolToInchiKey(mol)) if mol else None
    except Exception:
        canonical, inchi_key = smiles, None
    return {
        'smiles': canonical,
        'inchi_key': inchi_key,
        'similarity': float(similarity) if similarity else 0.0,
        'fingerprint': mol_fp,
        'frequency': frequency,
        'source': source
    }


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    target_smiles: str,
    target_inchi_key: str,
    target_fp: np.ndarray,
    fp_bits: int = 4096,
    fp_radius: int = 2,
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth.
    
    Computes:
    - Exact match (InChI key) at Top-1 and Top-10
    - Tanimoto similarity at Top-1 and Top-10
    
    Args:
        predictions: List of prediction dictionaries (sorted by frequency/similarity)
        target_smiles: Ground truth SMILES
        target_inchi_key: Ground truth InChI key (first block)
        target_fp: Ground truth fingerprint array
        fp_bits: Fingerprint bits
        fp_radius: Fingerprint radius
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'target_smiles': target_smiles,
        'target_inchi_key': target_inchi_key,
        'num_predictions': len(predictions),
        'exact_match_top1': 0.0,
        'exact_match_top10': 0.0,
        'tanimoto_top1': 0.0,
        'tanimoto_top10': 0.0,
        'tanimoto_mean': 0.0,
    }
    
    if not predictions:
        return results

    # Compute Tanimoto similarities to ground truth
    gt_similarities = []
    for pred in predictions:
        mol_fp = pred.get('fingerprint')
        if mol_fp is None:
            mol_fp = compute_morgan_fingerprint(pred['smiles'], fp_bits, fp_radius)
        sim = compute_tanimoto_similarity(target_fp, mol_fp)
        gt_similarities.append(sim)

    # Exact match (InChI key)
    pred_keys = [p.get('inchi_key') for p in predictions]
    results['exact_match_top1'] = 1.0 if (pred_keys and pred_keys[0] == target_inchi_key) else 0.0
    results['exact_match_top10'] = 1.0 if target_inchi_key in pred_keys[:10] else 0.0

    # Tanimoto similarity
    results['tanimoto_top1'] = gt_similarities[0] if gt_similarities else 0.0
    results['tanimoto_top10'] = max(gt_similarities[:10]) if gt_similarities else 0.0
    results['tanimoto_mean'] = float(np.mean(gt_similarities)) if gt_similarities else 0.0

    # Top predictions detail
    results['top_predictions'] = [
        {
            'smiles': p['smiles'],
            'inchi_key': p.get('inchi_key'),
            'similarity_to_gt': gt_similarities[i] if i < len(gt_similarities) else 0.0,
            'frequency': p.get('frequency', 0),
            'source': p.get('source'),
        }
        for i, p in enumerate(predictions[:5])
    ]

    return results


def compute_aggregate_statistics(
    results: List[Dict[str, Any]],
    elapsed_time: float,
) -> Dict[str, Any]:
    """
    Compute aggregate statistics from per-spectrum results.
    
    Args:
        results: List of per-spectrum result dictionaries
        elapsed_time: Total elapsed time in seconds
        
    Returns:
        Dictionary with aggregate statistics
    """
    n = len(results)
    if n == 0:
        return {'error': 'No results', 'total_spectra': 0}

    agg = {
        'total_spectra': n,
        'elapsed_time_seconds': float(elapsed_time),
        'spectra_per_second': float(n / elapsed_time) if elapsed_time > 0 else 0.0,
        
        # Core metrics
        'exact_match_top1': float(np.mean([r.get('exact_match_top1', 0.0) for r in results])),
        'exact_match_top10': float(np.mean([r.get('exact_match_top10', 0.0) for r in results])),
        'tanimoto_top1_mean': float(np.mean([r.get('tanimoto_top1', 0.0) for r in results])),
        'tanimoto_top10_mean': float(np.mean([r.get('tanimoto_top10', 0.0) for r in results])),
        
        # MIST encoder quality
        'mist_tanimoto_mean': float(np.mean([r.get('mist_tanimoto', 0.0) for r in results])),
        
        # Formula matching stats (use total_formula_matched for actual counts,
        # NOT formula_matches_collected which includes padding with non-matched molecules)
        'avg_formula_matches': float(np.mean([r.get('total_formula_matched', 0) for r in results])),
        'avg_predictions_collected': float(np.mean([r.get('formula_matches_collected', 0) for r in results])),
        'avg_total_generated': float(np.mean([r.get('total_generated', 0) for r in results])),
        'formula_match_success_rate': float(np.mean([
            1.0 if r.get('total_formula_matched', 0) > 0 else 0.0 for r in results
        ])),
        
        # Average attempts to first match (for successful cases)
        'avg_attempts_to_match': 0.0,
        'never_matched_rate': 0.0,
    }
    
    # Compute attempts statistics using actual formula matches (not padded count)
    successful = [r for r in results if r.get('total_formula_matched', 0) > 0]
    failed = [r for r in results if r.get('total_formula_matched', 0) == 0]
    
    if successful:
        avg_attempts = float(np.mean([
            r.get('total_generated', 0) / max(r.get('total_formula_matched', 1), 1)
            for r in successful
        ]))
        agg['avg_attempts_to_match'] = avg_attempts
    
    agg['never_matched_rate'] = len(failed) / n if n > 0 else 0.0
    
    return agg
