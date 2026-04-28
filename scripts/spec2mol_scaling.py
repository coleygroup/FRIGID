#!/usr/bin/env python
"""
Spec2Mol benchmark with ICEBERG inference-time scaling.

This script extends the standard Spec2Mol benchmark with ICEBERG-guided
refinement for improved structure elucidation performance.

The algorithm ("Spectrum-Error Guided Refinement"):
1. Generate diverse candidate molecules (B samples per round)
2. Select top K unique candidates by fingerprint similarity  
3. Simulate mass spectra with ICEBERG
4. Identify hallucinated peaks and map to atoms/tokens
5. Create M masked versions per molecule and refine with DLM
6. Repeat for R rounds
7. Rank all unique candidates by fingerprint similarity

Usage:
    python scripts/spec2mol_scaling.py \
        --config configs/spec2mol_benchmark.yaml \
        --mist-checkpoint path/to/mist.ckpt \
        --dlm-checkpoint path/to/dlm.ckpt \
        --iceberg-gen-ckpt path/to/model1.ckpt \
        --iceberg-inten-ckpt path/to/model2.ckpt \
        --num-rounds 3 \
        --batch-size 128
"""

import os
import sys
import argparse
import warnings
import time
import random
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union, Set

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Ensure src/ is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add the repo-local ms-pred submodule to the import path when not installed.
ms_pred_src_path = os.path.join(project_root, 'ms-pred', 'src')
if os.path.exists(ms_pred_src_path) and ms_pred_src_path not in sys.path:
    sys.path.insert(0, ms_pred_src_path)

from dlm.sampler import Sampler
from dlm.iceberg_sampler import IcebergSampler, IcebergConfig, ScalingConfig, normalize_instrument_type
from dlm.utils.benchmark_utils import (
    normalize_formula,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    get_inchikey_first_block,
    binarize_fingerprint,
    build_prediction_entry,
    evaluate_predictions,
    compute_aggregate_statistics,
    load_token_model,
    get_molecular_formula,
)
from mist.models import SpectraEncoderGrowing
from mist.data import (
    get_paired_spectra,
    get_paired_featurizer,
    SpectraMolDataset,
    PresetSpectraSplitter,
)
from mist.data.datasets import get_paired_loader
from ms_pred.dag_pred.iceberg_elucidation import load_real_spec

import ms_pred.common as common

# Suppress RDKit warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spec2Mol benchmark with ICEBERG inference-time scaling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Standard benchmark args
    parser.add_argument('--config', type=str, default='configs/spec2mol_benchmark.yaml')
    parser.add_argument('--mist-checkpoint', type=str, help='MIST encoder checkpoint')
    parser.add_argument('--dlm-checkpoint', type=str, help='DLM decoder checkpoint')
    parser.add_argument('--data-dir', type=str, help='Spec data directory')
    parser.add_argument('--fp-threshold', type=float, help='FP binarization threshold')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--split', type=str, choices=['val', 'test'])
    parser.add_argument('--max-spectra', type=int, default=None)
    parser.add_argument('--softmax-temp', type=float)
    parser.add_argument('--randomness', type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--use-shared-cross-attention',
        action='store_true',
        help='Use shared cross-attention mode'
    )
    parser.add_argument(
        '--token-model',
        type=str,
        default=None,
        help='Path to token count prediction model (.joblib)'
    )
    parser.add_argument(
        '--sigma-lambda',
        type=float,
        default=3.0,
        help='Lambda multiplier for NGBoost sigma-based sampling range'
    )
    
    # ICEBERG args
    parser.add_argument(
        '--iceberg-gen-ckpt',
        type=str,
        default='',
        help='Path to ICEBERG generator checkpoint (model 1)'
    )
    parser.add_argument(
        '--iceberg-inten-ckpt',
        type=str,
        default='',
        help='Path to ICEBERG intensity checkpoint (model 2)'
    )
    parser.add_argument(
        '--iceberg-python-path',
        type=str,
        default='python',
        help='Python path for ICEBERG subprocess'
    )
    parser.add_argument(
        '--iceberg-batch-size',
        type=int,
        default=8,
        help='Batch size for ICEBERG prediction'
    )
    parser.add_argument(
        '--iceberg-gpu',
        type=int,
        nargs='+',
        default=[0],
        help='GPU(s) for ICEBERG'
    )
    parser.add_argument(
        '--iceberg-results-dir',
        type=str,
        default='/tmp/iceberg_scaling',
        help='Directory for intermediate ICEBERG results'
    )
    
    # Scaling args
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Total samples per scaling round (B)'
    )
    parser.add_argument(
        '--num-unique-to-refine',
        type=int,
        default=16,
        help='Number of unique molecules to refine per round (K)'
    )
    parser.add_argument(
        '--masks-per-molecule',
        type=int,
        default=4,
        help='Number of masked versions per molecule (M)'
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=3,
        help='Number of refinement rounds (R)'
    )
    parser.add_argument(
        '--mask-prob',
        type=float,
        default=0.5,
        help='Probability of masking each bad token'
    )
    parser.add_argument(
        '--top-k-halluc-peaks',
        type=int,
        default=3,
        help='Number of hallucinated peaks to consider'
    )
    parser.add_argument(
        '--halluc-inten-threshold',
        type=float,
        default=0.05,
        help='Intensity threshold for hallucination detection'
    )
    parser.add_argument(
        '--collision-energies',
        type=int,
        nargs='+',
        default=[10, 20, 30, 40, 50],
        help='Collision energies for ICEBERG simulation'
    )
    parser.add_argument(
        '--nce',
        action='store_true',
        help='Treat collision energies as normalized collision energies'
    )
    parser.add_argument(
        '--ppm',
        type=int,
        default=20,
        help='Parts-per-million tolerance'
    )
    parser.add_argument(
        '--num-bins',
        type=int,
        default=15000,
        help='Number of bins for spectrum binning'
    )
    parser.add_argument(
        '--upper-limit',
        type=int,
        default=1500,
        help='Upper m/z limit for spectrum binning'
    )
    parser.add_argument(
        '--masking-strategy',
        type=str,
        choices=['simple', 'intensity_weighted', 'score_based'],
        default='simple',
        help=(
            'Masking strategy for identifying hallucinated atoms. '
            '"simple": Uses peak distance to experimental peaks. '
            '"intensity_weighted": Uses composite scoring with intensity weighting, '
            'fragment-aware atom scoring, and neighbor extension. '
            '"score_based": Scores atoms positively for matched peaks and negatively '
            'for unmatched peaks, using ICEBERG fragments directly.'
        )
    )
    parser.add_argument(
        '--max-output-preds',
        type=int,
        default=100,
        help='Maximum predictions to output per spectrum'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--proc-idx',
        type=int,
        default=0,
        help='Process index for parallel execution'
    )
    parser.add_argument(
        '--n-proc',
        type=int,
        default=1,
        help='Total number of processes for parallel execution'
    )
    parser.add_argument(
        '--buddy-formula-path',
        type=str,
        default=None,
        help='Path to BUDDY formula predictions TSV file. If provided, uses predicted '
             'formulas instead of ground truth for generation.'
    )
    parser.add_argument(
        '--incl-unknown-instrument',
        action='store_true',
        help='Whether to include "Unknown" instrument type in ICEBERG predictions'
    )
    
    return parser.parse_args()


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        'mist_encoder': {
            'checkpoint': 'checkpoints/mist_encoder.ckpt',
            'hidden_size': 512,
            'peak_attn_layers': 2,
            'num_heads': 8,
            'spectra_dropout': 0.1,
            'output_size': 4096,
            'form_embedder': 'pos-cos',
            'set_pooling': 'cls',
            'refine_layers': 4,
            'pairwise_featurization': True,
            'embed_instrument': False,
            'inten_transform': 'float',
            'magma_modulo': 2048,
            'inten_prob': 0.1,
            'remove_prob': 0.5,
            'cls_type': 'ms1',
            'max_peaks': None,
            'spec_features': 'peakformula',
            'mol_features': 'fingerprint',
            'top_layers': 1,
        },
        'dlm': {
            'checkpoint': 'checkpoints/dlm.ckpt',
        },
        'data': {
            'datadir': 'data/msg',
            'labels_file': 'data/msg/labels.tsv',
            'split_file': 'data/msg/split.tsv',
            'spec_folder': 'data/msg/spec_files',
            'subform_folder': 'data/msg/subformulae/default_subformulae',
        },
        'fingerprint': {
            'bits': 4096,
            'radius': 2,
            'threshold': 0.172,
        },
        'generation': {
            'softmax_temp': 1.0,
            'randomness': 0.1,
            'top_k': 100,
        },
        'evaluation': {
            'split': 'test',
            'max_spectra': None,
        },
        'output': {
            'results_dir': 'results/spec2mol_scaling',
        },
    }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    defaults = get_default_config()
    for key in defaults:
        if key not in config:
            config[key] = defaults[key]
        elif isinstance(defaults[key], dict):
            for subkey in defaults[key]:
                if subkey not in config[key]:
                    config[key][subkey] = defaults[key][subkey]
    return config


def merge_config_with_args(config: dict, args) -> dict:
    """Override config with command-line arguments."""
    if args.mist_checkpoint:
        config['mist_encoder']['checkpoint'] = args.mist_checkpoint
    if args.dlm_checkpoint:
        config['dlm']['checkpoint'] = args.dlm_checkpoint
    if args.data_dir:
        config['data']['datadir'] = args.data_dir
        config['data']['labels_file'] = os.path.join(args.data_dir, 'labels.tsv')
        config['data']['split_file'] = os.path.join(args.data_dir, 'split.tsv')
        config['data']['spec_folder'] = os.path.join(args.data_dir, 'spec_files')
        config['data']['subform_folder'] = os.path.join(args.data_dir, 'subformulae/default_subformulae')
    if args.fp_threshold is not None:
        config['fingerprint']['threshold'] = args.fp_threshold
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir
    if args.split:
        config['evaluation']['split'] = args.split
    if args.softmax_temp is not None:
        config['generation']['softmax_temp'] = args.softmax_temp
    if args.randomness is not None:
        config['generation']['randomness'] = args.randomness
    return config


def load_mist_encoder(config: dict, device: torch.device) -> torch.nn.Module:
    """Load MIST encoder from checkpoint."""
    checkpoint_path = config['checkpoint']
    print(f"\nLoading MIST encoder from: {checkpoint_path}")

    encoder = SpectraEncoderGrowing(
        form_embedder=config.get('form_embedder', 'pos-cos'),
        output_size=config.get('output_size', 4096),
        hidden_size=config.get('hidden_size', 512),
        spectra_dropout=config.get('spectra_dropout', 0.1),
        peak_attn_layers=config.get('peak_attn_layers', 2),
        num_heads=config.get('num_heads', 8),
        set_pooling=config.get('set_pooling', 'cls'),
        refine_layers=config.get('refine_layers', 4),
        pairwise_featurization=config.get('pairwise_featurization', True),
        embed_instrument=config.get('embed_instrument', False),
        inten_transform=config.get('inten_transform', 'float'),
        magma_modulo=config.get('magma_modulo', 2048),
        inten_prob=config.get('inten_prob', 0.1),
        remove_prob=config.get('remove_prob', 0.5),
        cls_type=config.get('cls_type', 'ms1'),
        spec_features=config.get('spec_features', 'peakformula'),
        mol_features=config.get('mol_features', 'fingerprint'),
        top_layers=config.get('top_layers', 1),
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            if not state_dict:
                state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        encoder.load_state_dict(state_dict, strict=False)
        print('✓ Loaded MIST encoder weights')
    else:
        print(f"WARNING: MIST checkpoint not found: {checkpoint_path}")

    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def load_dlm_sampler(config: dict, use_shared_cross_attention: bool = False) -> Sampler:
    """Load DLM decoder as Sampler."""
    checkpoint_path = config['checkpoint']
    print(f"\nLoading DLM decoder from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DLM checkpoint not found: {checkpoint_path}")
    
    config_overrides = None
    if use_shared_cross_attention:
        print("Using SHARED cross-attention mode")
        config_overrides = {'use_shared_cross_attention': True}
    
    sampler = Sampler(checkpoint_path, config_overrides=config_overrides)
    
    if sampler.model.use_fingerprint_conditioning:
        print('✓ Fingerprint conditioning: ENABLED')
    else:
        print('WARNING: Fingerprint conditioning is NOT enabled!')
    
    if sampler.model.use_formula_conditioning:
        print('✓ Formula conditioning: ENABLED')
    
    return sampler


def load_spec_data(
    config: dict,
    encoder_config: dict,
    split: str = 'test',
    shuffle=False,
    n_proc: int = 1,
    proc_idx: int = 0,
) -> Tuple[SpectraMolDataset, List, Dict[str, str], Dict[str, str]]:
    """
    Load Spec dataset.

    Returns:
        Tuple of (dataset, split_data, spec_to_instrument, spec_to_ionization)
        - dataset: SpectraMolDataset
        - split_data: List of (spectrum, molecule) pairs
        - spec_to_instrument: Dict mapping spectrum name -> instrument type
        - spec_to_ionization: Dict mapping spectrum name -> ionization type
    """
    print(f"\nLoading {split} split...")
    
    datadir = config['datadir']
    labels_file = config.get('labels_file', os.path.join(datadir, 'labels.tsv'))
    split_file = config.get('split_file', os.path.join(datadir, 'split.tsv'))
    spec_folder = config.get('spec_folder', os.path.join(datadir, 'spec_files'))
    subform_folder = config.get('subform_folder', os.path.join(datadir, 'default_subformulae'))

    for path, name in [
        (labels_file, 'labels'), 
        (split_file, 'split'),
        (spec_folder, 'spec_folder'), 
        (subform_folder, 'subform_folder')
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required {name} not found: {path}")

    # Load labels to get instrument information
    labels_df = pd.read_csv(labels_file, sep='\t')
    
    # Build spec -> instrument and spec -> ionization mappings
    spec_to_instrument: Dict[str, str] = {}
    spec_to_ionization: Dict[str, str] = {}
    if 'instrument' in labels_df.columns:
        has_ionization_col = 'ionization' in labels_df.columns
        if not has_ionization_col:
            print("Warning: 'ionization' column not found in labels.tsv, defaulting to '[M+H]+' for all")

        for _, row in labels_df.iterrows():
            spec_name = row['spec']
            raw_instrument = row.get('instrument', None)
            # Normalize instrument type (handles variants like "QTOF (LCMS)")
            instrument = normalize_instrument_type(raw_instrument)
            spec_to_instrument[spec_name] = instrument

            # Read ionization type
            if has_ionization_col:
                raw_ionization = row.get('ionization', None)
                if raw_ionization and isinstance(raw_ionization, str) and raw_ionization.strip():
                    spec_to_ionization[spec_name] = raw_ionization.strip()
                else:
                    spec_to_ionization[spec_name] = '[M+H]+'
            else:
                spec_to_ionization[spec_name] = '[M+H]+'

        print(f"Loaded instrument information for {len(spec_to_instrument)} spectra")

        # Print instrument distribution
        from collections import Counter
        instrument_counts = Counter(spec_to_instrument.values())
        print(f"  Instrument distribution: {dict(instrument_counts)}")

        ionization_counts = Counter(spec_to_ionization.values())
        print(f"  Ionization distribution: {dict(ionization_counts)}")
    else:
        raise ValueError("Warning: 'instrument' column not found in labels.tsv, using 'Unknown' for all")

    spectra_list, mol_list = get_paired_spectra(
        labels_file=labels_file,
        spec_folder=spec_folder,
        prog_bars=True,
        split_file=split_file,
        split_val=split,
    )
    full_dataset = list(zip(spectra_list, mol_list))
    print(f"Loaded {len(full_dataset)} total spectra-molecule pairs")

    splitter = PresetSpectraSplitter(split_file=split_file)
    _, (train_data, val_data, test_data) = splitter.get_splits(full_dataset)

    split_map = {'train': train_data, 'val': val_data, 'test': test_data}
    split_data = split_map[split]
    print(f"Using {len(split_data)} samples from {split} split")

    if shuffle:
        print("Shuffling split data")
        random.Random(42).shuffle(split_data)

    split_data = split_data[proc_idx::n_proc]


    featurizer = get_paired_featurizer(
        spec_features='peakformula',
        mol_features='fingerprint',
        subform_folder=subform_folder,
        fp_names=['morgan4096'],
        magma_modulo=encoder_config.get('magma_modulo', 2048),
        cls_type=encoder_config.get('cls_type', 'ms1'),
        inten_transform=encoder_config.get('inten_transform', 'float'),
        inten_prob=encoder_config.get('inten_prob', 0.1),
        remove_prob=encoder_config.get('remove_prob', 0.5),
        max_peaks=encoder_config.get('max_peaks', None),
    )

    return SpectraMolDataset(split_data, featurizer), split_data, spec_to_instrument, spec_to_ionization


def load_real_spectrum(spec_path: str) -> Optional[common.CompositeMassSpec]:
    """Load real spectrum from .ms file."""
    if not os.path.exists(spec_path):
        return None
    try:
        meta, comp_spec = common.parse_spectra(spec_path)
        return comp_spec
    except Exception as e:
        print(f"Warning: Failed to load spectrum from {spec_path}: {e}")
        return None


def load_buddy_formulas(buddy_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Load BUDDY formula predictions from TSV file.
    
    Args:
        buddy_path: Path to BUDDY predictions TSV file
        
    Returns:
        Dict mapping spectrum identifier -> list of (formula, probability) tuples
        Probabilities are [0.8, 0.05, 0.05, 0.05, 0.05] for valid formulas,
        with probability mass redistributed to top-1 if some are null/invalid.
    """
    if not os.path.exists(buddy_path):
        raise FileNotFoundError(f"BUDDY formula file not found: {buddy_path}")
    
    print(f"\nLoading BUDDY formula predictions from: {buddy_path}")
    
    df = pd.read_csv(buddy_path)
    
    # Base probabilities for top-5 formulas
    base_probs = [0.8, 0.05, 0.05, 0.05, 0.05]
    
    buddy_formulas: Dict[str, List[Tuple[str, float]]] = {}
    
    for _, row in df.iterrows():
        identifier = row['identifier']
        
        # Collect valid formulas
        valid_formulas = []
        for i in range(1, 6):
            col_name = f'formula_rank_{i}'
            if col_name in row and pd.notna(row[col_name]):
                formula = str(row[col_name]).strip()
                if formula and formula.lower() not in ('nan', 'none', ''):
                    # Normalize the formula
                    normalized = normalize_formula(formula)
                    if normalized:
                        valid_formulas.append((normalized, i - 1))  # (formula, rank_index)
        
        if not valid_formulas:
            # No valid formulas - skip this spectrum
            continue
        
        # Assign probabilities
        # Redistribute probability mass from invalid formulas to top-1
        total_invalid_prob = sum(base_probs[i] for i in range(len(base_probs)) 
                                  if i >= len(valid_formulas) or 
                                  not any(vf[1] == i for vf in valid_formulas))
        
        formula_probs = []
        for formula, rank_idx in valid_formulas:
            prob = base_probs[rank_idx]
            # Add redistributed probability to top-1
            if rank_idx == 0:
                prob += total_invalid_prob
            formula_probs.append((formula, prob))
        
        # Normalize probabilities to sum to 1
        total_prob = sum(p for _, p in formula_probs)
        if total_prob > 0:
            formula_probs = [(f, p / total_prob) for f, p in formula_probs]
        
        buddy_formulas[identifier] = formula_probs
    
    print(f"Loaded BUDDY formulas for {len(buddy_formulas)} spectra")
    
    # Report statistics
    num_with_5 = sum(1 for v in buddy_formulas.values() if len(v) == 5)
    num_with_fewer = sum(1 for v in buddy_formulas.values() if len(v) < 5)
    print(f"  - {num_with_5} spectra with all 5 candidate formulas")
    print(f"  - {num_with_fewer} spectra with fewer than 5 candidate formulas")
    
    return buddy_formulas


def sample_formula_from_candidates(
    candidate_formulas: List[Tuple[str, float]],
) -> str:
    """
    Sample a formula from candidate formulas based on their probabilities.
    
    Args:
        candidate_formulas: List of (formula, probability) tuples
        
    Returns:
        Sampled formula string
    """
    if not candidate_formulas:
        raise ValueError("No candidate formulas provided")
    
    formulas = [f for f, _ in candidate_formulas]
    probs = [p for _, p in candidate_formulas]
    
    # Normalize probabilities just in case
    total = sum(probs)
    probs = [p / total for p in probs]
    
    return np.random.choice(formulas, p=probs)


def rank_candidates_with_formula_priority(
    smiles_list: List[str],
    target_fp: np.ndarray,
    target_formulas: Union[str, List[str], Set[str]],
    fp_bits: int = 4096,
    fp_radius: int = 2,
    top_k: int = 100,
) -> List[str]:
    """
    Rank candidates prioritizing formula matches, then by Tanimoto similarity.
    
    Strategy:
    1. Separate candidates into formula matches and non-matches
    2. Rank each group by Tanimoto similarity
    3. Take top-K from formula matches first
    4. Fill remaining slots with non-matches if needed
    
    Args:
        smiles_list: List of candidate SMILES
        target_fp: Target fingerprint
        target_formulas: Target molecular formula(s) - can be a single formula string,
                        a list of formulas, or a set of formulas. Candidates matching
                        ANY of the provided formulas are prioritized.
        fp_bits: Fingerprint bits
        fp_radius: Fingerprint radius
        top_k: Maximum number of predictions to return
        
    Returns:
        List of ranked SMILES strings (up to top_k)
    """
    if not smiles_list:
        return []
    
    # Normalize target_formulas to a set for efficient lookup
    if isinstance(target_formulas, str):
        formula_set = {target_formulas}
    elif isinstance(target_formulas, (list, tuple)):
        formula_set = set(target_formulas)
    else:
        formula_set = target_formulas
    
    # Calculate similarities and formulas for all candidates
    formula_matches = []
    formula_non_matches = []
    
    for smi in smiles_list:
        fp = compute_morgan_fingerprint(smi, fp_bits, fp_radius)
        if fp is None:
            continue
        
        sim = compute_tanimoto_similarity(target_fp, fp)
        formula = get_molecular_formula(smi)
        
        if formula in formula_set:
            formula_matches.append((smi, sim))
        else:
            formula_non_matches.append((smi, sim))
    
    # Sort each group by similarity (descending)
    formula_matches.sort(key=lambda x: x[1], reverse=True)
    formula_non_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Build final ranked list: formula matches first, then non-matches
    ranked_smiles = [smi for smi, _ in formula_matches]
    
    # Fill remaining slots with non-matches if needed
    if len(ranked_smiles) < top_k:
        remaining = top_k - len(ranked_smiles)
        ranked_smiles.extend([smi for smi, _ in formula_non_matches[:remaining]])
    else:
        ranked_smiles = ranked_smiles[:top_k]
    
    return ranked_smiles


def save_intermediate_predictions(
    round_results: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    round_idx: int,
    max_preds: int = 100,
):
    """
    Save intermediate predictions CSV after a round completes.
    
    Args:
        round_results: Dict mapping round_idx -> list of per-spectrum result dicts
        output_dir: Output directory
        round_idx: Current round index (0-indexed)
        max_preds: Maximum predictions per spectrum
    """
    results = round_results.get(round_idx, [])
    if not results:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate actual max predictions across all spectra
    actual_max = min(
        max(len(r.get('all_matched_smiles', [])) for r in results) if results else 0,
        max_preds
    )
    
    rows = []
    for r in results:
        row = {
            'true_smiles': r.get('target_smiles', ''),
            'name': r.get('spec_name', ''),
        }
        matched = r.get('all_matched_smiles', [])[:max_preds]
        for i in range(actual_max):
            col_name = f'pred_smiles_{i + 1}'
            row[col_name] = matched[i] if i < len(matched) else ''
        rows.append(row)
    
    columns = ['true_smiles', 'name'] + [f'pred_smiles_{i + 1}' for i in range(actual_max)]
    df = pd.DataFrame(rows, columns=columns)
    
    # Save with round number in filename (1-indexed for user-friendliness)
    pred_path = os.path.join(output_dir, f'predictions_round_{round_idx + 1}.csv')
    df.to_csv(pred_path, index=False)
    print(f"Saved round {round_idx + 1} predictions to: {pred_path}")


def save_round_timing(
    round_timing: Dict[int, Dict[str, float]],
    output_dir: str,
    num_samples: int,
):
    """
    Save timing data for all completed rounds to JSON file.
    
    Creates a JSON file with cumulative GPU time per round, which can be used
    to plot accuracy vs. GPU time/sample.
    
    Args:
        round_timing: Dict mapping round_idx -> timing_info dict
        output_dir: Output directory
        num_samples: Number of test samples
    """
    if not round_timing:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build timing summary
    timing_summary = {
        'num_samples': num_samples,
        'rounds': {},
    }
    
    for round_idx, timing_info in sorted(round_timing.items()):
        timing_summary['rounds'][f'round_{round_idx + 1}'] = {
            'round_time_seconds': timing_info.get('round_time_seconds', 0.0),
            'cumulative_time_seconds': timing_info.get('cumulative_time_seconds', 0.0),
            'time_per_sample_seconds': timing_info.get('time_per_sample_seconds', 0.0),
            'round_times_all': timing_info.get('round_times', []),
        }
    
    timing_path = os.path.join(output_dir, 'round_timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Saved timing data to: {timing_path}")


def run_scaling_benchmark(
    mist_encoder: torch.nn.Module,
    iceberg_sampler: IcebergSampler,
    dataset: SpectraMolDataset,
    split_data: List,
    config: dict,
    device: torch.device,
    spec_folder: str,
    output_dir: str,
    max_spectra: Optional[int] = None,
    max_output_preds: int = 100,
    verbose: bool = False,
    buddy_formulas: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    spec_to_instrument: Optional[Dict[str, str]] = None,
    spec_to_ionization: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
    """
    Run the scaling benchmark with batched ICEBERG calls.

    This implementation processes ALL test samples together, batching ICEBERG
    predictions by (collision energy, instrument, ionization) groups across all molecules for efficiency.

    Note: Token model configuration is now handled by IcebergSampler.

    Args:
        buddy_formulas: Optional dict mapping spectrum name -> list of (formula, prob) tuples.
                       If provided, uses predicted formulas instead of ground truth for generation.
        spec_to_instrument: Optional dict mapping spectrum name -> instrument type.
                           If provided, uses instrument type for ICEBERG predictions.
        spec_to_ionization: Optional dict mapping spectrum name -> ionization/adduct type.
                           If provided, uses ionization type for ICEBERG predictions.
    
    Returns:
        Tuple of (aggregate_stats, per_sample_results, round_timing)
        - aggregate_stats: Aggregated metrics across all samples
        - per_sample_results: List of per-sample evaluation results
        - round_timing: Dict mapping round_idx -> timing info dict with keys:
            - 'round_time_seconds': Time for this round
            - 'cumulative_time_seconds': Total time through this round
            - 'time_per_sample_seconds': Cumulative time / num_samples
    """
    
    use_buddy_formulas = buddy_formulas is not None
    
    # If spec_to_instrument/spec_to_ionization not provided, default to empty dict
    if spec_to_instrument is None:
        spec_to_instrument = {}
    if spec_to_ionization is None:
        spec_to_ionization = {}
    
    fp_cfg = config['fingerprint']
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    fp_threshold = fp_cfg['threshold']
    top_k = config['generation'].get('top_k', 100)
    num_rounds = iceberg_sampler.scaling_config.num_rounds

    print(f"\n{'='*70}")
    print('SPEC2MOL BENCHMARK WITH BATCHED ICEBERG SCALING')
    if use_buddy_formulas:
        print('** Using BUDDY predicted formulas (ground truth formula NOT used) **')
    print(f"{'='*70}")
    print(f"Total spectra in dataset: {len(dataset)}")
    print(f"Scaling config:")
    print(f"  - Batch size (B): {iceberg_sampler.scaling_config.batch_size}")
    print(f"  - Unique to refine (K): {iceberg_sampler.scaling_config.num_unique_to_refine}")
    print(f"  - Masks per molecule (M): {iceberg_sampler.scaling_config.masks_per_molecule}")
    print(f"  - Num rounds (R): {iceberg_sampler.scaling_config.num_rounds}")
    print(f"  - Mask probability: {iceberg_sampler.scaling_config.mask_prob}")
    print(f"  - Collision energies: {iceberg_sampler.scaling_config.collision_energies}")
    print(f"FP threshold: {fp_threshold}")
    if iceberg_sampler.token_model is not None:
        print(f"Token model: ENABLED (NGBoost={iceberg_sampler.is_ngboost}, sigma_lambda={iceberg_sampler.sigma_lambda})")
    print(f"{'='*70}\n")

    start_time = time.time()
    
    # =========================================================================
    # PHASE 1: Preprocess all test samples
    # =========================================================================
    print("Phase 1: Preprocessing all test samples...")
    
    dataloader = get_paired_loader(dataset, shuffle=False, batch_size=1, num_workers=0)
    test_samples = []
    sample_metadata = []  # Store additional info needed for evaluation
    skipped_no_buddy = 0
    
    for idx, batch in enumerate(tqdm(dataloader, total=max_spectra, desc='Loading spectra')):
        if max_spectra and idx >= max_spectra:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        spec, mol = split_data[idx]
        target_smiles = mol.get_smiles()
        target_inchi_key = get_inchikey_first_block(mol.get_inchikey())
        target_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target_smiles)))
        target_fp = compute_morgan_fingerprint(target_smiles, fp_bits, fp_radius)
        spec_name = spec.get_spec_name()
        
        if target_fp is None:
            print(f"Warning: Could not compute FP for {target_smiles}, skipping")
            continue

        # Handle BUDDY formulas
        candidate_formulas = None
        if use_buddy_formulas:
            # Look up BUDDY formulas for this spectrum
            candidate_formulas = buddy_formulas.get(spec_name)
            
            if candidate_formulas is None:
                skipped_no_buddy += 1
                if verbose:
                    print(f"Warning: No BUDDY formulas for {spec_name}, skipping")
                continue

        # Load real spectrum
        spec_path = os.path.join(spec_folder, f"{spec_name}.ms")
        real_specs = load_real_spec(spec_path, real_spec_type='ms', nce=iceberg_sampler.scaling_config.nce)
        
        if real_specs is None:
            print(f"Warning: Could not load spectrum for {spec_name}, skipping")
            continue

        # Encode spectra to fingerprint
        with torch.no_grad():
            pred_fp_probs, _ = mist_encoder(batch)
            pred_fp_probs = pred_fp_probs.cpu().numpy()[0]
        pred_fp_binary = binarize_fingerprint(pred_fp_probs, fp_threshold)

        # Build test sample dict (token lengths are now handled inside IcebergSampler)
        # When using BUDDY formulas, we pass candidate_formulas instead of target_formula
        # The IcebergSampler will sample from candidate formulas for each generation
        
        # Get instrument and ionization type for this spectrum
        instrument = spec_to_instrument.get(spec_name, 'Unknown')
        ionization = spec_to_ionization.get(spec_name, '[M+H]+')

        test_sample = {
            'name': spec_name,
            'real_specs': real_specs,
            'target_fp': pred_fp_binary,
            'target_smiles': target_smiles,
            'pred_fp_binary': pred_fp_binary,
            'instrument': instrument,
            'ionization': ionization,
        }
        
        if use_buddy_formulas:
            # Pass candidate formulas for sampling during generation
            test_sample['candidate_formulas'] = candidate_formulas
            # Use top-1 BUDDY formula as default (for when a single formula is needed)
            test_sample['target_formula'] = candidate_formulas[0][0]
        else:
            # Use ground truth formula
            test_sample['target_formula'] = target_formula
        
        test_samples.append(test_sample)
        
        # Store metadata for evaluation (always use ground truth formula for evaluation)
        # Also store candidate formulas for ranking when using BUDDY mode
        meta_entry = {
            'spec_name': spec_name,
            'target_smiles': target_smiles,
            'target_inchi_key': target_inchi_key,
            'target_formula': target_formula,  # Always ground truth for evaluation
            'target_fp': target_fp,
            'pred_fp_binary': pred_fp_binary,
        }
        if use_buddy_formulas and candidate_formulas:
            # Store BUDDY formulas for ranking (just the formula strings, not probabilities)
            meta_entry['candidate_formulas'] = [f for f, _ in candidate_formulas]
        sample_metadata.append(meta_entry)
    
    print(f"Loaded {len(test_samples)} valid test samples")
    if use_buddy_formulas and skipped_no_buddy > 0:
        print(f"  (Skipped {skipped_no_buddy} spectra with no BUDDY formulas)")
    
    # Track results per round for intermediate saving
    round_results: Dict[int, List[Dict[str, Any]]] = {r: [] for r in range(num_rounds)}
    
    # Track timing data per round
    round_timing: Dict[int, Dict[str, float]] = {}
    
    def round_callback(round_idx: int, sample_states: List[Dict], timing_info: Dict[str, float]):
        """Callback to save intermediate predictions and timing after each round."""
        print(f"\n  -> Saving intermediate predictions for round {round_idx + 1}...")
        
        # Store timing info for this round
        round_timing[round_idx] = timing_info.copy()
        
        round_results[round_idx] = []
        for i, state in enumerate(sample_states):
            meta = sample_metadata[i]
            
            # Rank candidates with formula priority
            # Use BUDDY formulas if available, otherwise use ground truth
            ranking_formulas = meta.get('candidate_formulas', meta['target_formula'])
            ranked_smiles = rank_candidates_with_formula_priority(
                smiles_list=list(state['all_unique_smiles']),
                target_fp=meta['pred_fp_binary'],
                target_formulas=ranking_formulas,
                fp_bits=fp_bits,
                fp_radius=fp_radius,
                top_k=max_output_preds,
            )
            
            round_results[round_idx].append({
                'target_smiles': meta['target_smiles'],
                'spec_name': meta['spec_name'],
                'all_matched_smiles': ranked_smiles,
            })
        
        # Save intermediate predictions
        save_intermediate_predictions(round_results, output_dir, round_idx, max_output_preds)
        
        # Save timing data
        save_round_timing(round_timing, output_dir, len(test_samples))
    
    # =========================================================================
    # PHASE 2: Run batched scaling
    # =========================================================================
    print(f"\nPhase 2: Running batched scaling with {num_rounds} rounds...")
    
    scaling_results = iceberg_sampler.run_scaling_all_samples(
        test_samples=test_samples,
        verbose=verbose,
        round_callback=round_callback,
    )
    
    # =========================================================================
    # PHASE 3: Evaluate results
    # =========================================================================
    print(f"\nPhase 3: Evaluating results...")
    
    all_results = []
    for i, (result, meta) in enumerate(zip(scaling_results, sample_metadata)):
        generated_smiles = result['generated_smiles']
        
        # Rank candidates with formula priority
        # Use BUDDY formulas if available, otherwise use ground truth
        ranking_formulas = meta.get('candidate_formulas', meta['target_formula'])
        ranked_smiles = rank_candidates_with_formula_priority(
            smiles_list=generated_smiles,
            target_fp=meta['pred_fp_binary'],
            target_formulas=ranking_formulas,
            fp_bits=fp_bits,
            fp_radius=fp_radius,
            top_k=max_output_preds,
        )
        
        # Build predictions list
        predictions = []
        for smi in ranked_smiles:
            fp = compute_morgan_fingerprint(smi, fp_bits, fp_radius)
            if fp is None:
                continue
            sim = compute_tanimoto_similarity(meta['pred_fp_binary'], fp)
            entry = build_prediction_entry(smi, sim, 1, 'scaling', fp_bits, fp_radius)
            if entry:
                predictions.append(entry)

        # Evaluate
        preds_eval = predictions[:top_k] if top_k and top_k > 0 else predictions
        eval_result = evaluate_predictions(
            preds_eval, 
            meta['target_smiles'], 
            meta['target_inchi_key'], 
            meta['target_fp'], 
            fp_bits, 
            fp_radius
        )
        
        # Add additional metrics
        eval_result['mist_tanimoto'] = compute_tanimoto_similarity(meta['target_fp'], meta['pred_fp_binary'])
        eval_result['spec_name'] = meta['spec_name']
        eval_result['proposal_smiles'] = predictions[0]['smiles'] if predictions else None
        eval_result['total_unique_generated'] = len(generated_smiles)
        eval_result['all_matched_smiles'] = ranked_smiles

        all_results.append(eval_result)

    elapsed = time.time() - start_time
    aggregate = compute_aggregate_statistics(all_results, elapsed)
    
    # Add round timing info to aggregate
    if round_timing:
        aggregate['round_timing'] = {
            f'round_{r+1}': {
                'round_time_seconds': t.get('round_time_seconds', 0.0),
                'cumulative_time_seconds': t.get('cumulative_time_seconds', 0.0),
                'time_per_sample_seconds': t.get('time_per_sample_seconds', 0.0),
            }
            for r, t in round_timing.items()
        }

    return aggregate, all_results, round_timing


def save_predictions_csv(
    results: List[Dict[str, Any]],
    output_dir: str,
    max_preds: int = 100,
):
    """Save predictions CSV."""
    actual_max = min(
        max(len(r.get('all_matched_smiles', [])) for r in results) if results else 0,
        max_preds
    )
    
    rows = []
    for r in results:
        row = {
            'true_smiles': r.get('target_smiles', ''),
            'name': r.get('spec_name', ''),
        }
        matched = r.get('all_matched_smiles', [])[:max_preds]
        for i in range(actual_max):
            col_name = f'pred_smiles_{i + 1}'
            row[col_name] = matched[i] if i < len(matched) else ''
        rows.append(row)
    
    columns = ['true_smiles', 'name'] + [f'pred_smiles_{i + 1}' for i in range(actual_max)]
    df = pd.DataFrame(rows, columns=columns)
    
    pred_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")


def save_results(
    aggregate: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: str,
    config: dict,
    max_output_preds: int = 100,
):
    """Save benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregate statistics
    stats_path = os.path.join(output_dir, 'aggregate_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved aggregate statistics to: {stats_path}")

    # Save config
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save detailed results
    save_data = []
    for r in results:
        rc = {k: v for k, v in r.items() if k not in ('top_predictions', 'all_matched_smiles')}
        save_data.append(rc)
    
    details_path = os.path.join(output_dir, 'detailed_results.csv')
    pd.DataFrame(save_data).to_csv(details_path, index=False)
    print(f"Saved detailed results to: {details_path}")
    
    # Save predictions CSV
    save_predictions_csv(results, output_dir, max_output_preds)


def print_summary(aggregate: Dict[str, Any]):
    """Print benchmark summary."""
    print(f"\n{'='*70}")
    print('BENCHMARK SUMMARY')
    print(f"{'='*70}")
    
    print(f"\nTotal spectra: {aggregate.get('total_spectra', 0)}")
    print(f"Time: {aggregate.get('elapsed_time_seconds', 0):.1f}s")
    print(f"Speed: {aggregate.get('spectra_per_second', 0):.2f} spectra/s")
    
    print(f"\n--- MIST Encoder Quality ---")
    print(f"Mean Tanimoto (pred vs GT FP): {aggregate.get('mist_tanimoto_mean', 0):.4f}")
    
    print(f"\n--- Core Metrics ---")
    print(f"Exact Match Top-1: {aggregate.get('exact_match_top1', 0)*100:.2f}%")
    print(f"Exact Match Top-10: {aggregate.get('exact_match_top10', 0)*100:.2f}%")
    print(f"Tanimoto Top-1: {aggregate.get('tanimoto_top1_mean', 0):.4f}")
    print(f"Tanimoto Top-10: {aggregate.get('tanimoto_top10_mean', 0):.4f}")
    
    print(f"\n--- Scaling Statistics ---")
    if 'avg_total_unique_generated' in aggregate:
        print(f"Avg unique generated: {aggregate.get('avg_total_unique_generated', 0):.1f}")
    
    # Print per-round timing if available
    if 'round_timing' in aggregate:
        print(f"\n--- Per-Round GPU Timing ---")
        for round_name, timing in sorted(aggregate['round_timing'].items()):
            cumulative = timing.get('cumulative_time_seconds', 0.0)
            per_sample = timing.get('time_per_sample_seconds', 0.0)
            print(f"{round_name}: cumulative={cumulative:.2f}s, per_sample={per_sample:.4f}s")
    
    print(f"{'='*70}")


def main():
    args = parse_args()
    config = merge_config_with_args(load_config(args.config), args)
    max_spectra = args.max_spectra or config.get('evaluation', {}).get('max_spectra')

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load token model if provided
    token_model = None
    token_features = None
    is_ngboost = False
    if args.token_model:
        token_model, token_features, is_ngboost = load_token_model(args.token_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("config['mist_encoder']:", config['mist_encoder'])
    dataset, split_data, spec_to_instrument, spec_to_ionization = load_spec_data(
        config['data'],
        config['mist_encoder'],
        config['evaluation']['split'],
        shuffle=True,
        n_proc=args.n_proc,
        proc_idx=args.proc_idx,
    )

    # Load models
    mist_encoder = load_mist_encoder(config['mist_encoder'], device)
    sampler = load_dlm_sampler(config['dlm'], args.use_shared_cross_attention)

    # Create ICEBERG config
    iceberg_config = IcebergConfig(
        gen_ckpt=args.iceberg_gen_ckpt,
        inten_ckpt=args.iceberg_inten_ckpt,
        python_path=args.iceberg_python_path,
        cuda_devices=args.iceberg_gpu,
        batch_size=args.iceberg_batch_size,
        num_gpu_workers=len(args.iceberg_gpu),
        num_cpu_workers=8,
        ppm=args.ppm,
        num_bins=args.num_bins,
        upper_limit=args.upper_limit,
    )

    # Create scaling config
    scaling_config = ScalingConfig(
        batch_size=args.batch_size,
        num_unique_to_refine=args.num_unique_to_refine,
        masks_per_molecule=args.masks_per_molecule,
        num_rounds=args.num_rounds,
        mask_prob=args.mask_prob,
        top_k_halluc_peaks=args.top_k_halluc_peaks,
        halluc_inten_threshold=args.halluc_inten_threshold,
        softmax_temp=config['generation']['softmax_temp'],
        randomness=config['generation']['randomness'],
        collision_energies=args.collision_energies,
        nce=args.nce,
    )

    # Create ICEBERG sampler (with token model for length prediction)
    iceberg_sampler = IcebergSampler(
        sampler=sampler,
        iceberg_config=iceberg_config,
        scaling_config=scaling_config,
        results_dir=args.iceberg_results_dir,
        token_model=token_model,
        token_features=token_features,
        is_ngboost=is_ngboost,
        sigma_lambda=args.sigma_lambda,
        masking_strategy_name=args.masking_strategy,
        incl_unknown_instrument=args.incl_unknown_instrument,
    )

    # Load BUDDY formulas if provided
    buddy_formulas = None
    if args.buddy_formula_path:
        buddy_formulas = load_buddy_formulas(args.buddy_formula_path)

    # Run benchmark
    aggregate, results, round_timing = run_scaling_benchmark(
        mist_encoder=mist_encoder,
        iceberg_sampler=iceberg_sampler,
        dataset=dataset,
        split_data=split_data,
        config=config,
        device=device,
        spec_folder=config['data']['spec_folder'],
        output_dir=config['output']['results_dir'],
        max_spectra=max_spectra,
        max_output_preds=args.max_output_preds,
        verbose=args.verbose,
        buddy_formulas=buddy_formulas,
        spec_to_instrument=spec_to_instrument,
        spec_to_ionization=spec_to_ionization,
    )

    # Save and print results
    save_results(aggregate, results, config['output']['results_dir'], config, args.max_output_preds)
    print_summary(aggregate)
    
    # Print timing summary
    if round_timing:
        print(f"\n--- Round Timing Summary ---")
        num_samples = len(results)
        for round_idx, timing in sorted(round_timing.items()):
            cumulative = timing.get('cumulative_time_seconds', 0.0)
            per_sample = timing.get('time_per_sample_seconds', 0.0)
            print(f"Round {round_idx + 1}: cumulative={cumulative:.2f}s, per_sample={per_sample:.4f}s")
    
    print(f"\nResults saved to: {config['output']['results_dir']}/")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nBenchmark interrupted.')
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
