#!/usr/bin/env python
"""
Spec2Mol benchmark with separate MIST encoder and DLM decoder checkpoints.

Evaluates end-to-end spectra-to-molecule performance using:
1. MIST encoder: Predicts fingerprints from spectra
2. DLM decoder: Generates molecules conditioned on predicted fingerprints

Key metrics:
- Exact match (InChI key) at Top-1 and Top-10
- Tanimoto similarity at Top-1 and Top-10
- Formula match rate and statistics
- MIST encoder quality (predicted FP vs GT FP)

Usage:
    python scripts/benchmark_spec2mol.py \
        --config configs/spec2mol_benchmark.yaml \
        --mist-checkpoint path/to/mist.ckpt \
        --dlm-checkpoint path/to/dlm.ckpt
"""

import os
import sys
import argparse
import warnings
import time
import random
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

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

from dlm.sampler import Sampler
from dlm.utils.benchmark_utils import (
    normalize_formula,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    get_inchikey_first_block,
    binarize_fingerprint,
    generate_with_formula_filter,
    build_prediction_entry,
    evaluate_predictions,
    compute_aggregate_statistics,
    load_token_model,
    predict_token_count,
)
from mist.models import SpectraEncoderGrowing
from mist.data import (
    get_paired_spectra,
    get_paired_featurizer,
    SpectraMolDataset,
    PresetSpectraSplitter,
)
from mist.data.datasets import get_paired_loader

# Suppress RDKit warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spec2Mol benchmark (separate encoder/decoder)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/spec2mol_benchmark.yaml')
    parser.add_argument('--mist-checkpoint', type=str, help='MIST encoder checkpoint')
    parser.add_argument('--dlm-checkpoint', type=str, help='DLM decoder checkpoint')
    parser.add_argument('--data-dir', type=str, help='Spec data directory')
    parser.add_argument('--fp-threshold', type=float, help='FP binarization threshold')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--split', type=str, choices=['val', 'test'])
    parser.add_argument('--max-spectra', type=int, default=None)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--softmax-temp', type=float)
    parser.add_argument('--randomness', type=float)
    parser.add_argument('--formula-matches', type=int, help='Required formula matches per spectrum')
    parser.add_argument('--max-attempts', type=int, help='Max generation attempts per spectrum')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--use-shared-cross-attention',
        action='store_true',
        help='Use shared cross-attention mode'
    )
    parser.add_argument(
        '--n-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use for inference (data parallelism)'
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
        help='Lambda multiplier for NGBoost sigma-based sampling range (mean ± sigma*lambda)'
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
            'batch_size': 16,
            'softmax_temp': 1.0,
            'randomness': 0.1,
            'top_k': 100,
        },
        'formula_filter': {
            'n_required': 20,
            'max_attempts': 300,
        },
        'evaluation': {
            'split': 'test',
            'max_spectra': 500,
        },
        'output': {
            'results_dir': 'results/spec2mol_benchmark',
        },
    }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Merge with defaults
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
    if args.batch_size is not None:
        config['generation']['batch_size'] = args.batch_size
    if args.softmax_temp is not None:
        config['generation']['softmax_temp'] = args.softmax_temp
    if args.randomness is not None:
        config['generation']['randomness'] = args.randomness
    if args.formula_matches is not None:
        config['formula_filter']['n_required'] = args.formula_matches
    if args.max_attempts is not None:
        config['formula_filter']['max_attempts'] = args.max_attempts
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
    
    # Prepare config overrides for checkpoint loading
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
    
    # Report cross-attention mode
    use_shared = getattr(sampler.model, 'use_shared_cross_attention', False)
    print(f"Cross-attention mode: {'SHARED' if use_shared else 'INDEPENDENT'}")
    
    return sampler


def load_spec_data(
    config: dict,
    encoder_config: dict,
    split: str = 'test',
    shuffle: bool = False,
) -> Tuple[SpectraMolDataset, List]:
    """Load Spec dataset."""
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
    )

    return SpectraMolDataset(split_data, featurizer), split_data


def _worker_process_spectra(
    gpu_id: int,
    indices: List[int],
    dataset: SpectraMolDataset,
    split_data: List,
    config: dict,
    use_shared_cross_attention: bool,
    result_queue,
    token_model=None,
    token_features=None,
    is_ngboost: bool = False,
    sigma_lambda: float = 3.0,
):
    """
    Worker function for multi-GPU processing.
    Each worker processes a subset of spectra on a specific GPU.
    """
    import torch.multiprocessing as mp
    
    # Set device for this worker
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    # Load models on this GPU
    print(config['mist_encoder'])
    mist_encoder = load_mist_encoder(config['mist_encoder'], device)
    sampler = load_dlm_sampler(config['dlm'], use_shared_cross_attention)
    
    gen_cfg = config['generation']
    fp_cfg = config['fingerprint']
    filter_cfg = config['formula_filter']
    
    batch_size = gen_cfg['batch_size']
    softmax_temp = gen_cfg['softmax_temp']
    randomness = gen_cfg['randomness']
    top_k = gen_cfg.get('top_k', 100)
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    fp_threshold = fp_cfg['threshold']
    
    dataloader = get_paired_loader(dataset, shuffle=False, batch_size=1, num_workers=0)
    results = []
    
    # Create a mapping from indices to dataloader items
    all_batches = list(dataloader)
    
    for idx in tqdm(indices, desc=f'GPU {gpu_id}', position=gpu_id):
        batch = all_batches[idx]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        spec, mol = split_data[idx]
        target_smiles = mol.get_smiles()
        target_inchi_key = get_inchikey_first_block(mol.get_inchikey())
        target_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target_smiles)))
        target_fp = compute_morgan_fingerprint(target_smiles, fp_bits, fp_radius)
        
        if target_fp is None:
            print(f"Warning: Could not compute FP for {target_smiles}, skipping")
            continue
        
        # Encode spectra to fingerprint
        with torch.no_grad():
            pred_fp_probs, _ = mist_encoder(batch)
            pred_fp_probs = pred_fp_probs.cpu().numpy()[0]
        pred_fp_binary = binarize_fingerprint(pred_fp_probs, fp_threshold)
        
        # Generate with formula filtering
        matched_smiles, matched_sims, total_gen, total_valid, total_matched, counter, last_valid = \
            generate_with_formula_filter(
                sampler=sampler,
                fingerprint_array=pred_fp_binary,
                target_formula=target_formula,
                target_smiles=target_smiles,
                n_required=filter_cfg['n_required'],
                max_attempts=filter_cfg['max_attempts'],
                batch_size=batch_size,
                softmax_temp=softmax_temp,
                randomness=randomness,
                fp_bits=fp_bits,
                fp_radius=fp_radius,
                token_model=token_model,
                token_features=token_features,
                is_ngboost=is_ngboost,
                sigma_lambda=sigma_lambda,
            )
        
        # Build predictions list
        sim_records = defaultdict(list)
        for smi, sim in zip(matched_smiles, matched_sims):
            sim_records[smi].append(sim)
        
        predictions = []
        if sim_records:
            for smi, sims in sim_records.items():
                entry = build_prediction_entry(smi, np.mean(sims), len(sims), 'formula', fp_bits, fp_radius)
                if entry:
                    predictions.append(entry)
            predictions.sort(key=lambda x: (x['frequency'], x['similarity']), reverse=True)
            source = 'formula'
        else:
            source = 'fallback'
            if last_valid:
                fallback_fp = compute_morgan_fingerprint(last_valid, fp_bits, fp_radius)
                sim = compute_tanimoto_similarity(pred_fp_binary, fallback_fp) if fallback_fp is not None else 0.0
                entry = build_prediction_entry(last_valid, sim, 1, 'fallback', fp_bits, fp_radius)
                if entry:
                    predictions.append(entry)
        
        # Evaluate
        preds_eval = predictions[:top_k] if top_k and top_k > 0 else predictions
        result = evaluate_predictions(preds_eval, target_smiles, target_inchi_key, target_fp, fp_bits, fp_radius)
        
        # Add additional metrics
        result['mist_tanimoto'] = compute_tanimoto_similarity(target_fp, pred_fp_binary)
        result['spec_name'] = spec.get_spec_name()
        result['proposal_smiles'] = predictions[0]['smiles'] if predictions else None
        result['proposal_source'] = source if predictions else None
        result['formula_matches_collected'] = len(matched_smiles)
        result['total_generated'] = total_gen
        result['total_valid'] = total_valid
        result['total_formula_matched'] = total_matched
        result['generation_time'] = gen_time
        result['original_idx'] = idx  # Track original index for ordering
        # Store all matched SMILES (already sorted by similarity descending)
        result['all_matched_smiles'] = matched_smiles
        
        results.append(result)
    
    result_queue.put((gpu_id, results))


def run_benchmark_multi_gpu(
    config: dict,
    dataset: SpectraMolDataset,
    split_data: List,
    n_gpus: int,
    use_shared_cross_attention: bool,
    max_spectra: Optional[int] = None,
    token_model=None,
    token_features=None,
    is_ngboost: bool = False,
    sigma_lambda: float = 3.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the benchmark across multiple GPUs using multiprocessing.
    
    Note: mist_encoder parameter is not used - each worker process loads its own
    models on its assigned GPU to avoid memory duplication on GPU 0.
    """
    import torch.multiprocessing as mp
    
    gen_cfg = config['generation']
    fp_cfg = config['fingerprint']
    filter_cfg = config['formula_filter']
    
    print(f"\n{'='*70}")
    print('SPEC2MOL BENCHMARK (Multi-GPU)')
    print(f"{'='*70}")
    print(f"Total spectra: {len(dataset)}")
    print(f"Number of GPUs: {n_gpus}")
    print(f"Formula matches required: {filter_cfg['n_required']}")
    print(f"Max attempts: {filter_cfg['max_attempts']}")
    print(f"Batch size: {gen_cfg['batch_size']}")
    print(f"Temperature: {gen_cfg['softmax_temp']}")
    print(f"Randomness: {gen_cfg['randomness']}")
    print(f"FP threshold: {fp_cfg['threshold']}")
    if token_model is not None:
        print(f"Token model: ENABLED (NGBoost={is_ngboost}, sigma_lambda={sigma_lambda})")
    print(f"{'='*70}\n")
    
    num_to_process = min(len(dataset), max_spectra) if max_spectra else len(dataset)
    all_indices = list(range(num_to_process))
    
    # Split indices across GPUs
    indices_per_gpu = [[] for _ in range(n_gpus)]
    for i, idx in enumerate(all_indices):
        indices_per_gpu[i % n_gpus].append(idx)
    
    print(f"Distributing {num_to_process} spectra across {n_gpus} GPUs:")
    for gpu_id, indices in enumerate(indices_per_gpu):
        print(f"  GPU {gpu_id}: {len(indices)} spectra")
    
    start_time = time.time()
    
    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=_worker_process_spectra,
            args=(
                gpu_id,
                indices_per_gpu[gpu_id],
                dataset,
                split_data,
                config,
                use_shared_cross_attention,
                result_queue,
                token_model,
                token_features,
                is_ngboost,
                sigma_lambda,
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results from all workers
    all_results = []
    for _ in range(n_gpus):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
        print(f"GPU {gpu_id} completed with {len(results)} results")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Sort results by original index to maintain order
    all_results.sort(key=lambda x: x.get('original_idx', 0))
    
    # Remove the temporary index field
    for r in all_results:
        r.pop('original_idx', None)
    
    elapsed = time.time() - start_time
    aggregate = compute_aggregate_statistics(all_results, elapsed)
    
    # Add generation timing statistics
    total_gen_time = sum(r.get('generation_time', 0.0) for r in all_results)
    aggregate['total_generation_time'] = total_gen_time
    aggregate['generation_time_percentage'] = (total_gen_time / elapsed * 100) if elapsed > 0 else 0.0
    
    # Add source distribution
    aggregate['proposal_source_counts'] = Counter([
        r['proposal_source'] for r in all_results if r.get('proposal_source')
    ]).most_common()
    aggregate['n_gpus'] = n_gpus
    
    return aggregate, all_results


def run_benchmark(
    mist_encoder: torch.nn.Module,
    sampler: Sampler,
    dataset: SpectraMolDataset,
    split_data: List,
    config: dict,
    device: torch.device,
    max_spectra: Optional[int] = None,
    token_model=None,
    token_features=None,
    is_ngboost: bool = False,
    sigma_lambda: float = 3.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run the benchmark."""
    gen_cfg = config['generation']
    fp_cfg = config['fingerprint']
    filter_cfg = config['formula_filter']
    
    batch_size = gen_cfg['batch_size']
    softmax_temp = gen_cfg['softmax_temp']
    randomness = gen_cfg['randomness']
    top_k = gen_cfg.get('top_k', 100)
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    fp_threshold = fp_cfg['threshold']

    print(f"\n{'='*70}")
    print('SPEC2MOL BENCHMARK')
    print(f"{'='*70}")
    print(f"Total spectra: {len(dataset)}")
    print(f"Formula matches required: {filter_cfg['n_required']}")
    print(f"Max attempts: {filter_cfg['max_attempts']}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {softmax_temp}")
    print(f"Randomness: {randomness}")
    print(f"FP threshold: {fp_threshold}")
    print(f"{'='*70}\n")

    dataloader = get_paired_loader(dataset, shuffle=False, batch_size=1, num_workers=0)
    all_results = []
    start_time = time.time()
    num_to_process = min(len(dataset), max_spectra) if max_spectra else len(dataset)

    for idx, batch in enumerate(tqdm(dataloader, total=num_to_process, desc='Processing spectra')):
        if max_spectra and idx >= max_spectra:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        spec, mol = split_data[idx]
        target_smiles = mol.get_smiles()
        target_inchi_key = get_inchikey_first_block(mol.get_inchikey())
        target_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target_smiles)))
        target_fp = compute_morgan_fingerprint(target_smiles, fp_bits, fp_radius)
        
        if target_fp is None:
            print(f"Warning: Could not compute FP for {target_smiles}, skipping")
            continue

        # Encode spectra to fingerprint
        with torch.no_grad():
            pred_fp_probs, _ = mist_encoder(batch)
            pred_fp_probs = pred_fp_probs.cpu().numpy()[0]
        pred_fp_binary = binarize_fingerprint(pred_fp_probs, fp_threshold)

        # Generate with formula filtering
        matched_smiles, matched_sims, total_gen, total_valid, total_matched, counter, last_valid, gen_time = \
            generate_with_formula_filter(
                sampler=sampler,
                fingerprint_array=pred_fp_binary,
                target_formula=target_formula,
                target_smiles=target_smiles,
                n_required=filter_cfg['n_required'],
                max_attempts=filter_cfg['max_attempts'],
                batch_size=batch_size,
                softmax_temp=softmax_temp,
                randomness=randomness,
                fp_bits=fp_bits,
                fp_radius=fp_radius,
                token_model=token_model,
                token_features=token_features,
                is_ngboost=is_ngboost,
                sigma_lambda=sigma_lambda,
            )

        # Build predictions list
        sim_records = defaultdict(list)
        for smi, sim in zip(matched_smiles, matched_sims):
            sim_records[smi].append(sim)

        predictions = []
        if sim_records:
            for smi, sims in sim_records.items():
                entry = build_prediction_entry(smi, np.mean(sims), len(sims), 'formula', fp_bits, fp_radius)
                if entry:
                    predictions.append(entry)
            predictions.sort(key=lambda x: (x['frequency'], x['similarity']), reverse=True)
            source = 'formula'
        else:
            source = 'fallback'
            if last_valid:
                fallback_fp = compute_morgan_fingerprint(last_valid, fp_bits, fp_radius)
                sim = compute_tanimoto_similarity(pred_fp_binary, fallback_fp) if fallback_fp is not None else 0.0
                entry = build_prediction_entry(last_valid, sim, 1, 'fallback', fp_bits, fp_radius)
                if entry:
                    predictions.append(entry)

        # Evaluate
        preds_eval = predictions[:top_k] if top_k and top_k > 0 else predictions
        result = evaluate_predictions(preds_eval, target_smiles, target_inchi_key, target_fp, fp_bits, fp_radius)
        
        # Add additional metrics
        result['mist_tanimoto'] = compute_tanimoto_similarity(target_fp, pred_fp_binary)
        result['spec_name'] = spec.get_spec_name()
        result['proposal_smiles'] = predictions[0]['smiles'] if predictions else None
        result['proposal_source'] = source if predictions else None
        result['formula_matches_collected'] = len(matched_smiles)
        result['total_generated'] = total_gen
        result['total_valid'] = total_valid
        result['total_formula_matched'] = total_matched
        result['generation_time'] = gen_time
        # Store all matched SMILES (already sorted by similarity descending)
        result['all_matched_smiles'] = matched_smiles

        all_results.append(result)

    elapsed = time.time() - start_time
    aggregate = compute_aggregate_statistics(all_results, elapsed)
    
    # Add generation timing statistics
    total_gen_time = sum(r.get('generation_time', 0.0) for r in all_results)
    aggregate['total_generation_time'] = total_gen_time
    aggregate['generation_time_percentage'] = (total_gen_time / elapsed * 100) if elapsed > 0 else 0.0
    aggregate['total_elapsed_time_seconds'] = elapsed
    
    # Add source distribution
    aggregate['proposal_source_counts'] = Counter([
        r['proposal_source'] for r in all_results if r.get('proposal_source')
    ]).most_common()

    return aggregate, all_results


def save_predictions_csv(
    results: List[Dict[str, Any]],
    output_dir: str,
):
    """
    Save predictions CSV with columns: true_smiles, name, pred_smiles_1, ..., pred_smiles_K.
    
    K is the maximum number of matched SMILES across all spectra.
    Predictions are ordered by similarity to true fingerprint (descending).
    """
    # Find maximum number of predictions
    max_preds = max(
        len(r.get('all_matched_smiles', [])) for r in results
    ) if results else 0
    
    # Build rows
    rows = []
    for r in results:
        row = {
            'true_smiles': r.get('target_smiles', ''),
            'name': r.get('spec_name', ''),
        }
        matched = r.get('all_matched_smiles', [])
        for i in range(max_preds):
            col_name = f'pred_smiles_{i + 1}'
            row[col_name] = matched[i] if i < len(matched) else ''
        rows.append(row)
    
    # Create DataFrame with ordered columns
    columns = ['true_smiles', 'name'] + [f'pred_smiles_{i + 1}' for i in range(max_preds)]
    df = pd.DataFrame(rows, columns=columns)
    
    pred_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")


def save_results(
    aggregate: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: str,
    config: dict
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

    # Save detailed results (excluding large lists)
    save_data = []
    for r in results:
        rc = {k: v for k, v in r.items() if k not in ('top_predictions', 'all_matched_smiles')}
        if 'top_predictions' in r:
            rc['top_prediction_smiles'] = [p['smiles'] for p in r['top_predictions']]
        save_data.append(rc)
    
    details_path = os.path.join(output_dir, 'detailed_results.csv')
    pd.DataFrame(save_data).to_csv(details_path, index=False)
    print(f"Saved detailed results to: {details_path}")
    
    # Save predictions CSV (true_smiles, name, pred_smiles_1, ..., pred_smiles_K)
    save_predictions_csv(results, output_dir)


def print_summary(aggregate: Dict[str, Any]):
    """Print benchmark summary."""
    print(f"\n{'='*70}")
    print('BENCHMARK SUMMARY')
    print(f"{'='*70}")
    
    print(f"\nTotal spectra: {aggregate.get('total_spectra', 0)}")
    print(f"Time: {aggregate.get('elapsed_time_seconds', 0):.1f}s")
    print(f"Speed: {aggregate.get('spectra_per_second', 0):.2f} spectra/s")
    print(f"\n--- Generation Timing ---")
    print(f"Time in generation functions: {aggregate.get('total_generation_time', 0):.1f}s")
    print(f"Percentage of total time: {aggregate.get('generation_time_percentage', 0):.1f}%")
    
    print(f"\n--- MIST Encoder Quality ---")
    print(f"Mean Tanimoto (pred vs GT FP): {aggregate.get('mist_tanimoto_mean', 0):.4f}")
    
    print(f"\n--- Core Metrics ---")
    print(f"Exact Match Top-1: {aggregate.get('exact_match_top1', 0)*100:.2f}%")
    print(f"Exact Match Top-10: {aggregate.get('exact_match_top10', 0)*100:.2f}%")
    print(f"Tanimoto Top-1: {aggregate.get('tanimoto_top1_mean', 0):.4f}")
    print(f"Tanimoto Top-10: {aggregate.get('tanimoto_top10_mean', 0):.4f}")
    
    print(f"\n--- Formula Matching ---")
    print(f"Success rate: {aggregate.get('formula_match_success_rate', 0)*100:.2f}%")
    print(f"Never matched: {aggregate.get('never_matched_rate', 0)*100:.2f}%")
    print(f"Avg actual formula matches: {aggregate.get('avg_formula_matches', 0):.1f}")
    print(f"Avg predictions collected (incl. padding): {aggregate.get('avg_predictions_collected', 0):.1f}")
    print(f"Avg generations: {aggregate.get('avg_total_generated', 0):.1f}")
    print(f"Avg attempts per match (successful only): {aggregate.get('avg_attempts_to_match', 0):.1f}")
    
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

    # Determine number of GPUs to use
    n_gpus = args.n_gpus
    if n_gpus > 1 and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if n_gpus > available_gpus:
            print(f"Warning: Requested {n_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
            n_gpus = available_gpus
        print(f"Using {n_gpus} GPUs for parallel inference")
    else:
        n_gpus = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data (needed for both single and multi-GPU modes)
    dataset, split_data = load_spec_data(
        config['data'],
        config['mist_encoder'],
        config['evaluation']['split'],
        shuffle=False,
    )

    # Run benchmark (with multi-GPU support if n_gpus > 1)
    if n_gpus > 1:
        # In multi-GPU mode, each worker loads its own models on its assigned GPU
        # Don't load models in the main process to avoid double memory usage on GPU 0
        aggregate, results = run_benchmark_multi_gpu(
            config, dataset, split_data,
            n_gpus, args.use_shared_cross_attention, max_spectra,
            token_model, token_features, is_ngboost, args.sigma_lambda
        )
    else:
        # Single GPU mode: load models in main process
        mist_encoder = load_mist_encoder(config['mist_encoder'], device)
        sampler = load_dlm_sampler(config['dlm'], args.use_shared_cross_attention)
        aggregate, results = run_benchmark(
            mist_encoder, sampler, dataset, split_data,
            config, device, max_spectra,
            token_model, token_features, is_ngboost, args.sigma_lambda
        )

    # Save and print results
    save_results(aggregate, results, config['output']['results_dir'], config)
    print_summary(aggregate)
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
