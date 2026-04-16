#!/usr/bin/env python3
"""
Multi-GPU launcher for spec2mol_scaling.py

This script automatically splits the scaling benchmark across multiple GPUs,
managing job distribution and output logging.

Usage:
    python run_scaling_multi_gpu.py \
        --config <config.yaml> \
        --mist-checkpoint <path> \
        --dlm-checkpoint <path> \
        --output-dir <output_dir> \
        --gpus 0 1 2 3 \
        [other spec2mol_scaling.py arguments]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional
import time


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU launcher for spec2mol_scaling.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on 4 GPUs (single server)
    python run_scaling_multi_gpu.py \\
        --config configs/monte_spec2mol_benchmark_canopus.yaml \\
        --mist-checkpoint checkpoints/canopus_msg-synth.pt \\
        --dlm-checkpoint checkpoints/frigid_base.ckpt \\
        --output-dir ./scale/experiment_1 \\
        --gpus 0 1 2 3 \\
        --batch-size 128 \\
        --num-rounds 3

    # Run on all available GPUs (single server)
    python run_scaling_multi_gpu.py \\
        --config configs/monte_spec2mol_benchmark_canopus.yaml \\
        --mist-checkpoint checkpoints/canopus_msg-synth.pt \\
        --dlm-checkpoint checkpoints/frigid_base.ckpt \\
        --output-dir ./scale/experiment_1 \\
        --gpus all \\
        --batch-size 128

    # Multi-server example (16 total GPUs: 8 on server A, 8 on server B)
    # On server A (GPUs 0-7, proc_idx 0-7):
    python run_scaling_multi_gpu.py \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --n-proc 16 --offset 0 \\
        --output-dir /shared/storage/experiment_1 \\
        ...

    # On server B (GPUs 0-7, proc_idx 8-15):
    python run_scaling_multi_gpu.py \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --n-proc 16 --offset 8 \\
        --output-dir /shared/storage/experiment_1 \\
        ...
        """)
    
    # GPU configuration
    parser.add_argument(
        '--gpus',
        type=str,
        nargs='+',
        required=True,
        help='GPUs to use: space-separated list (e.g., 0 1 2 3) or "all"'
    )
    
    # Multi-server configuration
    parser.add_argument(
        '--n-proc',
        type=int,
        default=None,
        help='Total number of processes across all servers. If not specified, defaults to number of GPUs on this server.'
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Offset to add to proc_idx for this server. Use this when running on multiple servers.'
    )
    
    # Core arguments (required)
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--mist-checkpoint', type=str, required=True, help='MIST checkpoint path')
    parser.add_argument('--dlm-checkpoint', type=str, required=True, help='DLM checkpoint path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    
    # Other common arguments
    parser.add_argument('--split', type=str, default='test', help='Data split')
    parser.add_argument('--max-spectra', type=int, help='Max spectra to process')
    parser.add_argument('--softmax-temp', type=float, default=1.0, help='Softmax temperature')
    parser.add_argument('--randomness', type=float, default=0.5, help='Randomness factor')
    parser.add_argument('--fp-threshold', type=float, help='Fingerprint threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-shared-cross-attention', action='store_true', help='Use shared cross-attention')
    
    # Token model arguments
    parser.add_argument('--token-model', type=str, help='Token count prediction model')
    parser.add_argument('--sigma-lambda', type=float, default=3.0, help='Variance multiplier for NGBoost')
    
    # ICEBERG arguments
    parser.add_argument('--iceberg-gen-ckpt', type=str, help='ICEBERG generation checkpoint')
    parser.add_argument('--iceberg-inten-ckpt', type=str, help='ICEBERG intensity checkpoint')
    parser.add_argument('--iceberg-python-path', type=str, help='ICEBERG Python executable path')
    parser.add_argument('--iceberg-batch-size', type=int, default=8, help='ICEBERG batch size')
    parser.add_argument('--iceberg-gpu', type=int, nargs='+', help='GPU(s) for ICEBERG')
    parser.add_argument('--iceberg-results-dir', type=str, default='/tmp/iceberg_scaling', help='ICEBERG results directory')
    
    # BUDDY formula prediction arguments
    parser.add_argument('--buddy-formula-path', type=str, default=None,
                        help='Path to BUDDY formula predictions TSV file. If provided, uses predicted formulas.')

    parser.add_argument('--incl-unknown-instrument', action='store_true', default=False,
                        help='Whether to include "Unknown" instrument type in ICEBERG predictions')
    
    # Scaling arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (B)')
    parser.add_argument('--num-unique-to-refine', type=int, default=16, help='Unique to refine (K)')
    parser.add_argument('--masks-per-molecule', type=int, default=4, help='Masks per molecule (M)')
    parser.add_argument('--num-rounds', type=int, default=3, help='Number of rounds (R)')
    parser.add_argument('--mask-prob', type=float, default=0.5, help='Mask probability')
    parser.add_argument('--top-k-halluc-peaks', type=int, default=3, help='Top K hallucinated peaks')
    parser.add_argument('--collision-energies', type=int, nargs='+', help='Collision energies')
    parser.add_argument('--nce', action='store_true', help='Use NCE (vs eV)')
    parser.add_argument('--masking-strategy', type=str, default='simple', help='Masking strategy')
    
    # Execution arguments
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--wait', action='store_true', help='Wait for all jobs to complete')
    parser.add_argument('--max-wait', type=int, default=604800, help='Max wait time in seconds')
    
    # Pass through any other arguments
    parser.add_argument('--extra-args', type=str, default='', help='Extra arguments to pass through')
    
    return parser.parse_args()


def parse_gpu_list(gpu_list: List[str]) -> List[int]:
    """Parse GPU list (list of strings from argparse nargs='+')."""
    # Handle special case: ['all'] means all GPUs
    if len(gpu_list) == 1 and gpu_list[0].lower() == 'all':
        # Try to detect number of GPUs
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                num_gpus = int(result.stdout.strip().split('\n')[0])
                return list(range(num_gpus))
        except Exception as e:
            print(f"Warning: Could not detect GPUs: {e}")
            return [0]
    
    # Parse list of string integers
    try:
        return sorted(list(set(int(x) for x in gpu_list)))
    except ValueError:
        print(f"Error: Invalid GPU specification: {gpu_list}")
        sys.exit(1)


def build_command(
    spec2mol_path: str,
    args: argparse.Namespace,
    gpu_id: int,
    proc_idx: int,
    n_proc: int,
    log_file: str,
) -> str:
    """Build the command to run."""
    cmd_parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python",
        spec2mol_path,
        f"--config {args.config}",
        f"--mist-checkpoint {args.mist_checkpoint}",
        f"--dlm-checkpoint {args.dlm_checkpoint}",
        f"--output-dir {args.output_dir}/{proc_idx}",
        f"--split {args.split}",
        f"--softmax-temp {args.softmax_temp}",
        f"--randomness {args.randomness}",
        f"--seed {args.seed}",
        f"--proc-idx {proc_idx}",
        f"--n-proc {n_proc}",
    ]
    
    if args.max_spectra:
        cmd_parts.append(f"--max-spectra {args.max_spectra}")
    
    if args.fp_threshold:
        cmd_parts.append(f"--fp-threshold {args.fp_threshold}")
    
    if args.use_shared_cross_attention:
        cmd_parts.append("--use-shared-cross-attention")
    
    if args.token_model:
        cmd_parts.append(f"--token-model {args.token_model}")
        cmd_parts.append(f"--sigma-lambda {args.sigma_lambda}")
    
    if args.iceberg_gen_ckpt:
        cmd_parts.append(f"--iceberg-gen-ckpt {args.iceberg_gen_ckpt}")
    
    if args.iceberg_inten_ckpt:
        cmd_parts.append(f"--iceberg-inten-ckpt {args.iceberg_inten_ckpt}")
    
    if args.iceberg_python_path:
        cmd_parts.append(f"--iceberg-python-path {args.iceberg_python_path}")
    
    cmd_parts.append(f"--iceberg-batch-size {args.iceberg_batch_size}")
    
    # For ICEBERG GPU: if user specified --iceberg-gpu, use that; otherwise default to the assigned GPU
    if args.iceberg_gpu is not None:
        cmd_parts.append(f"--iceberg-gpu {' '.join(map(str, args.iceberg_gpu))}")
    else:
        # Default to the GPU assigned to this subprocess
        cmd_parts.append(f"--iceberg-gpu {gpu_id}")
    
    if args.iceberg_results_dir:
        cmd_parts.append(f"--iceberg-results-dir {args.iceberg_results_dir}")
    
    # BUDDY formula predictions
    if args.buddy_formula_path:
        cmd_parts.append(f"--buddy-formula-path {args.buddy_formula_path}")

    if args.incl_unknown_instrument:
        cmd_parts.append(f"--incl-unknown-instrument")
    
    cmd_parts.append(f"--batch-size {args.batch_size}")
    cmd_parts.append(f"--num-unique-to-refine {args.num_unique_to_refine}")
    cmd_parts.append(f"--masks-per-molecule {args.masks_per_molecule}")
    cmd_parts.append(f"--num-rounds {args.num_rounds}")
    cmd_parts.append(f"--mask-prob {args.mask_prob}")
    cmd_parts.append(f"--top-k-halluc-peaks {args.top_k_halluc_peaks}")
    
    if args.collision_energies:
        cmd_parts.append(f"--collision-energies {' '.join(map(str, args.collision_energies))}")
    
    if args.nce:
        cmd_parts.append("--nce")
    
    cmd_parts.append(f"--masking-strategy {args.masking_strategy}")
    
    if args.verbose:
        cmd_parts.append("--verbose")
    
    if args.extra_args:
        cmd_parts.append(args.extra_args)
    
    # Redirect output to log file
    cmd = " ".join(cmd_parts) + f" > {log_file} 2>&1"
    
    return cmd


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse GPU list
    gpus = parse_gpu_list(args.gpus)
    n_gpus_this_server = len(gpus)
    
    # Determine total n_proc (across all servers) and offset
    n_proc = args.n_proc if args.n_proc is not None else n_gpus_this_server
    offset = args.offset
    
    print(f"Multi-GPU Launcher for spec2mol_scaling.py")
    print(f"=" * 60)
    print(f"GPUs on this server: {gpus}")
    print(f"Number of GPUs on this server: {n_gpus_this_server}")
    print(f"Total n_proc (across all servers): {n_proc}")
    print(f"Proc index offset: {offset}")
    print(f"Proc indices for this server: {list(range(offset, offset + n_gpus_this_server))}")
    print(f"Output base directory: {args.output_dir}")
    print(f"=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find spec2mol_scaling.py
    spec2mol_path = Path(__file__).parent / "scripts" / "spec2mol_scaling.py"
    if not spec2mol_path.exists():
        # Try alternative path
        spec2mol_path = Path("/path/to/scripts/spec2mol_scaling.py")
    
    if not spec2mol_path.exists():
        print(f"Error: Could not find spec2mol_scaling.py at {spec2mol_path}")
        sys.exit(1)
    
    print(f"Using spec2mol_scaling.py at: {spec2mol_path}")
    print()
    
    # Build and launch commands
    processes = []
    for local_idx, gpu_id in enumerate(gpus):
        # Calculate the global proc_idx (with offset for multi-server)
        proc_idx = offset + local_idx
        
        # Create subdirectory for this job
        job_dir = output_dir / str(proc_idx)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = job_dir / "job.log"
        
        # Build command
        cmd = build_command(
            str(spec2mol_path),
            args,
            gpu_id,
            proc_idx,
            n_proc,
            str(log_file),
        )
        
        print(f"Launching job {proc_idx} (local {local_idx}) on GPU {gpu_id}")
        print(f"  Output: {job_dir}")
        print(f"  Log: {log_file}")
        print(f"  Command: {cmd[:100]}...")
        print()
        
        # Launch process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append({
            'proc_idx': proc_idx,
            'gpu_id': gpu_id,
            'process': process,
            'log_file': log_file,
        })
    
    print(f"All {len(processes)} jobs launched!")
    print()
    
    if args.wait:
        print(f"Waiting for jobs to complete (timeout: {args.max_wait}s)...")
        print()
        
        start_time = time.time()
        completed = 0
        failed = 0
        
        while completed + failed < len(processes):
            for proc_info in processes:
                if proc_info['process'] is None:
                    continue
                
                returncode = proc_info['process'].poll()
                
                if returncode is not None:
                    # Process completed
                    if returncode == 0:
                        print(f"✓ Job {proc_info['proc_idx']} (GPU {proc_info['gpu_id']}) completed successfully")
                        completed += 1
                    else:
                        print(f"✗ Job {proc_info['proc_idx']} (GPU {proc_info['gpu_id']}) failed with code {returncode}")
                        print(f"  See {proc_info['log_file']} for details")
                        failed += 1
                    
                    proc_info['process'] = None
            
            elapsed = time.time() - start_time
            if elapsed > args.max_wait:
                print(f"Timeout reached ({args.max_wait}s)")
                break
            
            time.sleep(1)
        
        print()
        print(f"=" * 60)
        print(f"Results: {completed} completed, {failed} failed")
        print(f"=" * 60)
        
        if failed > 0:
            sys.exit(1)
    else:
        print("Jobs running in background. Use --wait to wait for completion.")
        print()
        print("To monitor progress:")
        for proc_info in processes:
            print(f"  tail -f {proc_info['log_file']}")


if __name__ == '__main__':
    main()
