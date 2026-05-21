"""misc_utils.py"""
from pathlib import Path
import sys
import copy
import logging
from typing import List, Iterable, Iterator
from itertools import islice

import torch
import numpy as np
import matplotlib.pyplot as plt

from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

import yaml
import pytorch_lightning as pl

from pytorch_lightning.utilities.rank_zero import rank_zero_only
try:
    from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
except ImportError:
    from pytorch_lightning.loggers import Logger as LightningLoggerBase
    from pytorch_lightning.loggers.logger import rank_zero_experiment


def compute_similarities_continuous(pred_fps, target_fps):
    """Compute Tanimoto and Cosine similarities using continuous values
    
    Uses continuous Tanimoto and Cosine similarity for probabilistic predictions.
    Returns: (mean_tanimoto, mean_cosine, tanimoto_list, cosine_list)
    """
    tanimoto_sims = []
    cosine_sims = []
            
    for pred_fp, target_fp in zip(pred_fps, target_fps):
        # Tanimoto for continuous: sum(min(a,b)) / sum(max(a,b))
        tanimoto_sim = np.sum(np.minimum(pred_fp, target_fp)) / (np.sum(np.maximum(pred_fp, target_fp)) + 1e-10)
        
        # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
        cosine_sim = np.dot(pred_fp, target_fp) / (np.linalg.norm(pred_fp) * np.linalg.norm(target_fp) + 1e-10)
            
        tanimoto_sims.append(tanimoto_sim)
        cosine_sims.append(cosine_sim)
                
    return np.mean(tanimoto_sims), np.mean(cosine_sims), tanimoto_sims, cosine_sims

def compute_similarities_binary(pred_fps, target_fps, fp_size, threshold=0.187):
    """Compute Tanimoto and Cosine similarities using RDKit with binary fingerprints
    
    Binarizes predictions using the specified threshold before computing similarities.
    Returns: (mean_tanimoto, mean_cosine, tanimoto_list, cosine_list)
    """
    tanimoto_sims = []
    cosine_sims = []
            
    for pred_fp, target_fp in zip(pred_fps, target_fps):
        # Binarize predictions with threshold
        pred_binary = (pred_fp > threshold).astype(np.uint8)
        target_binary = target_fp.astype(np.uint8)
        
        # Convert to RDKit ExplicitBitVect
        pred_bv = ExplicitBitVect(fp_size)
        target_bv = ExplicitBitVect(fp_size)
            
        # Set bits
        for i in range(len(pred_binary)):
            if pred_binary[i]:
                pred_bv.SetBit(i)
        for i in range(len(target_binary)):
            if target_binary[i]:
                target_bv.SetBit(i)
            
        # Compute similarities using RDKit
        tanimoto_sim = DataStructs.TanimotoSimilarity(pred_bv, target_bv)
        cosine_sim = DataStructs.CosineSimilarity(pred_bv, target_bv)
            
        tanimoto_sims.append(tanimoto_sim)
        cosine_sims.append(cosine_sim)
            
    return np.mean(tanimoto_sims), np.mean(cosine_sims), tanimoto_sims, cosine_sims

def plot_similarity_histogram(similarities, title, save_path, metric_name="Similarity"):
    """Plot and save histogram of similarity values"""
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(similarities)
    median_val = np.median(similarities)
    std_val = np.std(similarities)
    stats_text = f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class ConsoleLogger(LightningLoggerBase):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


def setup_train(save_dir: Path, kwargs):
    """setup.

    Set seed, define logger, dump args, & update kwargs for debug

    """
    # Seed everything
    #pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Define default root dir
    setup_logger(save_dir, debug=kwargs["debug"])

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(yaml_args)

    # Dump args
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Hard code max_count for debugging!
    if kwargs.get("debug") == "test":
        kwargs["max_epochs"] = 3
        kwargs["max_count"] = 100
    elif kwargs.get("debug") == "test_overfit":
        kwargs["min_epochs"] = 1000
        kwargs["max_epochs"] = None
        kwargs["max_count"] = 100
    else:
        kwargs["max_count"] = None
        pass


def setup_logger(save_dir, log_name="output.log", debug=False):
    """Create output directory"""

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_file = save_dir / log_name

    if debug is not False:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode="trunc")
    return tuple(reversed(out))


def np_clamp(x, _min=-100):
    x = np.ones_like(x) * x
    x[x <= _min] = _min
    return x


def clamped_log_np(x, _min=-100):
    res = np.log(x)
    return np_clamp(res, _min=_min)


def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size""" ""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


def pad_packed_tensor(input, lengths, value):
    """pad_packed_tensor"""
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    # Initialize a tensor with an index for every value in the array
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    # Row shifts
    row_shifts = torch.cumsum(max_len - lengths, 0)

    # Calculate shifts for second row, third row... nth row (not the n+1th row)
    # Expand this out to match the shape of all entries after the first row
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    # Add this to the list of inds _after_ the first row
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0] :] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def reverse_packed_tensor(packed_tensor, lengths):
    """reverse_packed_tensor.

    Args:
        packed tensor: Batch x  length x feat_dim
        lengths : Batch
    Return:
        [batch,length] x feat_dim
    """
    device = packed_tensor.device
    batch_size, batch_len, feat_dim = packed_tensor.shape
    max_length = torch.arange(batch_len).to(device)
    indices = max_length.unsqueeze(0).expand(batch_size, batch_len)
    bool_mask = indices < lengths.unsqueeze(1)
    output = packed_tensor[bool_mask]
    return output


def unpack_bits(vec, num_bits):
    return np.unpackbits(vec, axis=-1)[..., -num_bits:]