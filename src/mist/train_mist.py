""" train_mist.py

Train spectrum transformer model

"""
import os
import yaml
import logging
import pickle
from pathlib import Path
import argparse
import pickle

import torch
import numpy as np
from tqdm import tqdm

from mist.models import mist_model
from mist.data import datasets, splitter, featurizers
from mist import utils, parsing


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parsing.add_base_args(parser)
    parsing.add_dataset_args(parser)
    parsing.add_train_args(parser)
    parsing.add_mist_args(parser)
    return parser.parse_args()


def run_training():
    """run_training."""
    # Get args
    args = get_args()
    kwargs = args.__dict__

    exp_name = kwargs.get("name", "default_exp")
    kwargs["save_dir"] = os.path.join(kwargs.get("save_dir"), exp_name)
    save_dir = Path(kwargs.get("save_dir"))
    
    utils.setup_train(save_dir, kwargs)

    # Get model class
    model_class = mist_model.MistNet
    kwargs["model"] = model_class.__name__
    kwargs["spec_features"] = model_class.spec_features()
    kwargs["mol_features"] = model_class.mol_features()
    kwargs["dataset_type"] = model_class.dataset_type()

    # Get featurizers
    paired_featurizer = featurizers.get_paired_featurizer(**kwargs)


    spectra_mol_pairs = datasets.get_paired_spectra(**kwargs)
    spectra_mol_pairs = list(zip(*spectra_mol_pairs))

    # Split data
    my_splitter = splitter.get_splitter(**kwargs)

    # Redefine splitter s.t. this splits three times and remove subsetting
    split_name, (train, val, test) = my_splitter.get_splits(spectra_mol_pairs)

    for name, _data in zip(["train", "val", "test"], [train, val, test]):
        logging.info(f"Len of {name}: {len(_data)}")

    train_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=train, featurizer=paired_featurizer, **kwargs
    )
    val_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=val, featurizer=paired_featurizer, **kwargs
        )
    test_dataset = datasets.SpectraMolDataset(
        spectra_mol_list=test, featurizer=paired_featurizer, **kwargs
    )
        
    spec_dataloader_module = datasets.SpecDataModule(
        train_dataset, val_dataset, test_dataset, **kwargs
    )

    # Create model
    model = model_class(**kwargs)

    logging.info(f"Starting fold: {split_name}")

    test_loss = model.train_model(
        spec_dataloader_module,
        log_name=exp_name,
        log_version=split_name,
        **kwargs,
    )

    # save model.spectra_encoder to kwargs["save_dir"]
    torch.save(model.spectra_encoder.state_dict(), save_dir / "spectra_encoder.pt")

if __name__ == "__main__":
    import time

    start_time = time.time()
    run_training()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
