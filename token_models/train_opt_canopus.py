#!/usr/bin/env python3
"""
Optimize NGBoost models using CANOPUS training set and evaluate on CANOPUS val sets.
"""

import os
import json

import pandas as pd

import token_utils

# --- 0. THREADING CONFIGURATION ---
# IMPORTANT: Limit each individual job to 1 core so we can run 196 jobs in parallel
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Number of parallel Optuna jobs
N_OPTUNA_JOBS = 48

# Path to save optimization results
OUTPUT_DIR = 'models/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Paths for cached features
CANOPUS_TRAIN_FEATURES_PATH = 'data/canopus_train_features.csv'
CANOPUS_VAL_FEATURES_PATH = 'data/canopus_val_features.csv'

SUB_FEATURES = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S"]
FEATURES = ["As", "B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Se", "Si"]

if __name__ == "__main__":
    if os.path.exists(CANOPUS_TRAIN_FEATURES_PATH):
        print(f"Loading cached training features from {CANOPUS_TRAIN_FEATURES_PATH}")
        train_df = pd.read_csv(CANOPUS_TRAIN_FEATURES_PATH)
    else:
        print(f"Processing CANOPUS training set...")
        labels = pd.read_csv("../data/canopus/labels.tsv", sep="\t")
        splits = pd.read_csv("../data/canopus/splits/canopus_hplus_100_0.tsv", sep="\t")
        train_names = splits[splits['split'] == 'train'].name
        train_smiles = labels[labels['spec'].isin(train_names)]['smiles']
        train_df = token_utils.process_smiles_to_features(train_smiles, FEATURES)
        train_df.to_csv(CANOPUS_TRAIN_FEATURES_PATH, index=False)
        print(f"Saved training features to {CANOPUS_TRAIN_FEATURES_PATH}")

    if os.path.exists(CANOPUS_VAL_FEATURES_PATH):
        print(f"Loading cached val features from {CANOPUS_VAL_FEATURES_PATH}")
        val_df = pd.read_csv(CANOPUS_VAL_FEATURES_PATH)
    else:
        print(f"Processing CANOPUS val set...")
        labels = pd.read_csv("../data/canopus/labels.tsv", sep="\t")
        splits = pd.read_csv("../data/canopus/splits/canopus_hplus_100_0.tsv", sep="\t")
        val_names = splits[splits['split'] == 'val'].name
        val_smiles = labels[labels['spec'].isin(val_names)]['smiles']
        val_df = token_utils.process_smiles_to_features(val_smiles, FEATURES)
        val_df.to_csv(CANOPUS_VAL_FEATURES_PATH, index=False)
        print(f"Saved val features to {CANOPUS_VAL_FEATURES_PATH}")

    X_train = train_df[FEATURES]
    y_train = train_df['n_tokens']
    X_test = val_df[FEATURES]
    y_test = val_df['n_tokens']

    token_utils.run_optimization(X_train, y_train, X_test, y_test, FEATURES, "CANOPUS", OUTPUT_DIR, N_OPTUNA_JOBS, n_trials=1024, sub_feature_cols=SUB_FEATURES)