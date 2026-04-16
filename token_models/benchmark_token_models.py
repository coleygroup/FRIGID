#!/usr/bin/env python3
"""
Benchmark token length prediction models on test sets.

Evaluates multiple NGBoost models and reports MAE, MSE, R^2, and NLL metrics.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import token_utils

# ====== CONFIGURATION ======

# Dataset selection: 'MSG' or 'CANOPUS'
DATASET = 'MSG'

# Model configurations: (name, checkpoint_path)
MODELS = [
    ("MSG Baseline", "models/best_ngboost_MSG.joblib"),
    ("MSG + PubChem", "models/best_ngboost_MSG_pubchem.joblib"),
]


# ===========================

def load_test_data(dataset, features):
    """Load test set features for the specified dataset."""
    cache_path = f'data/{dataset.lower()}_test_features.csv'

    if os.path.exists(cache_path):
        print(f"Loading cached test features from {cache_path}")
        test_df = pd.read_csv(cache_path)
    else:
        print(f"Processing {dataset} test set...")

        if dataset == 'MSG':
            labels = pd.read_csv("../data/msg/labels.tsv", sep="\t")
            splits = pd.read_csv("../data/msg/split.tsv", sep="\t")
            test_names = splits[splits['split'] == 'test'].name
        elif dataset == 'CANOPUS':
            labels = pd.read_csv("../data/canopus/labels.tsv", sep="\t")
            splits = pd.read_csv("../data/canopus/splits/canopus_hplus_100_0.tsv", sep="\t")
            test_names = splits[splits['split'] == 'test'].name
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        test_smiles = labels[labels['spec'].isin(test_names)]['smiles']
        test_df = token_utils.process_smiles_to_features(test_smiles, features)
        test_df.to_csv(cache_path, index=False)
        print(f"Saved test features to {cache_path}")

    X_test = test_df[features]
    y_test = test_df['n_tokens']

    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics for a model."""
    # Get predictions
    dists = model.pred_dist(X_test)
    y_pred = dists.loc

    # Compute metrics
    nll = -dists.logpdf(y_test.values).mean()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {'NLL': nll, 'MSE': mse, 'MAE': mae, 'R2': r2}


def print_results_table(results):
    """Print a nicely formatted results table."""
    # Determine column widths
    name_width = max(len("Model"), max(len(r['name']) for r in results))

    # Print header
    print("\n" + "=" * (name_width + 55))
    print(f"{'Model':<{name_width}}  {'MAE':>8}  {'MSE':>8}  {'R²':>8}  {'NLL':>8}")
    print("=" * (name_width + 55))

    # Print results
    for result in results:
        metrics = result['metrics']
        print(f"{result['name']:<{name_width}}  "
              f"{metrics['MAE']:>8.4f}  "
              f"{metrics['MSE']:>8.4f}  "
              f"{metrics['R2']:>8.4f}  "
              f"{metrics['NLL']:>8.4f}")

    print("=" * (name_width + 55))
    print()


def main():
    print(f"\n{'='*60}")
    print(f"Token Length Model Benchmark - {DATASET} Test Set")
    print(f"{'='*60}\n")

    # Load test data
    print(f"Loading {DATASET} test data...")
    X_test, y_test = load_test_data(DATASET)
    print(f"Test set size: {len(X_test)} samples\n")

    # Evaluate all models
    results = []

    for name, checkpoint_path in MODELS:
        feat_path = checkpoint_path.replace('.joblib', '_features.json')
        feat = None
        if os.path.exists(feat_path):
            with open(feat_path, 'r') as f:
                feat = json.load(f)

        X_test, y_test = load_test_data(DATASET, features=feat)
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"Evaluating {name}...")

        # Load model
        model = joblib.load(checkpoint_path)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            'name': name,
            'metrics': metrics
        })

    # Print results table
    if results:
        print_results_table(results)
    else:
        print("\n⚠ No models were successfully evaluated.\n")


if __name__ == "__main__":
    main()
