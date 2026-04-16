# Token Length Prediction Models

This folder contains NGBoost models for predicting SAFE token sequence lengths from molecular formulae. These models are used by the FRIGID-base sampler to estimate appropriate sequence lengths during generation.

## Overview

FRIGID generates molecules as SAFE (Sequential Attachment-based Fragment Embedding) sequences. To initialize the generation process, the model needs to predict how many tokens the final molecule will contain. This prediction is based solely on the molecular formula (element counts: C, H, N, O, etc.).

**Key components:**
- **NGBoost regressors**: Probabilistic models that predict token count distributions
- **Features**: Element counts (Br, C, Cl, F, H, I, N, O, P, S)
- **Training data**: Either benchmark train sets or PubChem auxiliary data
- **Output**: Mean and uncertainty estimates for token counts

## Installation

These scripts require `optuna` for hyperparameter optimization, which is not included in the base FRIGID environment:

```bash
# Activate your FRIGID environment
conda activate frigid  # or your environment name

# Install optuna
pip install optuna
```

## Directory Structure

```
token_models/
├── README.md                               # This file
├── download_pubchem.sh                     # Download PubChem CID-SMILES database
├── generate_training_data.py               # Filter PubChem for auxiliary training data
├── token_utils.py                          # Shared utilities (feature extraction, optimization)
├── train_opt_canopus.py                    # Train model on CANOPUS test set
├── train_opt_msg.py                        # Train model on MSG test set
├── train_opt_canopus_pubchem.py            # Train model with PubChem auxiliary data (CANOPUS val)
├── train_opt_msg_pubchem.py                # Train model with PubChem auxiliary data (MSG val)
├── data/                                   # Generated training data and cached features
│   ├── canopus_train.csv                   # PubChem molecules matching CANOPUS formulae
│   ├── msg_train.csv                       # PubChem molecules matching MSG formulae
│   ├── *_features.csv                      # Cached feature matrices (auto-generated)
└── models/                                 # Trained model checkpoints
```

## Quick Start

### Option 1: Train on Benchmark Test Sets (Faster)

Use molecules from the CANOPUS or MSG test sets as training data:

```bash
cd token_models/

# Train CANOPUS model (uses molecules from ../data/canopus/)
python train_opt_canopus.py

# Train MSG model (uses molecules from ../data/msg/)
python train_opt_msg.py
```

**Pros:** Fast, no data download required
**Cons:** Smaller training set, potential overfitting to test distribution

### Option 2: Train with PubChem Auxiliary Data (Recommended)

Use a large auxiliary dataset from PubChem to train more generalizable models:

```bash
cd token_models/

# Step 1: Download PubChem database (~6 GB compressed, ~15 GB uncompressed)
bash download_pubchem.sh

# Step 2: Generate auxiliary training data
# This filters PubChem molecules to match formulae in test sets
# while excluding exact 2D InChI key matches (prevents data leakage)
python generate_training_data.py

# Step 3: Train models with auxiliary data
python train_opt_canopus_pubchem.py  # CANOPUS variant
python train_opt_msg_pubchem.py      # MSG variant
```

**Pros:** Larger training set, better generalization
**Cons:** Requires downloading and processing PubChem (~123M molecules)

## Detailed Workflow

### 1. Data Preparation (PubChem Method)

The `generate_training_data.py` script creates auxiliary training sets by:

1. Loading test set molecules from `../data/canopus/` and `../data/msg/`
2. Extracting unique formulae and InChI keys from test sets
3. Filtering PubChem to find molecules with:
   - Same molecular formula as test set molecules
   - Different InChI keys (prevents data leakage)
4. Saving filtered molecules to `data/canopus_train.csv` and `data/msg_train.csv`

### 2. Feature Extraction

All training scripts use `token_utils.py` to extract features:

- **Input:** SMILES strings
- **Processing:**
  - Convert SMILES → SAFE sequence
  - Tokenize with FRIGID's tokenizer
  - Count tokens (target variable)
  - Parse molecular formula → element counts (features)
- **Output:** Feature matrix with columns: `[Br, C, Cl, F, H, I, N, O, P, S, n_tokens]`

Features are cached to `data/*_features.csv` to speed up repeated runs.

### 3. Hyperparameter Optimization

Training scripts use Optuna to optimize NGBoost hyperparameters:

**Search space:**
- Base learner: Decision tree vs. Ridge regression
- For trees: max_depth (2-14), min_samples_leaf (1-50)
- For linear: Ridge alpha (0.01-10.0)
- NGBoost: n_estimators (100-2500), learning_rate (0.001-0.2)
- Regularization: minibatch_frac (0.1-1.0), col_sample (0.25-1.0)

**Objective:** Minimize negative log-likelihood (NLL) on validation set

**Configuration:**
- Trials: 1024 per run (configurable via `n_trials` parameter)
- Parallel jobs: 48-64 (configurable via `N_OPTUNA_JOBS`)
- Threading: Limited to 1 core per trial for efficient parallelization

### 4. Model Selection

During optimization:
- Each trial trains an NGBoost model with different hyperparameters
- Models are evaluated on the validation set (NLL metric)
- Best model is automatically saved to `models/best_ngboost_<DATASET>.joblib`
- Thread-safe saving ensures no race conditions during parallel optimization

### 5. Final Evaluation

After optimization completes, the best model is loaded and evaluated:

**Metrics:**
- **NLL**: Negative log-likelihood (primary optimization target)
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination

## Troubleshooting



### Import Errors

If you see `ModuleNotFoundError: No module named 'optuna'`:
- Install optuna: `pip install optuna`

If you see import errors for `dlm.utils.utils_data`:
- Ensure you're running from the `frigid/` directory or have proper PYTHONPATH
- Run: `export PYTHONPATH=/path/to/frigid:$PYTHONPATH`

Depending on what directory you run the scripts from, you may need to update the data paths, which use relative imports.

### Out of Memory

If training crashes with OOM errors:
- Reduce `N_OPTUNA_JOBS` in training scripts
- Reduce `MAX_WORKERS` in `generate_training_data.py`
- Reduce `CHUNK_SIZE` in `generate_training_data.py`

### Slow Performance

If optimization is very slow:
- Check that `OMP_NUM_THREADS=1` is set (prevents oversubscription)
- Verify `N_OPTUNA_JOBS` matches your CPU count
- Consider reducing `n_trials` for faster runs

## Model Files

Pretrained models are saved in `models/`:

| Model | Training Data | Use Case |
|-------|--------------|----------|
| `best_ngboost_CANOPUS.joblib` | CANOPUS test set | Baseline model for CANOPUS benchmark |
| `best_ngboost_MSG.joblib` | MSG test set | Baseline model for MSG benchmark |
| `best_ngboost_CANOPUS_pubchem.joblib` | PubChem (CANOPUS formulae) | Better generalization for CANOPUS-like molecules |
| `best_ngboost_MSG_pubchem.joblib` | PubChem (MSG formulae) | Better generalization for MSG-like molecules |

Each model file has a corresponding `*_features.json` file listing the feature order.