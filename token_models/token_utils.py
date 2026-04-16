import re
import os
import json
import threading
import multiprocessing as mp
from functools import partial
from collections import defaultdict

import optuna
import joblib
import safe as sf
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from dlm.utils.utils_data import get_tokenizer


HF_CACHE_DIR = '../data/safes/'
tokenizer = get_tokenizer(HF_CACHE_DIR)
converter = sf.SAFEConverter(ignore_stereo=True)

def get_token_count(smiles):
    try:
        safe_str = converter.encoder(smiles, allow_empty=True)
        if safe_str is None: return None
        tokens = tokenizer.encode(safe_str)
        return len(tokens)
    except Exception:
        return None

def parse_formula(formula):
    atoms = defaultdict(int)
    if pd.isna(formula): return atoms
    pattern = r"([A-Z][a-z]?)(\d*)"
    try:
        for element, count in re.findall(pattern, formula):
            count = int(count) if count else 1
            atoms[element] += count
    except Exception:
        pass
    return atoms

def process_single_smiles(smiles, features_list):
    """Process a single SMILES string - worker function for parallel processing."""
    if pd.isna(smiles):
        print("Encountered NaN SMILES.")
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            return None
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        try:
            safe_str = converter.encoder(smiles, allow_empty=True)
            if safe_str is None:
                print("Could not convert SMILES to SAFE format.")
                return None
            tokens = tokenizer.encode(safe_str)
            n_tokens = len(tokens)
        except Exception as e:
            print(f"Error tokenizing SMILES {smiles}: {e}")
            return None
        
        atom_counts = parse_formula(formula)
        if not atom_counts:
            print(f"Could not parse molecular formula: {formula}")
            return None
        entry = {'smiles': smiles, 'n_tokens': n_tokens}
        for feature in features_list:
            entry[feature] = atom_counts.get(feature, 0)
        return entry
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

def init_worker():
    """Initialize worker process with its own tokenizer and converter."""
    global tokenizer, converter
    tokenizer = get_tokenizer(HF_CACHE_DIR)
    converter = sf.SAFEConverter(ignore_stereo=True)

def process_smiles_to_features(smiles_series, features_list, n_workers=None):
    """Convert SMILES to feature vectors for prediction - parallelized version"""
    if n_workers is None:
        n_workers = max(1, int(mp.cpu_count() * 0.75))
    
    smiles_list = list(smiles_series)
    
    with mp.Pool(n_workers, initializer=init_worker) as pool:
        worker_fn = partial(process_single_smiles, features_list=features_list)
        results = list(tqdm(
            pool.imap(worker_fn, smiles_list, chunksize=100),
            total=len(smiles_list),
            desc="Processing SMILES"
        ))
    
    processed_data = [r for r in results if r is not None]
    return pd.DataFrame(processed_data)

class ModelSaver:
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.best_score = float('inf')
        self.lock = threading.Lock()

    def check_and_save(self, model, score, trial_number, features):
        with self.lock:
            if score < self.best_score:
                print(f"    [TRIAL {trial_number}] New Best Test NLL! {score:.4f} (Prev: {self.best_score:.4f}) -> Saving Model...")
                self.best_score = score
                
                # Save Model
                model_filename = os.path.join(self.output_dir, f"best_ngboost_{self.dataset_name}.joblib")
                joblib.dump(model, model_filename)
                
                # Save Features
                features_filename = os.path.join(self.output_dir, f"best_ngboost_{self.dataset_name}_features.json")
                with open(features_filename, 'w') as f:
                    json.dump(features, f)
                return True
            return False
        
def objective(trial, X_train, y_train, X_test, y_test, feature_cols, sub_feature_cols, saver):
    # Feature Subset Selection (optional)
    if sub_feature_cols is not None:
        use_full_features = trial.suggest_categorical('use_full_features', [True, False])
        if use_full_features:
            selected_features = feature_cols
        else:
            selected_features = sub_feature_cols

        # Filter training and test data to selected features
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]
    else:
        selected_features = feature_cols
        X_train_subset = X_train
        X_test_subset = X_test

    # Base Learner Type
    base_learner_type = trial.suggest_categorical('base_learner_type', ['tree', 'linear'])

    if base_learner_type == 'tree':
        max_depth = trial.suggest_int('tree_max_depth', 2, 14)
        min_samples_leaf = trial.suggest_int('tree_min_samples_leaf', 1, 50)
        base_learner = DecisionTreeRegressor(
            criterion='friedman_mse',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        n_estimators = trial.suggest_int('n_estimators', 100, 2500, step=25)
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.2, log=True)

    else: # Linear
        alpha = trial.suggest_float('ridge_alpha', 0.01, 10.0, log=True)
        base_learner = Ridge(alpha=alpha, random_state=42)
        n_estimators = trial.suggest_int('n_estimators', 100, 2500, step=25)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)

    minibatch_frac = trial.suggest_float('minibatch_frac', 0.1, 1.0)
    col_sample = trial.suggest_float('col_sample', 0.25, 1.0)

    ngb = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=base_learner,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        minibatch_frac=minibatch_frac,
        col_sample=col_sample,
        verbose=False,
        random_state=42
    )

    ngb.fit(X_train_subset, y_train)

    dists = ngb.pred_dist(X_test_subset)
    nll = -dists.logpdf(y_test.values).mean()

    saver.check_and_save(ngb, nll, trial.number, selected_features)

    return nll

def run_optimization(X_train, y_train, X_test, y_test, feature_cols, dataset_name, output_dir, n_jobs, n_trials=50, sub_feature_cols=None):
    saver = ModelSaver(dataset_name, output_dir)

    print(f"\nStarting Parallel Optimization for {dataset_name} (Jobs: {n_jobs}, Trials: {n_trials})...")
    if sub_feature_cols is not None:
        print(f"  Full features: {feature_cols}")
        print(f"  Subset features: {sub_feature_cols}")
        print(f"  Optuna will search over both feature sets.")
    else:
        print(f"  Features: {feature_cols}")

    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test, feature_cols, sub_feature_cols, saver),
        n_trials=n_trials,
        n_jobs=n_jobs
    )

    print(f"\nOptimization Complete for {dataset_name}!")
    print(f"Best Test NLL: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")

    # Final Confirmation Load
    best_model_path = os.path.join(output_dir, f"best_ngboost_{dataset_name}.joblib")
    print(f"Loading best model from {best_model_path}...")
    best_model = joblib.load(best_model_path)

    # Load the feature list used by the best model
    features_filename = os.path.join(output_dir, f"best_ngboost_{dataset_name}_features.json")
    with open(features_filename, 'r') as f:
        best_features = json.load(f)

    print(f"Best model uses features: {best_features}")

    # Evaluate using the correct feature set
    X_test_subset = X_test[best_features]

    dists_test = best_model.pred_dist(X_test_subset)
    y_pred_test = dists_test.loc

    test_nll = -dists_test.logpdf(y_test.values).mean()
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n>>> FINAL OPTIMIZED RESULTS ({dataset_name}) <<<")
    print(f"NLL: {test_nll:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R2:  {test_r2:.4f}")

    return study
