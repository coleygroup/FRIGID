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
FP2Mol Validation Callback for DLM training.

This callback performs FP2Mol-style validation during training:
- Loads molecules from a CSV file
- Extracts formulas and fingerprints
- Generates molecules conditioned on formula + fingerprint
- Computes metrics: formula match rate, FP match rate, mean/median tanimoto
"""

import os
import re
import warnings
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm

# Suppress RDKit warnings
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')


ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')


def normalize_formula(formula: str) -> str:
    """Normalize molecular formula to consistent format (sorted by element)."""
    if not formula:
        return ""
    
    # Parse formula into atom counts
    atoms = {}
    for element, count in ELEMENT_PATTERN.findall(formula):
        if element:
            count = int(count) if count else 1
            atoms[element] = atoms.get(element, 0) + count
    
    # Rebuild formula in sorted order
    parts = []
    # Carbon and Hydrogen first (Hill system)
    for elem in ['C', 'H']:
        if elem in atoms:
            if atoms[elem] == 1:
                parts.append(elem)
            else:
                parts.append(f"{elem}{atoms[elem]}")
    
    # Rest alphabetically
    for elem in sorted(atoms.keys()):
        if elem not in ['C', 'H']:
            if atoms[elem] == 1:
                parts.append(elem)
            else:
                parts.append(f"{elem}{atoms[elem]}")
    
    return ''.join(parts)


def load_token_model(model_path: str) -> Tuple[Any, List[str], bool]:
    """
    Load token count prediction model from path.
    
    Returns:
        Tuple of (model, features, is_ngboost)
    """
    if not model_path or not os.path.exists(model_path):
        return None, None, False
    
    try:
        import joblib
        model = joblib.load(model_path)
        
        # Try to load features
        features_path = model_path.replace('.joblib', '_features.json')
        features = None
        if os.path.exists(features_path):
            import json
            with open(features_path, 'r') as f:
                features = json.load(f)
        else:
            features = getattr(model, 'feature_names_in_', None)
            if features is not None:
                features = list(features)
        
        # Check if NGBoost model
        is_ngboost = hasattr(model, 'pred_dist') or 'ngboost' in type(model).__module__.lower()
        
        return model, features, is_ngboost
    except Exception as e:
        print(f"[ValidationCallback] Warning: Failed to load token model: {e}")
        return None, None, False


def predict_token_count(model, features: List[str], formula: str, is_ngboost: bool, sigma_lambda: float = 3.0) -> Tuple[Optional[int], Optional[float]]:
    """
    Predict token count from formula.
    
    Returns:
        Tuple of (predicted_length, predicted_sigma)
    """
    if model is None or features is None:
        return None, None
    
    # Parse formula into atom counts
    atoms = defaultdict(int)
    try:
        for element, count in ELEMENT_PATTERN.findall(formula):
            if element:
                count = int(count) if count else 1
                atoms[element] += count
    except Exception:
        return None, None
    
    # Build feature vector
    vector = [atoms.get(f, 0) for f in features]
    
    try:
        if is_ngboost:
            dist = model.pred_dist([vector])
            pred_mean = dist.mean()[0]
            pred_std = dist.std()[0]
            return int(round(pred_mean)), pred_std ** 2  # Return variance
        else:
            pred = model.predict([vector])[0]
            return int(round(pred)), None
    except Exception:
        return None, None


class FP2MolValidationCallback(Callback):
    """
    Callback for FP2Mol-style validation during training.
    
    Generates molecules conditioned on formula + fingerprint from a held-out
    CSV file and computes metrics.
    """
    
    def __init__(
        self,
        csv_path: str,
        n_mols: int = 100,
        n_samples: int = 10,
        n_steps: int = 5000,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        batch_size: int = 10,
        token_model_path: Optional[str] = None,
        sigma_lambda: float = 3.0,
        fp_bits: int = 4096,
        fp_radius: int = 2
    ):
        """
        Initialize the validation callback.
        
        Args:
            csv_path: Path to CSV file with 'smiles' column
            n_mols: Number of molecules to validate on (random subset)
            n_samples: Number of molecules to generate per target
            n_steps: Run validation every n_steps
            softmax_temp: Softmax temperature for sampling
            randomness: Randomness parameter for sampling
            batch_size: Batch size for generation
            token_model_path: Path to token count model (.joblib)
            sigma_lambda: NGBoost sigma multiplier
            fp_bits: Number of fingerprint bits
            fp_radius: Fingerprint radius
        """
        super().__init__()
        self.csv_path = csv_path
        self.n_mols = n_mols
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.softmax_temp = softmax_temp
        self.randomness = randomness
        self.batch_size = batch_size
        self.token_model_path = token_model_path
        self.sigma_lambda = sigma_lambda
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        
        # Will be loaded on first validation
        self._validation_data = None
        self._token_model = None
        self._token_features = None
        self._is_ngboost = False
        self._sampler = None
        self._last_validation_step = -1
    
    def _load_validation_data(self) -> pd.DataFrame:
        """Load and prepare validation data from CSV."""
        if self._validation_data is not None:
            return self._validation_data
        
        print(f"\n[ValidationCallback] Loading validation data from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Standardize column names
        if 'SMILES' in df.columns:
            df = df.rename(columns={'SMILES': 'smiles'})
        if 'InChI' in df.columns:
            df = df.rename(columns={'InChI': 'inchi'})
        if 'INCHI' in df.columns:
            df = df.rename(columns={'INCHI': 'inchi'})
        
        # Check which column exists
        has_smiles = 'smiles' in df.columns
        has_inchi = 'inchi' in df.columns
        
        if not has_smiles and not has_inchi:
            raise ValueError(f"CSV must have 'smiles' or 'inchi' column. Found: {df.columns.tolist()}")
        
        # Convert InChI to SMILES if needed
        if has_inchi and not has_smiles:
            print("[ValidationCallback] Converting InChI to SMILES...")
            df['smiles'] = df['inchi'].apply(self._inchi_to_smiles)
        
        # Sample n_mols with seed=42
        if len(df) > self.n_mols:
            df = df.sample(n=self.n_mols, random_state=42).reset_index(drop=True)
        
        # Extract formulas and fingerprints
        results = []
        for smiles in df['smiles']:
            try:
                if pd.isna(smiles) or not smiles:
                    continue
                
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                
                # Get formula
                formula = rdMolDescriptors.CalcMolFormula(mol)
                formula = normalize_formula(formula)
                
                # Get Morgan fingerprint (for conditioning)
                fp_bitvect = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                fp_array = np.zeros((self.fp_bits,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp_bitvect, fp_array)
                
                # Get RDKit topological fingerprint (for close/meaningful match metrics)
                fpgen = AllChem.GetRDKitFPGenerator()
                rdkit_fp = fpgen.GetFingerprint(mol)
                
                # Canonical SMILES
                canonical_smiles = Chem.MolToSmiles(mol)
                
                results.append({
                    'smiles': canonical_smiles,
                    'formula': formula,
                    'fingerprint_array': fp_array,
                    'fingerprint_bitvect': fp_bitvect,
                    'rdkit_fp': rdkit_fp
                })
            except Exception:
                continue
        
        self._validation_data = pd.DataFrame(results)
        print(f"[ValidationCallback] Loaded {len(self._validation_data)} valid molecules for validation")
        
        return self._validation_data
    
    def _inchi_to_smiles(self, inchi: str) -> Optional[str]:
        """Convert InChI to SMILES using RDKit."""
        try:
            if pd.isna(inchi) or not inchi:
                return None
            mol = Chem.MolFromInchi(str(inchi))
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None
    
    def _load_token_model(self):
        """Load token count prediction model if specified."""
        if self._token_model is not None or not self.token_model_path:
            return
        
        self._token_model, self._token_features, self._is_ngboost = load_token_model(
            self.token_model_path
        )
        if self._token_model is not None:
            model_type = "NGBoost" if self._is_ngboost else "sklearn"
            print(f"[ValidationCallback] Loaded token model ({model_type})")
    
    def _create_sampler(self, model: L.LightningModule):
        """Create a sampler from the current model state."""
        # Import here to avoid circular imports
        from dlm.sampler import Sampler
        import pickle
        import itertools
        
        # Create a lightweight sampler wrapper that uses the model directly
        class ModelSampler:
            """Lightweight sampler wrapper using the model directly."""
            
            def __init__(self, model: L.LightningModule):
                self.model = model
                self.tokenizer = model.tokenizer
                self.mdlm = model.mdlm
                self.device = model.device
                self.mask_index = model.mask_index
                self.bos_index = model.bos_index
                self.eos_index = model.eos_index
                self.pad_index = model.tokenizer.pad_token_id
                self.fingerprint_bits = model.fingerprint_bits
                self.max_position_embeddings = model.config.model.get('max_position_embeddings', 256)
                self.use_bracket_safe = model.config.training.get('use_bracket_safe', False)
                
                # Load length distribution for fallback
                try:
                    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
                    with open(os.path.join(root_dir, 'data/len.pk'), 'rb') as f:
                        self.seq_len_list = pickle.load(f)
                except FileNotFoundError:
                    self.seq_len_list = list(range(20, 100))
            
            def _prepare_fingerprint_batch(self, fingerprint, batch_size):
                """Prepare fingerprint for batch generation."""
                if fingerprint is None:
                    return None
                
                if torch.is_tensor(fingerprint):
                    fp = fingerprint.to(device=self.device, dtype=torch.float32)
                else:
                    fp = torch.as_tensor(fingerprint, device=self.device, dtype=torch.float32)
                
                if fp.dim() == 1:
                    fp = fp.unsqueeze(0)
                
                if fp.size(0) == 1 and batch_size > 1:
                    fp = fp.expand(batch_size, -1)
                
                return fp
            
            def _insert_mask(self, x, num_samples, min_add_len=18, target_lengths=None):
                """Insert mask tokens into the input sequence."""
                import random
                
                if target_lengths is None:
                    target_lengths = [
                        max(random.choice(self.seq_len_list), len(x[0]) + min_add_len)
                        for _ in range(num_samples)
                    ]
                elif len(target_lengths) == 1 and num_samples > 1:
                    target_lengths = target_lengths * num_samples
                
                x = x[0]  # Remove batch dimension
                current_len = len(x)
                x_new = []
                
                for i in range(num_samples):
                    target_len = target_lengths[i] if i < len(target_lengths) else target_lengths[-1]
                    add_seq_len = max(target_len - current_len, min_add_len)
                    add_seq_len = min(add_seq_len, self.max_position_embeddings - current_len)
                    add_seq_len = max(add_seq_len, min_add_len)
                    
                    x_new.append(torch.hstack([
                        x[:-1],
                        torch.full((add_seq_len,), self.mask_index),
                        x[-1:],
                    ]))
                
                pad_len = max([len(xx) for xx in x_new])
                x_new = [
                    torch.hstack([xx, torch.full((pad_len - len(xx),), self.pad_index)])
                    for xx in x_new
                ]
                return torch.stack(x_new)
            
            @torch.no_grad()
            def generate(self, x, softmax_temp=1.2, randomness=2, formula=None, fingerprint=None):
                """Generate molecules from masked input."""
                from dlm.utils.utils_chem import safe_to_smiles
                from dlm.utils.bracket_safe_converter import bracketsafe2safe
                
                x = x.to(self.device)
                num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
                attention_mask = x != self.pad_index
                
                if formula is not None and isinstance(formula, str):
                    formula = [formula] * x.size(0)
                
                fingerprint_tensor = None
                fingerprint_mask = None
                if fingerprint is not None:
                    fingerprint_tensor = self._prepare_fingerprint_batch(fingerprint, x.size(0))
                    fingerprint_mask = torch.ones(
                        fingerprint_tensor.size(0),
                        device=fingerprint_tensor.device,
                        dtype=torch.float32
                    )
                
                for i in range(num_steps):
                    logits = self.model(
                        x,
                        attention_mask,
                        formula=formula,
                        fingerprint=fingerprint_tensor,
                        fingerprint_mask=fingerprint_mask
                    )
                    x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)
                
                # Decode to SAFE strings
                samples = self.tokenizer.batch_decode(x, skip_special_tokens=True)
                
                # Convert to SMILES
                if self.use_bracket_safe:
                    samples = [safe_to_smiles(bracketsafe2safe(s), fix=True) for s in samples]
                else:
                    samples = [safe_to_smiles(s, fix=True) for s in samples]
                
                # Remove None and take the largest fragment
                samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
                return samples
            
            @torch.no_grad()
            def unified_conditioned_generation(
                self,
                formula=None,
                fingerprint=None,
                num_samples=1,
                softmax_temp=0.8,
                randomness=0.5,
                target_lengths=None,
                min_add_len=40
            ):
                """Generate molecules with unified conditioning."""
                original_training = self.model.training
                self.model.eval()
                
                # Prepare masked input
                x = torch.hstack([
                    torch.full((1, 1), self.bos_index),
                    torch.full((1, 1), self.eos_index)
                ])
                x = self._insert_mask(x, num_samples, min_add_len=min_add_len, target_lengths=target_lengths)
                x = x.to(self.device)
                
                samples = self.generate(
                    x,
                    softmax_temp,
                    randomness,
                    formula=formula,
                    fingerprint=fingerprint
                )
                
                if original_training:
                    self.model.train()
                
                return samples
        
        return ModelSampler(model)
    
    def _generate_for_target(
        self,
        sampler,
        formula: str,
        fingerprint_array: np.ndarray,
        num_samples: int
    ) -> List[str]:
        """Generate molecules for a single target."""
        import random
        
        all_samples = []
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        # Predict token count if model available
        predicted_len = None
        predicted_sigma = None
        if self._token_model is not None:
            predicted_len, predicted_sigma = predict_token_count(
                self._token_model, self._token_features, formula,
                self._is_ngboost, self.sigma_lambda
            )
        
        for _ in range(num_batches):
            current_batch_size = min(self.batch_size, num_samples - len(all_samples))
            if current_batch_size <= 0:
                break
            
            try:
                # Determine target lengths
                target_lengths = None
                if predicted_len:
                    if self._is_ngboost and predicted_sigma is not None:
                        std_dev = np.sqrt(predicted_sigma * self.sigma_lambda)
                        target_lengths = [
                            max(1, int(round(np.random.normal(predicted_len, std_dev))))
                            for _ in range(current_batch_size)
                        ]
                    else:
                        low = max(1, predicted_len - 3)
                        high = predicted_len + 3
                        target_lengths = [
                            random.randint(low, high)
                            for _ in range(current_batch_size)
                        ]
                
                samples = sampler.unified_conditioned_generation(
                    formula=formula,
                    fingerprint=fingerprint_array,
                    num_samples=current_batch_size,
                    softmax_temp=self.softmax_temp,
                    randomness=self.randomness,
                    target_lengths=target_lengths,
                    min_add_len=2
                )
                all_samples.extend(samples)
            except Exception as e:
                print(f"[ValidationCallback] Warning: Batch generation failed: {e}")
                all_samples.extend([None] * current_batch_size)
        
        return all_samples[:num_samples]
    
    def _evaluate_generation(
        self,
        target_formula: str,
        target_fp_bitvect,
        target_fp_array: np.ndarray,
        target_rdkit_fp,
        generated_smiles: List[str]
    ) -> Dict[str, Any]:
        """Evaluate generated molecules against target."""
        # RDKit fingerprint generator for close/meaningful match
        fpgen = AllChem.GetRDKitFPGenerator()
        
        results = {
            'valid_molecules': 0,
            'formula_matches': 0,
            'fingerprint_exact_matches': 0,
            'tanimoto_similarities': [],
            'rdkit_tanimoto_similarities': [],  # For close/meaningful match
            'best_rdkit_tanimoto': 0.0  # Track best RDKit tanimoto for this target
        }
        
        for smiles in generated_smiles:
            if not smiles or (isinstance(smiles, float) and pd.isna(smiles)):
                continue
            
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                
                results['valid_molecules'] += 1
                
                # Check formula
                gen_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(mol))
                if gen_formula == target_formula:
                    results['formula_matches'] += 1
                
                # Check Morgan fingerprint
                gen_fp_bitvect = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
                gen_fp_array = np.zeros((self.fp_bits,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(gen_fp_bitvect, gen_fp_array)
                
                # Exact match
                if np.array_equal(gen_fp_array, target_fp_array):
                    results['fingerprint_exact_matches'] += 1
                
                # Morgan Tanimoto similarity
                tanimoto = DataStructs.TanimotoSimilarity(target_fp_bitvect, gen_fp_bitvect)
                results['tanimoto_similarities'].append(float(tanimoto))
                
                # RDKit topological fingerprint Tanimoto (for close/meaningful match)
                gen_rdkit_fp = fpgen.GetFingerprint(mol)
                rdkit_tanimoto = DataStructs.TanimotoSimilarity(target_rdkit_fp, gen_rdkit_fp)
                results['rdkit_tanimoto_similarities'].append(float(rdkit_tanimoto))
                results['best_rdkit_tanimoto'] = max(results['best_rdkit_tanimoto'], rdkit_tanimoto)
            
            except Exception:
                continue
        
        return results
    
    def _run_validation(self, trainer: L.Trainer, pl_module: L.LightningModule) -> Dict[str, float]:
        """Run FP2Mol validation and return metrics."""
        # Load data and token model on first run
        df = self._load_validation_data()
        self._load_token_model()
        
        # Create sampler from current model
        sampler = self._create_sampler(pl_module)
        
        # Put model in eval mode
        pl_module.eval()
        
        # Get distributed info
        world_size = trainer.world_size
        rank = trainer.global_rank
        
        # Split data across GPUs
        total_targets = len(df)
        targets_per_rank = (total_targets + world_size - 1) // world_size
        start_idx = rank * targets_per_rank
        end_idx = min(start_idx + targets_per_rank, total_targets)
        
        # Get this rank's subset
        df_subset = df.iloc[start_idx:end_idx]
        
        # Aggregate metrics for this rank
        total_valid = 0
        total_formula_matches = 0
        total_fp_exact_matches = 0
        all_tanimoto = []
        all_rdkit_tanimoto = []
        total_close_matches = 0  # RDKit tanimoto >= 0.675
        total_meaningful_matches = 0  # RDKit tanimoto >= 0.4
                
        # Use tqdm only on rank 0
        iterator = df_subset.iterrows()
        if rank == 0:
            iterator = tqdm(iterator, total=len(df_subset), desc="Validation")
        
        for idx, row in iterator:
            target_formula = row['formula']
            target_fp_array = row['fingerprint_array']
            target_fp_bitvect = row['fingerprint_bitvect']
            target_rdkit_fp = row['rdkit_fp']
            
            # Generate molecules
            generated = self._generate_for_target(
                sampler,
                target_formula,
                target_fp_array,
                self.n_samples
            )
            
            # Evaluate
            eval_results = self._evaluate_generation(
                target_formula,
                target_fp_bitvect,
                target_fp_array,
                target_rdkit_fp,
                generated
            )
            
            total_valid += eval_results['valid_molecules']
            total_formula_matches += eval_results['formula_matches']
            total_fp_exact_matches += eval_results['fingerprint_exact_matches']
            all_tanimoto.extend(eval_results['tanimoto_similarities'])
            all_rdkit_tanimoto.extend(eval_results['rdkit_tanimoto_similarities'])
            
            # Close match: best RDKit tanimoto >= 0.675
            if eval_results['best_rdkit_tanimoto'] >= 0.675:
                total_close_matches += 1
            # Meaningful match: best RDKit tanimoto >= 0.4
            if eval_results['best_rdkit_tanimoto'] >= 0.4:
                total_meaningful_matches += 1
        
        # Put model back in training mode
        pl_module.train()
        
        # Gather results from all ranks if distributed
        if world_size > 1:
            import torch.distributed as dist
            
            # Gather scalar counts
            device = pl_module.device
            local_counts = torch.tensor([
                total_valid, 
                total_formula_matches, 
                total_fp_exact_matches,
                total_close_matches,
                total_meaningful_matches,
                len(df_subset)  # targets processed by this rank
            ], dtype=torch.float64, device=device)
            
            # All-reduce to sum across ranks
            dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
            
            total_valid = int(local_counts[0].item())
            total_formula_matches = int(local_counts[1].item())
            total_fp_exact_matches = int(local_counts[2].item())
            total_close_matches = int(local_counts[3].item())
            total_meaningful_matches = int(local_counts[4].item())
            
            # Gather tanimoto lists - use all_gather for variable-length tensors
            # Convert to tensors
            local_tanimoto = torch.tensor(all_tanimoto, dtype=torch.float64, device=device) if all_tanimoto else torch.tensor([], dtype=torch.float64, device=device)
            local_rdkit_tanimoto = torch.tensor(all_rdkit_tanimoto, dtype=torch.float64, device=device) if all_rdkit_tanimoto else torch.tensor([], dtype=torch.float64, device=device)
            
            # Gather sizes first
            local_size = torch.tensor([len(local_tanimoto)], dtype=torch.long, device=device)
            all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            
            local_rdkit_size = torch.tensor([len(local_rdkit_tanimoto)], dtype=torch.long, device=device)
            all_rdkit_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_rdkit_sizes, local_rdkit_size)
            
            # Pad to max size and gather
            max_size = max(s.item() for s in all_sizes)
            if max_size > 0:
                padded_tanimoto = torch.zeros(max_size, dtype=torch.float64, device=device)
                padded_tanimoto[:len(local_tanimoto)] = local_tanimoto
                gathered_tanimoto = [torch.zeros(max_size, dtype=torch.float64, device=device) for _ in range(world_size)]
                dist.all_gather(gathered_tanimoto, padded_tanimoto)
                
                # Unpad and combine
                all_tanimoto = []
                for i, t in enumerate(gathered_tanimoto):
                    size = all_sizes[i].item()
                    all_tanimoto.extend(t[:size].cpu().tolist())
            
            max_rdkit_size = max(s.item() for s in all_rdkit_sizes)
            if max_rdkit_size > 0:
                padded_rdkit = torch.zeros(max_rdkit_size, dtype=torch.float64, device=device)
                padded_rdkit[:len(local_rdkit_tanimoto)] = local_rdkit_tanimoto
                gathered_rdkit = [torch.zeros(max_rdkit_size, dtype=torch.float64, device=device) for _ in range(world_size)]
                dist.all_gather(gathered_rdkit, padded_rdkit)
                
                # Unpad and combine
                all_rdkit_tanimoto = []
                for i, t in enumerate(gathered_rdkit):
                    size = all_rdkit_sizes[i].item()
                    all_rdkit_tanimoto.extend(t[:size].cpu().tolist())
        
        # Compute final metrics (only meaningful on rank 0, but compute on all for simplicity)
        total_generated = total_targets * self.n_samples
        metrics = {
            'val/formula_match_rate': 100.0 * total_formula_matches / total_valid if total_valid > 0 else 0.0,
            'val/fp_exact_match_rate': 100.0 * total_fp_exact_matches / total_valid if total_valid > 0 else 0.0,
            'val/mean_tanimoto': 100.0 * np.mean(all_tanimoto) if all_tanimoto else 0.0,
            'val/median_tanimoto': 100.0 * np.median(all_tanimoto) if all_tanimoto else 0.0,
            'val/validity_rate': 100.0 * total_valid / total_generated if total_generated > 0 else 0.0,
            # Close match: at least one generated molecule has RDKit tanimoto >= 0.675
            'val/close_match_rate': 100.0 * total_close_matches / total_targets if total_targets > 0 else 0.0,
            # Meaningful match: at least one generated molecule has RDKit tanimoto >= 0.4
            'val/meaningful_match_rate': 100.0 * total_meaningful_matches / total_targets if total_targets > 0 else 0.0
        }
        
        return metrics
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx
    ):
        """Check if validation should run after this step."""
        current_step = trainer.global_step
        
        # Check if it's time for validation (all ranks must participate for distributed validation)
        if current_step > 0 and current_step % self.n_steps == 0 and current_step != self._last_validation_step:
            self._last_validation_step = current_step
            
            try:
                metrics = self._run_validation(trainer, pl_module)
                
                # Log metrics only on rank 0
                if trainer.global_rank == 0:
                    for name, value in metrics.items():
                        pl_module.log(name, value, prog_bar=False, sync_dist=False, rank_zero_only=True)
                
                
            except Exception as e:
                if trainer.global_rank == 0:
                    print(f"\n[ValidationCallback] Warning: Validation failed with error: {e}")
                    import traceback
                    traceback.print_exc()
