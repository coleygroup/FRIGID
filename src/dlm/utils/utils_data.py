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


import os

os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
import datasets
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from safe.tokenizer import SAFETokenizer
from rdkit import RDLogger
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from dlm.utils.bracket_safe_converter import safe2bracketsafe
from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
RDLogger.DisableLog('rdApp.*')


ROOT_DIR = os.getcwd()


def _get_rank_and_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size


def get_last_checkpoint(save_dir):
    if not save_dir or not os.path.exists(save_dir):
        return None

    candidates = []
    for filename in os.listdir(save_dir):
        if not filename.endswith('.ckpt'):
            continue
        stem = filename[:-5]
        try:
            step = int(stem)
        except ValueError:
            continue
        candidates.append((step, filename))

    if not candidates:
        return None

    _, last_filename = max(candidates, key=lambda x: x[0])
    return os.path.join(save_dir, last_filename)
    

def get_tokenizer(hf_cache_dir=None):
    try:
        tk = SAFETokenizer.from_pretrained(
            'datamol-io/safe-gpt',
            cache_dir=hf_cache_dir,
            local_files_only=True  # Force using only local cached files, if throws error, set this to False and then re-run
        ).get_pretrained()
    except Exception as e:
        print("Error loading tokenizer from HF cache:", e)
        tk = SAFETokenizer.from_pretrained(
            'datamol-io/safe-gpt',
            cache_dir=hf_cache_dir,
            local_files_only=False
        ).get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk


class Collator:
    def __init__(self, config):
        self.tokenizer = get_tokenizer(config.data.get('hf_cache_dir', None))
        self.remove_stereo = config.data.get('remove_stereo', False)
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')
        self.use_formula_conditioning = config.model.get('use_formula_conditioning', False)
        self.use_fingerprint_conditioning = config.model.get('use_fingerprint_conditioning', False)
        self.fingerprint_bits = config.model.get('fingerprint_bits', 4096)
        self.fingerprint_radius = config.model.get('fingerprint_radius', 2)
        # Load exclusion set for filtering test set molecules
        self.exclude_inchi = self._load_exclude_set(config)
        if self.exclude_inchi:
            print(f"Loaded {len(self.exclude_inchi)} molecules for exclusion filtering")

    def _load_exclude_set(self, config):
        """Load InChI exclusion set from file."""
        exclude_file = config.data.get('exclude_inchikeys_file', None)
        if exclude_file and os.path.exists(exclude_file):
            return frozenset(pd.read_csv(exclude_file)['inchi'].tolist())
        return None
    
    def __call__(self, examples):
        if self.remove_stereo:
            for example in examples:
                mol = Chem.MolFromSmiles(example['input'])
                if mol is not None:
                    Chem.RemoveStereochemistry(mol)
                    smi = smiles_to_safe(Chem.MolToSmiles(mol))
                    if smi is not None:
                        example['input'] = smi

        if self.use_bracket_safe:
            for example in examples: example['input'] = safe2bracketsafe(example['input'])

        batch = self.tokenizer([example['input'] for example in examples],
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        del batch['token_type_ids']

        smiles_cache = None
        exclude_mask = None  # Track which samples should be excluded
        
        if self.use_formula_conditioning or self.use_fingerprint_conditioning:
            smiles_cache = []
            for example in examples:
                safe_str = example['input']
                smiles = safe_to_smiles(safe_str, fix=True)
                smiles_cache.append(smiles)
            
            # Check for exclusion if exclusion set is loaded
            if self.exclude_inchi is not None:
                exclude_mask = []
                for smiles in smiles_cache:
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                    inchi = Chem.MolToInchiKey(mol) if mol else None
                    if inchi and inchi.split('-')[0] in self.exclude_inchi:
                        exclude_mask.append(0.0)  # Exclude this sample
                    else:
                        exclude_mask.append(1.0)  # Keep this sample
        
        # Extract molecular formulas if conditioning is enabled
        if self.use_formula_conditioning:
            formulas = []
            for i, smiles in enumerate(smiles_cache):
                # If sample is excluded, set formula to None
                if exclude_mask is not None and exclude_mask[i] == 0.0:
                    formulas.append(None)
                    continue
                try:
                    if smiles:
                        # Parse SMILES and extract formula
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                            formulas.append(formula)
                        else:
                            formulas.append(None)
                    else:
                        formulas.append(None)
                except Exception as e:
                    # If formula extraction fails, use None
                    formulas.append(None)
            
            batch['formula'] = formulas

        if self.use_fingerprint_conditioning:
            fingerprint_tensors = []
            fingerprint_masks = []
            for i, smiles in enumerate(smiles_cache):
                fingerprint_tensor, success_flag = self._safe_to_fingerprint(smiles)
                fingerprint_tensors.append(fingerprint_tensor)
                # Apply exclusion mask: if sample is excluded, set mask to 0
                if exclude_mask is not None and exclude_mask[i] == 0.0:
                    fingerprint_masks.append(0.0)
                else:
                    fingerprint_masks.append(success_flag)
            if fingerprint_tensors:
                batch['fingerprint'] = torch.stack(fingerprint_tensors)
                batch['fingerprint_mask'] = torch.tensor(
                    fingerprint_masks,
                    dtype=torch.float32
                )
        
        # Add exclude_mask to batch for potential use in loss calculation
        if exclude_mask is not None:
            batch['exclude_mask'] = torch.tensor(exclude_mask, dtype=torch.float32)
        
        return batch
    
    def _safe_to_fingerprint(self, smiles):
        device = 'cpu'
        if smiles is None:
            return torch.zeros(self.fingerprint_bits, dtype=torch.float32, device=device), 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(self.fingerprint_bits, dtype=torch.float32, device=device), 0.0
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.fingerprint_radius,
                nBits=self.fingerprint_bits
            )
            arr = np.zeros((self.fingerprint_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return torch.from_numpy(arr), 1.0
        except Exception:
            return torch.zeros(self.fingerprint_bits, dtype=torch.float32, device=device), 0.0
    

class UserDataset(datasets.Dataset):
    def __init__(self, data_path):
        if not data_path:
            raise ValueError(
                "config.data.data_path must be set when data.use_safe_hf is False."
            )
        data_path = os.fspath(data_path)
        with open(data_path) as f:
            self.safe_list = f.readlines()
        self.safe_list = [s.strip('\n') for s in self.safe_list]
        
    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, indices):
        return {'input': self.safe_list[i] for i in indices}
    

def get_dataloader(config):
    if config.data.use_safe_hf:
        if config.data.hf_cache_dir is not None:
            print("Using HF cache dir:", config.data.hf_cache_dir)
            os.environ.setdefault('HF_DATASETS_CACHE', str(config.data.hf_cache_dir))
            os.environ.setdefault('HF_HUB_CACHE', str(config.data.hf_cache_dir))
        
        stream_ds = datasets.load_dataset(
            'datamol-io/safe-gpt', 
            streaming=True, 
            split='train', 
            cache_dir=config.data.hf_cache_dir,
            download_mode='reuse_cache_if_exists'
        )

        # Removed stream_shuffle_buffer implementation as requested

        rank, world_size = _get_rank_and_world_size()
        if world_size > 1:
            stream_ds = split_dataset_by_node(stream_ds, rank=rank, world_size=world_size)
            print(f"Splitting streaming dataset across nodes: rank {rank} / {world_size}")

        resume_step = config.loader.get('resume_step', 0)
        num_workers = config.loader.num_workers
        persistent_workers = config.loader.get('persistent_workers', True)
        if resume_step > 0:
            # Calculate items to skip per process
            # Each process consumes config.loader.batch_size items per step
            skip_count = resume_step * config.loader.batch_size
            print(f"Rank {rank}: Skipping {skip_count} samples (Resume step: {resume_step})")
            stream_ds = stream_ds.skip(skip_count)
            if num_workers > 0:
                # HF SkipExamplesIterable used after .skip() does not implement shard_data_sources,
                # so multiple DataLoader workers will raise NotImplementedError.
                print("Resume detected with streaming dataset: forcing num_workers=0 to avoid sharding errors.")
                num_workers = 0
                persistent_workers = False
        if num_workers == 0:
            persistent_workers = False

        dataloader = torch.utils.data.DataLoader(
            stream_ds,
            batch_size=config.loader.batch_size,
            collate_fn=Collator(config),
            num_workers=num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,  # streaming
            persistent_workers=persistent_workers)
        
        return dataloader
    
    # User-defined dataset
    user_data_path = config.data.get('data_path', None)
    if not user_data_path:
        raise ValueError(
            "config.data.data_path must be provided when data.use_safe_hf is False."
        )
    return torch.utils.data.DataLoader(
        UserDataset(user_data_path),
        batch_size=config.loader.batch_size,
        collate_fn=Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True)

