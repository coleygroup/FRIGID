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
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add src directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra
import lightning as L
import omegaconf
import torch
from dlm.model import DLM
from dlm.utils.utils_data import get_dataloader, get_last_checkpoint
from dlm.utils.validation_callback import FP2MolValidationCallback

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver('home', lambda: os.path.expanduser('~'))


def load_model_weights_only(model: DLM, checkpoint_path: str, strict: bool = False):
    """
    Load only the model weights from a checkpoint, ignoring optimizer state,
    training progress, and other PyTorch Lightning state.
    
    Args:
        model: The DLM model to load weights into
        checkpoint_path: Path to the checkpoint file
        strict: Whether to strictly enforce that the keys in state_dict match
                the keys returned by model.state_dict(). Default False to allow
                loading checkpoints with slight architecture differences.
    
    Returns:
        The model with loaded weights
    """
    print(f"[train] Loading model weights only from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"[train] Warning: Missing keys when loading weights: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
    
    if unexpected_keys:
        print(f"[train] Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
    
    print(f"[train] Successfully loaded model weights (strict={strict})")
    return model


@hydra.main(version_base=None,
    config_path="../configs",
    config_name="base",
)
def train(config):
    wandb_logger = None
    if config.wandb.name is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb)
    
    if config.training.get('use_bracket_safe'):
        config.model.vocab_size += 2

    model = DLM(config)
    
    # Check if we should load weights only (without training state)
    weights_only_path = config.get('load_weights_only', None)
    if weights_only_path is not None:
        strict = config.get('load_weights_strict', False)
        model = load_model_weights_only(model, weights_only_path, strict=strict)
    
    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    
    resume_step = 0
    if ckpt_path is not None:
        try:
            resume_step = int(os.path.basename(ckpt_path).split('.')[0])
            print(f"Resuming from step {resume_step}")
        except Exception as e:
            print(f"Warning: Could not parse step from checkpoint filename: {e}")
            
    with omegaconf.open_dict(config.loader):
        config.loader.resume_step = resume_step
    train_dataloader = get_dataloader(config)
    
    # Build callbacks list
    callbacks = [hydra.utils.instantiate(config.callback)]
    
    # Add FP2Mol validation callback if enabled
    if config.get('validation', {}).get('enabled', False):
        val_config = config.validation
        validation_callback = FP2MolValidationCallback(
            csv_path=val_config.csv_path,
            n_mols=val_config.get('n_mols', 100),
            n_samples=val_config.get('n_samples', 10),
            n_steps=val_config.get('n_steps', 5000),
            softmax_temp=val_config.get('softmax_temp', 0.8),
            randomness=val_config.get('randomness', 0.5),
            batch_size=val_config.get('batch_size', 10),
            token_model_path=val_config.get('token_model', None),
            sigma_lambda=val_config.get('sigma_lambda', 3.0),
            fp_bits=val_config.get('fp_bits', 4096),
            fp_radius=val_config.get('fp_radius', 2)
        )
        callbacks.append(validation_callback)
        print(f"[train] FP2Mol validation enabled: every {val_config.n_steps} steps on {val_config.n_mols} molecules")
    
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate({'_target_': 'lightning.pytorch.strategies.DDPStrategy',
                                          'find_unused_parameters': False}),
        logger=wandb_logger,
        enable_progress_bar=True)
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    

if __name__ == '__main__':
    train()
