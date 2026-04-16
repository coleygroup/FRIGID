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
Utilities for packing and managing Spec2Mol checkpoints.

This module provides functions to:
1. Pack separate MIST encoder and DLM decoder into unified checkpoint
2. Verify checkpoint structure
3. Extract individual components from unified checkpoint
"""

import os
from typing import Dict, Any, Optional

import yaml
import torch


def get_default_encoder_config() -> Dict[str, Any]:
    """Return default MIST encoder configuration."""
    return {
        'form_embedder': 'pos-cos',
        'output_size': 4096,
        'hidden_size': 512,
        'spectra_dropout': 0.1,
        'peak_attn_layers': 2,
        'num_heads': 8,
        'set_pooling': 'cls',
        'refine_layers': 4,
        'pairwise_featurization': True,
        'embed_instrument': False,
        'inten_transform': 'float',
        'magma_modulo': 2048,
        'inten_prob': 0.1,
        'remove_prob': 0.5,
        'cls_type': 'ms1',
        'spec_features': 'peakformula',
        'mol_features': 'fingerprint',
        'top_layers': 1,
    }


def extract_mist_encoder_state(
    checkpoint_path: str, 
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Extract MIST encoder state_dict from checkpoint.
    
    Handles both full training checkpoints (with 'encoder.' prefix) and 
    standalone encoder checkpoints.
    
    Args:
        checkpoint_path: Path to MIST checkpoint
        device: Device to load checkpoint to
        
    Returns:
        Encoder state dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MIST checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if any(k.startswith('encoder.') for k in state_dict.keys()):
            encoder_state = {
                k.replace('encoder.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('encoder.')
            }
            if encoder_state:
                return encoder_state
        return state_dict
    else:
        return checkpoint


def extract_dlm_decoder_state(
    checkpoint_path: str, 
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Extract DLM decoder checkpoint content.
    
    Returns the full checkpoint dict to preserve hyperparameters.
    
    Args:
        checkpoint_path: Path to DLM checkpoint
        device: Device to load checkpoint to
        
    Returns:
        Full DLM checkpoint dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DLM checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def pack_checkpoint(
    mist_checkpoint: str,
    dlm_checkpoint: str,
    output_path: str,
    encoder_config: Optional[Dict[str, Any]] = None,
    fingerprint_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Pack MIST encoder and DLM decoder into a unified checkpoint.
    
    Args:
        mist_checkpoint: Path to MIST encoder checkpoint
        dlm_checkpoint: Path to DLM decoder checkpoint
        output_path: Output path for unified checkpoint
        encoder_config: MIST encoder hyperparameters (uses defaults if None)
        fingerprint_config: Fingerprint settings (uses defaults if None)
        
    Returns:
        Path to created checkpoint
    """
    print(f"Loading MIST encoder from: {mist_checkpoint}")
    encoder_state = extract_mist_encoder_state(mist_checkpoint)
    print(f"  Loaded {len(encoder_state)} encoder parameters")

    print(f"Loading DLM decoder from: {dlm_checkpoint}")
    decoder_checkpoint = extract_dlm_decoder_state(dlm_checkpoint)
    print(f"  Loaded decoder checkpoint")

    # Use defaults if not provided
    if encoder_config is None:
        encoder_config = get_default_encoder_config()
    
    if fingerprint_config is None:
        fingerprint_config = {
            'bits': 4096,
            'radius': 2,
            'threshold': 0.172,
        }

    # Build unified checkpoint
    united_checkpoint = {
        'checkpoint_type': 'spec2mol_united',
        'version': '1.0',
        'encoder_state_dict': encoder_state,
        'encoder_config': encoder_config,
        'decoder_checkpoint': decoder_checkpoint,
        'fingerprint_config': fingerprint_config,
    }

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving unified checkpoint to: {output_path}")
    torch.save(united_checkpoint, output_path)
    
    print("\n" + "=" * 60)
    print("UNIFIED CHECKPOINT CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Encoder parameters: {len(encoder_state)}")
    print(f"Fingerprint config: {fingerprint_config}")
    print(f"Output file: {output_path}")
    print(f"Output size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print("=" * 60)
    
    return output_path


def verify_checkpoint(checkpoint_path: str) -> bool:
    """
    Verify the unified checkpoint structure.
    
    Args:
        checkpoint_path: Path to unified checkpoint
        
    Returns:
        True if checkpoint is valid
        
    Raises:
        AssertionError: If checkpoint structure is invalid
    """
    print(f"Verifying checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    assert checkpoint.get('checkpoint_type') == 'spec2mol_united', "Invalid checkpoint type"
    assert 'encoder_state_dict' in checkpoint, "Missing encoder_state_dict"
    assert 'encoder_config' in checkpoint, "Missing encoder_config"
    assert 'decoder_checkpoint' in checkpoint, "Missing decoder_checkpoint"
    assert 'fingerprint_config' in checkpoint, "Missing fingerprint_config"

    print("✓ Checkpoint structure verified")
    print(f"  - Encoder state keys: {len(checkpoint['encoder_state_dict'])}")
    print(f"  - Encoder config keys: {list(checkpoint['encoder_config'].keys())}")
    print(f"  - Fingerprint config: {checkpoint['fingerprint_config']}")

    return True


def load_config(config_path: Optional[str]) -> dict:
    """Load YAML configuration if provided."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# CLI interface for packing checkpoints
def main():
    """Command-line interface for packing checkpoints."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pack MIST encoder and DLM decoder into unified Spec2Mol checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mist-checkpoint', type=str, required=True,
        help='Path to MIST encoder checkpoint (.pt)'
    )
    parser.add_argument(
        '--dlm-checkpoint', type=str, required=True,
        help='Path to DLM decoder checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--output', type=str, default='checkpoints/spec2mol_united.ckpt',
        help='Output path for unified checkpoint'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Optional config YAML to embed encoder hyperparameters'
    )

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    encoder_config = get_default_encoder_config()
    if config and 'mist_encoder' in config:
        encoder_config.update({
            k: v for k, v in config['mist_encoder'].items() 
            if k != 'checkpoint'
        })
    
    fingerprint_config = {'bits': 4096, 'radius': 2, 'threshold': 0.172}
    if config and 'fingerprint' in config:
        fingerprint_config.update(config['fingerprint'])

    pack_checkpoint(
        mist_checkpoint=args.mist_checkpoint,
        dlm_checkpoint=args.dlm_checkpoint,
        output_path=args.output,
        encoder_config=encoder_config,
        fingerprint_config=fingerprint_config,
    )

    verify_checkpoint(args.output)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
