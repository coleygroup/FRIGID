#!/bin/bash

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No conda environment is currently active"
    exit 1
fi

# Comment out all lines in the safe package __init__.py
sed -i 's/^/# /' "$CONDA_PREFIX/lib/python3.10/site-packages/safe/__init__.py"

# Import required packages
echo "from .converter import SAFEConverter, decode, encode" >> "$CONDA_PREFIX/lib/python3.10/site-packages/safe/__init__.py"

echo "Fixed safe package in environment: $CONDA_PREFIX"

# This is to fix the following error with SAFE due to the new version of transformers
#
#   File "scripts/train.py", line 23, in <module>
#     from dlm.model import DLM
#   File "src/dlm/model.py", line 29, in <module>
#     from dlm.utils.utils_data import get_tokenizer
#   File "src/dlm/utils/utils_data.py", line 20, in <module>
#     from safe.tokenizer import SAFETokenizer
#   File "/home/USER/mambaforge/envs/frigid/lib/python3.10/site-packages/safe/__init__.py", line 4, in <module>
#     from .sample import SAFEDesign
#   File "/home/USER/mambaforge/envs/frigid/lib/python3.10/site-packages/safe/sample.py", line 21, in <module>
#     from safe.trainer.model import SAFEDoubleHeadsModel
#   File "/home/USER/mambaforge/envs/frigid/lib/python3.10/site-packages/safe/trainer/__init__.py", line 2, in <module>
#     from . import model
#   File "/home/USER/mambaforge/envs/frigid/lib/python3.10/site-packages/safe/trainer/model.py", line 8, in <module>
#     from transformers.models.gpt2.modeling_gpt2 import (
# ImportError: cannot import name '_CONFIG_FOR_DOC' from 'transformers.models.gpt2.modeling_gpt2' (/home/USER/mambaforge/envs/frigid/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py)
