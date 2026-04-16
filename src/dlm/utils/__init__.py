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
DLM utilities module with lazy exports.

The utilities package has several heavy submodules (e.g., Spec2Mol) that
introduce circular imports when eagerly imported. To avoid this we expose the
public symbols via lazy loading so that importing ``dlm.utils`` remains
lightweight for core training code.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # These imports are only for static analyzers / IDEs and won't run at
    # runtime, so circular dependencies are not triggered.
    from .spec2mol import Spec2MolModel, Spec2MolSampler, load_spec2mol_model
    from .checkpoint import pack_checkpoint, verify_checkpoint
    from .benchmark_utils import (
        normalize_formula,
        compute_morgan_fingerprint,
        compute_tanimoto_similarity,
        generate_with_formula_filter,
    )

__all__ = [
    'Spec2MolModel',
    'Spec2MolSampler',
    'load_spec2mol_model',
    'pack_checkpoint',
    'verify_checkpoint',
    'normalize_formula',
    'compute_morgan_fingerprint',
    'compute_tanimoto_similarity',
    'generate_with_formula_filter',
]

_LAZY_IMPORTS = {
    'Spec2MolModel': ('dlm.utils.spec2mol', 'Spec2MolModel'),
    'Spec2MolSampler': ('dlm.utils.spec2mol', 'Spec2MolSampler'),
    'load_spec2mol_model': ('dlm.utils.spec2mol', 'load_spec2mol_model'),
    'pack_checkpoint': ('dlm.utils.checkpoint', 'pack_checkpoint'),
    'verify_checkpoint': ('dlm.utils.checkpoint', 'verify_checkpoint'),
    'normalize_formula': ('dlm.utils.benchmark_utils', 'normalize_formula'),
    'compute_morgan_fingerprint': ('dlm.utils.benchmark_utils', 'compute_morgan_fingerprint'),
    'compute_tanimoto_similarity': ('dlm.utils.benchmark_utils', 'compute_tanimoto_similarity'),
    'generate_with_formula_filter': ('dlm.utils.benchmark_utils', 'generate_with_formula_filter'),
}


def __getattr__(name: str) -> Any:
    """Dynamically import utilities only when requested."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache to avoid future lookups
        return value
    raise AttributeError(f"module 'dlm.utils' has no attribute '{name}'")


def __dir__():
    """Make lazy attributes discoverable via dir()."""
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))