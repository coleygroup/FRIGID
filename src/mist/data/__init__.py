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

"""Data loading and featurization for MIST encoder."""

from .data import Spectra, Mol
from .featurizers import (
    PairedFeaturizer,
    PeakFormula,
    FingerprintFeaturizer,
    get_paired_featurizer,
)
from .datasets import (
    SpectraMolDataset,
    get_paired_spectra,
)
from .splitter import PresetSpectraSplitter
from importlib import import_module


def __getattr__(name):
    optional_modules = {
        "HDF5SpectrumDataset": "hdf5_dataset",
        "HDF5Spec2MolDataModule": "hdf5_dataset",
        "IcebergMultiPartDataModule": "iceberg_dataset",
        "IcebergEpochCallback": "iceberg_dataset",
    }
    if name in optional_modules:
        value = getattr(import_module(f"{__name__}.{optional_modules[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Spectra",
    "Mol",
    "PairedFeaturizer",
    "PeakFormula",
    "FingerprintFeaturizer",
    "get_paired_featurizer",
    "SpectraMolDataset",
    "get_paired_spectra",
    "PresetSpectraSplitter",
    "HDF5SpectrumDataset",
    "HDF5Spec2MolDataModule",
    "IcebergMultiPartDataModule",
    "IcebergEpochCallback",
]
