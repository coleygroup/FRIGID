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

"""MIST Spectra Encoder for mass spectrometry to fingerprint prediction."""

from typing import Tuple

import torch
from torch import nn

from . import modules


class SpectraEncoder(nn.Module):
    """Standard spectra encoder for fingerprint prediction.

    Encodes mass spectrometry data into molecular fingerprints using
    a FormulaTransformer backbone with MLP prediction heads.

    Args:
        form_embedder: Type of formula embedding ("float", "pos-cos", etc.)
        output_size: Output fingerprint dimension
        hidden_size: Hidden dimension size
        spectra_dropout: Dropout probability
        top_layers: Number of MLP layers in prediction heads
        refine_layers: Number of refinement layers (unused in this version)
        magma_modulo: Dimension for fragment prediction
        **kwargs: Additional arguments passed to FormulaTransformer
    """

    def __init__(
        self,
        form_embedder: str = "float",
        output_size: int = 4096,
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        refine_layers: int = 0,
        magma_modulo: int = 2048,
        **kwargs,
    ):
        super(SpectraEncoder, self).__init__()

        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            spectra_dropout=spectra_dropout,
            form_embedder=form_embedder,
            **kwargs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))

        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))

        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        top_layer_parts = []
        for _ in range(top_layers - 1):
            top_layer_parts.append(nn.Linear(hidden_size, hidden_size))
            top_layer_parts.append(nn.ReLU())
            top_layer_parts.append(nn.Dropout(spectra_dropout))
        top_layer_parts.append(nn.Linear(hidden_size, output_size))
        top_layer_parts.append(nn.Sigmoid())
        spectra_predictor = nn.Sequential(*top_layer_parts)

        self.spectra_encoder = nn.ModuleList([spectra_encoder_main, fragment_predictor, spectra_predictor])

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the encoder.

        Args:
            batch: Dictionary containing spectra data with keys:
                - num_peaks: Number of peaks per spectrum
                - types: Peak types
                - instruments: Instrument indices
                - ion_vec: Adduct information
                - form_vec: Formula vectors
                - intens: Peak intensities

        Returns:
            Tuple of:
                - output: Predicted fingerprint tensor [batch, output_size]
                - aux_outputs: Dictionary with auxiliary predictions
        """
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)

        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}

        output = self.spectra_encoder[2](encoder_output)
        aux_outputs["h0"] = encoder_output

        return output, aux_outputs


class SpectraEncoderGrowing(nn.Module):
    """Spectra encoder with progressive fingerprint refinement.

    Uses a growing module to progressively refine fingerprint predictions
    from coarse to fine resolution, enabling multi-scale supervision.

    Args:
        form_embedder: Type of formula embedding
        output_size: Final fingerprint dimension
        hidden_size: Hidden dimension size
        spectra_dropout: Dropout probability
        top_layers: Number of MLP layers in prediction heads
        refine_layers: Number of progressive refinement stages
        magma_modulo: Dimension for fragment prediction
        **kwargs: Additional arguments passed to FormulaTransformer
    """

    def __init__(
        self,
        form_embedder: str = "float",
        output_size: int = 4096,
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        refine_layers: int = 0,
        magma_modulo: int = 2048,
        **kwargs,
    ):
        super(SpectraEncoderGrowing, self).__init__()

        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            spectra_dropout=spectra_dropout,
            form_embedder=form_embedder,
            **kwargs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))

        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))

        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        spectra_predictor = modules.FPGrowingModule(
            hidden_input_dim=hidden_size,
            final_target_dim=output_size,
            num_splits=refine_layers,
            reduce_factor=2,
        )

        self.spectra_encoder = nn.ModuleList([spectra_encoder_main, fragment_predictor, spectra_predictor])

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the growing encoder.

        Args:
            batch: Dictionary containing spectra data

        Returns:
            Tuple of:
                - output: Final predicted fingerprint [batch, output_size]
                - aux_outputs: Dictionary with:
                    - pred_frag_fps: Fragment fingerprint predictions
                    - int_preds: Intermediate predictions at each scale
                    - h0: Encoder hidden state
        """
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)
        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}

        output = self.spectra_encoder[2](encoder_output)
        intermediates = output[:-1]
        final_output = output[-1]
        aux_outputs["int_preds"] = intermediates
        output = final_output
        aux_outputs["h0"] = encoder_output

        return output, aux_outputs
