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
Custom BERT encoder with cross-attention for formula conditioning.

This module extends the standard BERT encoder to include cross-attention
layers that allow conditioning on molecular formula sequences.
"""

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertEncoder,
    BertPreTrainedModel,
    BertForMaskedLM
)
from transformers.models.bert.configuration_bert import BertConfig


class BertLayerWithCrossAttention(nn.Module):
    """
    BERT layer extended with optional cross-attention over conditioning tokens.
    
    This layer performs:
    1. Standard self-attention
    2. Cross-attention to formula conditioning sequence (if provided)
    3. Cross-attention to fingerprint conditioning sequence (if provided)
    4. Feed-forward network
    
    When both formula and fingerprint conditioning are used, they have
    independent cross-attention layers, allowing each to learn optimal
    attention patterns for their respective modalities.
    """
    
    def __init__(self, config, cross_attention_layer=None, fingerprint_cross_attention_layer=None):
        """
        Initialize layer.
        
        Args:
            config: BERT configuration
            cross_attention_layer: CrossAttentionLayer module for formula (optional)
            fingerprint_cross_attention_layer: CrossAttentionLayer module for fingerprint (optional)
        """
        super().__init__()
        
        # Standard BERT layer components
        self.bert_layer = BertLayer(config)
        
        # Cross-attention layer for formula conditioning (optional)
        self.cross_attention = cross_attention_layer
        self.has_cross_attention = cross_attention_layer is not None
        
        # Cross-attention layer for fingerprint conditioning (optional, independent)
        self.fingerprint_cross_attention = fingerprint_cross_attention_layer
        self.has_fingerprint_cross_attention = fingerprint_cross_attention_layer is not None
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        condition_embeddings=None,
        condition_mask=None,
        fingerprint_embeddings=None,
        fingerprint_mask=None,
    ):
        """
        Forward pass with optional cross-attention for both formula and fingerprint.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Self-attention mask
            condition_embeddings: Formula conditioning sequence embeddings (for cross-attention)
            condition_mask: Formula conditioning attention mask
            fingerprint_embeddings: Fingerprint conditioning sequence embeddings (for cross-attention)
            fingerprint_mask: Fingerprint conditioning attention mask
            ... (other BERT layer arguments)
        
        Returns:
            Tuple of (hidden_states, attentions) or just hidden_states
        """
        # Standard BERT self-attention and FFN
        layer_outputs = self.bert_layer(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        
        hidden_states = layer_outputs[0]
        
        # Apply formula cross-attention if conditioning is provided
        if self.has_cross_attention and condition_embeddings is not None:
            hidden_states = self.cross_attention(
                hidden_states=hidden_states,
                condition_embeddings=condition_embeddings,
                condition_mask=condition_mask
            )
        
        # Apply fingerprint cross-attention if conditioning is provided (independent from formula)
        if self.has_fingerprint_cross_attention and fingerprint_embeddings is not None:
            hidden_states = self.fingerprint_cross_attention(
                hidden_states=hidden_states,
                condition_embeddings=fingerprint_embeddings,
                condition_mask=fingerprint_mask
            )
        
        # Return in same format as BertLayer
        outputs = (hidden_states,) + layer_outputs[1:]
        return outputs


class BertEncoderWithCrossAttention(nn.Module):
    """
    BERT encoder with cross-attention layers for conditioning sequences.
    
    Supports independent cross-attention for formula and fingerprint conditioning,
    allowing each modality to have its own learned attention patterns.
    """
    
    def __init__(self, config, cross_attention_layers=None, fingerprint_cross_attention_layers=None):
        """
        Initialize encoder.
        
        Args:
            config: BERT configuration
            cross_attention_layers: List of CrossAttentionLayer modules for formula,
                                   one per BERT layer (optional)
            fingerprint_cross_attention_layers: List of CrossAttentionLayer modules for fingerprint,
                                               one per BERT layer (optional)
        """
        super().__init__()
        self.config = config
        
        # Create layers with cross-attention (both formula and fingerprint)
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            cross_attn = cross_attention_layers[i] if cross_attention_layers else None
            fp_cross_attn = fingerprint_cross_attention_layers[i] if fingerprint_cross_attention_layers else None
            self.layer.append(
                BertLayerWithCrossAttention(
                    config, 
                    cross_attention_layer=cross_attn,
                    fingerprint_cross_attention_layer=fp_cross_attn
                )
            )
        
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        condition_embeddings=None,
        condition_mask=None,
        fingerprint_embeddings=None,
        fingerprint_mask=None,
    ):
        """
        Forward pass through all layers.
        
        Args:
            hidden_states: Input embeddings
            condition_embeddings: Formula conditioning sequence embeddings (for cross-attention)
            condition_mask: Formula conditioning attention mask
            fingerprint_embeddings: Fingerprint conditioning sequence embeddings (for cross-attention)
            fingerprint_mask: Fingerprint conditioning attention mask
            ... (other BERT encoder arguments)
        
        Returns:
            Encoder outputs (hidden states, attentions, etc.)
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        
        next_decoder_cache = () if use_cache else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing not implemented for cross-attention yet
                raise NotImplementedError("Gradient checkpointing not supported with cross-attention")
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    condition_embeddings=condition_embeddings,
                    condition_mask=condition_mask,
                    fingerprint_embeddings=fingerprint_embeddings,
                    fingerprint_mask=fingerprint_mask,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Return as tuple (compatible with BERT)
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )


class BertModelWithCrossAttention(BertPreTrainedModel):
    """
    BERT model with cross-attention for generic conditioning sequences.
    
    This is a drop-in replacement for BertModel that adds cross-attention
    capabilities while maintaining compatibility with standard BERT.
    
    Supports two modes:
    1. Independent cross-attention: Formula and fingerprint have separate cross-attention layers
    2. Shared cross-attention: Formula and fingerprint embeddings are concatenated and use
       a single set of cross-attention layers
    """
    
    def __init__(self, config, add_pooling_layer=True, cross_attention_layers=None, 
                 fingerprint_cross_attention_layers=None, use_shared_cross_attention=False):
        """
        Initialize model.
        
        Args:
            config: BERT configuration
            add_pooling_layer: Whether to add pooling layer (default: True)
            cross_attention_layers: List of CrossAttentionLayer modules for formula
            fingerprint_cross_attention_layers: List of CrossAttentionLayer modules for fingerprint
            use_shared_cross_attention: If True, uses only cross_attention_layers for both
                                        formula and fingerprint (concatenated embeddings)
        """
        super().__init__(config)
        self.config = config
        self.use_shared_cross_attention = use_shared_cross_attention
        
        # Use standard BERT embeddings
        from transformers.models.bert.modeling_bert import BertEmbeddings
        self.embeddings = BertEmbeddings(config)
        
        # In shared mode, fingerprint_cross_attention_layers should be None
        # (fingerprint embeddings are concatenated with formula and use the same cross-attn layers)
        if use_shared_cross_attention:
            fingerprint_cross_attention_layers = None
        
        # Use custom encoder with cross-attention (both formula and fingerprint)
        self.encoder = BertEncoderWithCrossAttention(
            config, 
            cross_attention_layers,
            fingerprint_cross_attention_layers
        )
        
        # Pooler (for sequence classification, not needed for MLM)
        from transformers.models.bert.modeling_bert import BertPooler
        self.pooler = BertPooler(config) if add_pooling_layer else None
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        """Prune attention heads (not implemented for cross-attention)."""
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].bert_layer.attention.prune_heads(heads)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        condition_embeddings=None,
        condition_mask=None,
        fingerprint_embeddings=None,
        fingerprint_mask=None,
    ):
        """
        Forward pass with optional conditioning.
        
        Args:
            condition_embeddings: Formula conditioning sequence embeddings (batch, cond_len, hidden_size)
            condition_mask: Formula conditioning attention mask (batch, cond_len)
            fingerprint_embeddings: Fingerprint conditioning sequence embeddings (batch, fp_len, hidden_size)
            fingerprint_mask: Fingerprint conditioning attention mask (batch, fp_len)
            ... (standard BERT arguments)
        
        Returns:
            Model outputs (hidden states, pooled output, etc.)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Get extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        # Pass through encoder with both formula and fingerprint conditioning
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            condition_embeddings=condition_embeddings,
            condition_mask=condition_mask,
            fingerprint_embeddings=fingerprint_embeddings,
            fingerprint_mask=fingerprint_mask,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class BertForMaskedLMWithCrossAttention(BertPreTrainedModel):
    """
    BERT for Masked Language Modeling with cross-attention conditioning.
    
    Drop-in replacement for BertForMaskedLM with both formula and fingerprint 
    conditioning support via cross-attention layers.
    
    Supports two modes:
    1. Independent: Separate cross-attention layers for formula and fingerprint
    2. Shared: Single set of cross-attention layers for concatenated embeddings
    """
    
    def __init__(self, config, cross_attention_layers=None, fingerprint_cross_attention_layers=None,
                 use_shared_cross_attention=False):
        super().__init__(config)
        self.use_shared_cross_attention = use_shared_cross_attention
        
        # For MLM, we don't need the pooling layer
        self.bert = BertModelWithCrossAttention(
            config, 
            add_pooling_layer=False,
            cross_attention_layers=cross_attention_layers,
            fingerprint_cross_attention_layers=fingerprint_cross_attention_layers,
            use_shared_cross_attention=use_shared_cross_attention
        )
        
        # MLM head
        from transformers.models.bert.modeling_bert import BertOnlyMLMHead
        self.cls = BertOnlyMLMHead(config)
        
        # Initialize weights
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        condition_embeddings=None,
        condition_mask=None,
        fingerprint_embeddings=None,
        fingerprint_mask=None,
    ):
        """
        Forward pass with conditioning.
        
        Args:
            condition_embeddings: Formula conditioning sequence embeddings
            condition_mask: Formula conditioning attention mask
            fingerprint_embeddings: Fingerprint conditioning sequence embeddings
            fingerprint_mask: Fingerprint conditioning attention mask
            ... (standard BertForMaskedLM arguments)
        
        Returns:
            Loss and logits (if return_dict=False) or MaskedLMOutput
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            condition_embeddings=condition_embeddings,
            condition_mask=condition_mask,
            fingerprint_embeddings=fingerprint_embeddings,
            fingerprint_mask=fingerprint_mask,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        from transformers.modeling_outputs import MaskedLMOutput
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
