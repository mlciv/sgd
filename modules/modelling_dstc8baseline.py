# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : modelling_dstc8baseline.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict active_intent, requested_slots, slot goals
# --------------------------------------------------------------------

import logging
import collections
import re
import os
import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from modules.core.encoder_utils import EncoderUtils
from modules.core.schemadst_configuration import SchemaDSTConfig
from modules.dstc8baseline_output_interface import DSTC8BaselineOutputInterface
from modules.schema_embedding_generator import SchemaInputFeatures
from src import utils_schema
from utils import (
    torch_ext,
    data_utils,
    schema
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)

# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class DSTC8BaselineModel(PreTrainedModel, DSTC8BaselineOutputInterface):
    """
    Main entry of DSTC model, this model is based on official baseline, for cls matching model
    """

    config_class = SchemaDSTConfig
    base_model_prefix = ""
    # see the trick in https://github.com/huggingface/transformers/blob/cae641ff269478f74b5895d36fbc686f7074f2fb/transformers/modeling_utils.py#L457
    # 1.To support the base model part from a model
    # 1.1. should only load the basemodel part of a different deveried model
    # 1.2. the original BertModel has base_model_prefix="", and no attr called base_model_prefix; A derived model will have a base_model_prefix="bert/roberta", and have parameters started with $base_model_prefix
    # Assumption 1: A derived model has an attr called $bas_model_prefix$, also its state_dict has parameter started with $base_model_prefix$. Then it will not meet any of the condition in the code. staret_prefix and model_to_load will not change. But if the source model is not eaxactly the same as the target model, some parameters will show warning.
    # Assumption 2: A base model has no attr or parameters started with base_model_str.
    # 2. When loading, we will check base the target model and source state_dict
    # if the model has attribute called $base_model_prefix$, but state_dict has no $base_model_prefix$(the base mode itself). Then we only load the base model part.
    # 2.1. Target: A derived model, Source: A derived model
    # condition 1 and 2 will failed. no change to source and target
    # 2.2  Target: A derived model, Source: base model part.
    # condition 1 will fail: all state_dict will be used, no prefix will change
    # condition 2 will succeed. only the base_model part in the target model will be fill
    # 2.3  Target: A based model. Source: A base model
    # condition 1 will fail: all state_dict will be used, no prefix will change
    # condition 2 will fail. all target model will be filled
    # 2.4 Target: Base model. Source: Derived model
    # condition 1 will succeed, only the basemodel part of source derived model  will be used
    # condition 2 will fail. No change for the source model part.


    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, encoder=None, config=None, args=None):
        super(DSTC8BaselineModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = EncoderUtils.create_encoder(self.config)
            EncoderUtils.set_encoder_finetuning_status(self.encoder, args.finetuning_encoder)
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.embedding_dim = self.config.schema_embedding_dim
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        self.token_dropout = torch.nn.Dropout(self.config.token_dropout)
        self.intent_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.requested_slots_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.categorical_slots_status_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.categorical_slots_values_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.noncategorical_slots_status_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        # Project the combined embeddings to obtain logits.
        # for intent, one logits
        self.intent_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )

        # for requested slots, one logits
        self.requested_slots_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )

        # for categorical_slots, 3 logits
        self.categorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 3)
        )

        # for categorical_slot_values,
        self.categorical_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )

        # for non-categorical_slots, 3 logits
        self.noncategorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 3)
        )
        self.noncat_span_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, 2)
        )

        # for non-categorical span value
        if not args.no_cuda:
            self.cuda(args.device)

    def _encode_utterances(self, features, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        output = self.encoder(
            input_ids=features["utt"],
            attention_mask=features["utt_mask"],
            token_type_ids=features["utt_seg"])

        encoded_utterance = output[0][:, 0, :]
        encoded_tokens = output[0]
        # Apply dropout in training mode.
        if is_training:
            encoded_utterance = self.utterance_dropout(encoded_utterance)
            encoded_tokens = self.utterance_dropout(encoded_tokens)
        return encoded_utterance, encoded_tokens

    def _get_logits(self, element_embeddings, _encoded_utterance, utterance_proj, final_proj):
        """Get logits for elements by conditioning on utterance embedding.
        Args:
        element_embeddings: A tensor of shape (batch_size, num_elements,
        embedding_dim).
        num_classes: An int containing the number of classes for which logits are
        to be generated.
        Returns:
        A tensor of shape (batch_size, num_elements, num_classes) containing the
        logits.
        """
        _, num_elements, _ = element_embeddings.size()
        # Project the utterance embeddings.
        utterance_embedding = utterance_proj(_encoded_utterance)
        # Combine the utterance and element embeddings.
        repeat_utterance_embeddings = utterance_embedding.unsqueeze(1).expand(-1, num_elements, -1)
        # logger.info("element_embeddings:{}, repeat_utterance_embeddings:{}".format(element_embeddings.size(), repeat_utterance_embeddings.size()))
        utterance_element_emb = torch.cat((repeat_utterance_embeddings, element_embeddings), dim=2)
        return final_proj(utterance_element_emb)

    def _get_intents(self, features, _encoded_utterance):
        """Obtain logits for intents."""
        intent_embeddings = features[SchemaInputFeatures.get_embedding_tensor_name("intent")]
        # Add a trainable vector for the NONE intent.
        # [batch_size, max_num_intent, dim]
        _, max_num_intents, embedding_dim = intent_embeddings.size()
        # init a matrix
        null_intent_embedding = torch.empty(1, 1, embedding_dim, device=self.device)
        torch.nn.init.normal_(null_intent_embedding, std=0.02)
        batch_size = intent_embeddings.size()[0]
        repeated_null_intent_embedding = null_intent_embedding.expand(batch_size, 1, -1)
        intent_embeddings = torch.cat(
            (repeated_null_intent_embedding, intent_embeddings), dim=1)

        logits = self._get_logits(
            intent_embeddings, _encoded_utterance,
            self.intent_utterance_proj, self.intent_final_proj)
        # Shape: (batch_size, max_intents + 1)
        logits = logits.squeeze(-1)
        # Mask out logits for padded intents. 1 is added to account for NONE intent.
        # [batch_size, max_intent+1]
        mask = torch_ext.sequence_mask(
            features["intent_num"] + 1,
            maxlen=max_num_intents + 1, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(logits) * torch.finfo(torch.float16).max
        return torch.where(mask, logits, negative_logits)

    def _get_requested_slots(self, features, _encoded_utterance):
        """Obtain logits for requested slots."""
        slot_embeddings = features[SchemaInputFeatures.get_embedding_tensor_name("req_slot")]
        logits = self._get_logits(
            slot_embeddings, _encoded_utterance,
            self.requested_slots_utterance_proj, self.requested_slots_final_proj)
        return torch.squeeze(logits, dim=-1)

    def _get_categorical_slots_goals(self, features, _encoded_utterance):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots.
        slot_embeddings = features[SchemaInputFeatures.get_embedding_tensor_name("cat_slot")]
        status_logits = self._get_logits(slot_embeddings,
                                         _encoded_utterance,
                                         self.categorical_slots_status_utterance_proj,
                                         self.categorical_slots_status_final_proj)
        # Predict the goal value.
        # Shape: (batch_size, max_categorical_slots, max_categorical_values,
        # embedding_dim).
        value_embeddings = features[SchemaInputFeatures.get_embedding_tensor_name("cat_slot_value")]
        _, max_num_slots, max_num_values, embedding_dim = (
            value_embeddings.size())
        value_embeddings_reshaped = value_embeddings.view(-1, max_num_slots * max_num_values, embedding_dim)
        value_logits = self._get_logits(value_embeddings_reshaped,
                                        _encoded_utterance,
                                        self.categorical_slots_values_utterance_proj,
                                        self.categorical_slots_values_final_proj)
        # Reshape to obtain the logits for all slots.
        value_logits = value_logits.view(-1, max_num_slots, max_num_values)
        # Mask out logits for padded slots and values because they will be
        # softmaxed.
        mask = torch_ext.sequence_mask(
            features["cat_slot_value_num"],
            maxlen=max_num_values, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(value_logits) * torch.finfo(torch.float16).max
        value_logits = torch.where(mask, value_logits, negative_logits)
        return status_logits, value_logits

    def _get_noncategorical_slots_goals(self, features, _encoded_utterance, _encoded_tokens):
        """Obtain logits for status and slot spans for non-categorical slots."""
        # Predict the status of all non-categorical slots.
        slot_embeddings = features[SchemaInputFeatures.get_embedding_tensor_name("noncat_slot")]
        max_num_slots = slot_embeddings.size()[1]
        status_logits = self._get_logits(slot_embeddings,
                                         _encoded_utterance,
                                         self.noncategorical_slots_status_utterance_proj,
                                         self.noncategorical_slots_status_final_proj)

        # Predict the distribution for span indices.
        token_embeddings = _encoded_tokens
        max_num_tokens = token_embeddings.size()[1]
        tiled_token_embeddings = token_embeddings.unsqueeze(1).expand(-1, max_num_slots, -1, -1)
        tiled_slot_embeddings = slot_embeddings.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
        # Shape: (batch_size, max_num_slots, max_num_tokens, 2 * embedding_dim).
        slot_token_embeddings = torch.cat(
            [tiled_slot_embeddings, tiled_token_embeddings], dim=3)

        # Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.noncat_span_layer(slot_token_embeddings)
        # Mask out invalid logits for padded tokens.
        token_mask = features["utt_mask"]  # Shape: (batch_size, max_num_tokens).
        tiled_token_mask = token_mask.unsqueeze(1).unsqueeze(3).expand(-1, max_num_slots, -1, 2)
        negative_logits = -0.7 * torch.ones_like(span_logits) * torch.finfo(torch.float16).max
        #logger.info("span_logits:{}, token_mask: {} , titled_token_mask:{}".format(
        #    span_logits.size(), token_mask.size(), tiled_token_mask.size()))
        span_logits = torch.where(tiled_token_mask.bool(), span_logits, negative_logits)
        # Shape of both tensors: (batch_size, max_num_slots, max_num_tokens).
        span_start_logits = span_logits[:, :, :, 0]
        span_end_logits = span_logits[:, :, :, 1]
        return status_logits, span_start_logits, span_end_logits

    def forward(self, features, labels=None):
        """
        given input, output probibilities of each selection
        input: (input1_ids, input2_ids)
        In the sentence pair of Bert, token_type_ids indices to indicate first and second portions of the inputs.
        Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        corresponds to a `sentence B` token
        But in our case, we encode each sentence seperately.
        plan2 is the same, for AE, only input_ids is useful.
        """
        is_training = (labels is not None)
        _encoded_utterance, _encoded_tokens = self._encode_utterances(features, is_training)
        outputs = {}
        outputs["logit_intent_status"] = self._get_intents(features, _encoded_utterance)
        outputs["logit_req_slot_status"] = self._get_requested_slots(features, _encoded_utterance)
        cat_slot_status, cat_slot_value = self._get_categorical_slots_goals(features, _encoded_utterance)
        outputs["logit_cat_slot_status"] = cat_slot_status
        outputs["logit_cat_slot_value"] = cat_slot_value
        noncat_slot_status, noncat_span_start, noncat_span_end = (
            self._get_noncategorical_slots_goals(features, _encoded_utterance, _encoded_tokens))
        outputs["logit_noncat_slot_status"] = noncat_slot_status
        outputs["logit_noncat_slot_start"] = noncat_span_start
        outputs["logit_noncat_slot_end"] = noncat_span_end

        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
