# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : modelling_toptransformer.py
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

_NL_SEPARATOR = "|||"
# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class TopTransformerModel(PreTrainedModel, DSTC8BaselineOutputInterface):
    """
    Main entry of Classifier Model
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

    def __init__(self, config=None, args=None):
        super(TopTransformerModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
        self.encoder = EncoderUtils.create_encoder(self.config)
        setattr(self, self.base_model_prefix, self.encoder)
        self.embedding_dim = self.config.schema_embedding_dim
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        self.token_dropout = torch.nn.Dropout(self.config.token_dropout)
        if self.embedding_dim == self.config.d_model:
            self.projection_layer = torch.nn.Sequential()
        else:
            self.projection_layer = nn.Linear(self.embedding_dim, self.config.d_model)

        self.intent_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.requested_slots_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.categorical_slots_status_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.categorical_slots_values_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.noncategorical_slots_status_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        # Project the combined embeddings to obtain logits.
        # for intent, one logits
        self.intent_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.d_model, int(self.config.d_model/2)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.d_model/2), 1)
        )

        # for requested slots, one logits
        self.requested_slots_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.d_model, int(self.config.d_model/2)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.d_model/2), 1)
        )

        # for categorical_slots, 3 logits
        self.categorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.d_model, int(self.config.d_model/2)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.d_model/2), 3)
        )

        # for categorical_slot_values,
        self.categorical_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.d_model, int(self.config.d_model/2)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.d_model/2), 1)
        )

        # for non-categorical_slots, 3 logits
        self.noncategorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.d_model, int(self.config.d_model/2)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.d_model/2), 3)
        )
        self.noncat_span_layer = nn.Sequential(
            torch.nn.Linear(self.config.d_model, int(self.config.d_model/2)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.d_model/2), 1)
        )
        # for non-categorical span value
        if not args.no_cuda:
            self.cuda(args.device)

    def _encode_utterances(self, features, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        # logger.info("utt:{}, utt_mask:{}, utt_seg:{}".format(features["utt"], features["utt_mask"], features["utt_seg"]))
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

    def _encode_schema(self, features, schema_type, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        schema_input_ids = features[SchemaInputFeatures.get_input_ids_tensor_name(schema_type)]
        schema_input_shape = list(schema_input_ids.size())
        max_length = schema_input_shape[-1]
        schema_attention_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(schema_type)]
        schema_token_type_ids = features[SchemaInputFeatures.get_input_type_ids_tensor_name(schema_type)]
        assert schema_attention_mask.size()[-1] == max_length, "schema_attention_mask has wrong shape:{}".format(schema_attention_mask.size())
        assert schema_token_type_ids.size()[-1] == max_length, "schema_token_type_ids has wrong shape:{}".format(schema_token_type_ids.size())

        # logger.info("input_ids:{}, attention_mask:{}, token_type_ids :{}".format(
        #    schema_input_ids.size(), schema_attention_mask.size(), schema_token_type_ids.size()))
        output = self.encoder(
            input_ids=schema_input_ids.view(-1, schema_input_ids.size()[-1]),
            attention_mask=schema_attention_mask.view(-1, schema_input_ids.size()[-1]),
            token_type_ids=schema_token_type_ids.view(-1, schema_input_ids.size()[-1]))

        encoded_schema_cls = output[0][:, 0, :]
        embedding_dim = encoded_schema_cls.size()[-1]
        # adding the last dim
        schema_cls_shape = schema_input_shape[:-1]
        schema_cls_shape.append(embedding_dim)
        schema_input_shape.append(embedding_dim)
        encoded_schema_tokens = output[0]
        # Apply dropout in training mode.
        if is_training:
            encoded_schema_cls = self.utterance_dropout(encoded_schema_cls)
            encoded_schema_tokens = self.utterance_dropout(encoded_schema_tokens)
        return encoded_schema_cls.view(torch.Size(schema_cls_shape)), encoded_schema_tokens.view(torch.Size(schema_input_shape))

    def _get_logits(self, element_embeddings, element_mask, _encoded_tokens, utterance_mask, matching_layer, final_proj):
        """Get logits for elements by conditioning on utterance embedding.
        Args:
        element_embeddings: A tensor of shape (batch_size, num_elements,
        max_seq_length, embedding_dim).
        element_mask: (batch_size, max_seq_length)
        _encoded_tokens: (batch_size, max_u_len, embedding_dim)
        _utterance_mask: (batch_size, max_u_len)
        num_classes: An int containing the number of classes for which logits are
        to be generated.
        Returns:
        A tensor of shape (batch_size, num_elements, num_classes) containing the
        logits.
        """
        batch_size1, num_elements, max_seq_length, element_emb_dim = element_embeddings.size()
        batch_size2, max_u_len, u_emb_len = _encoded_tokens.size()
        assert batch_size1 == batch_size2, "batch_size not match between element_emb and utterance_emb"
        assert element_emb_dim == u_emb_len, "dim not much element_emb={} and utterance_emb={}".format(element_emb_dim, u_emb_len)
        # (batch_size, num_elements, max_seq_length, dim)
        expanded_encoded_tokens = _encoded_tokens.unsqueeze(1).expand(-1, num_elements, -1, -1)
        expanded_utterance_mask = utterance_mask.unsqueeze(1).expand(-1, num_elements, -1)
        # (batch_size, num_elements, max_seq_length + max_u_len, dim)
        # Here, we use the same bert, max_seq_length == max_u_len for now.
        # [CLS] pre_system[SEP] user_utt [SEP] padding.... [CLS] schema seq1 [SEP] schema seq2 [SEP] padding
        max_total_len = max_u_len + max_seq_length
        # (batch_size * num_elements, max_total_len, dim)
        utterance_element_pair_emb = torch.cat(
            (expanded_encoded_tokens, element_embeddings), dim=2).view(-1, max_total_len, u_emb_len)
        # (batch_size, num_elements, max_u_len+max_seq_length)
        utterance_element_pair_mask = torch.cat(
            (expanded_utterance_mask, element_mask),
            dim=2).view(-1, max_total_len).bool()

        projected_utterance_element_pair_emb = self.projection_layer(utterance_element_pair_emb.transpose(0, 1))
        # trans_output: (batch_size * num_elements, max_total_len, dim)
        trans_output = matching_layer(
            projected_utterance_element_pair_emb,
            src_key_padding_mask=utterance_element_pair_mask).transpose(0, 1)
        # use the first [CLS] for classification
        output = final_proj(trans_output[ :, 0, :])
        return output.view(batch_size1, num_elements, -1), trans_output.view(batch_size1, num_elements, max_total_len, -1)

    def _get_intents(self, features, _encoded_tokens, is_training):
        """Obtain logits for intents."""
        # service_id is the index of emb value
        # [service_num, max_intentnum, max_seq_length, dim]
        _, intent_embeddings = self._encode_schema(features, "intent", is_training)
        # self.device is the device for Dataparallel model,which device 0
        # Here we need the device current input on
        current_device = intent_embeddings.device
        intent_mask = features[SchemaInputFeatures.get_input_mask_tensor_name("intent")]
        # Add a trainable vector for the NONE intent.
        batch_size, max_num_intents, max_seq_length, embedding_dim = intent_embeddings.size()
        # init a matrix
        null_intent_embedding = torch.empty(1, 1, max_seq_length, embedding_dim, device=current_device)
        null_intent_mask = torch.zeros(1, 1, max_seq_length, dtype=torch.long, device=current_device)
        null_intent_mask[:, :, 0] = 1
        torch.nn.init.normal_(null_intent_embedding, std=0.02)
        repeated_null_intent_embedding = null_intent_embedding.expand(batch_size, 1, -1, -1)
        # batch_size, 1, max_seq_length]
        repeated_null_intent_mask = null_intent_mask.expand(batch_size, 1, -1)
        intent_embeddings = torch.cat(
            (repeated_null_intent_embedding, intent_embeddings), dim=1)
        intent_mask = torch.cat((repeated_null_intent_mask, intent_mask), dim=1)

        utterance_mask = features["utt_mask"]
        logits, _ = self._get_logits(
            intent_embeddings, intent_mask,
            _encoded_tokens, utterance_mask,
            self.intent_matching_layer, self.intent_final_proj)
        # Shape: (batch_size, max_intents + 1)
        logits = logits.squeeze(-1)
        # Mask out logits for padded intents. 1 is added to account for NONE intent.
        # [batch_size, max_intent+1]
        mask = torch_ext.sequence_mask(
            features["intent_num"] + 1,
            maxlen=max_num_intents + 1, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(logits) * torch.finfo(torch.float16).max
        return torch.where(mask, logits, negative_logits)

    def _get_requested_slots(self, features, _encoded_tokens, is_training):
        """Obtain logits for requested slots."""
        _, slot_embeddings = self._encode_schema(features, "req_slot", is_training)
        slot_mask = features[SchemaInputFeatures.get_input_mask_tensor_name("req_slot")]
        utterance_mask = features["utt_mask"]
        logits, _ = self._get_logits(
            slot_embeddings, slot_mask,
            _encoded_tokens, utterance_mask,
            self.requested_slots_matching_layer, self.requested_slots_final_proj)
        return torch.squeeze(logits, dim=-1)

    def _get_categorical_slots_goals(self, features, _encoded_tokens, is_training):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots.
        _, slot_embeddings = self._encode_schema(features, "cat_slot", is_training)
        slot_mask = features[SchemaInputFeatures.get_input_mask_tensor_name("cat_slot")]
        utterance_mask = features["utt_mask"]
        status_logits, _ = self._get_logits(
            slot_embeddings, slot_mask,
            _encoded_tokens, utterance_mask,
            self.categorical_slots_status_matching_layer,
            self.categorical_slots_status_final_proj)
        # logger.info("status_logits:{}, utterance_mask:{}".format(status_logits.size(), utterance_mask.size()))
        # Predict the goal value.
        # Shape: (batch_size, max_categorical_slots, max_categorical_values,
        # embedding_dim).
        _, value_embeddings = self._encode_schema(features, "cat_slot_value", is_training)
        value_mask = features[SchemaInputFeatures.get_input_mask_tensor_name("cat_slot_value")]
        batch_size, max_num_slots, max_num_values, _, embedding_dim = value_embeddings.size()
        value_embeddings_reshaped = value_embeddings.view(batch_size, max_num_slots * max_num_values, -1, embedding_dim)
        value_mask_reshaped = value_mask.view(batch_size, max_num_slots * max_num_values, -1)
        value_logits, _ = self._get_logits(
            value_embeddings_reshaped, value_mask_reshaped,
            _encoded_tokens, utterance_mask,
            self.categorical_slots_values_matching_layer,
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

    def _get_noncategorical_slots_goals(self, features, _encoded_tokens, is_training):
        """Obtain logits for status and slot spans for non-categorical slots."""
        # Predict the status of all non-categorical slots.
        _, slot_embeddings = self._encode_schema(features, "noncat_slot", is_training)
        slot_mask = features[SchemaInputFeatures.get_input_mask_tensor_name("noncat_slot")]
        utterance_mask = features["utt_mask"]
        max_num_slots = slot_embeddings.size()[1]
        status_logits, matching_output = self._get_logits(
            slot_embeddings, slot_mask,
            _encoded_tokens, utterance_mask,
            self.noncategorical_slots_status_matching_layer,
            self.noncategorical_slots_status_final_proj)

        # Predict the distribution for span indices.
        # (batch_size, num_elements, max_u_len, dim)
        token_embeddings = matching_output[:, :, :_encoded_tokens.size()[1], :]
        # Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.noncat_span_layer(token_embeddings)
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
        # utterance encode is shared by all classifier
        _ , _encoded_tokens = self._encode_utterances(features, is_training)
        outputs = {}
        outputs["logit_intent_status"] = self._get_intents(features, _encoded_tokens, is_training)
        outputs["logit_req_slot_status"] = self._get_requested_slots(
            features, _encoded_tokens, is_training
        )
        cat_slot_status, cat_slot_value = self._get_categorical_slots_goals(
            features, _encoded_tokens, is_training
        )
        outputs["logit_cat_slot_status"] = cat_slot_status
        outputs["logit_cat_slot_value"] = cat_slot_value
        noncat_slot_status, noncat_span_start, noncat_span_end = (
            self._get_noncategorical_slots_goals(features, _encoded_tokens, is_training))
        outputs["logit_noncat_slot_status"] = noncat_slot_status
        outputs["logit_noncat_slot_start"] = noncat_span_start
        outputs["logit_noncat_slot_end"] = noncat_span_end

        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
