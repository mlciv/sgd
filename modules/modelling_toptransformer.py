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
from modules.core.encode_sep_utterance_schema_interface import EncodeSepUttSchemaInterface
from modules.schema_embedding_generator import SchemaInputFeatures
import modules.core.schema_constants as schema_constants
from src import utils_schema
from utils import (
    torch_ext,
    data_utils,
    schema,
    scalar_mix
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)

_NL_SEPARATOR = "|||"
# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class TopTransformerModel(PreTrainedModel, EncodeSepUttSchemaInterface, DSTC8BaselineOutputInterface):
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

    def __init__(self, config=None, args=None, encoder=None, utt_encoder=None, schema_encoder=None):
        super(TopTransformerModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
        # utt encoder
        if utt_encoder:
            self.utt_encoder = utt_encoder
        else:
            self.utt_encoder = EncoderUtils.create_encoder(self.config)
            EncoderUtils.set_encoder_finetuning_status(self.utt_encoder, args.encoder_finetuning)
        EncoderUtils.add_special_tokens(self.tokenizer, self.utt_encoder, schema_constants.USER_AGENT_SPECIAL_TOKENS)
        # schema_encoder
        if schema_encoder:
            self.schema_encoder = schema_encoder
        else:
            # TODO: now we don't consider a seperate encoder for schema
            # Given the matchng layer above, it seems sharing the embedding layer and encoding is good
            # But if the schema description is a graph or other style, we can consider a GCN or others
            new_schema_encoder = EncoderUtils.create_encoder(self.config)
            new_schema_encoder.embeddings = self.utt_encoder.embeddings
            self.schema_encoder = new_schema_encoder
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        if self.utterance_embedding_dim == self.config.d_model:
            self.utterance_projection_layer = torch.nn.Sequential()
        else:
            self.utterance_projection_layer = nn.Linear(self.embedding_dim, self.config.d_model)

        self.schema_embedding_dim = self.config.schema_embedding_dim
        self.schema_dropout = torch.nn.Dropout(self.config.schema_dropout)
        if self.schema_embedding_dim == self.config.d_model:
            self.schema_projection_layer = torch.nn.Sequential()
        else:
            # Here, we share the encoder with utt_encoder, hence,share the prejection too,
            # Later, to support seperate projection layer
            self.schema_projection_layer = self.utterance_projection_layer

        if self.config.bert_mix_layers > 1:
            self.scalar_utt_mix = scalar_mix.ScalarMix(self.config.bert_mix_layers, do_layer_norm=False)
            self.scalar_schema_mix = scalar_mix.ScalarMix(self.config.bert_mix_layers, do_layer_norm=False)
        else:
            self.scalar_utt_mix = None
            self.scalar_schema_mix = None

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

        self.cat_slots_status_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.cat_slots_values_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.noncat_slots_status_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        # Project the combined embeddings to obtain logits.
        # for intent, one logits
        self.intent_final_dropout = torch.nn.Dropout(self.config.final_dropout)
        self.intent_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 1)
        )

        # for requested slots, one logits
        self.requested_slots_final_dropout = torch.nn.Dropout(self.config.final_dropout)
        self.requested_slots_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 1)
        )

        # for categorical_slots, 3 logits
        self.cat_slots_final_dropout = torch.nn.Dropout(self.config.final_dropout)
        self.cat_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 3)
        )

        # for categorical_slot_values,
        self.cat_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 1)
        )

        # for non-categorical_slots, 3 logits
        self.noncat_slots_final_dropout = torch.nn.Dropout(self.config.final_dropout)
        self.noncat_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 3)
        )
        self.noncat_span_layer = nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 1)
        )
        # for non-categorical span value
        if not args.no_cuda:
            self.cuda(args.device)

    def _get_logits(self, element_embeddings, element_mask, _encoded_tokens, utterance_mask, matching_layer, final_proj, final_dropout_layer, is_training):
        """Get logits for elements by conditioning on utterance embedding.
        Args:
        element_embeddings: A tensor of shape (batch_size, num_elements,
        max_seq_length, embedding_dim).
        element_mask: (batch_size, num_elements, max_seq_length)
        _encoded_tokens: (batch_size, max_u_len, embedding_dim)
        _utterance_mask: (batch_size, max_u_len)
        num_classes: An int containing the number of classes for which logits are
        to be generated.
        Returns:
        A tensor of shape (batch_size, num_elements, num_classes) containing the
        logits.
        """
        # when it is flatten, num_elements is 1
        batch_size1, max_schema_num, max_seq_length, element_emb_dim = element_embeddings.size()
        batch_size2, max_u_len, utterance_dim = _encoded_tokens.size()
        assert batch_size1 == batch_size2, "batch_size not match between element_emb and utterance_emb"
        expanded_encoded_tokens = _encoded_tokens.unsqueeze(1).expand(batch_size2, max_schema_num, max_u_len, utterance_dim)
        expanded_utterance_mask = utterance_mask.unsqueeze(1).expand(batch_size2, max_schema_num, max_u_len)
        assert element_emb_dim == utterance_dim, "dim not much element_emb={} and utterance_emb={}".format(element_emb_dim, utterance_dim)
        max_total_len = max_u_len + max_seq_length
        # (batch_size * num_elements, max_total_len, dim)
        utterance_element_pair_emb = torch.cat(
            (self.utterance_projection_layer(expanded_encoded_tokens), self.schema_projection_layer(element_embeddings)),
            dim=2
        ).view(-1, max_total_len, element_emb_dim)
        # (batch_size * num_elements, max_u_len+max_seq_length)
        utterance_element_pair_mask = torch.cat(
            (expanded_utterance_mask, element_mask),
            dim=2).view(-1, max_total_len).bool()

        # trans_output: (batch_size, max_total_len, dim)
        trans_output = matching_layer(
            utterance_element_pair_emb.transpose(0, 1),
            src_key_padding_mask=utterance_element_pair_mask).transpose(0, 1).view(batch_size1, max_schema_num, max_total_len, -1)
        # TODO: for model type
        final_cls = trans_output[:, :, 0, :]
        if is_training:
            final_cls = final_dropout_layer(final_cls)
            trans_output = final_dropout_layer(trans_output)
        # TODO: use the first [CLS] for classification, if XLNET, it is the last one
        # batch_size, max_schema_num
        output = final_proj(final_cls)
        return output, trans_output

    def _get_intents(self, features, encoded_utt_tokens, encoded_utt_mask, is_training):
        """Obtain logits for intents."""
        # service_id is the index of emb value
        # [batch, max_intentnum, max_seq_length, dim]
        encoded_intent_cls, encoded_intent_tokens, encoded_intent_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, self.scalar_schema_mix, "intent", is_training)
        # self.device is the device for Dataparallel model,which device 0
        # Here we need the device current input on
        current_device = encoded_intent_tokens.device
        # Add a trainable vector for the NONE intent.
        batch_size, max_num_intents, max_seq_length, embedding_dim = encoded_intent_tokens.size()
        # init a matrix
        null_intent_embedding = torch.empty(1, 1, max_seq_length, embedding_dim, device=current_device)
        null_intent_mask = torch.zeros(1, 1, max_seq_length, dtype=torch.long, device=current_device)
        null_intent_mask[:, :, 0] = 1
        torch.nn.init.normal_(null_intent_embedding, std=0.02)
        repeated_null_intent_embedding = null_intent_embedding.expand(batch_size, 1, -1, -1)
        # batch_size, 1, max_seq_length]
        repeated_null_intent_mask = null_intent_mask.expand(batch_size, 1, -1)
        intent_embeddings = torch.cat(
            (repeated_null_intent_embedding, encoded_intent_tokens), dim=1)
        intent_mask = torch.cat((repeated_null_intent_mask, encoded_intent_mask), dim=1)

        logits, _ = self._get_logits(
            intent_embeddings, intent_mask,
            encoded_utt_tokens, encoded_utt_mask,
            self.intent_matching_layer, self.intent_final_proj, self.intent_final_dropout, is_training)
        # Shape: (batch_size, max_intents + 1)
        logits = logits.squeeze(-1)
        # Mask out logits for padded intents. 1 is added to account for NONE intent.
        # [batch_size, max_intent+1]
        mask = torch_ext.sequence_mask(
            features["intent_num"] + 1,
            maxlen=max_num_intents + 1, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(logits) * torch.finfo(torch.float16).max
        return torch.where(mask, logits, negative_logits)

    def _get_requested_slots(self, features, encoded_utt_tokens, encoded_utt_mask, is_training):
        """Obtain logits for requested slots."""
        encoded_req_slot_cls, encoded_req_slot_tokens, encoded_req_slot_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, self.scalar_schema_mix, "req_slot", is_training)
        logits, _ = self._get_logits(
            encoded_req_slot_tokens, encoded_req_slot_mask,
            encoded_utt_tokens, encoded_utt_mask,
            self.requested_slots_matching_layer, self.requested_slots_final_proj, self.requested_slots_final_dropout, is_training)
        return torch.squeeze(logits, dim=-1)

    def _get_categorical_slots_goals(self, features, encoded_utt_tokens, encoded_utt_mask, is_training):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots.
        encoded_cat_slot_cls, encoded_cat_slot_tokens, encoded_cat_slot_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, self.scalar_schema_mix, "cat_slot", is_training)
        status_logits, _ = self._get_logits(
            encoded_cat_slot_tokens, encoded_cat_slot_mask,
            encoded_utt_tokens, encoded_utt_mask,
            self.cat_slots_status_matching_layer,
            self.cat_slots_status_final_proj, self.cat_slots_final_dropout, is_training)
        # logger.info("status_logits:{}, utterance_mask:{}".format(status_logits.size(), utterance_mask.size()))
        # Predict the goal value.
        # Shape: (batch_size, max_categorical_slots, max_categorical_values,
        # embedding_dim).
        encoded_slot_value_cls, encoded_slot_value_tokens, encoded_slot_value_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, self.scalar_schema_mix, "cat_slot_value", is_training)
        batch_size, max_num_slots, max_num_values, _, embedding_dim = encoded_slot_value_tokens.size()
        value_embeddings_reshaped = encoded_slot_value_tokens.view(batch_size, max_num_slots * max_num_values, -1, embedding_dim)
        value_mask_reshaped = encoded_slot_value_mask.view(batch_size, max_num_slots * max_num_values, -1)
        value_logits, _ = self._get_logits(
            value_embeddings_reshaped, value_mask_reshaped,
            encoded_utt_tokens, encoded_utt_mask,
            self.cat_slots_values_matching_layer,
            self.cat_slots_values_final_proj, self.cat_slots_final_dropout, is_training)
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

    def _get_noncategorical_slots_goals(self, features, encoded_utt_tokens, encoded_utt_mask, is_training):
        """Obtain logits for status and slot spans for non-categorical slots."""
        # Predict the status of all non-categorical slots.
        encoded_noncat_slot_cls, encoded_noncat_slot_tokens, encoded_noncat_slot_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, self.scalar_schema_mix, "noncat_slot", is_training)
        max_num_slots = encoded_noncat_slot_tokens.size()[1]
        status_logits, matching_output = self._get_logits(
            encoded_noncat_slot_tokens, encoded_noncat_slot_mask,
            encoded_utt_tokens, encoded_utt_mask,
            self.noncat_slots_status_matching_layer,
            self.noncat_slots_status_final_proj, self.noncat_slots_final_dropout, is_training)

        # Predict the distribution for span indices.
        # (batch_size, num_elements, max_u_len, dim)
        token_embeddings = matching_output[:, :, :encoded_utt_tokens.size()[1], :]
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
        encoded_utt_cls, encoded_utt_tokens, encoded_utt_mask = self._encode_utterances(
            self.tokenizer, self.utt_encoder, features, self.utterance_dropout, self.scalar_utt_mix, is_training)
        outputs = {}
        outputs["logit_intent_status"] = self._get_intents(
            features, encoded_utt_tokens, encoded_utt_mask, is_training)
        outputs["logit_req_slot_status"] = self._get_requested_slots(
            features, encoded_utt_tokens, encoded_utt_mask, is_training
        )
        cat_slot_status, cat_slot_value = self._get_categorical_slots_goals(
            features, encoded_utt_tokens, encoded_utt_mask, is_training
        )
        outputs["logit_cat_slot_status"] = cat_slot_status
        outputs["logit_cat_slot_value"] = cat_slot_value
        noncat_slot_status, noncat_span_start, noncat_span_end = (
            self._get_noncategorical_slots_goals(
                features, encoded_utt_tokens, encoded_utt_mask, is_training))
        outputs["logit_noncat_slot_status"] = noncat_slot_status
        outputs["logit_noncat_slot_start"] = noncat_span_start
        outputs["logit_noncat_slot_end"] = noncat_span_end

        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
