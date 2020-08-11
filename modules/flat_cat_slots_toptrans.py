# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : flat_cat_slots_toptrans.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict cat slots, flatten examples
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
from modules.flat_dst_output_interface import FlatDSTOutputInterface
from modules.core.encode_sep_utterance_schema_interface import EncodeSepUttSchemaInterface
from modules.schema_embedding_generator import SchemaInputFeatures
import modules.core.schema_constants as schema_constants
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

class FlatCatSlotsTopTransModel(PreTrainedModel, EncodeSepUttSchemaInterface, FlatDSTOutputInterface):
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
        super(FlatCatSlotsTopTransModel, self).__init__(config=config)
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
            self.schema_encoder = self.utt_encoder
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
            self.schema_projection_layer =  self.utterance_projection_layer

        self.categorical_slots_values_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        self.categorical_slots_values_final_dropout = torch.nn.Dropout(self.config.final_dropout)
        # Project the combined embeddings to obtain logits.
        # then max p_active, if all p_active < 0.5, then it is null
        self.categorical_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.utterance_embedding_dim, 1)
        )

        if not args.no_cuda:
            self.cuda(args.device)

    def _get_logits(self, element_embeddings, element_mask, _encoded_tokens, utterance_mask, matching_layer, final_proj, is_training):
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
        # when it is flatten, num_elements is 1
        batch_size1, max_seq_length, element_emb_dim = element_embeddings.size()
        batch_size2, max_u_len, utterance_dim = _encoded_tokens.size()
        assert batch_size1 == batch_size2, "batch_size not match between element_emb and utterance_emb"
        assert element_emb_dim == utterance_dim, "dim not much element_emb={} and utterance_emb={}".format(element_emb_dim, utterance_dim)
        max_total_len = max_u_len + max_seq_length
        # (batch_size * num_elements, max_total_len, dim)
        utterance_element_pair_emb = torch.cat(
            (self.utterance_projection_layer(_encoded_tokens), self.schema_projection_layer(element_embeddings)), dim=1)
        # (batch_size, max_u_len+max_seq_length)
        utterance_element_pair_mask = torch.cat(
            (utterance_mask, element_mask),
            dim=1).view(-1, max_total_len).bool()

        # trans_output: (batch_size, max_total_len, dim)
        trans_output = matching_layer(
            utterance_element_pair_emb.transpose(0, 1),
            src_key_padding_mask=utterance_element_pair_mask).transpose(0, 1)
        # TODO: for model type
        final_cls = trans_output[:, 0, :]
        if is_training:
            final_cls = self.categorical_slots_values_final_dropout(final_cls)
        # TODO: use the first [CLS] for classification, if XLNET, it is the last one
        output = final_proj(final_cls)
        return output

    def _get_categorical_slots_goals(self, features, is_training):
        """Obtain logits for intents."""
        # we only use cls token for matching, either finetuned cls and fixed cls
        encoded_utt_cls, encoded_utt_tokens, encoded_utt_mask = self._encode_utterances(
            self.tokenizer, self.utt_encoder, features, self.utterance_dropout, is_training)
        encoded_schema_cls, encoded_schema_tokens, encoded_schema_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, "cat_slot", is_training)
        logits = self._get_logits(
            encoded_schema_tokens, encoded_schema_mask,
            encoded_utt_tokens, encoded_utt_mask,
            self.categorical_slots_values_matching_layer, self.categorical_slots_values_final_proj, is_training)
        # Shape: (batch_size, 1)
        logits = logits.squeeze(-1)
        # (batch_size, )
        return logits

    def forward(self, features, labels=None):
        is_training = (labels is not None)
        outputs = {}
        cat_slot_value_status = self._get_categorical_slots_goals(features, is_training)
        outputs["logit_cat_slot_value_status"] = cat_slot_value_status
        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
