# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : cat_slots_fusion.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict cat slot goals
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

class CatSlotsFusionModel(PreTrainedModel, EncodeSepUttSchemaInterface, DSTC8BaselineOutputInterface):
    """
    Main entry of Classifier Model
    """

    config_class = SchemaDSTConfig
    base_model_prefix = ""

    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config=None, args=None, encoder=None, utt_encoder=None, schema_encoder=None):
        super(CatSlotsFusionModel, self).__init__(config=config)
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

        # Project the combined embeddings to obtain logits.
        # for categorical_slots, 3 logits
        self.cat_slots_final_dropout = torch.nn.Dropout(self.config.final_dropout)
        self.cat_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 3)
        )

        # for categorical_slot_values,
        self.cat_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(int(self.config.d_model), 1)
        )

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
        cat_slot_status, cat_slot_value = self._get_categorical_slots_goals(
            features, encoded_utt_tokens, encoded_utt_mask, is_training
        )
        outputs["logit_cat_slot_status"] = cat_slot_status
        outputs["logit_cat_slot_value"] = cat_slot_value

        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
