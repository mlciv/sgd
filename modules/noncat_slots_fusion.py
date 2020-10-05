# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : noncat_slots_fusion.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict noncat slot goals
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
from modules.fixed_schema_cache import FixedSchemaCacheEncoder
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

class NonCatSlotsFusionModel(PreTrainedModel, EncodeSepUttSchemaInterface, DSTC8BaselineOutputInterface):
    """
    Main entry of Classifier Model
    """

    config_class = SchemaDSTConfig
    base_model_prefix = ""

    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config=None, args=None, encoder=None, utt_encoder=None, schema_encoder=None):
        super(NonCatSlotsFusionModel, self).__init__(config=config)
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

        self.noncat_slots_status_matching_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(self.config.d_model, self.config.nhead, self.config.dim_feedforward),
            num_layers=self.config.num_matching_layer,
            norm=torch.nn.LayerNorm(self.config.d_model)
        )

        # Project the combined embeddings to obtain logits.
        # for intent, one logits
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

    @classmethod
    def _encode_schema(cls, tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training):
        return FixedSchemaCacheEncoder._encode_schema(tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training)

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
