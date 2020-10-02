# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : flat_active_dual_encoder.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict active_intent, flatten examples
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
from modules.core.matcher import ConcatenateMatcher, BilinearMatcher, DotProductMatcher
from modules.fixed_schema_cache import FixedSchemaCacheEncoder
from modules.core.schemadst_configuration import SchemaDSTConfig
from modules.flat_dst_output_interface import FlatDSTOutputInterface
from modules.core.encode_sep_utterance_schema_interface import EncodeSepUttSchemaInterface
from modules.dstc8baseline_output_interface import DSTC8BaselineOutputInterface
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

# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class FlatActiveIntentDualEncoder(PreTrainedModel, EncodeSepUttSchemaInterface, FixedSchemaCacheEncoder, FlatDSTOutputInterface):
    """
    Main entry of Classifier Model
    """
    config_class = SchemaDSTConfig
    base_model_prefix = ""
    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config=None, args=None, encoder=None, utt_encoder=None, schema_encoder=None):
        super(FlatActiveIntentDualEncoder, self).__init__(config=config)
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
            if self.config.schema_embedding_type in ["token", "flat_token"]:
                self.schema_encoder = None
            else:
                new_schema_encoder = EncoderUtils.create_encoder(self.config)
                new_schema_encoder.embeddings = self.utt_encoder.embeddings
                self.schema_encoder = new_schema_encoder
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        self.intent_seq2_key = self.config.intent_seq2_key
        self.schema_embedding_dim = self.config.schema_embedding_dim
        self.schema_dropout = torch.nn.Dropout(self.config.schema_dropout)
        # we always project intent schema
        if self.schema_embedding_dim != self.utterance_embedding_dim:
            self.schema_projection_layer = torch.nn.Sequential(
                torch.nn.Linear(self.intent_embedding_size, self.utterance_embedding_dim)
            )
        else:
            self.schema_projection_layer = torch.nn.Sequential()

        self.schema_embedding_dim = self.config.schema_embedding_dim
        self.schema_dropout = torch.nn.Dropout(self.config.schema_dropout)
        self.utterance_projection_layer = torch.nn.Sequential()

        # for cls matching, mix is None
        self.scalar_utt_mix = None
        self.scalar_schema_mix = None

        if self.config.matcher == "ConcatenateMatcher":
            self.intent_matching_layer = ConcatenateMatcher(encoding_dim=self.utterance_embedding_dim)
        elif self.config.matcher == "DotProductMatcher":
            self.intent_matching_layer = DotProductMatcher(encoding_dim=self.utterance_embedding_dim)
        elif self.config.matcher == "BilinearMatcher":
            self.intent_matching_layer = BilinearMatcher(encoding_dim=self.utterance_embedding_dim)
        else:
            raise NotImplementedError("Not implemented for matcher {}".format(self.config.matcher))

        self.intent_final_dropout = torch.nn.Sequential()
        # Project the combined embeddings to obtain logits.
        # then max p_active, if all p_active < 0.5, then it is null
        if self.intent_matching_layer.outputDim == 1:
            self.intent_final_proj = torch.nn.Sequential(
            )
        else:
            self.intent_final_proj = torch.nn.Sequential(
                torch.nn.Linear(self.intent_matching_layer.outputDim, 1)
            )

        if not args.no_cuda:
            self.cuda(args.device)

    def _get_logits(self, schema_rep, utterance_rep, matching_layer, final_proj, is_training):
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
        batch_size1, element_emb_dim = schema_rep.size()
        batch_size2, utterance_dim = utterance_rep.size()
        assert batch_size1 == batch_size2, "batch_size not match between element_emb and utterance_emb"
        projected_schema_rep = self.schema_projection_layer(schema_rep)
        if is_training:
            utterance_rep = self.utterance_dropout(utterance_rep)
            projected_schema_rep = self.schema_dropout(projected_schema_rep)
        # batch_size x num_cat_value, output_dim
        pair_encodings = matching_layer(utterance_rep, projected_schema_rep)
        logits = final_proj(pair_encodings)
        logits = logits.view(batch_size1, -1)
        return logits

    def _get_intents(self, features, is_training):
        """Obtain logits for intents."""
        encoded_utt_cls, encoded_utt_tokens, encoded_utt_mask = self._encode_utterances(
            self.tokenizer, self.utt_encoder, features, self.utterance_dropout, self.scalar_utt_mix, is_training)
        encoded_schema_cls, encoded_schema_tokens, encoded_schema_mask = self._encode_schema(
            self.tokenizer, self.schema_encoder, features, self.schema_dropout, self.scalar_schema_mix, self.intent_seq2_key, is_training)
        logits = self._get_logits(
            encoded_schema_cls,
            encoded_utt_cls,
            self.intent_matching_layer, self.intent_final_proj, is_training)
        # Shape: (batch_size, 1)
        logits = logits.squeeze(-1)
        # (batch_size, )
        return logits

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
        outputs = {}
        outputs["logit_intent_status"] = self._get_intents(features, is_training)
        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )

    @classmethod
    def _encode_schema(cls, tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training):
        return FixedSchemaCacheEncoder._encode_schema(tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training)
