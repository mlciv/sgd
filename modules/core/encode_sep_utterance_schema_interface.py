# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : encode_sep_utterance_schema_pair_interface.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A encode interface for seperately encode utterance and schema with bert sentenct pair
# --------------------------------------------------------------------

import logging
import torch
import copy
from modules.schema_embedding_generator import SchemaInputFeatures
from src import utils_schema
from utils import (
    torch_ext
)
from transformers import BertTokenizer

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)


class EncodeSepUttSchemaInterface(object):

    @classmethod
    def _encode_utterances(cls, tokenzier, encoder, features, dropout_layer, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        # logger.info("utt:{}, utt_mask:{}, utt_seg:{}".format(features["utt"], features["utt_mask"], features["utt_seg"]))
        output = encoder(
            input_ids=features["utt"],
            attention_mask=features["utt_mask"],
            token_type_ids=features["utt_seg"])

        encoded_utterance = output[0][:, 0, :]
        encoded_tokens = output[0]
        # Apply dropout in training mode.
        if is_training:
            encoded_utterance = dropout_layer(encoded_utterance)
            encoded_tokens = dropout_layer(encoded_tokens)
        return encoded_utterance, encoded_tokens, features["utt_mask"]

    @classmethod
    def _encode_schema(cls, tokenizer, encoder, features, dropout_layer, schema_type, is_training):
        """
        Encode system and user utterances using BERT.
        return cls , and token embedding
        token_embedding always contains all the special tokens
        """
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        schema_input_ids = features[SchemaInputFeatures.get_input_ids_tensor_name(schema_type)]
        schema_input_shape = list(schema_input_ids.size())
        batch_size = schema_input_shape[0]
        max_length = schema_input_shape[-1]
        schema_input_ids = schema_input_ids.view(batch_size, -1, max_length)
        max_schema_num = schema_input_ids.size()[1]
        schema_attention_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(schema_type)]
        schema_token_type_ids = features[SchemaInputFeatures.get_input_type_ids_tensor_name(schema_type)]
        assert schema_attention_mask.size()[-1] == max_length, "schema_attention_mask has wrong shape:{}".format(schema_attention_mask.size())
        assert schema_token_type_ids.size()[-1] == max_length, "schema_token_type_ids has wrong shape:{}".format(schema_token_type_ids.size())

        schema_input_ids = schema_input_ids.view(batch_size, max_schema_num, -1)
        schema_attention_mask = schema_attention_mask.view(batch_size, max_schema_num, -1)
        # TODO: we need judge the type of encoder
        # For single sentence, there, we consider bert and roberta, both of them are [CLS]  X [SEP]
        if features["utt_seg"] is not None:
            schema_token_type_ids = schema_token_type_ids.view(batch_size, max_schema_num, -1)
        else:
            schema_token_type_ids = None

        if isinstance(tokenizer, BertTokenizer):
            # adding [CLS] in the begining
            max_total_length = 1 + max_length
            begingging_cls_ids = torch.ones(batch_size, max_schema_num, 1) * tokenizer.cls_token_id
            beginning_cls_attention_mask = torch.ones(batch_size, max_schema_num, 1)
            # changing token_type to 0 in seq1 in single sentence
            adjusted_token_type_ids = torch.zeros(batch_size, max_schema_num, max_total_length)
            adjusted_schema_input_ids = torch.cat((beginning_cls_ids, schema_input_ids), dim=2)
            adjusted_schema_attention_mask = torch.cat((begging_cls_attention_mask, schema_attention_mask), dim=2)
        else:
            adjusted_schema_input_ids = schema_input_ids
            adjusted_schema_attention_mask = schema_attention_mask
            adjusted_schema_token_type_ids = None

        output = encoder(
            input_ids=adjusted_schema_input_ids,
            attention_mask=adjusted_schema_attention_mask,
            token_type_ids=adjusted_schema_token_type_ids)

        cls_shape = copy.deepcopy(schema_input_shape)
        # cls ignore the length
        cls_shape[-1] = -1
        encoded_schema_cls = output[0][:, 0, :].view(cls_shape)

        token_shape = copy.deepcopy(schema_input_shape)
        token_shape[-1] = max_total_length
        token_shape.append(-1)
        # token_shape, schema_input_shape, -1
        encoded_schema_tokens = output[0].view(token_shape)
        # Apply dropout in training mode.
        if is_training:
            encoded_schema_cls = dropout_layer(encoded_schema_cls)
            encoded_schema_tokens = dropout_layer(encoded_schema_tokens)
        return encoded_schema_cls, encoded_schema_tokens, adjusted_schema_attention_mask
