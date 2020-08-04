# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : encode_utterance_schema_pair_interface.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A encode interface for encode utterance and schema with bert sentenct pair
# --------------------------------------------------------------------

import logging
import torch
import copy
from modules.schema_embedding_generator import SchemaInputFeatures
from src import utils_schema
from utils import (
    torch_ext
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)


class EncodeUttSchemaPairInterface(object):

    @classmethod
    def _encode_utterance_schema_pairs(cls, encoder, dropout_layer, features, schema_type, is_training):
        if SchemaInputFeatures.get_input_ids_tensor_name(schema_type) in features:
            return cls._concat_features_and_encode_pairs(encoder, dropout_layer, features, schema_type, is_training)
        else:
            return cls._encode_paired_features(encoder, dropout_layer, features, schema_type, is_training)

    @classmethod
    def _encode_paired_features(cls, encoder, dropout_layer, features, schema_type, is_training):
        # batch_size, max_length
        input_ids = features["input_ids"]
        input_masks = features["input_masks"]
        input_segments = features["input_segments"]

        output = encoder(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=input_segments)

        # batch_size, dim
        encoded_utt_schema_pair_cls = output[0][:, 0, :]
        # batch_size, max_length, dim
        encoded_utt_schema_pair_tokens = output[0]
        # Apply dropout in training mode.
        if is_training:
            encoded_utt_schema_pair_cls = dropout_layer(encoded_utt_schema_pair_cls)
            encoded_utt_schema_pair_tokens = dropout_layer(encoded_utt_schema_pair_tokens)
        return encoded_utt_schema_pair_cls, encoded_utt_schema_pair_tokens

    @classmethod
    def _concat_features_and_encode_pairs(cls, encoder, dropout_layer, features, schema_type, is_training):
        """
        Encode system and user utterances using BERT.
        return cls, token
        (ori_shape[:-1], -1)
        (ori_shape, embedding_dim)
        """
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        # construct input features
        schema_input_ids = features[SchemaInputFeatures.get_input_ids_tensor_name(schema_type)]
        schema_input_shape = list(schema_input_ids.size())
        batch_size = schema_input_shape[0]
        max_length = schema_input_shape[-1]
        schema_input_ids = schema_input_ids.view(batch_size, -1, max_length)
        # for cat slot value, max_schema_num = max_cat_slot x max_slot_value
        max_schema_num = schema_input_ids.size()[1]
        schema_attention_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(schema_type)]
        schema_token_type_ids = features[SchemaInputFeatures.get_input_type_ids_tensor_name(schema_type)]
        assert schema_attention_mask.size()[-1] == max_length, "schema_attention_mask has wrong shape:{}".format(schema_attention_mask.size())
        assert schema_token_type_ids.size()[-1] == max_length, "schema_token_type_ids has wrong shape:{}".format(schema_token_type_ids.size())

        # logger.info("input_ids:{}, attention_mask:{}, token_type_ids :{}".format(
        #    schema_input_ids.size(), schema_attention_mask.size(), schema_token_type_ids.size()))
        # when batchsize is too large,try to split and batch.
        # It tends that it may not help for the memory. It may help for the bert finetuning(batchsize, not sure yet)
        # (batch_size, max_schema_num, max_length)
        # for cat slot value, max_schema_num = max_cat_slot x max_slot_value
        expanded_utt_ids = features["utt"].unsqueeze(1).expand(batch_size, max_schema_num, -1)
        schema_input_ids = schema_input_ids.view(batch_size, max_schema_num, -1)
        utt_schema_pair_ids = torch.cat((expanded_utt_ids, schema_input_ids), dim=2).view(batch_size * max_schema_num, -1)
        max_total_length = utt_schema_pair_ids.size()[-1]
        expanded_utt_attention_mask = features["utt_mask"].unsqueeze(1).expand(batch_size, max_schema_num, -1)
        schema_attention_mask = schema_attention_mask.view(batch_size, max_schema_num, -1)
        utt_schema_pair_attention_mask = torch.cat((expanded_utt_attention_mask, schema_attention_mask), dim=2).view(batch_size * max_schema_num, -1)
        if features["utt_seg"] is not None:
            expanded_utt_seg_ids = features["utt_seg"].unsqueeze(1).expand(batch_size, max_schema_num, -1)
            schema_token_type_ids = schema_token_type_ids.view(batch_size, max_schema_num, -1)
            utt_schema_pair_seg_ids = torch.cat((expanded_utt_seg_ids, schema_token_type_ids), dim=2).view(batch_size * max_schema_num, -1)
        else:
            utt_schema_pair_seg_ids = None

        #logger.info("utt_scehma_pair_ids:{}, utt_schema_pair_attention_mask:{}, token_type_ids={}".format(
        #    utt_schema_pair_ids[0,:],
        #    utt_schema_pair_attention_mask[0,:],
        #    utt_schema_pair_seg_ids[0,:]))

        output = encoder(
            input_ids=utt_schema_pair_ids,
            attention_mask=utt_schema_pair_attention_mask,
            token_type_ids=utt_schema_pair_seg_ids)

        cls_shape = copy.deepcopy(schema_input_shape)
        cls_shape[-1] = -1
        encoded_utt_schema_pair_cls = output[0][:, 0, :].view(cls_shape)

        token_shape = copy.deepcopy(schema_input_shape)
        token_shape[-1] = max_total_length
        token_shape.append(-1)
        # token_shape, schema_input_shape, -1
        encoded_utt_schema_pair_tokens = output[0].view(token_shape)
        # Apply dropout in training mode.
        if is_training:
            encoded_utt_schema_pair_cls = dropout_layer(encoded_utt_schema_pair_cls)
            encoded_utt_schema_pair_tokens = dropout_layer(encoded_utt_schema_pair_tokens)
        return encoded_utt_schema_pair_cls, encoded_utt_schema_pair_tokens
