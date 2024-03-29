# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : encode_sep_utterance_schema_pair_interface.py
<<<<<<< HEAD
# Original Author    : cajie@amazon.com
=======
# Original Author    : jiessie.cao@gmail.com
>>>>>>> 4131baf55e48139fdc95ab20c573a529d9982b3d
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

<<<<<<< HEAD
=======
# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
>>>>>>> 4131baf55e48139fdc95ab20c573a529d9982b3d
class EncodeSepUttSchemaInterface(object):

    @classmethod
    def _encode_utterances(cls, tokenzier, encoder, features, dropout_layer, _scalar_mix, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        # logger.info("utt:{}, utt_mask:{}, utt_seg:{}".format(features["utt"], features["utt_mask"], features["utt_seg"]))
        last_encoder_layer, pooled_output, all_encoder_layers, _ = encoder(
            input_ids=features["utt"],
            attention_mask=features["utt_mask"],
            token_type_ids=features["utt_seg"],
            output_hidden_states=True,
            output_attentions=True,
        )

        if _scalar_mix is not None:
            # when not do layer norm, input_mask is not used
            selected_layers = list(all_encoder_layers[:_scalar_mix.mixture_size])
            mix = _scalar_mix(tensors=selected_layers, mask=features["utt_mask"])
        else:
            mix = last_encoder_layer

        encoded_utterance = mix[:, 0, :]
        encoded_tokens = mix
        # Apply dropout in training mode.
        if is_training:
            encoded_utterance = dropout_layer(encoded_utterance)
            encoded_tokens = dropout_layer(encoded_tokens)
        return encoded_utterance, encoded_tokens, features["utt_mask"]


    @classmethod
    def _encode_schema(cls, tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training):
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

        schema_attention_mask = schema_attention_mask.view(batch_size, max_schema_num, -1)
        # TODO: we need judge the type of encoder
        # For single sentence, there, we consider bert and roberta, both of them are [CLS]  X [SEP]
        if features["utt_seg"] is not None:
            schema_token_type_ids = schema_token_type_ids.view(batch_size, max_schema_num, -1)
        else:
            schema_token_type_ids = None

        # if is bert, we need to adjust it a little, by adding a begin cls
        if isinstance(tokenizer, BertTokenizer):
            # adding [CLS] in the begining
            max_total_length = 1 + max_length
            current_device = schema_input_ids.device
            begin_cls_ids = torch.ones(batch_size, max_schema_num, 1, device=current_device).long() * tokenizer.cls_token_id
            begin_cls_attention_mask = torch.ones(batch_size, max_schema_num, 1, device=current_device).long()
            # changing token_type to 0 in seq1 in single sentence
            adjusted_schema_token_type_ids = torch.zeros(batch_size, max_schema_num, max_total_length, device=current_device).long().view(batch_size*max_schema_num, -1)
            adjusted_schema_input_ids = torch.cat((begin_cls_ids, schema_input_ids), dim=2).long().view(batch_size*max_schema_num, -1)
            adjusted_schema_attention_mask = torch.cat((begin_cls_attention_mask, schema_attention_mask), dim=2).view(batch_size*max_schema_num, -1)
        else:
            adjusted_schema_input_ids = schema_input_ids.view(batch_size*max_schema_num, -1)
            adjusted_schema_attention_mask = schema_attention_mask.view(batch_size*max_schema_num, -1)
            adjusted_schema_token_type_ids = None

        last_encoder_layer, pooled_output, all_encoder_layers, _ = encoder(
            input_ids=adjusted_schema_input_ids,
            attention_mask=adjusted_schema_attention_mask,
            token_type_ids=adjusted_schema_token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

        if _scalar_mix is not None:
            # when not do layer norm, input_mask is not used
            mix = _scalar_mix(tensors=list(all_encoder_layers[:_scalar_mix.mixture_size]), mask=features["utt_mask"])
        else:
            mix = last_encoder_layer

        cls_shape = copy.deepcopy(schema_input_shape)
        # cls ignore the length
        cls_shape[-1] = -1
        encoded_schema_cls = mix[:, 0, :].view(cls_shape)

        token_shape = copy.deepcopy(schema_input_shape)
        token_shape[-1] = max_total_length
        token_shape.append(-1)
        # token_shape, schema_input_shape, -1
        encoded_schema_tokens = mix.view(token_shape)
        # Apply dropout in training mode.
        if is_training:
            encoded_schema_cls = dropout_layer(encoded_schema_cls)
            encoded_schema_tokens = dropout_layer(encoded_schema_tokens)
        return encoded_schema_cls, encoded_schema_tokens, adjusted_schema_attention_mask
