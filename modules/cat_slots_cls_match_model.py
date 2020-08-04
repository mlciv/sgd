# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : cat_slot_cls_match_model.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict categorical slots
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

class CatSlotsCLSMatchModel(PreTrainedModel, DSTC8BaselineOutputInterface):
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

    def __init__(self, config=None, args=None, encoder=None):
        super(CatSlotsCLSMatchModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = EncoderUtils.create_encoder(self.config)
            EncoderUtils.set_encoder_finetuning_status(self.encoder, args.encoder_finetuning)

        EncoderUtils.add_special_tokens(self.tokenizer, encoder, schema_constants.USER_AGENT_SPECIAL_TOKENS)
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.embedding_dim = self.config.schema_embedding_dim
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        self.token_dropout = torch.nn.Dropout(self.config.token_dropout)
        self.categorical_slots_status_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.categorical_slots_values_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
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

    def _encode_schema(self, features, schema_type, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        if self.config.schema_finetuning_type == "finetuning_cls":
            schema_input_ids = features[SchemaInputFeatures.get_input_ids_tensor_name(schema_type)]
            schema_input_shape = list(schema_input_ids.size())
            max_length = schema_input_shape[-1]
            schema_attention_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(schema_type)]
            schema_token_type_ids = features[SchemaInputFeatures.get_input_type_ids_tensor_name(schema_type)]
            assert schema_attention_mask.size()[-1] == max_length, "schema_attention_mask has wrong shape:{}".format(schema_attention_mask.size())
            assert schema_token_type_ids.size()[-1] == max_length, "schema_token_type_ids has wrong shape:{}".format(schema_token_type_ids.size())

            # logger.info("input_ids:{}, attention_mask:{}, token_type_ids :{}".format(
            #    schema_input_ids.size(), schema_attention_mask.size(), schema_token_type_ids.size()))
            # when batchsize is too large,try to split and batch.
            # It tends that it may not help for the memory. It may help for the bert finetuning(batchsize, not sure yet)
            total_schema_input_ids = schema_input_ids.view(-1, schema_input_ids.size()[-1])
            total_batch_size = total_schema_input_ids.size()[0]
            total_schema_attention_mask = schema_attention_mask.view(-1, schema_input_ids.size()[-1])
            total_schema_token_type_ids = schema_token_type_ids.view(-1, schema_input_ids.size()[-1])

            i = 0
            split_schema_cls = []
            split_schema_tokens = []
            while(i * self.config.schema_batch_size < total_batch_size):
                left = i * self.config.schema_batch_size
                right = min((i+1)*self.config.schema_batch_size, total_batch_size)
                i = i + 1
                # logger.info("encoding {} input: {} - {}".format(schema_type, left, right))
                output = self.encoder(
                    input_ids=total_schema_input_ids[left:right, :],
                    attention_mask=total_schema_attention_mask[left:right, :],
                    token_type_ids=total_schema_token_type_ids[left:right, :])

                split_schema_cls.append(output[0][:, 0, :])
                split_schema_tokens.append(output[0])

            # concat
            encoded_schema_cls = torch.cat(split_schema_cls, dim=0)
            encoded_schema_tokens = torch.cat(split_schema_tokens, dim=0)
            # adding the last dim
            embedding_dim = encoded_schema_cls.size()[-1]
            schema_cls_shape = schema_input_shape[:-1]
            schema_cls_shape.append(embedding_dim)
            schema_input_shape.append(embedding_dim)
            # Apply dropout in training mode.
            if is_training:
                encoded_schema_cls = self.utterance_dropout(encoded_schema_cls)
                encoded_schema_tokens = self.utterance_dropout(encoded_schema_tokens)
            schema_cls_return = encoded_schema_cls.view(torch.Size(schema_cls_shape))
            schema_token_return = encoded_schema_tokens.view(torch.Size(schema_input_shape))
        elif self.config.schema_finetuning_type == "fixed_cls":
            encoded_schema_cls = features[SchemaInputFeatures.get_embedding_tensor_name(schema_type)]
            # Apply dropout in training mode.
            if is_training:
                encoded_schema_cls = self.utterance_dropout(encoded_schema_cls)
            schema_cls_return = encoded_schema_cls
            schema_token_return = None
        else:
            raise NotImplementedError("{} is not supported in CLS matching".format(self.config.schema_finetuning_type))
        return schema_cls_return, schema_token_return

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

    def _get_categorical_slots_goals(self, features, _encoded_utterance, is_training):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots.
        slot_embeddings, _ = self._encode_schema(features, "cat_slot", is_training)
        status_logits = self._get_logits(slot_embeddings,
                                         _encoded_utterance,
                                         self.categorical_slots_status_utterance_proj,
                                         self.categorical_slots_status_final_proj)
        # Predict the goal value.
        # Shape: (batch_size, max_categorical_slots, max_categorical_values,
        # embedding_dim).
        value_embeddings, _ = self._encode_schema(features, "cat_slot_value", is_training)
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
        cat_slot_status, cat_slot_value = self._get_categorical_slots_goals(features, _encoded_utterance, is_training)
        outputs["logit_cat_slot_status"] = cat_slot_status
        outputs["logit_cat_slot_value"] = cat_slot_value
        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
