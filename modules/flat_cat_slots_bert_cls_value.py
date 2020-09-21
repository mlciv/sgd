# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : flat_cat_slots_bert_cls_value.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict active_intent
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
from modules.core.schemadst_configuration import SchemaDSTConfig
from modules.flat_dst_output_interface import FlatDSTOutputInterface
from modules.core.encode_utterance_schema_pair_interface import EncodeUttSchemaPairInterface
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

class FlatCatSlotsBERTCLSValueMatchModel(PreTrainedModel, EncodeUttSchemaPairInterface, FlatDSTOutputInterface):
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
        super(FlatCatSlotsBERTCLSValueMatchModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = EncoderUtils.create_encoder(self.config)
            EncoderUtils.set_encoder_finetuning_status(self.encoder, args.encoder_finetuning)
        EncoderUtils.add_special_tokens(self.tokenizer, self.encoder, schema_constants.USER_AGENT_SPECIAL_TOKENS)
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        # cat_value_seq2_key in ["cat_value_seq2", "cat_slot_value_seq2", "cat_slot_desc_value_seq2"],
        self.cat_value_embedding_key = self.config.cat_value_embedding_key

        self.cat_value_embedding_size = self.config.cat_value_embedding_size
        if self.cat_value_embedding_size != self.utterance_embedding_dim:
            self.categorical_slots_values_utterance_proj = torch.nn.Sequential(
                torch.nn.Linear(self.cat_value_embedding_size, self.utterance_embedding_dim)
            )
        else:
            self.categorical_slots_values_utterance_proj = torch.nn.Sequential(
            )

        if self.config.matcher == "ConcatenateMatcher":
            self.matcher = ConcatenateMatcher(encoding_dim=self.utterance_embedding_dim)
        elif self.config.matcher == "DotProductMatcher":
            self.matcher = DotProductMatcher(encoding_dim=self.utterance_embedding_dim)
        elif self.config.matcher == "BilinearMatcher":
            self.matcher = BilinearMatcher(encoding_dim=self.utterance_embedding_dim)
        else:
            raise NotImplementedError("Not implemented for mater {}".format(self.config.matcher))

        self.categorical_slots_status_utterance_proj = torch.nn.Sequential(
        )
        # for categorical_slots, 3 logits
        self.categorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.utterance_embedding_dim, 3)
        )
        # for categorical_slot_values,
        self.categorical_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.matcher.outputDim, 1)
        )

        if self.matcher.outputDim == 1:
            self.categorical_slots_values_final_proj = torch.nn.Sequential(
            )
        else:
            self.categorical_slots_values_final_proj = torch.nn.Sequential(
                torch.nn.Linear(self.macther.outputDim, 1)
            )

        self.value_dropout = torch.nn.Dropout(self.config.utterance_dropout)

        if not args.no_cuda:
            self.cuda(args.device)

    def _get_logits(self, utt_schema_pair_cls, utt_schema_pair_tokens, utterance_proj, final_proj, is_training):
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
        # logger.info("element_embeddings:{}, repeat_utterance_embeddings:{}".format(element_embeddings.size(), repeat_utterance_embeddings.size()))
        utt_schema_pair_cls = utterance_proj(utt_schema_pair_cls)
        if is_training:
            utt_schema_pair_cls = self.utterance_dropout(utt_schema_pair_cls)
        logits = final_proj(utt_schema_pair_cls)
        return logits

    def _get_value_logits(self, utt_schema_pair_cls, value_embeddings, value_proj, final_proj, is_training):
        """
        get logits for cat_values, with snt_slot_pair, and value_embeddings
        """
        batch_size, max_num_values, _ = value_embeddings.size()
        value_cls = value_proj(value_embeddings)
        if is_training:
            value_cls = self.value_dropout(value_cls)
        # batch_size x num_cat_value, output_dim
        pair_encodings = self.matcher(utt_schema_pair_cls, value_cls)
        logits = final_proj(pair_encodings)
        logits = logits.view(batch_size, max_num_values, -1)
        return logits

    def _get_categorical_slots_goals(self, features, is_training):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots value: active or nonactive
        # doncare is also one of the value
        # batch_size, embedding_dim
        # Shape: (batch_size, embedding_dim).
        utt_cat_slot_pair_cls, _ = self._encode_utterance_schema_pairs(
            self.encoder, self.utterance_dropout, features, "cat_slot", is_training)
        cat_slot_status_logits = self._get_logits(
            utt_cat_slot_pair_cls,
            None,
            self.categorical_slots_status_utterance_proj,
            self.categorical_slots_status_final_proj,
            is_training
        )

        value_embeddings = features[SchemaInputFeatures.get_embedding_tensor_name(self.cat_value_embedding_key)]
        # batch_size, max_cat_value_num
        cat_slot_value_logits = self._get_value_logits(
            utt_cat_slot_pair_cls,
            value_embeddings,
            self.categorical_slots_values_utterance_proj,
            self.categorical_slots_values_final_proj,
            is_training
        ).squeeze(-1)

        max_num_values = cat_slot_value_logits.size(-1)
        # Mask out logits for padded values because they will be
        # softmaxed.
        mask = torch_ext.sequence_mask(
            features["cat_slot_value_num"],
            maxlen=max_num_values, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(cat_slot_value_logits) * torch.finfo(torch.float16).max
        masked_cat_slot_value_logits = torch.where(mask, cat_slot_value_logits, negative_logits)
        return cat_slot_status_logits, masked_cat_slot_value_logits

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
        cat_slot_status_logits, cat_slot_value_logits = self._get_categorical_slots_goals(features, is_training)
        outputs["logit_cat_slot_status"] = cat_slot_status_logits
        outputs["logit_cat_slot_value"] = cat_slot_value_logits
        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
