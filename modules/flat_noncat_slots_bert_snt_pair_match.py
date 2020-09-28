# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : flat_noncat_slots_bert_snt_pair_match_model.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict noncat_slots
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

class FlatNonCatSlotsBERTSntPairMatchModel(PreTrainedModel, EncodeUttSchemaPairInterface, FlatDSTOutputInterface):
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
        super(FlatNonCatSlotsBERTSntPairMatchModel, self).__init__(config=config)
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
        self.noncat_slot_seq2_key = self.config.noncat_slot_seq2_key
        self.noncategorical_slots_status_utterance_proj = torch.nn.Sequential(
        )

        # for noncategorical_slots, 3 logits
        self.noncategorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.utterance_embedding_dim, 3)
        )

        # for noncategorical_,span layer
        self.noncat_span_layer = torch.nn.Sequential(
            torch.nn.Linear(self.utterance_embedding_dim, 2)
        )

        if not args.no_cuda:
            self.cuda(args.device)

    def _get_logits(self, utt_schema_pair_cls, utt_schema_pair_tokens, utterance_proj, final_proj, is_training):
        """Get logits for elements by conditioning on utterance embedding.
        mainly for status logits
        """
        utt_schema_pair_cls = utterance_proj(utt_schema_pair_cls)
        if is_training:
            utt_schema_pair_cls = self.utterance_dropout(utt_schema_pair_cls)
        utt_schema_pair_cls = final_proj(utt_schema_pair_cls)
        return utt_schema_pair_cls

    def _get_noncategorical_slots_goals(self, features, is_training):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots.
        # batch_size, max_noncat_slot, embedding_dim
        utt_noncat_slot_pair_cls, utt_noncat_slot_pair_tokens = self._encode_utterance_schema_pairs(
            self.encoder, self.utterance_dropout, features, self.noncat_slot_seq2_key, is_training)
        # batch_size, 3
        status_logits = self._get_logits(
            utt_noncat_slot_pair_cls,
            None,
            self.noncategorical_slots_status_utterance_proj,
            self.noncategorical_slots_status_final_proj,
            is_training
        )
        batch_size = features["utt"].size()[0]
        # Shape: (batch_size, max_seq_length, 2)
        span_logits = self.noncat_span_layer(utt_noncat_slot_pair_tokens)
        total_max_length = span_logits.size()[1]
        schema_attention_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(self.noncat_slot_seq2_key)]
        # all schema part should be masked as 0
        schema_attention_mask = torch.zeros_like(schema_attention_mask.view(batch_size, -1))
        # batch_size , max_seq_length
        utt_schema_pair_attention_mask = torch.cat((features["utt_mask"], schema_attention_mask), dim=1)
        assert utt_schema_pair_attention_mask.size()[1] == total_max_length, "length check"
        tiled_attention_mask = utt_schema_pair_attention_mask.unsqueeze(2).expand(-1, -1, 2)
        negative_logits = -0.7 * torch.ones_like(span_logits) * torch.finfo(torch.float16).max
        span_logits = torch.where(tiled_attention_mask.bool(), span_logits, negative_logits)
        # Shape of both tensors: (batch_size, max_seq_length).
        span_start_logits = span_logits[:, :, 0]
        span_end_logits = span_logits[:, :, 1]

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
        outputs = {}
        noncat_slot_status, noncat_span_start, noncat_span_end = (
            self._get_noncategorical_slots_goals(features, is_training))
        outputs["logit_noncat_slot_status"] = noncat_slot_status
        outputs["logit_noncat_slot_start"] = noncat_span_start
        outputs["logit_noncat_slot_end"] = noncat_span_end
        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )
