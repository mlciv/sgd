# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : noncat_slots_bert_snt_pair_match_model.py
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
from modules.dstc8baseline_output_interface import DSTC8BaselineOutputInterface
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

class NonCatSlotsBERTSntPairMatchModel(PreTrainedModel, EncodeUttSchemaPairInterface, DSTC8BaselineOutputInterface):
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
        super(NonCatSlotsBERTSntPairMatchModel, self).__init__(config=config)
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
        # logger.info("element_embeddings:{}, repeat_utterance_embeddings:{}".format(element_embeddings.size(), repeat_utterance_embeddings.size()))
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
            self.encoder, self.utterance_dropout, features, "noncat_slot", is_training)
        status_logits = self._get_logits(
            utt_noncat_slot_pair_cls,
            None,
            self.noncategorical_slots_status_utterance_proj,
            self.noncategorical_slots_status_final_proj,
            is_training
        )
        batch_size = features["utt"].size()[0]
        # batch_size, max_noncat_slot, 3
        status_logits = status_logits.view(batch_size, -1, 3)

        # Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.noncat_span_layer(utt_noncat_slot_pair_tokens)
        max_num_slots = span_logits.size()[1]
        # Mask out invalid logits for padded tokens.
        expanded_utt_attention_mask = features["utt_mask"].unsqueeze(1).expand(batch_size, max_num_slots, -1)
        schema_attention_mask = features[SchemaInputFeatures.get_input_mask_tensor_name("noncat_slot")]
        # all schema part should be masked as 0
        schema_attention_mask = torch.zeros_like(schema_attention_mask.view(batch_size, max_num_slots, -1))
        utt_schema_pair_attention_mask = torch.cat((expanded_utt_attention_mask, schema_attention_mask), dim=2)
        tiled_token_mask = utt_schema_pair_attention_mask.unsqueeze(3).expand(-1, max_num_slots, -1, 2)
        negative_logits = -0.7 * torch.ones_like(span_logits) * torch.finfo(torch.float16).max
        #logger.info("span_logits:{}, token_mask: {} , titled_token_mask:{}".format(
        #    span_logits.size(), token_mask.size(), tiled_token_mask.size()))
        span_logits = torch.where(tiled_token_mask.bool(), span_logits, negative_logits)
        # Shape of both tensors: (batch_size, max_num_slots, max_num_tokens).
        span_start_logits = span_logits[:, :, :, 0]
        span_end_logits = span_logits[:, :, :, 1]

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

    @classmethod
    def define_loss(cls, features, labels, outputs):
        """Obtain the loss of the model."""
        losses = {}
        # Non-categorical
        if "logit_noncat_slot_status" in outputs:
            # Non-categorical slot status.
            # Shape: (batch_size, max_num_noncat_slots, 3).
            noncat_slot_status_logits = outputs["logit_noncat_slot_status"]
            noncat_slot_status_labels = labels["noncat_slot_status"]
            max_num_noncat_slots = noncat_slot_status_labels.size()[-1]
            noncat_weights = torch_ext.sequence_mask(
                features["noncat_slot_num"],
                maxlen=max_num_noncat_slots,
                device=noncat_slot_status_logits.device,
                dtype=torch.float32).view(-1)
            # Logits for padded (invalid) values are already masked.
            noncat_slot_status_losses = torch.nn.functional.cross_entropy(
                noncat_slot_status_logits.view(-1, 3),
                noncat_slot_status_labels.view(-1).long(),
                reduction='none'
            )
            noncat_slot_status_loss = (noncat_slot_status_losses * noncat_weights).sum()
            # Non-categorical slot spans.
            # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
            span_start_logits = outputs["logit_noncat_slot_start"]
            span_start_labels = labels["noncat_slot_value_start"]
            max_num_tokens = span_start_logits.size()[-1]
            # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
            span_end_logits = outputs["logit_noncat_slot_end"]
            span_end_labels = labels["noncat_slot_value_end"]
            # Zero out losses for non-categorical slot spans when the slot status is not
            # active.
            # (batch_size, max_num_noncat_slots)
            noncat_loss_weight = (noncat_slot_status_labels == schema_constants.STATUS_ACTIVE).type(torch.float32).view(-1)
            active_noncat_value_weights = noncat_weights * noncat_loss_weight
            span_start_losses = torch.nn.functional.cross_entropy(
                span_start_logits.view(-1, max_num_tokens),
                span_start_labels.view(-1).long(), reduction='none')
            span_end_losses = torch.nn.functional.cross_entropy(
                span_end_logits.view(-1, max_num_tokens),
                span_end_labels.view(-1).long(), reduction='none')
            span_start_loss = (span_start_losses * active_noncat_value_weights).sum()
            span_end_loss = (span_end_losses * active_noncat_value_weights).sum()
            losses["loss_noncat_slot_status"] = noncat_slot_status_loss
            losses["loss_span_start"] = span_start_loss
            losses["loss_span_end"] = span_end_loss
        return losses

    @classmethod
    def define_predictions(cls, features, outputs):
        """Define model predictions."""
        predictions = {
            "example_id": features["example_id"].cpu().numpy(),
            "service_id": features["service_id"].cpu().numpy(),
        }

        # For non-categorical slots, the status of each slot and the indices for
        # spans are output.
        if "logit_noncat_slot_status" in outputs:
            predictions["noncat_slot_status"] = torch.argmax(
                outputs["logit_noncat_slot_status"], dim=-1)

        if "logit_noncat_slot_start" in outputs:
            start_scores = torch.nn.functional.softmax(outputs["logit_noncat_slot_start"], dim=-1)
            end_scores = torch.nn.functional.softmax(outputs["logit_noncat_slot_end"], dim=-1)
            _, max_num_slots, max_num_tokens = end_scores.size()
            batch_size = end_scores.size()[0]
            # Find the span with the maximum sum of scores for start and end indices.
            total_scores = (
                start_scores.unsqueeze(3) +
                end_scores.unsqueeze(2))
            # Mask out scores where start_index > end_index.
            # exclusive
            start_idx = torch.arange(0, max_num_tokens, device=total_scores.device).view(1, 1, -1, 1)
            end_idx = torch.arange(0, max_num_tokens, device=total_scores.device).view(1, 1, 1, -1)
            invalid_index_mask = (start_idx > end_idx).expand(batch_size, max_num_slots, -1, -1)
            # logger.info("invalid_index_mask:{}, total_scores:{}".format(invalid_index_mask.size(), total_scores.size()))
            total_scores = torch.where(invalid_index_mask, torch.zeros_like(total_scores), total_scores)
            max_span_index = torch.argmax(total_scores.view(-1, max_num_slots, max_num_tokens**2), dim=-1)
            span_start_index = (max_span_index.float() / max_num_tokens).floor().long()
            span_end_index = torch.fmod(max_span_index.float(), max_num_tokens).floor().long()
            predictions["noncat_slot_start"] = span_start_index
            predictions["noncat_slot_end"] = span_end_index
            # Add inverse alignments.
            predictions["noncat_alignment_start"] = features["noncat_alignment_start"]
            predictions["noncat_alignment_end"] = features["noncat_alignment_end"]
        return predictions

    @classmethod
    def define_oracle_predictions(cls, features, labels):
        """Define model predictions."""
        predictions = {
            "example_id": features["example_id"].cpu().numpy(),
            "service_id": features["service_id"].cpu().numpy(),
        }

        # For non-categorical slots, the status of each slot and the indices for
        # spans are output.
        predictions["noncat_slot_status"] = labels["noncat_slot_status"]
        predictions["noncat_slot_start"] = labels["noncat_slot_value_start"]
        predictions["noncat_slot_end"] = labels["noncat_slot_value_end"]
        # Add inverse alignments.
        predictions["noncat_alignment_start"] = features["noncat_alignment_start"]
        predictions["noncat_alignment_end"] = features["noncat_alignment_end"]
        return predictions
