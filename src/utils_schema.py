# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset reader and tokenization-related utilities for baseline model."""

from __future__ import absolute_import
from __future__ import division

import collections
import json
import os
import re
import csv
import math
import logging
import numpy as np

from utils import schema
from utils import data_utils
from modules.schema_dialog_processor import SchemaDialogProcessor, DEFAULT_MAX_SEQ_LENGTH
import torch
import torch.nn as nn
from modules.core.schema_dst_example import (
    ActiveIntentExample,
    RequestedSlotExample,
    CatSlotExample,
    CatSlotFullStateExample,
    CatSlotValueExample, NonCatSlotExample, SchemaDSTExample
)

from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """
    A single set of features of the data, Here we build the baseline model first
    Then we will consider how merge all into a unified model
    # For unified model, all the labels will share the input ids, seg and mask, which is following the bert-input
    # all the schema features from loaded from the pregenerated schema features
    """
    def __init__(self, example_id, service_id,
                 utt_ids, utt_seg, utt_mask,
                 cat_slot_num, cat_slot_status, cat_slot_value_num,
                 cat_slot_value, noncat_slot_num, noncat_slot_status,
                 noncat_slot_start, noncat_slot_end, noncat_alignment_start,
                 noncat_alignment_end, req_slot_num, req_slot_status,
                 intent_num, intent_status):
        # example_info
        self.example_id = example_id
        self.service_id = service_id
        self.utt_ids = utt_ids
        self.utt_seg = utt_seg
        self.utt_mask = utt_mask
        self.cat_slot_num = cat_slot_num
        self.cat_slot_status = cat_slot_status
        self.cat_slot_value_num = cat_slot_value_num
        self.cat_slot_value = cat_slot_value
        self.noncat_slot_num = noncat_slot_num
        self.noncat_slot_status = noncat_slot_status
        self.noncat_slot_value_start = noncat_slot_start
        self.noncat_slot_value_end = noncat_slot_end
        self.noncat_alignment_start = noncat_alignment_start
        self.noncat_alignment_end = noncat_alignment_end
        self.req_slot_num = req_slot_num
        self.req_slot_status = req_slot_status
        self.intent_num = intent_num
        self.intent_status = intent_status

class ActiveIntentSntPairInputFeatures(object):
    """
    A single set of features for active intent data
    """
    def __init__(self, example_id, service_id,
                 input_ids, input_seg, input_mask,
                 intent_id, intent_status):
        # example_info
        self.example_type = 1
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_seg = input_seg
        self.input_mask = input_mask
        self.intent_id = intent_id
        self.intent_status = intent_status

class RequestedSlotSntPairInputFeatures(object):
    """
    A single set of features for req_slots data
    """
    def __init__(self, example_id, service_id,
                 input_ids, input_seg, input_mask,
                 requested_slot_id, requested_slot_status):
        # example_info
        self.example_type = 2
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_seg = input_seg
        self.input_mask = input_mask
        self.requested_slot_id = requested_slot_id
        self.requested_slot_status = requested_slot_status

class CatSlotFullStateSntPairInputFeatures(object):
    """
    A single set of features for cat slot data
    """
    def __init__(self, example_id, service_id,
                 input_ids, input_seg, input_mask,
                 cat_slot_id, num_cat_slot_values, cat_slot_value):
        # example_info
        self.example_type = 6
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_seg = input_seg
        self.input_mask = input_mask
        self.cat_slot_id = cat_slot_id
        self.num_cat_slot_values = num_cat_slot_values
        self.cat_slot_value = cat_slot_value

class CatSlotSntPairInputFeatures(object):
    """
    A single set of features for cat slot data
    """
    def __init__(self, example_id, service_id,
                 input_ids, input_seg, input_mask,
                 cat_slot_id, num_cat_slot_values, cat_slot_status, cat_slot_value):
        # example_info
        self.example_type = 5
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_seg = input_seg
        self.input_mask = input_mask
        self.cat_slot_id = cat_slot_id
        self.num_cat_slot_values = num_cat_slot_values
        self.cat_slot_status = cat_slot_status
        self.cat_slot_value = cat_slot_value

class CatSlotValueSntPairInputFeatures(object):
    """
    A single set of features for cat slot data
    """
    def __init__(self, example_id, service_id,
                 input_ids, input_seg, input_mask,
                 cat_slot_id, cat_slot_value_id, cat_slot_value_status):
        # example_info
        self.example_type = 3
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_seg = input_seg
        self.input_mask = input_mask
        self.cat_slot_id = cat_slot_id
        self.cat_slot_value_id = cat_slot_value_id
        self.cat_slot_value_status = cat_slot_value_status

class NonCatSlotSntPairInputFeatures(object):
    """
    A single set of features for noncat slot data
    """
    def __init__(self, example_id, service_id,
                 input_ids, input_seg, input_mask,
                 noncat_slot_id, noncat_start_char_idx, noncat_end_char_idx,
                 noncat_slot_status, noncat_slot_value_start, noncat_slot_value_end):
        # example_info
        self.example_type = 4
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_seg = input_seg
        self.input_mask = input_mask
        self.noncat_slot_id = noncat_slot_id
        self.noncat_start_char_idx = noncat_start_char_idx
        self.noncat_end_char_idx = noncat_end_char_idx
        self.noncat_slot_status = noncat_slot_status
        self.noncat_slot_value_start = noncat_slot_value_start
        self.noncat_slot_value_end = noncat_slot_value_end


def convert_noncat_slot_examples_to_features(
        examples, dataset_config, max_seq_length, is_training, return_dataset):
    """Convert a set of `CatSlotExample` to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations
    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing noncat_slot example %d of %d", ex_index, len(examples))
        feature = NonCatSlotSntPairInputFeatures(
            ex.example_id,
            ex.service_id,
            ex.input_ids,
            ex.input_seg,
            ex.input_mask,
            ex.noncat_slot_id,
            ex.noncat_start_char_idx,
            ex.noncat_end_char_idx,
            ex.noncat_slot_status,
            ex.noncat_slot_value_start,
            ex.noncat_slot_value_end)
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        all_example_types = torch.tensor([f.example_type for f in features], dtype=torch.uint8)
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # snt_pair features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.input_seg for f in features], dtype=torch.long)

        # noncat slot
        all_noncat_slot_ids = torch.tensor([f.noncat_slot_id for f in features], dtype=torch.long)
        all_noncat_alignment_start = torch.tensor([f.noncat_start_char_idx for f in features], dtype=torch.long)
        all_noncat_alignment_end = torch.tensor([f.noncat_end_char_idx for f in features], dtype=torch.long)
        all_noncat_slot_status = torch.tensor([f.noncat_slot_status for f in features], dtype=torch.long)
        # noncat_slot_value
        all_noncat_slot_value_start = torch.tensor([f.noncat_slot_value_start for f in features], dtype=torch.long)
        all_noncat_slot_value_end = torch.tensor([f.noncat_slot_value_end for f in features], dtype=torch.long)


        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_noncat_slot_ids,
                all_noncat_alignment_start,
                all_noncat_alignment_end,
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_noncat_slot_ids,
                all_noncat_alignment_start,
                all_noncat_alignment_end,
                all_noncat_slot_status,
                all_noncat_slot_value_start,
                all_noncat_slot_value_end
            )

        return features, dataset

    return features

def convert_cat_slot_value_examples_to_features(
        examples, dataset_config, max_seq_length, is_training, return_dataset):
    """Convert a set of `CatSlotExample` to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations
    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing cat_slot example %d of %d", ex_index, len(examples))
        feature = CatSlotValueSntPairInputFeatures(
            ex.example_id,
            ex.service_id,
            ex.input_ids,
            ex.input_seg,
            ex.input_mask,
            ex.cat_slot_id,
            ex.cat_slot_value_id,
            ex.cat_slot_value_status)
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        all_example_types = torch.tensor([f.example_type for f in features], dtype=torch.uint8)
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # snt_pair features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.input_seg for f in features], dtype=torch.long)

        # cat slot
        all_cat_slot_ids = torch.tensor([f.cat_slot_id for f in features], dtype=torch.long)
        all_cat_slot_value_ids = torch.tensor([f.cat_slot_value_id for f in features], dtype=torch.long)

        # value status
        all_cat_slot_value_status = torch.tensor([f.cat_slot_value_status for f in features], dtype=torch.long)


        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_ids,
                all_cat_slot_value_ids
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_ids,
                all_cat_slot_value_ids,
                all_cat_slot_value_status
            )
        return features, dataset

    return features


def convert_cat_slot_examples_to_features(
        examples, dataset_config, max_seq_length, is_training, return_dataset):
    """Convert a set of `CatSlotExample` to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations
    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing cat_slot example %d of %d", ex_index, len(examples))
        feature = CatSlotSntPairInputFeatures(
            ex.example_id,
            ex.service_id,
            ex.input_ids,
            ex.input_seg,
            ex.input_mask,
            ex.cat_slot_id,
            ex.num_cat_slot_values,
            ex.cat_slot_status,
            ex.cat_slot_value
        )
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        all_example_types = torch.tensor([f.example_type for f in features], dtype=torch.uint8)
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # snt_pair features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.input_seg for f in features], dtype=torch.long)

        # cat slot
        all_cat_slot_ids = torch.tensor([f.cat_slot_id for f in features], dtype=torch.long)
        all_num_cat_slot_values = torch.tensor([f.num_cat_slot_values for f in features], dtype=torch.long)
        # cat_slot_value
        all_cat_slot_status = torch.tensor([f.cat_slot_status for f in features], dtype=torch.long)
        all_cat_slot_value = torch.tensor([f.cat_slot_value for f in features], dtype=torch.long)


        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_ids,
                all_num_cat_slot_values
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_ids,
                all_num_cat_slot_values,
                all_cat_slot_status,
                all_cat_slot_value,
            )
        return features, dataset

    return features

def convert_cat_slot_full_state_examples_to_features(
        examples, dataset_config, max_seq_length, is_training, return_dataset):
    """Convert a set of `CatSlotExample` to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations
    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing cat_slot example %d of %d", ex_index, len(examples))
        feature = CatSlotFullStateSntPairInputFeatures(
            ex.example_id,
            ex.service_id,
            ex.input_ids,
            ex.input_seg,
            ex.input_mask,
            ex.cat_slot_id,
            ex.num_cat_slot_values,
            ex.cat_slot_value
        )
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        all_example_types = torch.tensor([f.example_type for f in features], dtype=torch.uint8)
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # snt_pair features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.input_seg for f in features], dtype=torch.long)

        # cat slot
        all_cat_slot_ids = torch.tensor([f.cat_slot_id for f in features], dtype=torch.long)
        all_num_cat_slot_values = torch.tensor([f.num_cat_slot_values for f in features], dtype=torch.long)
        # cat_slot_value
        all_cat_slot_value = torch.tensor([f.cat_slot_value for f in features], dtype=torch.long)


        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_ids,
                all_num_cat_slot_values
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_ids,
                all_num_cat_slot_values,
                all_cat_slot_value,
            )
        return features, dataset

    return features

def convert_requested_slot_examples_to_features(
        examples, dataset_config, max_seq_length, is_training, return_dataset):
    """Convert a set of `SchemaDSTExample to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations
    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing req_slot example %d of %d", ex_index, len(examples))
        feature = RequestedSlotSntPairInputFeatures(
            ex.example_id,
            ex.service_id,
            ex.input_ids,
            ex.input_seg,
            ex.input_mask,
            ex.requested_slot_id,
            ex.requested_slot_status)
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        all_example_types = torch.tensor([f.example_type for f in features], dtype=torch.uint8)
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # snt_pair features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.input_seg for f in features], dtype=torch.long)

        # active_intent
        all_requested_slot_ids = torch.tensor([f.requested_slot_id for f in features], dtype=torch.long)
        all_requested_slot_status = torch.tensor([f.requested_slot_status for f in features], dtype=torch.long)

        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_requested_slot_ids
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_requested_slot_ids,
                all_requested_slot_status
            )

        return features, dataset

    return features

def convert_active_intent_examples_to_features(
        examples, dataset_config, max_seq_length, is_training, return_dataset):
    """Convert a set of `SchemaDSTExample to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations
    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing active_intent example %d of %d", ex_index, len(examples))
        feature = ActiveIntentSntPairInputFeatures(
            ex.example_id,
            ex.service_id,
            ex.input_ids,
            ex.input_seg,
            ex.input_mask,
            ex.intent_id,
            ex.intent_status)
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        # for intent example, the flag is 1
        all_example_types = torch.tensor([f.example_type for f in features], dtype=torch.uint8)
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # snt_pair features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.input_seg for f in features], dtype=torch.long)

        # active_intent
        all_intent_ids = torch.tensor([f.intent_id for f in features], dtype=torch.long)
        all_intent_status = torch.tensor([f.intent_status for f in features], dtype=torch.long)

        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_intent_ids
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_types,
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_intent_ids,
                all_intent_status
            )

        return features, dataset

    return features

def convert_schema_dst_examples_to_features(examples,
                                 dataset_config,
                                 max_seq_length,
                                 is_training,
                                 return_dataset):
    """Convert a set of `SchemaDSTExample to features In the google
    baseline, all features including the utterance ids are actually
    preprocessed when creaing the dialog examples.  TODO: we can do
    any other expansions later for other combinations

    """
    features = []
    for (ex_index, ex) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing schema_dst example %d of %d", ex_index, len(examples))
        feature = InputFeatures(
            ex.example_id,
            ex.service_schema.service_id,
            ex.utterance_ids,
            ex.utterance_segment,
            ex.utterance_mask,
            ex.num_categorical_slots,
            ex.categorical_slot_status,
            ex.num_categorical_slot_values,
            ex.categorical_slot_values,
            ex.num_noncategorical_slots,
            ex.noncategorical_slot_status,
            ex.noncategorical_slot_value_start,
            ex.noncategorical_slot_value_end,
            ex.start_char_idx,
            ex.end_char_idx,
            ex.num_slots,
            ex.requested_slot_status,
            ex.num_intents,
            ex.intent_status)
        features.append(feature)

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        # flags or ids for the example
        all_example_ids = torch.tensor([list(f.example_id.encode("utf-8")) for f in features], dtype=torch.uint8)
        all_service_ids = torch.tensor([f.service_id for f in features], dtype=torch.long)

        # dialogue history features
        all_input_ids = torch.tensor([f.utt_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.utt_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.utt_seg for f in features], dtype=torch.long)

        # categorical slots features
        all_cat_slot_num = torch.tensor([f.cat_slot_num for f in features], dtype=torch.int)
        all_cat_slot_value_num = torch.tensor([f.cat_slot_value_num for f in features], dtype=torch.long)
        all_cat_slot_status = torch.tensor([f.cat_slot_status for f in features], dtype=torch.int)
        all_cat_slot_values = torch.tensor([f.cat_slot_value for f in features], dtype=torch.long)

        # non_categorical_slots features
        all_noncat_slot_num = torch.tensor([f.noncat_slot_num for f in features], dtype=torch.int)
        all_noncat_alignment_start = torch.tensor([f.noncat_alignment_start for f in features], dtype=torch.long)
        all_noncat_alignment_end = torch.tensor([f.noncat_alignment_end for f in features], dtype=torch.long)
        all_noncat_slot_status = torch.tensor([f.noncat_slot_status for f in features], dtype=torch.int)
        all_noncat_slot_start = torch.tensor([f.noncat_slot_value_start for f in features], dtype=torch.long)
        all_noncat_slot_end = torch.tensor([f.noncat_slot_value_end for f in features], dtype=torch.long)

        # requested_slots
        all_req_slot_num = torch.tensor([f.req_slot_num for f in features], dtype=torch.long)
        all_req_slot_status = torch.tensor([f.req_slot_status for f in features], dtype=torch.long)
        # active_intent
        all_intent_num = torch.tensor([f.intent_num for f in features], dtype=torch.long)
        all_intent_status = torch.tensor([f.intent_status for f in features], dtype=torch.long)

        if not is_training:
            dataset = torch.utils.data.TensorDataset(
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_num,
                all_cat_slot_value_num,
                all_noncat_slot_num,
                all_noncat_alignment_start,
                all_noncat_alignment_end,
                all_req_slot_num,
                all_intent_num
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                all_example_ids,
                all_service_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cat_slot_num,
                all_cat_slot_value_num,
                all_noncat_slot_num,
                all_noncat_alignment_start,
                all_noncat_alignment_end,
                all_req_slot_num,
                all_intent_num,
                # results
                all_cat_slot_status,
                all_cat_slot_values,
                all_noncat_slot_status,
                all_noncat_slot_start,
                all_noncat_slot_end,
                all_req_slot_status,
                all_intent_status
            )

        return features, dataset

    return features

def convert_examples_to_features(examples,
                                 dataset_config,
                                 max_seq_length,
                                 is_training,
                                 return_dataset):
    if len(examples) == 0:
        raise RuntimeError("No examples to convert into features")
    else:
        if isinstance(examples[0], SchemaDSTExample):
            return convert_schema_dst_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        elif isinstance(examples[0], ActiveIntentExample):
            return convert_active_intent_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        elif isinstance(examples[0], RequestedSlotExample):
            return convert_requested_slot_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        elif isinstance(examples[0], CatSlotValueExample):
            return convert_cat_slot_value_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        elif isinstance(examples[0], CatSlotExample):
            return convert_cat_slot_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        elif isinstance(examples[0], CatSlotFullStateExample):
            return convert_cat_slot_full_state_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        elif isinstance(examples[0], NonCatSlotExample):
            return convert_noncat_slot_examples_to_features(
                examples, dataset_config, max_seq_length, is_training, return_dataset)
        else:
            raise NotImplementedError(" example type {} is not supported".format(type(examples[0])))
