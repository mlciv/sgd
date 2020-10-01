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
from modules.core.schema_input_features import SchemaInputFeatures
import modules.core.schema_constants as schema_constants
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

def assemble_schema_features_into_inputs(inputs, batch, schema_tensors, args, config):
    """
    add batch and schema features into inputs, it will change inputs, and return labels
    """
    all_examples_types = batch[0]
    if all_examples_types[0] == 1:
        # active_intent
        inputs["intent_id"] = batch[6]
        intent_key = config.intent_seq2_key if "intent_seq2_key" in config.__dict__ else "intent"
        intent_input_embs_key = SchemaInputFeatures.get_embedding_tensor_name(intent_key)
        intent_input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(intent_key)
        intent_input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(intent_key)
        intent_input_type_ids_key = SchemaInputFeatures.get_input_type_ids_tensor_name(intent_key)
        # max_service_num, max_intent_num, max_seq_lenth -> (batch_size, max_intent_num, max_seq_length)
        all_intent_ids_in_batch = schema_tensors[intent_input_ids_key].to(args.device).index_select(0, inputs["service_id"])
        _, max_intent_num, max_seq_length = all_intent_ids_in_batch.size()
        # batch_size, 1, max_seq_length
        intent_indices = inputs["intent_id"].view(-1, 1, 1).expand(-1, 1, max_seq_length)
        # (batch_size, max_intent_num, max_seq_length) -> (batch_size, 1, max_seq_length)
        inputs[intent_input_ids_key] = all_intent_ids_in_batch.gather(1, intent_indices).squeeze(1)
        all_intent_mask_in_batch = schema_tensors[intent_input_mask_key].to(args.device).index_select(0, inputs["service_id"])
        inputs[intent_input_mask_key] = all_intent_mask_in_batch.gather(1, intent_indices).squeeze(1)
        all_intent_seg_in_batch = schema_tensors[intent_input_type_ids_key].to(args.device).index_select(0, inputs["service_id"])
        inputs[intent_input_type_ids_key] = all_intent_seg_in_batch.gather(1, intent_indices).squeeze(1)

        if intent_input_embs_key in schema_tensors:
            all_intent_embs_in_batch = schema_tensors[intent_input_embs_key].to(args.device).index_select(0, inputs["service_id"])
            _, _, max_enc_length, enc_dim = all_intent_embs_in_batch.size()
            emb_intent_indices = inputs["intent_id"].view(-1, 1, 1, 1).expand(-1, 1, max_enc_length, enc_dim)
            inputs[intent_input_embs_key] = all_intent_embs_in_batch.gather(1, emb_intent_indices).squeeze(1)

        if len(batch) > 7:
            # results
            labels = {
                "intent_status": batch[7]
            }
        else:
            labels = None
    elif all_examples_types[0] == 2:
        # req slot
        inputs["req_slot_id"] = batch[6]
        req_slot_key = config.req_slot_seq2_key if "req_slot_seq2_key" in config.__dict__ else "req_slot"
        req_slot_input_embs_key = SchemaInputFeatures.get_embedding_tensor_name(req_slot_key)
        req_slot_input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(req_slot_key)
        req_slot_input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(req_slot_key)
        req_slot_input_type_ids_key = SchemaInputFeatures.get_input_type_ids_tensor_name(req_slot_key)
        # max_service_num, max_req_slot_num, max_seq_lenth -> (batch_size, max_req_slot_num, max_seq_length)
        all_req_slot_ids_in_batch = schema_tensors[req_slot_input_ids_key].to(args.device).index_select(0, inputs["service_id"])
        _, max_req_slot_num, max_seq_length = all_req_slot_ids_in_batch.size()
        req_slot_indices = inputs["req_slot_id"].view(-1, 1, 1).expand(-1, 1, max_seq_length)
        inputs[req_slot_input_ids_key] = all_req_slot_ids_in_batch.gather(1, req_slot_indices).squeeze(1)
        all_req_slot_mask_in_batch = schema_tensors[req_slot_input_mask_key].to(args.device).index_select(0, inputs["service_id"])
        inputs[req_slot_input_mask_key] = all_req_slot_mask_in_batch.gather(1, req_slot_indices).squeeze(1)
        all_req_slot_seg_in_batch = schema_tensors[req_slot_input_type_ids_key].to(args.device).index_select(0, inputs["service_id"])
        inputs[req_slot_input_type_ids_key] = all_req_slot_seg_in_batch.gather(1, req_slot_indices).squeeze(1)
        if req_slot_input_embs_key in schema_tensors:
            all_req_slot_embs_in_batch = schema_tensors[req_slot_input_embs_key].to(args.device).index_select(0, inputs["service_id"])
            _, _, max_enc_length, enc_dim = all_req_slot_ids_in_batch.size()
            emb_req_slot_indices = inputs["req_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_enc_length, enc_dim)
            inputs[req_slot_input_embs_key] = all_req_slot_embs_in_batch.gather(1, emb_req_slot_indices).squeeze(1)

        if len(batch) > 7:
            # results
            labels = {
                "req_slot_status": batch[7]
            }
        else:
            labels = None
    elif all_examples_types[0] == 3:
        # cat slot
        inputs["cat_slot_id"] = batch[6]
        cat_slot_key = config.cat_slot_seq2_key if "cat_slot_seq2_key" in config.__dict__ else "cat_slot"
        cat_slot_input_embs_key = SchemaInputFeatures.get_embedding_tensor_name(cat_slot_key)
        cat_slot_input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(cat_slot_key)
        cat_slot_input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(cat_slot_key)
        cat_slot_input_type_ids_key = SchemaInputFeatures.get_input_type_ids_tensor_name(cat_slot_key)
        # max_service_num, max_cat_slot, max_seq_length => batch_size, max_cat_slot, nax_seq_length
        all_cat_slot_ids_in_batch = schema_tensors[cat_slot_input_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        _, max_cat_slot_num, max_seq_length = all_cat_slot_ids_in_batch.size()
        # batch_size => batch_size,1,max_seq_length
        cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1).expand(-1, 1, max_seq_length)
        # batch_size, max_cat_slot , max_seq_length => batch_size, max_seq_length
        inputs[cat_slot_input_ids_key] = all_cat_slot_ids_in_batch.gather(1, cat_slot_indices).squeeze(1)
        # max_service_name, max_cat_slot, max_seq_length =>  base_size, max_cat_slot, nax_seq_length
        all_cat_slot_mask_in_batch = schema_tensors[cat_slot_input_mask_key].to(
            args.device).index_select(0, inputs["service_id"])
        # batch_size, max_seq_length
        inputs[cat_slot_input_mask_key] = all_cat_slot_mask_in_batch.gather(1, cat_slot_indices).squeeze(1)
        all_cat_slot_seg_in_batch = schema_tensors[cat_slot_input_type_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        inputs[cat_slot_input_type_ids_key] = all_cat_slot_seg_in_batch.gather(1, cat_slot_indices).squeeze(1)
        if cat_slot_input_embs_key in schema_tensors:
            all_cat_slot_embs_in_batch = schema_tensors[cat_slot_input_embs_key].to(args.device).index_select(0, inputs["service_id"])
            _, _, max_enc_length, enc_dim = all_cat_slot_embs_in_batch.size()
            embs_cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1).expand(-1, 1, max_enc_length, enc_dim)
            inputs[cat_slot_input_embs_key] = all_cat_slot_embs_in_batch.gather(1, embs_cat_slot_indices).squeeze(1)
        # cat slot value
        inputs["cat_slot_value_id"] = batch[7]
        # max_service_num, max_cat_slot_num, max_value_num,  max_seq_lenth -> (batch_size, max_cat_slot_num, max_cat_value_num, max_seq_length)
        if "cat_value_seq2_key" in config.__dict__:
            input_embs_key = SchemaInputFeatures.get_embedding_tensor_name(config.cat_value_seq2_key)
            input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(config.cat_value_seq2_key)
            input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(config.cat_value_seq2_key)
            input_type_key = SchemaInputFeatures.get_input_type_ids_tensor_name(config.cat_value_seq2_key)
            all_cat_slot_value_ids_in_batch = schema_tensors[input_ids_key].to(
                args.device).index_select(0, inputs["service_id"])
            _, _, max_cat_value_num, max_seq_length = all_cat_slot_value_ids_in_batch.size()
            cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_cat_value_num, max_seq_length)
            cat_slot_value_indices = inputs["cat_slot_value_id"].view(-1, 1, 1).expand(-1, 1, max_seq_length)
            # (batch_size, max_cat_slot_num, max_value_num, max_seq_length) -> (batch_size, max_cat_value_num, max_seq_length)
            inputs[input_ids_key] = all_cat_slot_value_ids_in_batch.gather(
                1, cat_slot_indices).squeeze(1).gather(1, cat_slot_value_indices).squeeze(1)
            all_cat_slot_value_mask_in_batch = schema_tensors[input_mask_key].to(
                args.device).index_select(0, inputs["service_id"])
            inputs[input_mask_key] = all_cat_slot_value_mask_in_batch.gather(
                1, cat_slot_indices).squeeze(1).gather(1, cat_slot_value_indices).squeeze(1)
            all_cat_slot_value_seg_in_batch = schema_tensors[input_type_key].to(
                args.device).index_select(0, inputs["service_id"])
            inputs[input_type_key] = all_cat_slot_value_seg_in_batch.gather(
                1, cat_slot_indices).squeeze(1).gather(1, cat_slot_value_indices).squeeze(1)
            if input_embs_key in schema_tensors:
                # batch_size, cat, value, max_seq, dim
                all_cat_slot_value_embs_in_batch = schema_tensors[input_embs_key].to(args.device).index_select(0, inputs["service_id"])
                _, _, _, max_enc_length, enc_dim = all_cat_slot_value_embs_in_batch.size()
                emb_cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1, 1).expand(-1, 1, max_cat_value_num, max_enc_length, enc_dim)
                embs_cat_slot_value_indices = inputs["cat_slot_value_id"].view(-1, 1, 1, 1).expand(-1, 1, max_enc_length, enc_dim)
                inputs[input_embs_key] = all_cat_slot_value_embs_in_batch.gather(1, emb_cat_slot_indices).squeeze(1).gather(1, embs_cat_slot_value_indices).squeeze(1)

        elif "cat_value_embedding_key" in config.__dict__:
            input_embedding_key = SchemaInputFeatures.get_embedding_tensor_name(config.cat_value_embedding_key)
            # batch_size, max_cat, max_cat_value, max_length
            all_cat_slot_value_embeddings_in_batch = schema_tensors[input_embedding_key].to(
                args.device).index_select(0, inputs["service_id"])
            _, _, max_cat_value_num, embedding_dim = all_cat_slot_value_embeddings_in_batch.size()
            cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_cat_value_num, embedding_dim)
            # batch_size, 1,  max_cat_value, dim
            inputs[input_embedding_key] = all_cat_slot_value_embeddings_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)

        if len(batch) > 8:
            # results
            all_cat_slot_value_status = batch[8]
            # logger.info("all_cat_slot_value_status:{}".format(all_cat_slot_value_status))
            labels = {
                "cat_slot_value_status": all_cat_slot_value_status
            }
        else:
            labels = None
    elif all_examples_types[0] == 6:
        # cat slots, one cat slot statu, and another a single value for the gt cat_slot_value( without special value shift)
        inputs["cat_slot_id"] = batch[6]
        inputs["cat_slot_value_num"] = batch[7]
        cat_slot_key = config.cat_slot_seq2_key if "cat_slot_seq2_key" in config.__dict__ else "cat_slot"
        cat_slot_input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(cat_slot_key)
        cat_slot_input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(cat_slot_key)
        cat_slot_input_type_ids_key = SchemaInputFeatures.get_input_type_ids_tensor_name(cat_slot_key)
        all_cat_slot_ids_in_batch = schema_tensors[cat_slot_input_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        # cat slot value
        inputs["cat_slot_value_id"] = batch[7]
        # batch_size, max_cat, max_cat_value, max_length
        if "cat_value_seq2_key" in config.__dict__:
            input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(config.cat_value_seq2_key)
            input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(config.cat_value_seq2_key)
            input_type_key = SchemaInputFeatures.get_input_type_ids_tensor_name(config.cat_value_seq2_key)
            all_cat_slot_value_ids_in_batch = schema_tensors[input_ids_key].to(
                args.device).index_select(0, inputs["service_id"])
            _, _, max_cat_value_num, max_seq_length = all_cat_slot_value_ids_in_batch.size()
            cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_cat_value_num, max_seq_length)
            # batch_size, max_cat_value - special_value, max_length
            inputs[input_ids_key] = all_cat_slot_value_ids_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)
            all_cat_slot_value_mask_in_batch = schema_tensors[input_mask_key].to(
                args.device).index_select(0, inputs["service_id"])
            # batch_size, max_cat_value, max_length
            inputs[input_mask_key] = all_cat_slot_value_mask_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)
            # batch_size, max_cat, max_cat_value, max_length
            all_cat_slot_value_seg_in_batch = schema_tensors[input_type_key].to(
                args.device).index_select(0, inputs["service_id"])
            # batch_size, max_cat_value, max_length
            inputs[input_type_key] = all_cat_slot_value_seg_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)
        elif "cat_value_embedding_key" in config.__dict__:
            input_embedding_key = SchemaInputFeatures.get_embedding_tensor_name(config.cat_value_embedding_key)
            # batch_size, max_cat, max_cat_value, max_length
            all_cat_slot_value_embeddings_in_batch = schema_tensors[input_embedding_key].to(
                args.device).index_select(0, inputs["service_id"])
            _, _, max_cat_value_num, embedding_dim = all_cat_slot_value_embeddings_in_batch.size()
            cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_cat_value_num, embedding_dim)
            # batch_size, 1,  max_cat_value, dim
            # batch_size, 1,  max_cat_value, dim
            inputs[input_embedding_key] = all_cat_slot_value_embeddings_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)
        if len(batch) > 8:
            # results
            all_cat_slot_value = batch[8]
            labels = {
                "cat_slot_value_full_state": all_cat_slot_value
            }
        else:
            labels = None
    elif all_examples_types[0] == 5:
        # cat slots, one cat slot statu, and another a single value for the gt cat_slot_value( without special value shift)
        inputs["cat_slot_id"] = batch[6]
        inputs["cat_slot_value_num"] = batch[7]
        cat_slot_key = config.cat_slot_seq2_key if "cat_slot_seq2_key" in config.__dict__ else "cat_slot"
        cat_slot_input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(cat_slot_key)
        cat_slot_input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(cat_slot_key)
        cat_slot_input_type_ids_key = SchemaInputFeatures.get_input_type_ids_tensor_name(cat_slot_key)
        all_cat_slot_ids_in_batch = schema_tensors[cat_slot_input_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        _, max_cat_slot_num, max_seq_length = all_cat_slot_ids_in_batch.size()
        cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1).expand(-1, 1, max_seq_length)
        inputs[cat_slot_input_ids_key] = all_cat_slot_ids_in_batch.gather(1, cat_slot_indices).squeeze(1)
        all_cat_slot_mask_in_batch = schema_tensors[cat_slot_input_mask_key].to(
            args.device).index_select(0, inputs["service_id"])
        inputs[cat_slot_input_mask_key] = all_cat_slot_mask_in_batch.gather(1, cat_slot_indices).squeeze(1)
        all_cat_slot_seg_in_batch = schema_tensors[cat_slot_input_type_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        inputs[cat_slot_input_type_ids_key] = all_cat_slot_seg_in_batch.gather(1, cat_slot_indices).squeeze(1)
        if "cat_value_seq2_key" in config.__dict__:
            input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(config.cat_value_seq2_key)
            input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(config.cat_value_seq2_key)
            input_type_key = SchemaInputFeatures.get_input_type_ids_tensor_name(config.cat_value_seq2_key)
            # batch_size, max_cat, max_cat_value, max_length
            all_cat_slot_value_ids_in_batch = schema_tensors[input_ids_key].to(
                args.device).index_select(0, inputs["service_id"])
            _, _, max_cat_value_num, max_seq_length = all_cat_slot_value_ids_in_batch.size()
            cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_cat_value_num, max_seq_length)
            # batch_size, max_cat_value - special_value, max_length
            inputs[input_ids_key] = all_cat_slot_value_ids_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)[:, schema_constants.SPECIAL_CAT_VALUE_OFFSET:, :]
            all_cat_slot_value_mask_in_batch = schema_tensors[input_mask_key].to(
                args.device).index_select(0, inputs["service_id"])
            # batch_size, max_cat_value, max_length
            inputs[input_mask_key] = all_cat_slot_value_mask_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)[:, schema_constants.SPECIAL_CAT_VALUE_OFFSET:, :]
            # batch_size, max_cat, max_cat_value, max_length
            all_cat_slot_value_seg_in_batch = schema_tensors[input_type_key].to(
                args.device).index_select(0, inputs["service_id"])
            # batch_size, max_cat_value, max_length
            inputs[input_type_key] = all_cat_slot_value_seg_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)[:, schema_constants.SPECIAL_CAT_VALUE_OFFSET:, :]
        elif "cat_value_embedding_key" in config.__dict__:
            input_embedding_key = SchemaInputFeatures.get_embedding_tensor_name(config.cat_value_embedding_key)
            all_cat_slot_value_embeddings_in_batch = schema_tensors[input_embedding_key].to(
                args.device).index_select(0, inputs["service_id"])
            _, _, max_cat_value_num, embedding_dim = all_cat_slot_value_embeddings_in_batch.size()
            cat_slot_indices = inputs["cat_slot_id"].view(-1, 1, 1, 1).expand(-1, 1, max_cat_value_num, embedding_dim)
            # batch_size, 1,  max_cat_value, dim
            inputs[input_embedding_key] = all_cat_slot_value_embeddings_in_batch.gather(
                1, cat_slot_indices
            ).squeeze(1)
        if len(batch) > 8:
            # results
            all_cat_slot_status = batch[8]
            all_cat_slot_value = batch[9]
            labels = {
                "cat_slot_status": all_cat_slot_status,
                "cat_slot_value": all_cat_slot_value
            }
        else:
            labels = None
    elif all_examples_types[0] == 4:
        # noncat slot
        inputs["noncat_slot_id"] = batch[6]
        inputs["noncat_alignment_start"] = batch[7]
        inputs["noncat_alignment_end"] = batch[8]
        noncat_slot_key = config.noncat_slot_seq2_key if "noncat_slot_seq2_key" in config.__dict__ else "noncat_slot"
        noncat_slot_input_ids_key = SchemaInputFeatures.get_input_ids_tensor_name(noncat_slot_key)
        noncat_slot_input_mask_key = SchemaInputFeatures.get_input_mask_tensor_name(noncat_slot_key)
        noncat_slot_input_type_ids_key = SchemaInputFeatures.get_input_type_ids_tensor_name(noncat_slot_key)
        all_noncat_slot_ids_in_batch = schema_tensors[noncat_slot_input_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        _, max_noncat_slot_num, max_seq_length = all_noncat_slot_ids_in_batch.size()
        noncat_slot_indices = inputs["noncat_slot_id"].view(-1, 1, 1).expand(-1, 1, max_seq_length)
        inputs[noncat_slot_input_ids_key] = all_noncat_slot_ids_in_batch.gather(1, noncat_slot_indices).squeeze(1)
        all_noncat_slot_mask_in_batch = schema_tensors[noncat_slot_input_mask_key].to(
            args.device).index_select(0, inputs["service_id"])
        inputs[noncat_slot_input_mask_key] = all_noncat_slot_mask_in_batch.gather(1, noncat_slot_indices).squeeze(1)
        all_noncat_slot_seg_in_batch = schema_tensors[noncat_slot_input_type_ids_key].to(
            args.device).index_select(0, inputs["service_id"])
        inputs[noncat_slot_input_type_ids_key] = all_noncat_slot_seg_in_batch.gather(1, noncat_slot_indices).squeeze(1)
        if len(batch) > 9:
            # results
            all_noncat_slot_status = batch[9]
            all_noncat_slot_value_start = batch[10]
            all_noncat_slot_value_end = batch[11]
            labels = {
                "noncat_slot_status": all_noncat_slot_status,
                "noncat_slot_value_start": all_noncat_slot_value_start,
                "noncat_slot_value_end": all_noncat_slot_value_end
            }
        else:
            labels = None
    return labels
