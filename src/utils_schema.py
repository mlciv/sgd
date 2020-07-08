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

from utils import schema
from utils import data_utils
import torch
# from bert import tokenization

from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

logger = logging.getLogger(__name__)

STR_DONTCARE = "dontcare"
# The maximum total input sequence length after WordPiece tokenization.
DEFAULT_MAX_SEQ_LENGTH = 128

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2

# Name of the file containing all predictions and their corresponding frame
# metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


def load_dialogues(dialog_json_filepaths):
    """Obtain the list of all dialogues from specified json files."""
    dialogs = []
    for dialog_json_filepath in sorted(dialog_json_filepaths):
        with open(dialog_json_filepath) as f:
            dialogs.extend(json.load(f))
    return dialogs


class PaddingSchemaDSTExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """
    pass


class SchemaDSTC8Processor(DataProcessor):
    """
    Data generator for dstc8 dialogues.
    Implemented based transformer DataProcessor
    """
    def __init__(self,
                 dataset_config,
                 tokenizer,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                 log_data_warnings=False):
        self._log_data_warnings = log_data_warnings
        self._dataset_config = dataset_config
        # BERT tokenizer
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    @property
    def dataset_config(self):
        return self._dataset_config

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors.
        Args:
           tensor_dict: Keys and values should match the corresponding Glue
           tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir, train_file_name):
        """
        Gets a collection of :class:`SchemaDSTExample` for the test set.
        """
        return self.get_dialog_examples(data_dir, train_file_name)

    def get_dev_examples(self, data_dir, dev_file_name):
        """
        Gets a collection of :class:`SchemaDSTExample` for the dev set.
        """
        return self.get_dialog_examples(data_dir, dev_file_name)

    def get_test_examples(self, data_dir, test_file_name):
        """
        Gets a collection of :class:`SchemaDSTExample` for the test set.
        # consider the unannotated data in the future
        """
        return self.get_dialog_examples(data_dir, test_file_name)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_input_dialog_files(self, data_dir, dataset):
        dialog_paths = [
            os.path.join(data_dir, dataset,
                         "dialogues_{:03d}.json".format(i))
            for i in self._dataset_config.file_ranges[dataset]
        ]
        return dialog_paths

    @staticmethod
    def get_schema_file(data_dir, dataset):
        """
        return the schema file given the data_dir and split
        """
        return os.path.join(data_dir, dataset, "schema.json")

    @staticmethod
    def get_schemas(data_dir, dataset):
        """
        get all the schema given data_dir and datasplit
        """
        schema_path = SchemaDSTC8Processor.get_schema_file(data_dir, dataset)
        schemas = schema.Schema(schema_path)
        return schemas

    def get_dialog_examples(self, data_dir, dataset):
        """Return a list of `SchemaDSTExample`s of the data splits' dialogues.
        Args:
        data_dir: the folder store the data, it have three data splits
        dataset: str. can be "train", "dev", or "test".
        Returns:
        examples: a list of `SchemaDSTExample`s.
        """
        dialogs = self.get_whole_dialogs(data_dir, dataset)
        schemas = SchemaDSTC8Processor.get_schemas(data_dir, dataset)

        examples = []
        for dialog_idx, dialog in enumerate(dialogs):
            if dialog_idx % 1000 == 0:
                logger.info("Processed %d dialogs.", dialog_idx)
            examples.extend(
                self._create_examples_from_dialog(dialog, schemas, dataset))
        return examples

    def _create_examples_from_dialog(self, dialog, schemas, dataset):
        """
        Create examples for every turn in the dialog.
        """
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            # Only user turn have the frame information to predict
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if turn_idx > 0:
                    system_turn = dialog["turns"][turn_idx - 1]
                    system_utterance = system_turn["utterance"]
                    system_frames = {f["service"]: f for f in system_turn["frames"]}
                else:
                    # The first turn is always the u
                    system_utterance = ""
                    system_frames = {}
                # a global turn id
                # turnuid ="split-dialogue_id-turn_idx"
                turn_id = "{:<5}-{:<12}-{:<3}".format(
                    dataset, dialog_id,
                    SchemaDSTC8Processor.format_turn_idx(turn_idx))
                # create an example given current turn, user utterance, system_utterances
                # here, it only use the previous system utterance and current user utterance
                # TODO: support long context example
                turn_examples, prev_states = self._create_examples_from_turn(
                    turn_id, system_utterance, user_utterance, system_frames,
                    user_frames, prev_states, schemas)
                examples.extend(turn_examples)
        return examples

    def _get_state_update(self, current_state, prev_state):
        """
        get the state increments between current_state and prev_state
        """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    @classmethod
    def format_turn_idx(cls, turn_idx):
        turn_id = "{:02d}".format(turn_idx)
        return turn_id

    def _create_examples_from_turn(self, turn_id, system_utterance,
                                   user_utterance, system_frames, user_frames,
                                   prev_states, schemas):
        """
        Creates an example for each frame in the user turn.
        """
        # after tokenization, it generatet subwords, and the indices to its tokens
        system_tokens, system_alignments, system_inv_alignments = (
            self._tokenize(system_utterance))
        user_tokens, user_alignments, user_inv_alignments = (
            self._tokenize(user_utterance))
        states = {}
        # create a base example without sepecific dialogue and schema info
        base_example = SchemaDSTExample(
            dataset_config=self._dataset_config,
            max_seq_length=self._max_seq_length,
            tokenizer=self._tokenizer,
            log_data_warnings=self._log_data_warnings)
        base_example.example_id = turn_id
        # Here, only use the system and user utterance as input
        base_example.add_utterance_features(system_tokens,
                                            system_inv_alignments,
                                            user_tokens,
                                            user_inv_alignments)
        examples = []
        # In one turn, it may have multiple turns
        for service, user_frame in user_frames.items():
            # Create an example for this service.
            example = base_example.make_copy_with_utterance_features()
            service_id = schemas.get_service_id(service)
            # In one turn, there will be multiple frames, usually they
            # are from different service, use the joint turn_id and
            # service
            example.example_id = "{}-{:<20}".format(turn_id, service)
            example.service_id = service_id
            example.service_schema = schemas.get_service_schema(service)
            system_frame = system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            # find state updates with current and prev states
            state_update = self._get_state_update(
                state, prev_states.get(service, {}))
            states[service] = state
            # Populate features in the example.
            example.add_categorical_slots(state_update)
            # The input tokens to bert are in the format [CLS] [S1] [S2] ... [SEP]
            # [U1] [U2] ... [SEP] [PAD] ... [PAD]. For system token indices a bias of
            # 1 is added for the [CLS] token and for user tokens a bias of 2 +
            # len(system_tokens) is added to account for [CLS], system tokens and
            # [SEP].
            if isinstance(self._tokenizer, RobertaTokenizer):
                special_bias = (1, 3)
            else:
                special_bias = (1, 2)
            user_span_boundaries = self._find_subword_indices(
                state_update, user_utterance, user_frame["slots"], user_alignments,
                user_tokens, special_bias[1] + len(system_tokens))
            if system_frame is not None:
                system_span_boundaries = self._find_subword_indices(
                    state_update, system_utterance, system_frame["slots"],
                    system_alignments, system_tokens, special_bias[0])
            else:
                system_span_boundaries = {}

            example.add_noncategorical_slots(
                state_update,
                system_span_boundaries,
                user_span_boundaries)
            example.add_requested_slots(user_frame)
            example.add_intents(user_frame)
            examples.append(example)
        return examples, states

    def _find_subword_indices(self, slot_values, utterance, char_slot_spans,
                              alignments, subwords, bias):
        """Find indices for subwords corresponding to slot values."""
        span_boundaries = {}
        for slot, values in slot_values.items():
            # Get all values present in the utterance for the specified slot.
            value_char_spans = {}
            for slot_span in char_slot_spans:
                if slot_span["slot"] == slot:
                    value = utterance[slot_span["start"]:slot_span["exclusive_end"]]
                    start_tok_idx = alignments[slot_span["start"]]
                    end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                    if 0 <= start_tok_idx < len(subwords):
                        end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                        value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
            for v in values:
                if v in value_char_spans:
                    span_boundaries[slot] = value_char_spans[v]
                    break
        return span_boundaries

    def _tokenize(self, utterance):
        """Tokenize the utterance using word-piece tokenization used by BERT.
        Args:
        utterance: A string containing the utterance to be tokenized.
        Returns:
        bert_tokens: A list of tokens obtained by word-piece tokenization of the
        utterance.
        alignments: A dict mapping indices of characters corresponding to start
        and end positions of words (not subwords) to corresponding indices in
        bert_tokens list.
        inverse_alignments: A list of size equal to bert_tokens. Each element is a
        tuple containing the index of the starting and inclusive ending
        character of the word corresponding to the subword. This list is used
        during inference to map word-piece indices to spans in the original
        utterance.
        """
        # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
        # direct concatenation of all the tokens in the sequence will be the
        # original string.
        tokens = data_utils._naive_tokenize(utterance)
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = []
        # These lists store inverse alignments to be used during inference.
        bert_tokens_start_chars = []
        bert_tokens_end_chars = []
        for token in tokens:
            if token.strip():
                subwords = self._tokenizer.tokenize(token)
                # Store the alignment for the index of starting character and the
                # inclusive ending character of the token.
                alignments[char_index] = len(bert_tokens)
                bert_tokens_start_chars.extend([char_index] * len(subwords))
                bert_tokens.extend(subwords)
                # The inclusive ending character index corresponding to the word.
                inclusive_char_end = char_index + len(token) - 1
                alignments[inclusive_char_end] = len(bert_tokens) - 1
                bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
            char_index += len(token)
        inverse_alignments = list(
            zip(bert_tokens_start_chars, bert_tokens_end_chars))
        return bert_tokens, alignments, inverse_alignments

    def get_whole_dialogs(self, data_dir, dataset):
        """Get the number of dilaog examples in the data split.
        Args:
        data_dir: str, the main folder store the data with three subdirs, traing/dev/test
        dataset: str. can be "train", "dev", or "test".
        Returns:
        example_count: int. number of examples in the specified dataset.
        """
        dialog_paths = self.get_input_dialog_files(data_dir, dataset)
        dials = load_dialogues(dialog_paths)
        return dials

    def get_num_dialog_examples(self, data_dir, dataset):
        """Get the number of dilaog examples in the data split.
        Args:
        data_dir: str, the main folder store the data with three subdirs, traing/dev/test
        dataset: str. can be "train", "dev", or "test".
        Returns:
        example_count: int. number of examples in the specified dataset.
        """
        example_count = 0
        dials = self.get_whole_dialogs(data_dir, dataset)
        for dialog in dials:
            for turn in dialog["turns"]:
                if turn["speaker"] == "USER":
                    example_count += len(turn["frames"])
        return example_count


class SchemaDSTExample(object):
    """An example for training/inference."""
    def __init__(self, dataset_config,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                 service_schema=None, example_id="NONE",
                 service_id="NONE",
                 tokenizer=None,
                 log_data_warnings=False):
        """Constructs an SchemaDSTExample.
        Args:
        dataset_config: DataConfig object denoting the config of the dataset.
        max_seq_length: The maximum length of the sequence. Sequences longer than
        this value will be truncated.
        service_schema: A ServiceSchema object wrapping the schema for the service
        corresponding to this example.
        example_id: Unique identifier for the example.
        tokenizer: A tokenizer object that has convert_tokens_to_ids and
        convert_ids_to_tokens methods. It must be non-None when
        log_data_warnings: If True, warnings generted while processing data are
        logged. This is useful for debugging data processing.
        """
        # The corresponding service schme object, which can be used to obatin the intent ans slots
        self.service_schema = service_schema
        # example_id - global_turn_id + service_name
        # global_turn_id = split-dialogue_id-turn_idx
        self.example_id = example_id
        self.service_id = service_id
        # max seq_length for bert sequence encoding
        self._max_seq_length = max_seq_length
        # tokenizer used token the utterance
        self._tokenizer = tokenizer
        # whether log data warning
        self._log_data_warnings = log_data_warnings
        # the dataset config, which contains the range of dataset for single or multiple domain
        self._dataset_config = dataset_config
        # prepare for bert input
        # The id of each subword in the vocabulary for BERT.
        self.utterance_ids = [0] * self._max_seq_length
        # Denotes the identity of the sequence. Takes values 0 (system utterance)
        # and 1 (user utterance).
        self.utterance_segment = [0] * self._max_seq_length
        # Mask which takes the value 0 for padded tokens and 1 otherwise.
        self.utterance_mask = [0] * self._max_seq_length

        # Start and inclusive end character indices in the original utterance
        # corresponding to the tokens. This is used to obtain the character indices
        # from the predicted subword indices during inference.
        # NOTE: A positive value indicates the character indices in the user
        # utterance whereas a negative value indicates the character indices in the
        # system utterance. The indices are offset by 1 to prevent ambiguity in the
        # 0 index, which could be in either the user or system utterance by the
        # above convention. Now the 0 index corresponds to padded tokens.
        self.start_char_idx = [0] * self._max_seq_length
        self.end_char_idx = [0] * self._max_seq_length

        # Number of categorical slots present in the service.
        self.num_categorical_slots = 0
        # The status of each categorical slot in the service.
        # Each slot has thress slot status: off, dontcare, active
        # off , means no new assignment for this slot, keeping unchanged
        # doncare, means no preference for the slot, hence, a special slot value doncare for it
        # active, means this slot will predict a new value and get assigned in the next stage.
        self.categorical_slot_status = [STATUS_OFF] * dataset_config.max_num_cat_slot
        # Number of values taken by each categorical slot. This is to check each availible slot
        self.num_categorical_slot_values = [0] * dataset_config.max_num_cat_slot
        # The index of the correct value for each categorical slot.
        self.categorical_slot_values = [0] * dataset_config.max_num_cat_slot

        # Number of non-categorical slots present in the service.
        self.num_noncategorical_slots = 0
        # The status of each non-categorical slot in the service.
        self.noncategorical_slot_status = [STATUS_OFF] * dataset_config.max_num_noncat_slot
        # The index of the starting subword corresponding to the slot span for a
        # non-categorical slot value.
        self.noncategorical_slot_value_start = [0] * dataset_config.max_num_noncat_slot
        # The index of the ending (inclusive) subword corresponding to the slot span
        # for a non-categorical slot value.
        self.noncategorical_slot_value_end = [0] * dataset_config.max_num_noncat_slot

        # Total number of slots present in the service. All slots are included here
        # since every slot can be requested
        self.num_slots = 0
        # Takes value 1 if the corresponding slot is requested, 0 otherwise.
        self.requested_slot_status = [STATUS_OFF] * (
            dataset_config.max_num_cat_slot + dataset_config.max_num_noncat_slot)

        # Total number of intents present in the service.
        self.num_intents = 0
        # Takes value 1 if the intent is active, 0 otherwise.
        self.intent_status = [STATUS_OFF] * dataset_config.max_num_intent

    def __str__(self):
        """
        return the to string
        """
        return self.__repr__()

    def __repr__(self):
        """
        more rich to string
        """
        summary_dict = self.readable_summary
        return json.dumps(summary_dict, sorted_keys=True)

    @property
    def readable_summary(self):
        """
        Get a readable dict that summarizes the attributes of an SchemaDSTExample.
        """
        seq_length = sum(self.utterance_mask)
        utt_toks = self._tokenizer.convert_ids_to_tokens(
            self.utterance_ids[:seq_length])
        utt_tok_mask_pairs = list(
            zip(utt_toks, self.utterance_segment[:seq_length]))
        active_intents = [
            self.service_schema.get_intent_from_id(idx)
            for idx, s in enumerate(self.intent_status)
            if s == STATUS_ACTIVE
        ]
        if len(active_intents) > 1:
            raise ValueError(
                "Should not have multiple active intents in a single service.")
        active_intent = active_intents[0] if active_intents else ""
        slot_values_in_state = {}
        for idx, status in enumerate(self.categorical_slot_status):
            if status == STATUS_ACTIVE:
                value_id = self.categorical_slot_values[idx]
                cat_slot = self.service_schema.get_categorical_slot_from_id(idx)
                slot_values_in_state[cat_slot] = self.service_schema.get_categorical_slot_value_from_id(
                    idx, value_id)
            elif status == STATUS_DONTCARE:
                slot_values_in_state[cat_slot] = STR_DONTCARE

        for idx, status in enumerate(self.noncategorical_slot_status):
            if status == STATUS_ACTIVE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                start_id = self.noncategorical_slot_value_start[idx]
                end_id = self.noncategorical_slot_value_end[idx]
                # Token list is consisted of the subwords that may start with "##". We
                # remove "##" to reconstruct the original value. Note that it's not a
                # strict restoration of the original string. It's primarily used for
                # debugging.
                # ex. ["san", "j", "##ose"] --> "san jose"
                readable_value = " ".join(utt_toks[start_id:end_id + 1]).replace(" ##", "")
                slot_values_in_state[slot] = readable_value
            elif status == STATUS_DONTCARE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                slot_values_in_state[slot] = STR_DONTCARE

        summary_dict = {
            "utt_tok_mask_pairs": utt_tok_mask_pairs,
            "utt_len": seq_length,
            "num_categorical_slots": self.num_categorical_slots,
            "num_categorical_slot_values": self.num_categorical_slot_values,
            "num_noncategorical_slots": self.num_noncategorical_slots,
            "service_name": self.service_schema.service_name,
            "active_intent": active_intent,
            "slot_values_in_state": slot_values_in_state
        }
        return summary_dict

    def add_utterance_features(self, system_tokens, system_inv_alignments,
                               user_tokens, user_inv_alignments):
        """Add utterance related features input to bert.  Note: this method
        modifies the system tokens and user_tokens in place to make
        their total length <= the maximum input length for BERT model.

        Args:
        system_tokens: a list of strings which represents system
        utterance.
        system_inv_alignments: a list of tuples which
        denotes the start and end charater of the tpken that a bert
        token originates from in the original system utterance.
        user_tokens: a list of strings which represents user
        utterance.
        user_inv_alignments: a list of tuples which
        denotes the start and end charater of the token that a bert
        token originates from in the original user utterance.
        """
        # Make user-system utterance input (in BERT format)
        # Input sequence length for utterance BERT encoder
        max_utt_len = self._max_seq_length
        # Modify lengths of sys & usr utterance so that length of total utt
        # (including [CLS], [SEP], [SEP]) is no more than max_utt_len
        # For Roberta, we need to change this.
        if isinstance(self._tokenizer, RobertaTokenizer):
            special_token_num = 4
        else:
            special_token_num = 3
        is_too_long = data_utils.truncate_seq_pair(system_tokens, user_tokens, max_utt_len - special_token_num)
        if is_too_long and self._log_data_warnings:
            logger.info("Utterance sequence truncated in example id - %s.",
                        self.example_id)

        # Construct the tokens, segment mask and valid token mask which will be
        # input to BERT, using the tokens for system utterance (sequence A) and
        # user utterance (sequence B).
        utt_subword = []
        utt_seg = []
        utt_mask = []
        start_char_idx = []
        end_char_idx = []

        utt_subword.append(self._tokenizer.cls_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(system_tokens):
            utt_subword.append(subword)
            utt_seg.append(0)
            utt_mask.append(1)
            st, en = system_inv_alignments[subword_idx]
            start_char_idx.append(-(st + 1))
            end_char_idx.append(-(en + 1))

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        # for roberta, TODO
        if isinstance(self._tokenizer, RobertaTokenizer):
            utt_subword.append(self._tokenizer.cls_token)
            utt_seg.append(1)
            utt_mask.append(1)
            start_char_idx.append(0)
            end_char_idx.append(0)

        for subword_idx, subword in enumerate(user_tokens):
            utt_subword.append(subword)
            utt_seg.append(1)
            utt_mask.append(1)
            st, en = user_inv_alignments[subword_idx]
            start_char_idx.append(st + 1)
            end_char_idx.append(en + 1)

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(1)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)
        # convert subwords to ids
        utterance_ids = self._tokenizer.convert_tokens_to_ids(utt_subword)

        # Zero-pad up to the BERT input sequence length.
        while len(utterance_ids) < max_utt_len:
            utterance_ids.append(self._tokenizer.pad_token_id)
            utt_seg.append(self._tokenizer.pad_token_type_id)
            utt_mask.append(0)
            start_char_idx.append(0)
            end_char_idx.append(0)
        self.utterance_ids = utterance_ids
        self.utterance_segment = utt_seg
        self.utterance_mask = utt_mask
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx

    def make_copy_with_utterance_features(self):
        """
        Make a copy of the current example with utterance features.
        """
        new_example = SchemaDSTExample(
            dataset_config=self._dataset_config,
            max_seq_length=self._max_seq_length,
            service_schema=self.service_schema,
            example_id=self.example_id,
            service_id=self.service_id,
            tokenizer=self._tokenizer,
            log_data_warnings=self._log_data_warnings)

        new_example.utterance_ids = list(self.utterance_ids)
        new_example.utterance_segment = list(self.utterance_segment)
        new_example.utterance_mask = list(self.utterance_mask)
        new_example.start_char_idx = list(self.start_char_idx)
        new_example.end_char_idx = list(self.end_char_idx)
        return new_example

    def add_categorical_slots(self, state_update):
        """
        For every state update, Add features and labels for categorical slots.
        """
        categorical_slots = self.service_schema.categorical_slots
        self.num_categorical_slots = len(categorical_slots)
        for slot_idx, slot in enumerate(categorical_slots):
            values = state_update.get(slot, [])
            # Add categorical slot value features.
            slot_values = self.service_schema.get_categorical_slot_values(slot)
            self.num_categorical_slot_values[slot_idx] = len(slot_values)
            if not values:
                # the status is off, it means no new assignment for the slot
                self.categorical_slot_status[slot_idx] = STATUS_OFF
            elif values[0] == STR_DONTCARE:
                # use a spaecial value dontcare
                self.categorical_slot_status[slot_idx] = STATUS_DONTCARE
            else:
                self.categorical_slot_status[slot_idx] = STATUS_ACTIVE
                # here it only use the first values
                self.categorical_slot_values[slot_idx] = (
                    self.service_schema.get_categorical_slot_value_id(slot, values[0]))

    def add_noncategorical_slots(self, state_update, system_span_boundaries,
                                 user_span_boundaries):
        """
        Add features for non-categorical slots.
        Here only consider the spans in the last user and system turns
        """
        noncategorical_slots = self.service_schema.non_categorical_slots
        self.num_noncategorical_slots = len(noncategorical_slots)
        for slot_idx, slot in enumerate(noncategorical_slots):
            values = state_update.get(slot, [])
            if not values:
                self.noncategorical_slot_status[slot_idx] = STATUS_OFF
            elif values[0] == STR_DONTCARE:
                self.noncategorical_slot_status[slot_idx] = STATUS_DONTCARE
            else:
                self.noncategorical_slot_status[slot_idx] = STATUS_ACTIVE
                # Add indices of the start and end tokens for the first encountered
                # value. Spans in user utterance are prioritized over the system
                # utterance. If a span is not found, the slot value is ignored.
                if slot in user_span_boundaries:
                    start, end = user_span_boundaries[slot]
                elif slot in system_span_boundaries:
                    start, end = system_span_boundaries[slot]
                else:
                    # A span may not be found because the value was cropped out or because
                    # the value was mentioned earlier in the dialogue. Since this model
                    # only makes use of the last two utterances to predict state updates,
                    # it will fail in such cases.
                    if self._log_data_warnings:
                        logger.info(
                            "Slot values %s not found in user or system utterance in "
                            + "example with id - %s, service_id: %s .",
                            str(values), self.example_id, self.service_id)
                    continue
                self.noncategorical_slot_value_start[slot_idx] = start
                self.noncategorical_slot_value_end[slot_idx] = end

    def add_requested_slots(self, frame):
        """
        requested_slots can be only slot in the schema definitions
        """
        all_slots = self.service_schema.slots
        self.num_slots = len(all_slots)
        for slot_idx, slot in enumerate(all_slots):
            if slot in frame["state"]["requested_slots"]:
                self.requested_slot_status[slot_idx] = STATUS_ACTIVE

    def add_intents(self, frame):
        """
        active intents for this turn frame examples
        """
        all_intents = self.service_schema.intents
        self.num_intents = len(all_intents)
        for intent_idx, intent in enumerate(all_intents):
            if intent == frame["state"]["active_intent"]:
                self.intent_status[intent_idx] = STATUS_ACTIVE

class InputFeatures(object):
    """
    A single set of features of the data, Here we build the baseline model first
    Then we will consider how merge all into a unified model
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


def convert_examples_to_features(examples,
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
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        if isinstance(example, PaddingSchemaDSTExample):
            ex = SchemaDSTExample(dataset_config=dataset_config)
        else:
            ex = example

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
