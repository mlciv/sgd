from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

from modules.core.schema_constants import *
from modules.core.schema_dst_example import SchemaDSTExample
from utils import schema
from utils import data_utils
from utils import evaluate_utils
import os
import json

import logging

logger = logging.getLogger(__name__)

class SchemaDialogProcessor(DataProcessor):
    """
    Data generator for dstc8 dialogues.
    Implemented based transformer DataProcessor
    """
    def __init__(self,
                 dataset_config,
                 tokenizer,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                 max_schema_seq_length=MAX_SCHEMA_SEQ_LENGTH,
                 log_data_warnings=False, dialog_cxt_length=2):
        self._log_data_warnings = log_data_warnings
        self._dataset_config = dataset_config
        # BERT tokenizer
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._max_schema_seq_length = max_schema_seq_length
        # the dialogue history used(not including current user utterance)
        self.dial_cxt_length = dialog_cxt_length
        self.metrics = evaluate_utils.CORE_METRIC_SUBKEYS

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

    def get_schema_file(self, data_dir, dataset):
        """
        return the schema file given the data_dir and split
        """
        return os.path.join(data_dir, dataset, self._dataset_config.schema_file)

    def get_schemas(self, data_dir, dataset):
        """
        get all the schema given data_dir and datasplit
        """
        schema_path = self.get_schema_file(data_dir, dataset)
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
        schemas = self.get_schemas(data_dir, dataset)

        examples = []
        for dialog_idx, dialog in enumerate(dialogs):
            if dialog_idx % 1000 == 0:
                logger.info("Processed %d dialogs.", dialog_idx)
            examples.extend(
                self._create_examples_from_dialog(dialog, schemas, dataset))
        return examples

    @classmethod
    def format_turn_idx(cls, turn_idx):
        turn_id = "{:02d}".format(turn_idx)
        return turn_id

    def _create_examples_from_dialog(self, dialog, schemas, dataset):
        """
        Create examples for every turn in the dialog.
        In this implementation, we only use the system and user turn as sentence pair
        """
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        utterances = []
        system_frames = {}
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            # Only user turn have the frame information to predict
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_tokens, user_alignments, user_inv_alignments = (self._tokenize(user_utterance))
                user_frames = {f["service"]: f for f in turn["frames"]}
                utterances.append((USER_SPECIAL_TOKEN, user_utterance, user_tokens,
                                        user_alignments, user_inv_alignments, user_frames))
                # a global turn id
                # turnuid ="split-dialogue_id-turn_idx"
                turn_id = "{:<5}-{:<20}-{:<3}".format(
                    dataset, dialog_id,
                    SchemaDialogProcessor.format_turn_idx(turn_idx))
                turn_examples, prev_states = self._create_examples_from_turn(
                    turn_id, utterances, prev_states, schemas)
                examples.extend(turn_examples)
                # dialogue history will not including the current user utterance.
            else:
                system_utterance = turn["utterance"]
                system_tokens, system_alignments, system_inv_alignments = (self._tokenize(system_utterance))
                system_frames = {f["service"]: f for f in turn["frames"]}
                utterances.append((SYSTEM_SPECIAL_TOKEN, system_utterance, system_tokens,
                                          system_alignments, system_inv_alignments, system_frames))
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

    def _create_examples_from_turn(self, turn_id, utterances,
                                   prev_states, schemas):
        """
        Creates an example for each frame in the user turn.
        """
        states = {}
        _, current_user_utterance, current_user_tokens, \
            current_user_alignments, current_user_inv_alignements, current_user_frames = utterances[-1]
        if len(utterances) > 1:
            _, current_system_utterance, current_system_tokens, \
                current_system_alignments, current_system_inv_alignements, current_system_frames = utterances[-2]
        else:
            current_system_frames = {}
        # create a base example without sepecific dialogue and schema info
        base_example = SchemaDSTExample(
            dataset_config=self._dataset_config,
            max_seq_length=self._max_seq_length,
            max_schema_seq_length=self._max_schema_seq_length,
            tokenizer=self._tokenizer,
            log_data_warnings=self._log_data_warnings,
            dial_cxt_length=self.dial_cxt_length
        )
        base_example.example_id = turn_id
        start_turn, start_turn_subtoken_offset, global_subtoken_offsets = base_example.add_dial_history_features(utterances)
        # add utterance features
        examples = []
        # In current user turn, it may have multiple frames
        for service, user_frame in current_user_frames.items():
            # Create an example for this service.
            example = base_example.make_copy_with_utterance_features()
            service_id = schemas.get_service_id(service)
            # In one turn, there will be multiple frames, usually they
            # are from different service, use the joint turn_id and
            # service
            example.example_id = "{}-{:<20}".format(turn_id, service)
            example.service_id = service_id
            example.service_schema = schemas.get_service_schema(service)
            # here only assuming in domain carry over.
            system_frame = current_system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            # find state updates with current and prev states
            state_update = self._get_state_update(
                state, prev_states.get(service, {}))
            states[service] = state
            # Populate features in the example.
            cat_slot_examples = example.add_categorical_slots(state_update)

            # The input tokens to bert are in the format [CLS] utterance history + current user utterance [SEP]
            # [schema_description][SEP] [PAD] ... [PAD].
            # 50% are from the current user utterance, 40% are from the system utterance.
            # 10% are from the previous utterances. We match with previous utterance only when
            # it failed to match user and system utterance.
            # Here, we make sure that handle those slots already in the frames
            # when the frames have already offer the spans
            user_span_boundaries = self._find_subword_indices(
                state_update, current_user_utterance, user_frame["slots"], current_user_alignments,
                current_user_tokens, global_subtoken_offsets[-1])
            if system_frame is not None and len(global_subtoken_offsets) > 1:
                system_span_boundaries = self._find_subword_indices(
                    state_update, current_system_utterance, system_frame["slots"],
                    current_system_alignments, current_system_tokens, global_subtoken_offsets[-2])
            else:
                system_span_boundaries = {}

            # TODO: may carry over cross service here, and any other resources.
            # cross domain carry over is not easyto do here, no slot carryover relations
            # not easy to find the slot matching.
            # Here, in future we just do the string match to find all other spans
            noncat_slot_examples = example.add_noncategorical_slots(
                state_update,
                system_span_boundaries,
                user_span_boundaries, utterances, start_turn, start_turn_subtoken_offset, global_subtoken_offsets)
            req_slot_examples = example.add_requested_slots(user_frame)
            intent_examples = example.add_intents(user_frame)
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
                    if "start" not in slot_span:
                        logger.warning("utterance:{} ".format(utterance))
                        continue
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
        dials = data_utils.load_dialogues(dialog_paths)
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
