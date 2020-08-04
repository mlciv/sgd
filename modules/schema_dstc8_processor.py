from __future__ import absolute_import
from __future__ import division

import csv
import logging

from modules.schema_dialog_processor import SchemaDialogProcessor
from modules.core.schema_constants import *

from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

logger = logging.getLogger(__name__)

class SchemaDSTC8Processor(SchemaDialogProcessor):
    """
    Data generator for dstc8 dialogues. This processor will generate a set of examples for each turn
    Implemented based transformer DataProcessor
    """
    def __init__(self,
                 dataset_config,
                 tokenizer,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                 log_data_warnings=False, dialog_cxt_length=1):
        super(SchemaDSTC8Processor, self).__init__(
            dataset_config=dataset_config, tokenizer=tokenizer, max_seq_length=max_seq_length,
            log_data_warnings=log_data_warnings, dialog_cxt_length=dialog_cxt_length)

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors.
        Args:
           tensor_dict: Keys and values should match the corresponding Glue
           tensorflow_dataset examples.
        """
        raise NotImplementedError()

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
        In this implementation, we only use the system and user turn as sentence pair
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
                    SchemaDialogProcessor.format_turn_idx(turn_idx))
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
