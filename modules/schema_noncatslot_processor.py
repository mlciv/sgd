from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

from modules.core.schema_constants import *
from modules.core.schema_dst_example import SchemaDSTExample
from modules.schema_dialog_processor import SchemaDialogProcessor
from utils import schema
from utils import data_utils
import os
import json

import logging

logger = logging.getLogger(__name__)

class SchemaNonCatSlotProcessor(SchemaDialogProcessor):
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
        super(SchemaNonCatSlotProcessor, self).__init__(
            dataset_config=dataset_config, tokenizer=tokenizer, max_seq_length=max_seq_length,
            log_data_warnings=log_data_warnings, dialog_cxt_length=dialog_cxt_length)

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
                turn_id = "{:<5}-{:<12}-{:<3}".format(
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
        start_turn, offsets = base_example.add_dial_history_features(utterances)
        # add utterance features
        all_noncat_slot_examples = []
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
            # here only assuming in domcin carry over.
            system_frame = current_system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            states[service] = state
            state_update = self._get_state_update(
                state, prev_states.get(service, {}))
            # The input tokens to bert are in the format [CLS] utterance history + current user utterance [SEP]
            # [schema_description][SEP] [PAD] ... [PAD].
            # 50% are from the current user utterance, 40% are from the system utterance.
            # 10% are from the previous utterances. We match with previous utterance only when
            # it failed to match user and system utterance.
            # Here, we make sure that handle those slots already in the frames
            user_span_boundaries = self._find_subword_indices(
                state_update, current_user_utterance, user_frame["slots"], current_user_alignments,
                current_user_tokens, offsets[-1])
            if system_frame is not None and len(offsets) > 1:
                system_span_boundaries = self._find_subword_indices(
                    state_update, current_system_utterance, system_frame["slots"],
                    current_system_alignments, current_system_tokens, offsets[-2])
            else:
                system_span_boundaries = {}

            # TODO: may carry over cross service here, and any other resources.
            # Here, in future we just do the string match to find all other spans
            noncat_slot_examples = example.add_noncategorical_slots(
                state_update,
                system_span_boundaries,
                user_span_boundaries)
            all_noncat_slot_examples.extend(noncat_slot_examples)

        return all_noncat_slot_examples, states
