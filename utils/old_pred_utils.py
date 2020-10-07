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

"""Prediction and evaluation-related utility functions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import os
import logging
import torch
import modules.core.schema_constants as schema_constants

from utils import schema
from src import utils_schema

logger = logging.getLogger(__name__)

REQ_SLOT_THRESHOLD = 0.5
ACTIVE_INTENT_THRESHOLD = 0.5
CAT_VALUE_THRESHOLD = 0.5


def get_predicted_dialog(dialog, all_predictions, schemas):
    """Update labels in a dialogue based on model predictions.
    Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_predictions: A dict mapping prediction name to the predicted value. See
      SchemaGuidedDST class for the contents of this dict.
    schemas: A Schema object wrapping all the schemas for the dataset.
    Returns:
    A json object containing the dialogue with labels predicted by the model.
    """
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.
    dialog_id = dialog["dialogue_id"]
    # The slot values tracked for each service.
    all_slot_values = collections.defaultdict(dict)
    # concatenating without any token
    global_utterance_for_index = ''
    for turn_idx, turn in enumerate(dialog["turns"]):
        # inclusive
        current_utt_start_char_index = len(global_utterance_for_index)
        global_utterance_for_index = global_utterance_for_index + turn["utterance"]
        # exclusive
        current_utt_end_char_index = len(global_utterance_for_index)
        if turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = (dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else "")
            for frame in turn["frames"]:
                dial_key = (dialog_id, turn_idx, frame["service"])
                if dial_key in all_predictions:
                    predictions = all_predictions[(dialog_id, turn_idx, frame["service"])]
                else:
                    logger.warn("{} not found in predictions".format(dial_key))
                    continue
                # logger.info("dial_id:{}, turn_idx:{}, predictions:{}".format(dialog_id, turn_idx, predictions))
                slot_values = all_slot_values[frame["service"]]
                service_schema = schemas.get_service_schema(frame["service"])
                # Remove the slot spans and state if present.
                gd_slots = frame.pop("slots", None)
                gd_state = frame.pop("state", None)

                # The baseline model doesn't predict slot spans. Only state predictions
                # are added.
                state = {}

                # Add prediction for active intent. Offset is subtracted to account for
                # NONE intent.
                active_intent = "NONE"
                if "intent_status" in predictions:
                    if isinstance(predictions["intent_status"], int) or (isinstance(predictions["intent_status"], torch.Tensor) and predictions["intent_status"].dim() == 0):
                        active_intent_id = predictions["intent_status"]
                        active_intent = (
                            service_schema.get_intent_from_id(active_intent_id - 1)
                            if active_intent_id else "NONE")
                    else:
                        # like req_slot using sigmoid
                        has_intent = any([p_active > ACTIVE_INTENT_THRESHOLD for p_active in predictions["intent_status"]])
                        if has_intent:
                            # add negative score for padding, to make sure the intent ids are valid
                            active_intent_id = torch.argmax(predictions["intent_status"]) if isinstance(predictions["intent_status"], torch.Tensor) else torch.argmax(torch.FloatTensor(predictions["intent_status"]))
                            active_intent = service_schema.get_intent_from_id(active_intent_id)
                        else:
                            active_intent = "NONE"
                state["active_intent"] = active_intent

                # Add prediction for requested slots.
                requested_slots = []
                if "req_slot_status" in predictions:
                    for slot_idx, slot in enumerate(service_schema.slots):
                        if predictions["req_slot_status"][slot_idx] > REQ_SLOT_THRESHOLD:
                            requested_slots.append(slot)
                state["requested_slots"] = requested_slots

                # Add prediction for user goal (slot values).
                # Categorical slots.
                # used for global state not for incremental
                # logger.info("dial_key={} old_slots_values={}, gd_value={}".format(dial_key, slot_values, gd_state["slot_values"]))
                new_cat_slot_values = {}
                if "cat_slot_status" in predictions:
                    for slot_idx, slot in enumerate(service_schema.categorical_slots):
                        slot_status = predictions["cat_slot_status"][slot_idx]
                        cat_values = service_schema.get_categorical_slot_values(slot)
                        if slot_status == schema_constants.STATUS_DONTCARE:
                            slot_values[slot] = schema_constants.STR_DONTCARE
                        elif slot_status == schema_constants.STATUS_ACTIVE:
                            value_idx = predictions["cat_slot_value"][slot_idx]
                            if len(cat_values) > value_idx >= 0:
                                slot_values[slot] = cat_values[value_idx]
                        # logger.info("dial_key={} slots_status={}, slot_idx={}, slot={}, value_id={}, gd_value={}".format(dial_key, slot_status, slot_idx, slot, predictions["cat_slot_value"][slot_idx], gd_state["slot_values"].get(slot, "NONE")))
                elif "cat_slot_value_full_state" in predictions:
                    for slot_idx, slot in enumerate(service_schema.categorical_slots):
                        value_id = predictions["cat_slot_value_full_state"][slot_idx]
                        value_idx = value_id - schema_constants.SPECIAL_CAT_VALUE_OFFSET
                        cat_values = service_schema.get_categorical_slot_values(slot)
                        if value_id == schema_constants.VALUE_UNKNOWN_ID:
                            # not adding the slot
                            continue
                        if value_id == schema_constants.VALUE_DONTCARE_ID:
                            new_cat_slot_values[slot] = schema_constants.STR_DONTCARE
                        else:
                            if 0 <= value_idx < len(cat_values):
                                new_cat_slot_values[slot] = cat_values[value_idx]
                elif "cat_slot_value_status" in predictions:
                    # for flattent case, we don't do incremental predciton, always predict the frame in current turn
                    for slot_idx, slot in enumerate(service_schema.categorical_slots):
                        value_id = torch.argmax(torch.FloatTensor(predictions["cat_slot_value_status"][slot_idx]))
                        value_idx = value_id - schema_constants.SPECIAL_CAT_VALUE_OFFSET
                        cat_values = service_schema.get_categorical_slot_values(slot)
                        # logger.info("{} has value: predictions['cat_slots_value_status']={}, slot_idx={}, slot={}, cat_values={}, value_id={}, gd_value={}".format(dial_key, predictions["cat_slot_value_status"][slot_idx], slot_idx, slot, cat_values, value_id, gd_state["slot_values"].get(slot, "NONE")))
                        if value_id == schema_constants.VALUE_DONTCARE_ID:
                            # dontcare
                            new_cat_slot_values[slot] = schema_constants.STR_DONTCARE
                        elif value_id == schema_constants.VALUE_UNKNOWN_ID:
                            # the current value is empty
                            continue
                        else:
                            if 0 <= value_idx < len(cat_values):
                                new_cat_slot_values[slot] = cat_values[value_idx]

                # We didn't do turn-level slot tagging F1 here.
                # Non-categorical slots.
                slots = []
                if "noncat_slot_status" in predictions:
                    for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
                        slot_status = predictions["noncat_slot_status"][slot_idx]
                        if slot_status == schema_constants.STATUS_DONTCARE:
                            slot_values[slot] = schema_constants.STR_DONTCARE
                        elif slot_status == schema_constants.STATUS_ACTIVE:
                            tok_start_idx = predictions["noncat_slot_start"][slot_idx]
                            tok_end_idx = predictions["noncat_slot_end"][slot_idx]
                            if tok_start_idx < 0 or tok_end_idx < 0 or  \
                               tok_start_idx >= predictions["noncat_alignment_start"].size()[0] or \
                               tok_end_idx >= predictions["noncat_alignment_end"].size()[0]:
                                continue
                            ch_start_idx = predictions["noncat_alignment_start"][tok_start_idx].item()
                            ch_end_idx = predictions["noncat_alignment_end"][tok_end_idx].item()
                            #if ch_start_idx >= current_utt_start_char_index and ch_end_idx <= current_utt_end_char_index:
                            #    # this slot is in the current utterance
                            #    slot_span = {}
                            #    slot_span["slot"] = slot
                            #    # shift the char index by one
                            #    slot_span["start"] = ch_start_idx - current_utt_start_char_index - 1
                            #    slot_span["exclusive_end"] = ch_end_idx - current_utt_start_char_index
                            #    slots.append(slot_span)

                            # shift the char index by one
                            # slot_values[slot] = global_utterance_for_index[ch_start_idx - 1:ch_end_idx]
                            # TO SUPPORT old utternace features
                            if ch_start_idx < 0 and ch_end_idx < 0:
                                # Add span from the system utterance.
                                slot_values[slot] = (
                                    system_utterance[-ch_start_idx - 1:-ch_end_idx])
                            elif ch_start_idx > 0 and ch_end_idx > 0:
                                # Add span from the user utterance.
                                slot_values[slot] = (user_utterance[ch_start_idx - 1:ch_end_idx])
                            # directly use global offset
                # Create a new dict to avoid overwriting the state in previous turns
                # because of use of same objects.
                frame["slots"] = slots
                # logger.info("gd_slots:{}, pred_slots:{}".format(gd_slots, slots))
                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                if "cat_slot_value_status" in predictions or "cat_slot_value_full_state" in predictions:
                    for s, v in new_cat_slot_values.items():
                        state["slot_values"][s] = [v]

                # logger.info("dial_key:{}, slot_values:{}, gd_state:{}".format(dial_key, slot_values, gd_state))
                frame["state"] = state
    return dialog


def get_predictions_index_dict(predictions, ):
    """
    get the prediction indexed into a dict
    """
    if isinstance(predictions, dict):
        # if it is already orangized as dict, just return it
        return predictions
    else:
        all_predictions = {}
        for idx, prediction in enumerate(predictions):
            if idx % 500 == 0:
                logger.debug("Processed %d examples.", idx)
            _, dialog_id, turn_id, service_name = [x.rstrip() for x in bytes(prediction["example_id"].tolist()).decode("utf-8").split("-")]
            turn_idx = int(turn_id)
            # logger.info("dialogue_id:{}, turn_idx,:{}, service_name:{}".format(dialog_id, turn_idx, service_name))
            all_predictions[(dialog_id, turn_idx, service_name)] = prediction
    return all_predictions


def get_all_prediction_dialogues(origin_dialogs, indexed_predictions, schemas):
    """
    use original dialogues, schema, and indexed prediction (via get_prediction_index_dic)
    to get all the predicted dialogs
    """
    pred_dialogs = []
    for dial in origin_dialogs:
        pred_dialogs.append(get_predicted_dialog(dial, indexed_predictions, schemas))
    return pred_dialogs


def write_predictions_to_file(predictions, input_json_files, schema_json_file,
                              output_dir):
    """Write the predicted dialogues as json files.
    Args:
    predictions: An iterator containing model predictions. This is the output of
      the predict method in the estimator.
    input_json_files: A list of json paths containing the dialogues to run
      inference on.
    schema_json_file: Path for the json file containing the schemas.
    output_dir: The directory where output json files will be created.
    """
    logger.info("Writing predictions to %s.", output_dir)
    schemas = schema.Schema(schema_json_file)
    # Index all predictions.
    indexed_predictions = get_predictions_index_dict(predictions)
    # Read each input file and write its predictions.
    for input_file_path in input_json_files:
        with open(input_file_path) as f:
            dialogs = json.load(f)
            pred_dialogs = get_all_prediction_dialogues(dialogs, indexed_predictions, schemas)
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(
                pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
