from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os
import collections
import tensorflow as tf
from schema_guided_dst import schema
from schema_guided_dst.baseline.config import *
from schema_guided_dst.baseline import data_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_folder", None,
    "data_folder in which all schema files and data files exist")

flags.DEFINE_string(
    "task_name", None,
    "task_name must be in {}".format(','.join(DATASET_CONFIG.keys())))

flags.DEFINE_string(
    "split", None,
    "split as train, dev, test")

def is_boolean(slot, service_schema):
    if slot in service_schema.all_categorical_slots:
        values = service_schema.get_all_categorical_slot_values(slot)
        # not sure the order
        if len(values) == 2 and "True" in values and "False" in values:
            return True
        else:
            return False
    else:
        return False

def is_numeric(slot, service_schema):
    if slot in service_schema.all_categorical_slots:
        values = service_schema.get_all_categorical_slot_values(slot)
        return all([value.isdigit() for value in values])
    return False


def is_span_based(slot, service_schema):
    if slot in service_schema.all_non_categorical_slots:
        return True
    else:
        return False


def make_full_slot_name(service_name, slot_name):
    return service_name + "@" + slot_name


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    split = FLAGS.split
    task_name = FLAGS.task_name
    schema_file = os.path.join(FLAGS.data_folder, split, "schema.json")
    schemas = schema.Schema(schema_file)
    cnt = collections.Counter()
    span_based_slots = {}
    boolean_slots = {}
    numeric_slots = {}
    text_based_slots = {}
    for service_name in schemas.services:
        service_schema = schemas.get_service_schema(service_name)
        for slot_name in service_schema._slots:
            name = make_full_slot_name(service_schema._service_name, slot_name)
            if is_span_based(slot_name, service_schema):
                cnt["span_based"] += 1
                if slot_name in service_schema.state_slots:
                    cnt["span_base_state"] += 1
                for slot in service_schema._schema_json["slots"]:
                    if slot_name == slot["name"]:
                        span_based_slots[name] = slot
            elif is_boolean(slot_name, service_schema):
                cnt["boolean"] += 1
                if slot_name in service_schema.state_slots:
                    cnt["boolean_state"] += 1
                for slot in service_schema._schema_json["slots"]:
                    if slot_name == slot["name"]:
                        boolean_slots[name] = slot
            elif is_numeric(slot_name, service_schema):
                cnt["numeric"] += 1
                if slot_name in service_schema.state_slots:
                    cnt["numeric_state"] += 1
                for slot in service_schema._schema_json["slots"]:
                    if slot_name == slot["name"]:
                        numeric_slots[name] = slot
            else:
                cnt["text_based"] += 1
                if slot_name in service_schema.state_slots:
                    cnt["text_state"] += 1
                for slot in service_schema._schema_json["slots"]:
                    if slot_name == slot["name"]:
                        text_based_slots[name] = slot

    tf.logging.info("schema cnt: %s", json.dumps(cnt, indent=4))
    tf.logging.info("span_based_slots: %s", json.dumps(span_based_slots, indent=4))
    tf.logging.info("boolean_slots: %s", json.dumps(boolean_slots, indent=4))
    tf.logging.info("numeric_slots: %s", json.dumps(numeric_slots, indent=4))
    tf.logging.info("text_based_slots: %s", json.dumps(text_based_slots, indent=4))

    # check how many those slots are in the dialogue state.
    # read all dialogues
    turn_cnt = collections.Counter()
    slot_cnt = collections.Counter()
    slot_values_cnt = collections.Counter()
    requested_slots_cnt = collections.Counter()
    span_slots_cnt = collections.Counter()
    intent_cnt = collections.Counter()
    multiple_frames_cnt = collections.Counter()
    dialog_paths = [
        os.path.join(FLAGS.data_folder, split,
                     "dialogues_{:03d}.json".format(i))
        for i in DATASET_CONFIG[task_name].file_ranges[split]
    ]
    # check user turns
    dialogs = data_utils.load_dialogues(dialog_paths)
    tf.logging.info("total dialogues: %d", len(dialogs))
    for dialog in dialogs:
        last_active_intent = ""
        span_slots_in_user_turn = {}
        span_slots_in_previous_state = {}
        for turn in dialog["turns"]:
            if turn["speaker"] == "USER":
                turn_cnt["USER"] += 1
                # for multiple frames
                if len(turn["frames"]) > 1:
                    multiple_frames_cnt[len(turn["frames"])] += 1
                    services_intents = set([ '@'.join([f["service"], f["state"]["active_intent"]]) for f in turn["frames"]])
                    # multiple different services.
                    if len(services_intents) > 1:
                        multiple_frames_cnt[','.join(services_intents)] += 1

                # for slots
                for frame in turn["frames"]:
                    # check each frame
                    service = frame["service"]
                    service_schema = schemas.get_service_schema(service)
                    state = frame["state"]
                    span_slot_names_in_current_turn = [i['slot'] for i in frame["slots"]]
                    active_intent = state["active_intent"]
                    intent_cnt["active_intent"] += 1
                    if active_intent != last_active_intent:
                        intent_cnt["intent_change"] += 1
                    if active_intent == "NONE":
                        intent_cnt["NONE_intent"] += 1
                    last_active_intent = active_intent
                    requested_slots = state["requested_slots"]
                    requested_slots_cnt[len(requested_slots)] += 1
                    slot_cnt[len(state["slot_values"])] += 1
                    for slot, slot_values in state["slot_values"].items():
                        # general carry over
                        name = make_full_slot_name(service, slot)
                        slot_values_cnt[len(slot_values)] += 1
                        if name in span_based_slots:
                            slot_cnt["span_based"] += 1
                            if slot in service_schema.state_slots:
                                slot_cnt["span_based_state"] += 1
                            # if a slot is span_based slot, but the lost in the turn-level slots,
                            if slot not in span_slot_names_in_current_turn:
                                # this span slot is not from current turns
                                span_slots_cnt["span_not_current_turn"] += 1
                                if slot in span_slots_in_previous_state:
                                    if len(set(slot_values) & set(span_slots_in_previous_state[slot][2])):
                                        # this slot filled in previous user turn with the same joint value
                                        span_slots_cnt["span_from_previous_state"] += 1
                                    else:
                                        span_slots_cnt["differnt_span_value_from_previous_state"] += 1
                                else:
                                    # this slot filled must be from the previous system turn
                                    span_slots_cnt["new_span_slot_from_system_turn"] += 1
                            else:
                                # this slot is from span in current turn
                                span_slots_cnt["new_span_in_current_turn"] += 1
                                span_slots_in_user_turn[slot] = (service, active_intent, slot_values)
                        elif name in boolean_slots:
                            if slot in service_schema.state_slots:
                                slot_cnt["boolean_state"] += 1
                            slot_cnt["boolean"] += 1
                        elif name in numeric_slots:
                            if slot in service_schema.state_slots:
                                slot_cnt["numeric_state"] += 1
                            slot_cnt["numeric"] += 1
                        else:
                            if slot in service_schema.state_slots:
                                slot_cnt["text_based_state"] += 1
                            slot_cnt["text_based"] += 1
                        span_slots_in_previous_state[slot] = (service, active_intent, slot_values)
            else:
                turn_cnt["SYSTEM"] += 1
                # check the slot from system turn

    tf.logging.info("turn cnt: %s", json.dumps(turn_cnt, indent=4))
    tf.logging.info("multi frames cnt: %s", json.dumps(multiple_frames_cnt, indent=4))
    tf.logging.info("slot cnt: %s", json.dumps(slot_cnt, indent=4))
    tf.logging.info("slot_values cnt: %s", json.dumps(slot_values_cnt, indent=4))
    tf.logging.info("span_slots cnt: %s", json.dumps(span_slots_cnt, indent=4))
    tf.logging.info("requested_slots cnt: %s", json.dumps(requested_slots_cnt, indent=4))
    tf.logging.info("intent cnt: %s", json.dumps(intent_cnt, indent=4))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_folder")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("split")
    tf.compat.v1.app.run(main)
