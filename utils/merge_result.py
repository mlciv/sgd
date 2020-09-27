from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os
import glob
import logging

from utils import schema
import argparse

logger = logging.getLogger(__name__)

def merge_predicted_dialog(all_predicted_dialogs_dict, schemas):
    """Update labels in a dialogue based on model predictions.
    Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_prediction_dialogs_dict: 0: active_intent, 1 requested_slot, 2:cat_slot_task, 3:noncat_slot_tasks
    Returns:
    A json object containing the dialogue with labels predicted by the model.
    """
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.

    active_intent_focus_dialog = all_predicted_dialogs_dict.get("active_intent", None)
    requested_slots_focus_dialog = all_predicted_dialogs_dict.get("requested_slots", None)
    cat_slots_focus_dialog = all_predicted_dialogs_dict.get("cat_slots", None)
    noncat_slots_focus_dialog = all_predicted_dialogs_dict.get("noncat_slots", None)

    pivot_dialog = None
    pivot_task = None
    if active_intent_focus_dialog:
        pivot_dialog = active_intent_focus_dialog
        pivot_task = "active_intent"
    elif requested_slots_focus_dialog:
        pivot_dialog = requested_slots_focus_dialog
        pivot_task = "requested_slots"
    elif cat_slots_focus_dialog:
        pivot_dialog = cat_slots_focus_dialog
        pivot_task = "cat_slots"
    elif noncat_slots_focus_dialog:
        pivot_dialog = noncat_slots_focus_dialog
        pivot_task = "noncat_slots"
    else:
        raise RuntimeError("All input dialogus are None")

    for turn_idx, turn in enumerate(pivot_dialog["turns"]):
        if turn["speaker"] == "USER":
            # user_utterance = turn["utterance"]
            # system_utterance = (
            #    dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else "")
            for frame_idx, frame in enumerate(turn["frames"]):
                pivot_slots = frame.pop("slots", None)
                pivot_state = frame.pop("state", None)

                service_schema = schemas.get_service_schema(frame["service"])
                state = {}
                # Add prediction for active intent. Offset is subtracted to account for
                # NONE intent.
                active_intent = "NONE"
                if active_intent_focus_dialog:
                    # if not None
                    # we have specific active_intent tasks, adopt their prediction for "active_intent"
                    if pivot_task != "active_intent":
                        active_intent = active_intent_focus_dialog["turns"][turn_idx]["frames"][frame_idx]["state"]["active_intent"]
                    else:
                        active_intent = pivot_state["active_intent"]

                state["active_intent"] = active_intent

                # Add prediction for requested slots.
                requested_slots = []
                if requested_slots_focus_dialog:
                    if pivot_task != "requested_slots":
                        requested_slots = requested_slots_focus_dialog["turns"][turn_idx]["frames"][frame_idx]["state"]["requested_slots"]
                    else:
                        requested_slots = pivot_state["requested_slots"]
                state["requested_slots"] = requested_slots

                # Add prediction for user goal (slot values).
                # Categorical slots.
                slot_values = {}
                slots = []
                if cat_slots_focus_dialog:
                    if pivot_task != "cat_slots":
                        predicted_cat_slot_values = cat_slots_focus_dialog["turns"][turn_idx]["frames"][frame_idx]["state"]["slot_values"]
                    else:
                        predicted_cat_slot_values = pivot_state["slot_values"]

                    for slot in service_schema.categorical_slots:
                        if slot in predicted_cat_slot_values:
                            slot_values[slot] = predicted_cat_slot_values[slot]

                if noncat_slots_focus_dialog:
                    if pivot_task != "noncat_slots":
                        slots = noncat_slots_focus_dialog["turns"][turn_idx]["frames"][frame_idx]["slots"]
                        predicted_noncat_slot_values = noncat_slots_focus_dialog["turns"][turn_idx]["frames"][frame_idx]["state"]["slot_values"]
                    else:
                        slots = pivot_slots
                        predicted_noncat_slot_values = pivot_state["slot_values"]

                    for slot in service_schema.non_categorical_slots:
                        if slot in predicted_noncat_slot_values:
                            slot_values[slot] = predicted_noncat_slot_values[slot]

                frame["slots"] = slots
                state["slot_values"] = {s: v for s, v in slot_values.items()}
                # logger.info("dial_key:{}, slot_values:{}".format(dial_key, slot_values))
                frame["state"] = state
    return pivot_dialog


def get_indexed_dialog_dict(dialogues):
    dial_dict = {}
    for dial in dialogues:
        dial_dict[dial["dialogue_id"]] = dial
    return dial_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--active_intent_prediction_dir",
        default=None,
        type=str,
        required=False,
        help="Directory in which all JSON files combined are predictions of the"
        + "evaluation set on active_intent ask. We need to merge then evaluate these JSON files"
        + " by DSTC8 metrics.")

    parser.add_argument(
        "--requested_slots_prediction_dir",
        default=None,
        type=str,
        required=False,
        help="Directory in which all JSON files combined are predictions of the"
        + "evaluation set on requested_slots ask. We need to merge then evaluate these JSON files"
        + " by DSTC8 metrics.")

    parser.add_argument(
        "--cat_slots_prediction_dir",
        default=None,
        type=str,
        required=False,
        help="Directory in which all JSON files combined are predictions of the"
        + "evaluation set on cat_slots ask. We need to merge then evaluate these JSON files"
        + " by DSTC8 metrics.")

    parser.add_argument(
        "--noncat_slots_prediction_dir",
        default=None,
        type=str,
        required=False,
        help="Directory in which all JSON files combined are predictions of the"
        + "evaluation set on cat_slots ask. We need to merge then evaluate these JSON files"
        + " by DSTC8 metrics.")

    parser.add_argument(
        "--schema_file",
        default=None,
        type=str,
        required=True,
        help="schema files for reference")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Output Directory for writing out")

    args = parser.parse_args()
    if args.active_intent_prediction_dir:
        pivot_dir = args.active_intent_prediction_dir
        pivot_key = "active_intent"
    elif args.requested_slots_prediction_dir:
        pivot_dir = args.requested_slots_prediction_dir
        pivot_key = "requested_sots"
    elif args.cat_slots_prediction_dir:
        pivot_dir = args.cat_slots_prediction_dir
        pivot_key = "cat_slots"
    elif args.noncat_slots_prediction_dir:
        pivot_dir = args.noncat_slots_prediction_dir
        pivot_key = "noncat_slots"
    else:
        raise RuntimeError("All input dirs are empty")

    schemas = schema.Schema(args.schema_file)
    for input_file_path in glob.iglob(pivot_dir + '/dialogues_*.json', recursive=False):
        input_file_name = os.path.basename(input_file_path)
        logger.info("{}, {}".format(input_file_path, input_file_name))
        merge_dict = {}
        with open(input_file_path) as f:
            pivot_dialogs = json.load(f)
            merge_dict[pivot_key] = get_indexed_dialog_dict(pivot_dialogs)

        if args.active_intent_prediction_dir and pivot_key != "active_intent":
            with open(os.path.join(args.active_intent_prediction_dir, input_file_name)) as f:
                dialogs = json.load(f)
                merge_dict["active_intent"] = get_indexed_dialog_dict(dialogs)

        if args.requested_slots_prediction_dir and pivot_key != "requested_slots":
            with open(os.path.join(args.requested_slots_prediction_dir, input_file_name)) as f:
                dialogs = json.load(f)
                merge_dict["requested_slots"] = get_indexed_dialog_dict(dialogs)

        if args.cat_slots_prediction_dir and pivot_key != "cat_slots":
            with open(os.path.join(args.cat_slots_prediction_dir, input_file_name)) as f:
                dialogs = json.load(f)
                merge_dict["cat_slots"] = get_indexed_dialog_dict(dialogs)

        if args.noncat_slots_prediction_dir and pivot_key != "noncat_slots":
            with open(os.path.join(args.noncat_slots_prediction_dir, input_file_name)) as f:
                dialogs = json.load(f)
                merge_dict["noncat_slots"] = get_indexed_dialog_dict(dialogs)

        merged_dialogs = []
        for dial in pivot_dialogs:
            local_merge_dict = {}
            for key, values in merge_dict.items():
                local_merge_dict[key] = values.get(dial["dialogue_id"], None)

            merged_dialogs.append(merge_predicted_dialog(local_merge_dict, schemas))

        output_file_path = os.path.join(args.output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(
                merged_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)

if __name__ == "__main__":
    main()
