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

"""Wrappers for schemas of different services."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import argparse
import os
import glob
from utils.schema import Schema

logger = logging.getLogger(__name__)


"""
Assuming that in the input folder, there schema.json file and dialogs
from https://github.com/jasonwu0731/trade-dst/blob/e6dce870361517a8b7c2865991e4a2303521c6b5/utils/utils_multiWOZ_DST.py#L245
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]):
                continue

How to hand multiple domain dialogue:
for train and dev:
   1) the dialogs containing only except domain will be removed, both in schema and dialog
   2) For the dialog containing mulitple domain, those frames from except domain will be removed, schema for except domain removed
for test:
   1) if except domain no in the dialog, skip it.
   2) only consider the except domain, schema contains only for except domain, others will not be considered
"""
def process_folder(input_folder, except_domain, output_folder, good_domains, mode):
    if except_domain is None and good_domains is None:
        return

    for input_schema_path in glob.iglob(input_folder + '/schema.json*', recursive=False):
        input_schema_name = os.path.basename(input_schema_path)
        output_schema_path = os.path.join(output_folder, input_schema_name)
        logger.info("processing {} for  {}, {}".format(except_domain, input_schema_path, output_schema_path))
        schema = Schema(input_schema_path)
        new_schemas = []
        for s in schema._schemas:
            # only consider good domains
            if s["service_name"] not in good_domains:
                logger.warning("{} is not in good_domains".format(s["service_name"]))
                continue
            # for train and dev, remove except domain
            if mode in ["train", "dev"]:
                if s["service_name"] == except_domain:
                    continue
                else:
                    new_schemas.append(s)
            else:
                # test, only keep the except domain
                if except_domain:
                    if s["service_name"] == except_domain:
                        new_schemas.append(s)
                    else:
                        continue
                else:
                    # no except domain, we add all for test
                    new_schemas.append(s)

        # save new_schemas
        with open(output_schema_path, "w") as f:
            json.dump(new_schemas, f, indent=2)
        logger.info("schema {} processed done, new_schemas = {}".format(output_schema_path, [s["service_name"] for s in new_schemas]))

    domain_counter = {}
    domain_combination_counter = {}
    turn_counter = {}
    all_new_dialogs = []
    for input_dialog_path in glob.iglob(input_folder + '/dialogues_*.json', recursive=False):
        input_dialog_name = os.path.basename(input_dialog_path)
        output_dialog_path = os.path.join(output_folder, input_dialog_name)
        logger.info("processing {} for {}, {}".format(except_domain, input_dialog_path, output_dialog_path))
        with open(input_dialog_path) as f:
            dialogs = json.load(f)
        new_dialogs = []
        skip_dialogs = []
        for i, dialog in enumerate(dialogs):
            # try to assign th services
            possible_services = set()
            for j, turn in enumerate(dialog["turns"]):
                for f, frame in enumerate(turn["frames"]):
                    actions_flag = (len(frame["actions"]) != 0)
                    slots_flag = (len(frame["slots"]) != 0)
                    if "state" in frame:
                        intent_flag = (frame["state"]["active_intent"] != 'NONE')
                        requested_flag = (len(frame["state"]["requested_slots"]) != 0)
                        slot_values_flag = (len(frame["state"]["slot_values"]) != 0)
                    else:
                        intent_flag = False
                        requested_flag = False
                        slot_values_flag = False

                    if actions_flag or slots_flag or intent_flag or requested_flag or slot_values_flag:
                        possible_services.add(frame["service"])
            ori_set = set(dialog["services"])
            if ori_set != possible_services:
                logger.warning("{}, {}, {}, {} different services: ori : {}, new : {}".format(input_dialog_name, i, dialog["dialogue_id"], dialog["services"], ori_set, possible_services))
                dialogs[i]["services"] = list(possible_services)

            dialogs[i]["services"] = sorted(dialogs[i]["services"])
            # if all service are not in good domains, remove it
            if all([s not in good_domains for s in dialog["services"]]):
                logger.warning("{}, {}, not in good  domain, {}, {}".format(input_dialog_name, i, dialog["dialogue_id"], dialog["services"]))
                skip_dialogs.append(dialog)
                continue
            # train and dev,
            if mode in ["train", "dev"]:
                if except_domain:
                    if dialog["services"] == [except_domain]:
                        # if only except domain for dialog, skip it
                        skip_dialogs.append(dialog)
                        continue
                    else:
                        # remove all the frames that related to except_domain
                        if except_domain in dialogs[i]["services"]:
                            dialogs[i]["services"].remove(except_domain)
                        for j, turn in enumerate(dialog["turns"]):
                            dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] != except_domain and f["service"] in good_domains]
                        new_dialogs.append(dialogs[i])
                else:
                    # remove all service not in good
                    dialogs[i]["services"] = [s for s in dialogs[i]["services"] if s in good_domains]
                    for j, turn in enumerate(dialog["turns"]):
                        dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in good_domains]
                    new_dialogs.append(dialogs[i])

            else:
                if except_domain:
                    if except_domain not in dialog["services"]:
                        continue
                    else:
                        dialogs[i]["services"] = [except_domain]
                        for j, turn in enumerate(dialog["turns"]):
                            dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] == except_domain]
                        new_dialogs.append(dialogs[i])
                else:
                    # no except domain, we just add all good domains
                    for j, turn in enumerate(dialog["turns"]):
                        dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in good_domains]
                    new_dialogs.append(dialogs[i])

        # save new_schemas
        all_new_dialogs.extend(new_dialogs)
        with open(output_dialog_path, "w") as f:
            json.dump(new_dialogs, f, indent=2)

        logger.info("dialog {} processed done, old_dialogs = {}, new_dialogs = {}, skip_dialogs = {}".format(output_dialog_path, len(dialogs), len(new_dialogs), len(skip_dialogs)))
    domain_counter, turn_counter, domain_combination_counter = stat_dialogs(all_new_dialogs)
    logger.info("{} processed new_dialogs done, domain_counter:{}, turn_counter:{}".format(input_folder, domain_counter, turn_counter))
    logger.info("{} processed new_dialogs done, domain_combination_counter:{}".format(input_folder, domain_combination_counter))

def stat_dialogs(dialogs):
    domain_counter = {}
    domain_combination_counter = {}
    turn_counter = {}
    for i, dialog in enumerate(dialogs):
        dialogs[i]["services"] = sorted(dialogs[i]["services"])
        joint_combines = ",".join(dialogs[i]["services"])
        if joint_combines not in domain_combination_counter:
            domain_combination_counter[joint_combines] = 1
        else:
            domain_combination_counter[joint_combines] += 1
        for s in dialogs[i]["services"]:
            if s not in domain_counter:
                domain_counter[s] = 1
            else:
                domain_counter[s] += 1
            for j, turn in enumerate(dialog["turns"]):
                for f, frame in enumerate(dialogs[i]["turns"][j]["frames"]):
                    actions_flag = (len(frame["actions"]) != 0)
                    slots_flag = (len(frame["slots"]) != 0)
                    if "state" in frame:
                        intent_flag = (frame["state"]["active_intent"] != 'NONE')
                        requested_flag = (len(frame["state"]["requested_slots"]) != 0)
                        slot_values_flag = (len(frame["state"]["slot_values"]) != 0)
                    else:
                        intent_flag = False
                        requested_flag = False
                        slot_values_flag = False

                    if (actions_flag or slots_flag or intent_flag or requested_flag or slot_values_flag):
                        if frame["service"] not in turn_counter:
                            turn_counter[frame["service"]] = 1
                        else:
                            turn_counter[frame["service"]] += 1
    return domain_counter, turn_counter, domain_combination_counter

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        required=True,
        help="the path of schema json file")

    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        required=True,
        help="the output_folder")

    parser.add_argument(
        "--heldout_domain",
        default=None,
        type=str,
        required=False,
        choices=["attraction", "hotel", "restaurant", "taxi", "train"],
        help="for evaluation.")


    args = parser.parse_args()
    logger.info("args:{}".format(args))
    all_domains = ['attraction', 'hotel', 'restaurant', 'taxi', 'train', 'hospital', 'bus', 'police']
    good_domains = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
    train_folder = os.path.join(args.folder, "train")
    dev_folder = os.path.join(args.folder, "dev")
    test_folder = os.path.join(args.folder, "test")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    output_train_folder = os.path.join(args.output_folder, "train")
    output_dev_folder = os.path.join(args.output_folder, "dev")
    output_test_folder = os.path.join(args.output_folder, "test")
    if not os.path.exists(output_train_folder):
        os.makedirs(output_train_folder)

    if not os.path.exists(output_dev_folder):
        os.makedirs(output_dev_folder)

    if not os.path.exists(output_test_folder):
        os.makedirs(output_test_folder)

    process_folder(train_folder, args.heldout_domain, output_train_folder, all_domains, mode="train")
    process_folder(dev_folder, args.heldout_domain, output_dev_folder, all_domains, mode="dev")
    process_folder(test_folder, args.heldout_domain, output_test_folder, all_domains, mode="test")

if __name__ == "__main__":
    main()
