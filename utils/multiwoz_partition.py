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
import random
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
def process_folder(input_folder, keep_domains, keep_domains1, keep_domains2, output_folder, heldout=[]):

    for input_schema_path in glob.iglob(input_folder + '/schema.json*', recursive=False):
        input_schema_name = os.path.basename(input_schema_path)
        output_schema_path = os.path.join(output_folder, input_schema_name)
        logger.info("processing {} for  {}, {}".format(keep_domains, input_schema_path, output_schema_path))
        schema = Schema(input_schema_path)
        new_schemas = []
        for s in schema._schemas:
            # only consider good domains
            if s["service_name"] not in keep_domains:
                logger.warning("{} is not in keep_domains".format(s["service_name"]))
                continue
            else:
                new_schemas.append(s)

        with open(output_schema_path, "w") as f:
            json.dump(new_schemas, f, indent=2)
        logger.info("schema {} processed done, new_schemas = {}".format(output_schema_path, [s["service_name"] for s in new_schemas]))

    all_new_dialogs = []
    all_keep_dialogs1 = []
    all_keep_dialogs2 = []
    all_heldout = []
    for input_dialog_path in glob.iglob(input_folder + '/dialogues_*.json', recursive=False):
        input_dialog_name = os.path.basename(input_dialog_path)
        output_dialog_path = os.path.join(output_folder, input_dialog_name)
        logger.info("processing {} for {}, {}".format(keep_domains, input_dialog_path, output_dialog_path))
        with open(input_dialog_path) as f:
            dialogs = json.load(f)
        new_dialogs = []
        keep_dialogs1 = []
        keep_dialogs2 = []
        for i, dialog in enumerate(dialogs):
            if all([s not in keep_domains for s in dialog["services"]]):
                # we are assuming that dev and test always contains service in train
                # 1) if all the service not in keep, remove it from train
                # 1.1) if all the service only in dev_keep, save to dev
                service_in_keep1 = all([s in keep_domains1 for s in dialog["services"]]) if keep_domains1 else False
                service_in_keep2 = all([s in keep_domains2 for s in dialog["services"]]) if keep_domains2 else False
                if service_in_keep2 and service_in_keep1:
                    # randomly add to keep1 or keep2
                    if random.random() > 0.5:
                        for j, turn in enumerate(dialog["turns"]):
                            # only keep those service in keep_domains
                            dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in keep_domains1]
                        keep_dialogs1.append(dialogs[i])
                    else:
                        for j, turn in enumerate(dialog["turns"]):
                            # only keep those service in keep_domains
                            dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in keep_domains2]
                        keep_dialogs2.append(dialogs[i])
                elif service_in_keep1:
                    for j, turn in enumerate(dialog["turns"]):
                        # only keep those service in keep_domains
                        dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in keep_domains1]
                    keep_dialogs1.append(dialogs[i])
                elif service_in_keep2:
                    for j, turn in enumerate(dialog["turns"]):
                        # only keep those service in keep_domains
                        dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in keep_domains2]
                    keep_dialogs2.append(dialogs[i])
                #1.2) if all the service only in test_keep, save to test
                #1.3) if in both dev and test, random 1/2 in test and dev
            elif all([s in keep_domains for s in dialog["services"]]):
                # 2) if all the service in keep, doing nothing, just add the dialog
                for j, turn in enumerate(dialog["turns"]):
                    # only keep those service in keep_domains
                    dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in keep_domains]
                new_dialogs.append(dialogs[i])
            else:
                # for partial cases
                # remove all the frames that related to except_domain, only keep the domain in keep_domains
                if any([s in heldout for s in dialogs[i]["services"]]):
                    # if containing some special heldout domains:
                    all_heldout.append(dialogs[i])
                else:
                    # remove all the frames that related to except_domain, only keep the domain in keep_domains
                    dialogs[i]["services"] = [s for s in dialogs[i]["services"] if s in keep_domains]
                    for j, turn in enumerate(dialog["turns"]):
                        # only keep those service in keep_domains
                        dialogs[i]["turns"][j]["frames"] = [f for f in dialogs[i]["turns"][j]["frames"] if f["service"] in keep_domains]
                    new_dialogs.append(dialogs[i])

        all_new_dialogs.extend(new_dialogs)
        with open(output_dialog_path, "w") as f:
            json.dump(new_dialogs, f, indent=2)

        if keep_dialogs1:
            all_keep_dialogs1.extend(keep_dialogs1)
        if keep_dialogs2:
            all_keep_dialogs2.extend(keep_dialogs2)

    domain_counter, turn_counter, domain_combination_counter = stat_dialogs(all_new_dialogs)
    logger.info("{} processed new_dialogs done, domain_counter:{}, turn_counter:{}".format(input_folder, domain_counter, turn_counter))
    logger.info("{} processed new_dialogs done, domain_combination_counter:{}".format(input_folder, domain_combination_counter))
    if all_keep_dialogs1:
        keep1_output_dialog_path = os.path.join(output_folder, "to_move.keep1")
        with open(keep1_output_dialog_path, "w") as f:
            json.dump(all_keep_dialogs1, f, indent=2)
        domain_counter, turn_counter, domain_combination_counter = stat_dialogs(all_keep_dialogs1)
        logger.info("{} processed keep1 done, domain_counter:{}, turn_counter:{}".format(input_folder, domain_counter, turn_counter))
        logger.info("{} processed keep1 done, domain_combination_counter:{}".format(input_folder, domain_combination_counter))
    if all_keep_dialogs2:
        keep2_output_dialog_path = os.path.join(output_folder, "to_move.keep2")
        with open(keep2_output_dialog_path, "w") as f:
            json.dump(all_keep_dialogs2, f, indent=2)
        domain_counter, turn_counter, domain_combination_counter = stat_dialogs(all_keep_dialogs2)
        logger.info("{} processed keep2 done, domain_counter:{}, turn_counter:{}".format(input_folder, domain_counter, turn_counter))
        logger.info("{} processed keep2 done, domain_combination_counter:{}".format(input_folder, domain_combination_counter))
    if all_heldout:
        heldout_output_dialog_path = os.path.join(output_folder, "to_move.heldout")
        with open(heldout_output_dialog_path, "w") as f:
            json.dump(all_heldout, f, indent=2)
        domain_counter, turn_counter, domain_combination_counter = stat_dialogs(all_heldout)
        logger.info("{} processed heldout done, domain_counter:{}, turn_counter:{}".format(input_folder, domain_counter, turn_counter))
        logger.info("{} processed heldout done, domain_combination_counter:{}".format(input_folder, domain_combination_counter))




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

    args = parser.parse_args()
    logger.info("args:{}".format(args))
    all_domains = ['attraction', 'hotel', 'restaurant', 'taxi', 'train', 'hospital', 'bus', 'police']
    train_keep_domains = ['restaurant', 'attraction', 'train']
    special_heldout = ['bus']
    dev_keep_domains = ['restaurant', 'attraction', 'train', 'hotel', 'taxi']
    test_keep_domains = ['restaurant', 'attraction', 'train', 'hotel', 'taxi', 'hospital', 'police', 'bus']
    # good_domains = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
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

    process_folder(train_folder, train_keep_domains, dev_keep_domains, test_keep_domains, output_train_folder, heldout=special_heldout)
    process_folder(dev_folder, dev_keep_domains, None, test_keep_domains, output_dev_folder)
    process_folder(test_folder, test_keep_domains, None, None, output_test_folder)

if __name__ == "__main__":
    main()
