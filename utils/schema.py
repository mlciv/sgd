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

logger = logging.getLogger(__name__)

class ServiceSchema(object):
    """
    A wrapper for schema for a service, which is loaded from a schema decription file
    """
    def __init__(self, schema_json, service_id=None):
        self._service_name = schema_json["service_name"]
        self._description = schema_json["description"]
        self._schema_json = schema_json
        self._service_id = service_id

        # Construct the vocabulary for intents, slots, categorical slots,
        # non-categorical slots and categorical slot values. These vocabs are used
        # for generating indices for their embedding matrix.
        # all the intent names
        self._intents = sorted(i["name"] for i in schema_json["intents"])
        # all the slots names
        self._slots = sorted(s["name"] for s in schema_json["slots"])
        # all the categorical slots in the schemas
        self._all_categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if s["is_categorical"])

        # all the categorical slots permitted in the slots
        self._categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]

            if s["is_categorical"] and s["name"] in self.state_slots)

        self._boolean_categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if self.is_boolean(s)
        )

        # all the non categorical slots in the schema file
        self._all_non_categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if not s["is_categorical"])

        # all the non_categorical slots permitted in the state
        self._non_categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if not s["is_categorical"] and s["name"] in self.state_slots)

        slot_schemas = {s["name"]: s for s in schema_json["slots"]}
        categorical_slot_values = {}
        categorical_slot_value_ids = {}
        for slot in self._categorical_slots:
            slot_schema = slot_schemas[slot]
            values = sorted(slot_schema["possible_values"])
            categorical_slot_values[slot] = values
            value_ids = {value: idx for idx, value in enumerate(values)}
            categorical_slot_value_ids[slot] = value_ids
        self._categorical_slot_values = categorical_slot_values
        self._categorical_slot_value_ids = categorical_slot_value_ids

        all_categorical_slot_values = {}
        all_categorical_slot_value_ids = {}
        for slot in self._all_categorical_slots:
            slot_schema = slot_schemas[slot]
            values = sorted(slot_schema["possible_values"])
            all_categorical_slot_values[slot] = values
            value_ids = {value: idx for idx, value in enumerate(values)}
            all_categorical_slot_value_ids[slot] = value_ids
        self._all_categorical_slot_values = all_categorical_slot_values
        self._all_categorical_slot_value_ids = all_categorical_slot_value_ids

    def is_boolean(self, slot):
        """
        check whether the slot is boolean slot
        """
        if slot in self._all_categorical_slots:
            values = self.get_all_categorical_slot_values(slot)
            # not sure the order
            if len(values) == 2 and "True" in values and "False" in values:
                return True
            else:
                return False
        else:
            return False

    def is_numeric(self, slot):
        """
        check whether the slot is numeric slot
        """
        if slot in self._all_categorical_slots:
            values = self.get_all_categorical_slot_values(slot)
            return all([value.isdigit() for value in values])
        return False

    def is_other_cat(self, slot):
        """
        neight numeric and boolean slot
        """
        return not (self.is_numeric(slot) or self.is_boolean(slot))

    @property
    def schema_json(self):
        return self._schema_json

    @property
    def state_slots(self):
        """
        Set of slots which are permitted to be in the dialogue state.
        """
        state_slots = set()
        for intent in self._schema_json["intents"]:
            state_slots.update(intent["required_slots"])
            state_slots.update(intent["optional_slots"])
        return state_slots

    @property
    def service_name(self):
        return self._service_name

    @property
    def service_id(self):
        """
        service id
        """
        return self._service_id

    @property
    def description(self):
        """
        The sercice description
        """
        return self._description

    @property
    def slots(self):
        """
        all slot names
        """
        return self._slots

    @property
    def intents(self):
        """
        all intent names
        """
        return self._intents

    @property
    def categorical_slots(self):
        """
        categorical slots permitted in the state in optional and required slots
        """
        return self._categorical_slots

    @property
    def all_categorical_slots(self):
        """
        all categorical slots defined in the schema, may or maynot in the state
        """
        return self._all_categorical_slots

    @property
    def non_categorical_slots(self):
        """
        non-categorical slots permitted in the state
        """
        return self._non_categorical_slots

    @property
    def all_non_categorical_slots(self):
        """
        all non-categorical slots defined in the schmea, may or maynot in the state
        """
        return self._all_non_categorical_slots

    def get_categorical_slot_values(self, slot):
        """
        use slot name to get the corresponding slot
        """
        return self._categorical_slot_values[slot]

    def get_all_categorical_slot_values(self, slot):
        """
        use slot name to get the corresponding slot
        """
        return self._all_categorical_slot_values[slot]

    def get_slot_from_id(self, slot_id):
        return self._slots[slot_id]

    def get_intent_from_id(self, intent_id):
        return self._intents[intent_id]

    def get_categorical_slot_from_id(self, slot_id):
        return self._categorical_slots[slot_id]

    def get_non_categorical_slot_from_id(self, slot_id):
        return self._non_categorical_slots[slot_id]

    def get_categorical_slot_value_from_id(self, slot_id, value_id):
        slot = self.categorical_slots[slot_id]
        return self._categorical_slot_values[slot][value_id]

    def get_categorical_slot_value_id(self, slot, value):
        """
        use slot name and value to get the corresponding value id
        """
        return self._all_categorical_slot_value_ids[slot][value]


class Schema(object):
    """
    Wrapper for schemas for all services in a dataset.
    """
    def __init__(self, schema_json_path):
        """
        Load the schema from the json file.
        """
        self.schema_json_path = schema_json_path
        with open(schema_json_path) as f:
            schemas = json.load(f)
        self._services = sorted(schema["service_name"] for schema in schemas)
        # the vocabulary for all the services, the service name: id
        self._services_vocab = {v: k for k, v in enumerate(self._services)}
        service_schemas = {}
        for schema in schemas:
            service = schema["service_name"]
            # construct ServiceSchema object
            service_schemas[service] = ServiceSchema(
                schema, service_id=self.get_service_id(service))
        self._service_schemas = service_schemas
        self._schemas = schemas

    def get_service_id(self, service):
        """
        get service id by name
        """
        return self._services_vocab[service]

    def get_service_from_id(self, service_id):
        """
        get service by id
        """
        return self._services[service_id]

    def get_service_schema(self, service):
        """
        get the corresponding ServiceSchema object by service name
        """
        return self._service_schemas[service]

    @property
    def services(self):
        """
        get all the service names
        """
        return self._services

    def save_to_file(self, file_path):
        """
        save the whole schema object into a json file
        """
        with open(file_path, "w") as f:
            json.dump(self._schemas, f, indent=2)

    def gen_servicename_empty_desc(self):
        for schema in self._schemas:
            schema["description"] = schema["service_name"]
            for slot in schema["slots"]:
                slot['description'] = ""
            for intent in schema["intents"]:
                intent['description'] = ""

        self.save_to_file(self.schema_json_path + ".servicename_empty_desc")

    def gen_empty_description(self):
        for schema in self._schemas:
            for slot in schema["slots"]:
                slot['description'] = ""
            for intent in schema["intents"]:
                intent['description'] = ""

        self.save_to_file(self.schema_json_path + ".empty")

    def gen_enrich(self):
        for schema in self._schemas:
            # No change to service description
            # schema["description"] = schema["service_name"]
            for slot in schema["slots"]:
                slot['ori_description'] = slot['description']
                slot['description'] = slot["ori_description"]
            for intent in schema["intents"]:
                intent['ori_description'] = intent['description']
                intent['description'] = intent["ori_description"]

        self.save_to_file(self.schema_json_path + ".enrich")

    def gen_index_name(self, processed_schema_file=None):
        foldername = os.path.dirname(self.schema_json_path)
        processed_schemas = {}
        all_intent_names = set()
        all_slot_names = set()
        if processed_schema_file:
            # matching with processed_schema
            for f in processed_schema_file:
                if len(f) == 0 or not f[0]:
                    continue
                logger.info("loading {}".format(f[0]))
                schema_obj = Schema(f[0])
                for _, s in schema_obj._service_schemas.items():
                    for intent in s.intents:
                        all_intent_names.add(intent)
                    for slot in s.slots:
                        all_slot_names.add(slot)
                processed_schemas.update(schema_obj._service_schemas)

        j = len(all_intent_names)
        i = len(all_slot_names)
        logger.info("all_intent_names:{}".format(sorted(all_intent_names, key=lambda x: int(x.split("_")[1]))))
        logger.info("all_slot_names:{}".format(sorted(all_slot_names, key=lambda x: int(x.split("_")[1]))))
        logger.info("initial_intent_index={}, initial_slot_index={}".format(j,i))
        intent_name_mappings = {}
        slot_name_mappings = {}
        for index, schema in enumerate(self._schemas):
            if schema["service_name"] in processed_schemas:
                # service has been processed
                self._schemas[index] = processed_schemas[schema["service_name"]].schema_json
                # need to update the intent slot mappings
                for intent in self._schemas[index]["intents"]:
                    intent_name_mappings[(schema["service_name"], intent["ori_name"])] = intent["name"]

                for slot in self._schemas[index]["slots"]:
                    slot_name_mappings[(schema["service_name"], slot["ori_name"])] = slot["name"]

                continue

            schema["description"] = ""
            for slot in schema["slots"]:
                slot['description'] = ""
                slot["ori_name"] = slot["name"]
                slot["name"] = "slot_{}".format(i)
                slot_name_mappings[(schema["service_name"], slot["ori_name"])] = slot["name"]
                i = i + 1

            for intent in schema["intents"]:
                intent['description'] = ""
                intent["ori_name"] = intent["name"]
                intent["name"] = "intent_{}".format(j)
                intent_name_mappings[(schema["service_name"], intent["ori_name"])] = intent["name"]
                j = j + 1
                # update optional and required
                for s_idx, slot in enumerate(intent["required_slots"]):
                    intent["required_slots"][s_idx] = slot_name_mappings[(schema["service_name"], slot)]
                tmp_opt_dict = {}
                for slot_key, value in intent["optional_slots"].items():
                    new_key = slot_name_mappings[(schema["service_name"], slot_key)]
                    tmp_opt_dict[new_key] = value
                intent["optional_slots"] = tmp_opt_dict
                for s_idx, slot in enumerate(intent["result_slots"]):
                    intent["result_slots"][s_idx] = slot_name_mappings[(schema["service_name"], slot)]

        logger.info("final_intent_index={}, final_slot_index={}".format(j,i))
        self.save_to_file(self.schema_json_path + ".index_name")
        # change the dialogue files:
        dialogue_paths = [f for f in glob.glob(os.path.join(foldername, "dialogues_*.json"))]

        for dialog_json_filepath in sorted(dialogue_paths):
            new_dialogs = []
            with open(dialog_json_filepath) as f:
                ori_dialogs = json.load(f)
                for dialog in ori_dialogs:
                    new_dialogs.append(self.update_dialogue_with_name_mappings(dialog, intent_name_mappings, slot_name_mappings))
            with open(dialog_json_filepath + ".index_name", "w") as f:
                json.dump(
                    new_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)


    def update_dialogue_with_name_mappings(self, dialog, intent_name_mappings, slot_name_mappings):
        for turn_idx, turn in enumerate(dialog["turns"]):
            for f_idx, frame in enumerate(turn["frames"]):
                # actions
                for act_idx, act in enumerate(frame["actions"]):
                    if act["slot"] != "intent" and act["slot"]:
                        # We found that sometimes the system act use a different slot , which is not in the schema.
                        # Hence, be careful to use the system act
                        ori_slot = act["slot"]
                        act["slot"] = slot_name_mappings.get((frame["service"], ori_slot), ori_slot)
                        if act["slot"] == ori_slot:
                            logger.info("{} {} {} is not changed in actions  due to missing the mappings".format(dialog["dialogue_id"], turn_idx, ori_slot))
                    else:
                        # here we also replace the "canonical_values""
                        for v_i, intent in enumerate(act["canonical_values"]):
                            act["canonical_values"][v_i] = intent_name_mappings.get((frame['service'], intent), intent)
                            if act["canonical_values"][v_i] == intent:
                                logger.info("{} {} {} is not changed in actions-canonical_values due to missing the mappings".format(dialog["dialogue_id"], turn_idx, intent))

                        for v_i, intent in enumerate(act["values"]):
                            act["values"][v_i] = intent_name_mappings.get((frame['service'], intent), intent)
                            if act["values"][v_i] == intent:
                                logger.info("{} {} {} is not changed in actions-values due to missing the mappings".format(dialog["dialogue_id"], turn_idx, intent))
                    frame["actions"][act_idx] = act
                # slots
                for s_idx, slot in enumerate(frame["slots"]):
                    slot["slot"] = slot_name_mappings[(frame['service'], slot["slot"])]
                    frame["slots"][s_idx] = slot
                # state
                if "state" in frame:
                    # active_intent
                    if frame["state"]["active_intent"] != "NONE":
                        frame["state"]["active_intent"] = intent_name_mappings[(frame['service'], frame["state"]["active_intent"])]
                    # req_slots
                    for r_i, req_slot in enumerate(frame["state"]["requested_slots"]):
                        frame["state"]["requested_slots"][r_i] = slot_name_mappings[(frame['service'], req_slot)]
                    # slot_values
                    new_slot_values = {}
                    for slot_key, value in frame["state"]["slot_values"].items():
                        new_slot_key = slot_name_mappings[(frame['service'], slot_key)]
                        new_slot_values[new_slot_key] = value

                    frame["state"]["slot_values"] = new_slot_values
                # "service_call"
                if "service_call" in frame:
                    frame["service_call"]["method"] = intent_name_mappings[(frame['service'], frame["service_call"]["method"])]
                    new_paras = {}
                    for slot_key, value in frame["service_call"]["parameters"].items():
                        new_slot_key = slot_name_mappings[(frame['service'], slot_key)]
                        new_paras[new_slot_key] = value
                    frame["service_call"]["parameters"] = new_paras
                if "service_results" in frame:
                    for sr_i, sr in enumerate(frame["service_results"]):
                        new_sr = {}
                        for slot_key, value in sr.items():
                            new_slot_key = slot_name_mappings[(frame['service'], slot_key)]
                            new_sr[new_slot_key] = value
                        frame["service_results"][sr_i] = new_sr

                    frame["service_call"]["parameters"] = new_paras
                turn["frames"][f_idx] = frame
            dialog["turns"][turn_idx] = turn
        return dialog


    def gen_all_name_only_description(self):
        for schema in self._schemas:
            schema["descirption"] = schema["service_name"]
            for slot in schema["slots"]:
                slot['description'] = slot["name"]
            for intent in schema["intents"]:
                intent['description'] = intent["name"]

        self.save_to_file(self.schema_json_path + ".all_name_only")

    def gen_name_only_description(self):
        for schema in self._schemas:
            for slot in schema["slots"]:
                slot['description'] = slot["name"]
            for intent in schema["intents"]:
                intent['description'] = intent["name"]

        self.save_to_file(self.schema_json_path + ".name_only")

    def gen_name_only_change(self):
        for schema in self._schemas:
            schema["description"] = schema["service_name"]
            for slot in schema["slots"]:
                slot['description'] = slot["name"]
            for intent in schema["intents"]:
                intent['description'] = intent["name"]

        self.save_to_file(self.schema_json_path + ".name_change")

    def load_back_translation_file(self, back_translation_file):
        desc_mapping_dict = {}
        with open(back_translation_file, "r") as f:
            back_translations = json.load(f)
            for bt in back_translations:
                desc_mapping_dict[bt["source_line"][:-1]] = bt["back_translations"][0]["bt"][:-2]

        for schema in self._schemas:
            ori_service_desc = schema["description"]
            schema["description"] = desc_mapping_dict.get(ori_service_desc, "NOT_FOUND:" + ori_service_desc)
            schema["description_ori"] = ori_service_desc
            for slot in schema["slots"]:
                ori_slot_desc = slot['description']
                slot["description_ori"] = ori_slot_desc
                slot["description"] = desc_mapping_dict.get(ori_slot_desc, "NOT_FOUND" + ori_slot_desc)
            for intent in schema["intents"]:
                ori_intent_desc = intent['description']
                intent["description_ori"] = ori_intent_desc
                intent["description"] = desc_mapping_dict.get(ori_intent_desc, "NOT_FOUND" + ori_intent_desc)

        self.save_to_file(self.schema_json_path + ".bt")

    def gen_question_nameonly(self):
        slot_name = []
        intent_name = []
        for schema in self._schemas:
            # No change to service description
            # schema["description"] = schema["service_name"]
            for slot in schema["slots"]:
                slot['ori_description'] = slot['description']
                slot['description'] = "What is the value of {} ?".format(slot['name'])
                slot_name.append(slot['name'])
            for intent in schema["intents"]:
                intent['ori_description'] = intent['description']
                intent['description'] = "Is the user intending to {} ?".format(intent['name'])
                intent_name.append(intent['name'])
        self.save_to_file(self.schema_json_path + ".question_nameonly")

    def gen_question_template(self):
        slot_name = []
        intent_name = []
        for schema in self._schemas:
            # No change to service description
            # schema["description"] = schema["service_name"]
            for slot in schema["slots"]:
                slot['ori_description'] = slot['description']
                slot['description'] = slot["name"]
                slot_name.append(slot['name'])
            for intent in schema["intents"]:
                intent['ori_description'] = intent['description']
                intent['description'] = intent["name"]
                intent_name.append(intent['name'])

        with open(self.schema_json_path + ".slot_name", "w") as f:
            f.write("\n".join(slot_name))

        with open(self.schema_json_path + ".intent_name", "w") as f:
            f.write("\n".join(intent_name))

        self.save_to_file(self.schema_json_path + ".question_template")

    def gen_empty_service_desc(self):
        for schema in self._schemas:
            # No change to service description
            schema["description"] = ""
        self.save_to_file(self.schema_json_path + ".empty_service_desc")

    def gen_empty_service_name_only(self):
        for schema in self._schemas:
            # No change to service description
            schema["description"] = ""
            for slot in schema["slots"]:
                slot['description'] = ""
            for intent in schema["intents"]:
                intent['description'] = ""
        self.save_to_file(self.schema_json_path + ".empty_service_name_only")

    def gen_servicename_as_desc(self):
        for schema in self._schemas:
            # No change to service description
            schema["description"] = schema["service_name"]
        self.save_to_file(self.schema_json_path + ".servicename_as_desc")

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema_json_path",
        default=None,
        type=str,
        required=True,
        help="the path of schema json file")

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        choices=["empty", "name_only", "question_nameonly", "index_name", "question_template", "back_translation", "enrich", "empty_service_desc", "servicename_as_desc", "empty_service_name_only"],
        help="for evaluation.")

    parser.add_argument(
        "--processed_schema",
        nargs="*",
        action="append",
        default=None,
        type=str,
        required=False,
        help="processed_schema_files to avoid duplicate annotating, indexed by service name")

    args = parser.parse_args()
    logger.info("args:{}".format(args))
    schema = Schema(args.schema_json_path)
    if args.task_name == "empty":
        schema.gen_empty_description()
    elif args.task_name == "name_only":
        schema.gen_name_only_description()
    elif args.task_name == "all_name_only":
        schema.gen_all_name_only_description()
    elif args.task_name == "name_change":
        schema.gen_name_change_description()
    elif args.task_name == "back_translation":
        schema_description_back_translation_path = args.schema_json_path + ".ori_desc.cs.backtranslated.sorted"
        schema.load_back_translation_file(schema_description_back_translation_path)
    elif args.task_name == "question_template":
        schema.gen_question_template()
    elif args.task_name == "question_nameonly":
        schema.gen_question_nameonly()
    elif args.task_name == "enrich":
        schema.gen_enrich()
    elif args.task_name == "index_name":
        schema.gen_index_name(args.processed_schema)
    elif args.task_name == "empty_service_desc":
        schema.gen_empty_service_desc()
    elif args.task_name == "servicename_as_desc":
        schema.gen_servicename_as_desc()
    elif args.task_name == "servicename_empty_desc":
        schema.gen_servicename_empty_desc()
    elif args.task_name == "empty_service_name_only":
        schema.gen_empty_service_name_only()



if __name__ == "__main__":
    main()
