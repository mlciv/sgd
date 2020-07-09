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