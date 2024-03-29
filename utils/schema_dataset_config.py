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

"""Code for configuring the model depending on the dataset."""
import collections

# Config object that contains the following info:
# file_ranges: the file ranges of train, dev, and test set.
# max_num_cat_slot: Maximum allowed number of categorical trackable slots for a
# service.
# max_num_noncat_slot: Maximum allowed number of non-categorical trackable slots
# for a service.
# max_num_value_per_cat_slot: Maximum allowed number of values per categorical
# trackable slot.
# max_num_intent: Maximum allowed number of intents for a service.
DatasetConfig = collections.namedtuple("DatasetConfig", [
    "file_ranges", "max_num_cat_slot", "max_num_noncat_slot",
    "max_num_value_per_cat_slot", "max_num_intent", "name", "schema_file"
])

DATASET_CONFIG = {
    "dstc8_sample":
        DatasetConfig(
            file_ranges={
                "train": range(1, 2),
                "dev": range(1, 2),
                "test": range(1, 2)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_sample",
            schema_file="schema.json"
        ),
    "dstc8_single_domain":
        DatasetConfig(
            file_ranges={
                "train": range(1, 44),
                "dev": range(1, 8),
                "test": range(1, 12)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_single_domain",
            schema_file="schema.json"
        ),
    "dstc8_multi_domain":
        DatasetConfig(
            file_ranges={
                "train": range(44, 128),
                "dev": range(8, 21),
                "test": range(12, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_multi_domain",
            schema_file="schema.json"
        ),
    "dstc8_all":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_all",
            schema_file="schema.json"
        ),
    "dstc8_question_nameonly":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_question_nameonly",
            schema_file="schema.json.question_nameonly"
        ),
       "dstc8_name_para":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_name_para",
            schema_file="schema.json.name_para"
        ),
     "dstc8_question_rich":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_question_rich",
            schema_file="schema.json.question_rich"
        ),
    "dstc8_empty":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_empty",
            schema_file="schema.json.empty"
        ),
    "dstc8_index_name":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_index_name",
            schema_file="schema.json.index_name"
        ),
    "dstc8_name_only":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_name_only",
            schema_file="schema.json.name_only"
        ),
    "dstc8_back_translation":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_back_translation",
            schema_file="schema.json.back_translation"
        ),
    "dstc8_question_template":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_question_template",
            schema_file="schema.json.question_template"
        ),
    "dstc8_enrich":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_entich",
            schema_file="schema.json.enrich"
        ),
    "dstc8_nameonly":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_nameonly",
            schema_file="schema.json.nameonly"
        ),
    "dstc8_empty_service_name_only":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_empty_service_name_only",
            schema_file="schema.json.empty_service_name_only"
        ),
    "dstc8_empty_service_desc":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_empty_service_desc",
            schema_file="schema.json.empty_service_desc"
        ),
    "dstc8_servicename_as_desc":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4,
            name="dstc8_servicename_as_desc",
            schema_file="schema.json.servicename_as_desc"
        ),
    "multiwoz21_all":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1,
            name="multiwoz21_all",
            schema_file="schema.json"
        ),
    "multiwoz21_index_name":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1,
            name="multiwoz21_index_name",
            schema_file="schema.json.index_name"
        ),
    "multiwoz21_question_nameonly":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1,
            name="multiwoz21_question_nameonly",
            schema_file="schema.json.question_nameonly"
        ),
    "multiwoz21_name_only":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1,
            name="multiwoz21_name_only",
            schema_file="schema.json.name_only"
        ),
    "multiwoz21_back_translation":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1,
            name="multiwoz21_back_translation",
            schema_file="schema.json.back_translation"
        ),
    # service:hotel, max_num_cat_slot:9, max_num_noncat_slot:1, max_num_value_per_cat_slot:8, max_num_intent:2
    # service:train, max_num_cat_slot:4, max_num_noncat_slot:2, max_num_value_per_cat_slot:13, max_num_intent:2
    # service:attraction, max_num_cat_slot:2, max_num_noncat_slot:1, max_num_value_per_cat_slot:12, max_num_intent:1
    # service:restaurant, max_num_cat_slot:4, max_num_noncat_slot:3, max_num_value_per_cat_slot:8, max_num_intent:2
    # service:hospital, max_num_cat_slot:0, max_num_noncat_slot:1, max_num_value_per_cat_slot:0, max_num_intent:1
    # service:taxi, max_num_cat_slot:0, max_num_noncat_slot:4, max_num_value_per_cat_slot:0, max_num_intent:1
    # service:bus, max_num_cat_slot:1, max_num_noncat_slot:3, max_num_value_per_cat_slot:1, max_num_intent:1
    # service:police, max_num_cat_slot:1, max_num_noncat_slot:0, max_num_value_per_cat_slot:1, max_num_intent:1
    # max_num_cat_slot:9, max_num_noncat_slot:4, max_num_value_per_cat_slot:13, max_num_intent:2
    "multiwoz22_all":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_all",
            schema_file="schema.json"
        ),
    "multiwoz22_good":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_good",
            schema_file="schema.json"
        ),
    "multiwoz22_zero_sample":
        DatasetConfig(
            file_ranges={
                "train": range(1, 2),
                "dev": range(1, 2),
                "test": range(1, 2)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_sample",
            schema_file="schema.json"
        ),
    "multiwoz22_zero":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero",
            schema_file="schema.json"
        ),
      "multiwoz22_zero_name_only":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_name_only",
            schema_file="schema.json.name_only"
        ),
    "multiwoz22_zero_index_name":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_index_name",
            schema_file="schema.json.index_name"
        ),
    "multiwoz22_zero_question_nameonly":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_question_nameonly",
            schema_file="schema.json.question_nameonly"
        ),
    "multiwoz22_zero_name_para":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_name_para",
            schema_file="schema.json.name_para"
        ),
    "multiwoz22_zero_question_rich":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_question_rich",
            schema_file="schema.json.question_rich"
        ),
    "multiwoz22_zero_back_translation":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_back_translation",
            schema_file="schema.json.back_translation"
        ),
    "multiwoz22_zero_defintion":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 4),
                "test": range(1, 5)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=7,
            max_num_value_per_cat_slot=13,
            max_num_intent=2,
            name="multiwoz22_zero_definition",
            schema_file="schema.json.definition"
        ),
}
