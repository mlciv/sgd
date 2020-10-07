#!/bin/bash

eval "$(pyenv init -)"
pyenv activate py-sgd
GOOGLE_ROOT_DIR=$CODE_BASE/google-research/
SGD_CODE_DIR=$CODE_BASE/sgd/
SGD_SCRIPTS_DIR=$SGD_CODE_DIR/Expt/sgd-scripts/
SGD_COMMAND_DIR=$SGD_CODE_DIR/Expt/sgd-scripts/commands/
SGD_CONFIG_DIR=$SGD_CODE_DIR/Expt/sgd-scripts/config/
SGD_JSON_CONFIG_DIR=$SGD_CODE_DIR/Expt/sgd-scripts/json_configs/
SGD_EXPT_DIR=$EXPT_BASE/schema-guided/
SGD_CACHE_DIR=$EXPT_BASE/schema-guided/cache_dir/
SGD_DATA_DIR=$SGD_EXPT_DIR/data/
DSTC8_DATA_DIR=$SGD_DATA_DIR/dstc8/
DSTC8_INDEX_NAME_DATA_DIR=$SGD_DATA_DIR/dstc8_index_name/
MULTIWOZ21_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.1
CONV_MULTIWOZ21_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.1_converted
MULTIWOZ22_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2
MULTIWOZ22_GOOD_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2_good
MULTIWOZ22_ATTRACTION_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2_attaction
MULTIWOZ22_hotel_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2_hotel
MULTIWOZ22_restaurant_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2_restaurant
MULTIWOZ22_taxi_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2_taxi
MULTIWOZ22_train_DATA_DIR=$SGD_DATA_DIR/MultiWOZ_2.2_train

export PYTHONPATH=$SGD_CODE_DIR:$PYTHONPATH

SGD_WORK_DIR=$SGD_EXPT_DIR/workdirs/
dialogues_example_dir=$SGD_EXPT_DIR/dialogues_example_dir/
dstc8_single_schema_embedding_dir=$SGD_EXPT_DIR/schema_embedding_dir/dstc8_single/
dstc8_all_schema_embedding_dir=$SGD_EXPT_DIR/schema_embedding_dir/dstc8_all/
multiwoz_schema_embedding_dir=$SGD_EXPT_DIR/schema_embedding_dir/multiwoz20_all/
