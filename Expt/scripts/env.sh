#!/bin/bash

eval "$(pyenv init -)"
pyenv activate py-sgd
GOOGLE_ROOT_DIR=$CODE_BASE/google-research/
BASE_EXPT_DIR=$EXPT_BASE/schema-guided/
BASE_DATA_DIR=$BASE_EXPT_DIR/data/
DSTC8_DATA_DIR=$BASE_EXPT_DIR/data/dstc8/
MULTIWOZ21_DATA_DIR=$BASE_EXPT_DIR/data/MultiWOZ_2.1
CONV_MULTIWOZ21_DATA_DIR=$BASE_EXPT_DIR/data/MultiWOZ_2.1_converted
BASE_WORK_DIR=$BASE_EXPT_DIR/workdirs_google/
bert_ckpt_dir=$BASE_EXPT_DIR/bert_model_dir/
dialogues_example_dir=$BASE_EXPT_DIR/dialogues_example_dir_google/
dstc8_single_schema_embedding_dir=$BASE_EXPT_DIR/schema_embedding_dir_google/dstc8_single/
dstc8_all_schema_embedding_dir=$BASE_EXPT_DIR/schema_embedding_dir_google/dstc8_all/
multiwoz_schema_embedding_dir=$BASE_EXPT_DIR/schema_embedding_dir_google/multiwoz21_all/
