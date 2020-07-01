#!/bin/bash

# source envoronment variables
. ../../utils/env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$PS_ROOT_DIR/

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=$DIR/$1
fi

### configurate data directory
if [ ! -f ${config_data} ]; then
  echo "${config_data} doesn't exist"
  exit $?
else
  . ${config_data}
  echo "run ${config_data}"
fi

### CHECK WORK & DATA DIR
if [ -e ${EXP_DIR} ]; then
  today=`date +%m-%d.%H:%M`
  mv ${EXP_DIR} ${EXP_DIR%?}_${today}
  echo "rename original training folder to "${EXP_DIR%?}_${today}
fi

mkdir -p ${EXP_DIR}
mkdir -p ${EXP_MODELS}
mkdir -p ${EXP_SUMMARY}
mkdir -p ${EXP_RESULTS}


# training arguments
pargs="--do_train \
--do_test \
--evaluate_during_training \
--eval_all_checkpoints \
--model_type=$MODEL_TYPE \
--config_name=$CONFIG_NAME \
--model_name_or_path=$MODEL_NAME_PATH \
--preprocessor_name=$PREPROCESSOR_NAME \
--encoder_model_name_or_path=$ENCODER_MODEL_NAME_PATH \
--task_name=$TASK_NAME \
--train_file=$TRAIN_FILE \
--dev_file=$DEV_FILE \
--test_file=$TEST_FILE \
--per_gpu_eval_batch_size=$PER_GPU_EVAL_BATCH_SIZE \
--per_gpu_train_batch_size=$PER_GPU_TRAIN_BATCH_SIZE \
--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
--learning_rate=$LEARNING_RATE \
--num_train_epochs=$NUM_TRAIN_EPOCHS \
--output_dir=$EXP_MODELS \
--cache_dir=$PS_DATA_DIR/finetuning/classification_data/cached/ \
--logging_steps=$LOGGING_STEPS \
--finetuning=$FINETUNING \
"

pushd $EVAL_DIR
CUDA_VISIBLE_DEVICES=$2 python finetuning/classify_pairs/run_classify.py $pargs 2>&1 &> ${EXP_DIR}/train.log
popd

