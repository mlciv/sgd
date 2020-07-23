#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

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
if [ ! -e ${EXP_DIR} ]; then
  echo ${EXP_DIR}" not exit"
fi

# evlauate arguments
pargs="
--do_eval \
--overwrite_output_dir \
--evaluate_during_training \
--eval_all_checkpoints \
--model_type=$MODEL_TYPE \
--config_name=$CONFIG_NAME \
--encoder_config_name=$ENCODER_CONFIG_NAME \
--encoder_finetuning=$ENCODER_FINETUNING \
--model_name_or_path=$2 \
--encoder_model_name_or_path=$ENCODER_MODEL_NAME_PATH \
--task_name=$TASK_NAME \
--output_metric_file=$OUT_METRIC_FILE \
--joint_acc_across_turn=$JOINT_ACC_ACROSS_TURN \
--use_fuzzy_match=$USE_FUZZY_MATCH \
--data_dir=$DATA_DIR \
--train_file=$TRAIN_FILE \
--dev_file=$DEV_FILE \
--test_file=$DEV_FILE \
--per_gpu_eval_batch_size=$PER_GPU_EVAL_BATCH_SIZE \
--per_gpu_train_batch_size=$PER_GPU_TRAIN_BATCH_SIZE \
--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
--learning_rate=$LEARNING_RATE \
--num_train_epochs=$NUM_TRAIN_EPOCHS \
--output_dir=$EXP_DIR \
--cache_dir=$CACHE_DIR \
--logging_steps=$LOGGING_STEPS \
--save_steps=$SAVE_STEPS \
--max_seq_length=$MAX_SEQ_LENGTH \
--warmup_portion=$WARMUP_PORTION \
"

pushd $EVAL_DIR
# CUDA_LAUNCH_BLOCKING=1
python src/run_schema.py $pargs 2>&1 &> ${EXP_DIR}/eval_dev.log
popd

