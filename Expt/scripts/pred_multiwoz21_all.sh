#!/bin/bash

. env.sh

task_name=$1
WORK_DIR=$BASE_WORK_DIR/$task_name

split=$2
ckpt_start=$3
ckpt_gap=$4
ckpt_end=$5
ckpt=`seq $ckpt_start $ckpt_gap $ckpt_end | paste -sd","`

echo $ckpt
### CHECK WORK & DATA DIR
if [ ! -e ${WORK_DIR} ]; then
  echo "training folder does not exist"${WORK_DIR%?}_${today}
fi

pargs="
--bert_ckpt_dir $bert_ckpt_dir \
--dstc8_data_dir $CONV_MULTIWOZ21_DATA_DIR \
--dialogues_example_dir $dialogues_example_dir \
--schema_embedding_dir $multiwoz_schema_embedding_dir \
--output_dir $WORK_DIR \
--dataset_split $split \
--run_mode predict \
--task_name $task_name \
--eval_ckpt $ckpt \
"

pushd $GOOGLE_ROOT_DIR
python -m schema_guided_dst.baseline.train_and_predict $pargs > ${WORK_DIR}/predict_$split.log 2>&1
popd
