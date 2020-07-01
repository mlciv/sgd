#!/bin/bash

. env.sh

task_name="dstc8_single_domain"
WORK_DIR=$BASE_WORK_DIR/$task_name

### CHECK WORK & DATA DIR
if [ -e ${WORK_DIR} ]; then
  today=`date +%m-%d.%H:%M`
  mv ${WORK_DIR} ${WORK_DIR%?}_${today}
  echo "rename original training folder to "${WORK_DIR%?}_${today}
fi

mkdir -p $WORK_DIR/models/

pargs="
--bert_ckpt_dir $bert_ckpt_dir \
--dstc8_data_dir $BASE_DATA_DIR \
--dialogues_example_dir $dialogues_example_dir \
--schema_embedding_dir $dstc8_single_schema_embedding_dir \
--output_dir $WORK_DIR \
--dataset_split train \
--run_mode train \
--task_name $task_name
"

pushd $GOOGLE_ROOT_DIR
python -m schema_guided_dst.baseline.train_and_predict $pargs > ${WORK_DIR}/train.log 2>&1
popd
