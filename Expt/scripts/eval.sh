#!/bin/bash

. env.sh

task_name=$3
WORK_DIR=$BASE_WORK_DIR/$task_name

### CHECK WORK & DATA DIR
if [ ! -e ${WORK_DIR} ]; then
  echo "training folder does not exist"${WORK_DIR%?}_${today}
fi

split=$1
prediction_dir=$WORK_DIR/$2

pargs="
--dstc8_data_dir $BASE_DATA_DIR \
--prediction_dir $prediction_dir \
--eval_set $split \
--output_metric_file $prediction_dir/eval.metrics  \
"

pushd $GOOGLE_ROOT_DIR
python -m schema_guided_dst.evaluate $pargs > $prediction_dir/eval_$split.log 2>&1
popd
