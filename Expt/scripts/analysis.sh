#!/bin/bash

. env.sh

task_name=$1
split=$2
ckpt_start=$3
ckpt_gap=$4
ckpt_end=$5
WORK_DIR=$BASE_WORK_DIR/$task_name


pargs="
--work_dir $WORK_DIR \
--split $split \
--task_name $task_name \
--start $ckpt_start \
--step $ckpt_gap \
--stop $ckpt_end \
"

pushd $GOOGLE_ROOT_DIR
python -m schema_guided_dst.analysis_results $pargs > $WORK_DIR/analysis_${split}_${ckpt_start}_${ckpt_gap}_${ckpt_end}.log 2>&1
popd
