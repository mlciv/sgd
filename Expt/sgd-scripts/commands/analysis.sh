#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

task_name=$1
split=$2
ckpt_start=$3
ckpt_gap=$4
ckpt_end=$5
work_dir=$6


pargs="
--work_dir=$work_dir \
--split=$split \
--task_name=$task_name \
--start=$ckpt_start \
--step=$ckpt_gap \
--stop=$ckpt_end \
"

pushd $EVAL_DIR
python utils/analysis_results.py $pargs 2>&1 > $work_dir/analysis_${split}_${ckpt_start}_${ckpt_gap}_${ckpt_end}.log
popd
