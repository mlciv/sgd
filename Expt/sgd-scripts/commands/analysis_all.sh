#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

task_name=$1
split=$2
work_dir=$3


pargs="
--work_dir=$work_dir \
--split=$split \
--task_name=$task_name \
--start=0 \
--step=0 \
--stop=0 \
"

pushd $EVAL_DIR
echo "begin analysis"
python utils/analysis_results.py $pargs 2>&1 &> $work_dir/analysis_${split}_all.log 
popd
