#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

data_folder=$1
task_name=$2
split=$3

pargs="
--data_folder $data_folder \
--task_name $task_name \
--split $split \
"

pushd $EVAL_DIR
python -m utils.analysis_schema $pargs > ${data_folder}/$split/${task_name}.stat.log 2>&1
popd
