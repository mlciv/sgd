#!/bin/bash

. env.sh

data_folder=$1
task_name=$2
split=$3

pargs="
--data_folder $data_folder \
--task_name $task_name \
--split $split \
"

pushd $GOOGLE_ROOT_DIR
python -m schema_guided_dst.analysis_schema $pargs > ${data_folder}/$split/${task_name}.stat.log 2>&1
popd
