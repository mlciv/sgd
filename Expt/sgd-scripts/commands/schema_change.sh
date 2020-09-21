#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

schema_json_path=$1
task_name=$2


pargs="
--schema_json_path=$schema_json_path \
--task_name=$task_name \
--processed_schema=$3 \
--processed_schema=$4 \
"

pushd $EVAL_DIR
python utils/schema.py $pargs 2>&1 
popd
