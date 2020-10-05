#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

mkdir -p $2

pargs="
--schema_file=$1 \
--output_dir=$2 \
--cat_slots_prediction_dir=$3 \
--noncat_slots_prediction_dir=$4
"

pushd $EVAL_DIR
python utils/merge_result.py $pargs 2>&1
popd
