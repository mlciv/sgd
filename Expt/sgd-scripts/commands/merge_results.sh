#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

mkdir -p $2

pargs="
--schema_file=$1 \
--output_dir=$2 \
--active_intent_prediction_dir=$3 \
--requested_slots_prediction_dir=$4 \
--cat_slots_prediction_dir=$5 \
--noncat_slots_prediction_dir=$6
"

pushd $EVAL_DIR
python utils/merge_result.py $pargs 2>&1
popd
