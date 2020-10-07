#!/bin/bash

. env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

split=$1
gold_dir=$2
prediction_dir=$3
schema_file_name=$4

pargs="
--dstc8_data_dir $gold_dir \
--prediction_dir $prediction_dir \
--schema_file_name $schema_file_name \
--eval_set $split \
--output_metric_file $prediction_dir/eval.metrics  \
--joint_acc_across_turn=x \
--use_fuzzy_match= \
"

pushd $EVAL_DIR
python utils/evaluate_utils.py $pargs > $prediction_dir/eval_$split.log 2>&1
popd
