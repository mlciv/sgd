#!/bin/bash

. env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

split=$1
gold_dir=$2
prediction_dir=$3

pargs="
--dstc8_data_dir $gold_dir \
--prediction_dir $prediction_dir \
--eval_set $split \
--output_metric_file $prediction_dir/eval.metrics  \
"

pushd $EVAL_DIR
python utils/evaluate_utils.py $pargs > $prediction_dir/eval_$split.log 2>&1
popd
