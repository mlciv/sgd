#!/bin/bash

. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

folder=$1
output_folder=$2

pargs="
--folder=$folder \
--output_folder=$output_folder \
"

pushd $EVAL_DIR
python utils/multiwoz_splitter.py $pargs 2>&1 
popd
