#!/bin/bash

. env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$SGD_CODE_DIR/

fileA=$1
fileB=$2
alpha=$3

pushd $EVAL_DIR
python utils/testSignificance.py $fileA $fileB $alpha
popd 
