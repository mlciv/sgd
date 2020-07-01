#!/bin/bash

task_name=$1
split=$2
ckpt_start=$3
ckpt_gap=$4
ckpt_end=$5

./pred_all.sh $task_name $split $ckpt_start $ckpt_gap $ckpt_end
./eval_all.sh $task_name $split $ckpt_start $ckpt_gap $ckpt_end
