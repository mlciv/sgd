#!/bin/bash

taskname=$1
split=$2
ckpt_start=$3
ckpt_gap=$4
ckpt_end=$5

for i in `seq $ckpt_start $ckpt_gap $ckpt_end`;
do
	folder_name=`echo pred_res_${i}_${split}_${taskname}_`
	echo $folder_name
	./eval.sh $split $folder_name $taskname
done
