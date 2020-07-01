#!/bin/bash

taskname="multiwoz21_all"
split=$1
ckpt_start=$2
ckpt_gap=$3
ckpt_end=$4

for i in `seq $ckpt_start $ckpt_gap $ckpt_end`;
do
	folder_name=`echo pred_res_${i}_${split}_${taskname}_MultiWOZ_2.1_converted`
	echo $folder_name
	./eval_multiwoz21.sh $split $folder_name
done
