#!/bin/bash

. env.sh

task_name=$1
split=$2
ckpt_start=$3
ckpt_gap=$4
ckpt_end=$5
WORK_DIR=$BASE_WORK_DIR/$task_name


pushd $WORK_DIR
for i in `seq $ckpt_start $ckpt_gap $ckpt_end`;
do
	folder_prefix=`echo model.ckpt-${i}`
	echo $folder_prefix
	rm -vf ${folder_prefix}.data-*
	rm -vf ${folder_prefix}.index
	rm -vf ${folder_prefix}.meta
	result_folder_prefix=`echo pred_res_${i}_${split}`
	echo ${result_folder_prefix}
	rm -vrf ${result_folder_prefix}_*
done
popd
