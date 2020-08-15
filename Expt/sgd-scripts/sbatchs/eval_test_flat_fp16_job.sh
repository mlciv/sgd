#!/bin/bash

#SBATCH --job-name=test_flat_fp16
#SBATCH --gres=gpu:2
#SBATCH --output=/home/utah/jiecao/dgx_jobs/sgd/test_flat_fp16.txt
#SBATCH --ntasks=1
#SBATCH --time=100:40:00
#SBATCH --mem=60G
pushd $CODE_BASE/sgd/Expt/sgd-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./eval_test_flat_fp16.sh $1 $2
popd
