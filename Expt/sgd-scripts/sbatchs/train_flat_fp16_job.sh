#!/bin/bash

#SBATCH --job-name=train_flat_fp16
#SBATCH --gres=gpu:2
#SBATCH --output=/home/utah/jiecao/dgx_jobs/sgd/train_flat_fp16.txt
#SBATCH --ntasks=1
#SBATCH --time=100:40:00
#SBATCH --mem=60G
pushd $CODE_BASE/sgd/Expt/sgd-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./train_flat_fp16.sh $1
popd
