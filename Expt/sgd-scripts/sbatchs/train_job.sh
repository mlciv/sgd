#!/bin/bash

#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/sgd/train.txt
#SBATCH --ntasks=1
#SBATCH --time=100:40:00
#SBATCH --mem=60G
pushd $CODE_BASE/sgd/Expt/sgd-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./train.sh $1
popd
