#! /bin/bash
# $1 is the configs folder path
# $2 is the command folder path
set -x
configs_folder=$1
commands_folder=$2
configs=`find ${configs_folder} -name "*.sh"`
commands=`find ${commands_folder} -name "train.sh" -o -name "train_fp16.sh" -o -name "train_from_ckpt.sh" -o -name "eval_dev.sh" -o -name "eval_test.sh" -o -name "eval_test_fp16.sh"`
ARG_COMMENT="# whether finetuning the encoder"
arg_name_lc="encoder_finetuning"
ARG_NAME="ENCODER_FINETUNING"
#ARG_VALUE=transformer:pamr:\{\"hidden_dim\":256,\"projection_dim\":256,\"feedforward_hidden_dim\":200,\"num_layers\":2,\"num_attention_heads\":4\}
ARG_VALUE=x

for i in $configs; do
  if grep -Fq "${ARG_COMMENT}" $i; then
    printf "$i already has already taken effect"
  else
    printf "${ARG_COMMENT}\n${ARG_NAME}=${ARG_VALUE}\n" >> $i
  fi
done

for j in ${commands[@]}; do
  if grep -Fq "${ARG_NAME}" $j; then
    printf "$j already take effect"
  else
    sed -i "/pargs=\"/a --${arg_name_lc}=\$\{${ARG_NAME}\} \\\\" $j
  fi
done

set +x
