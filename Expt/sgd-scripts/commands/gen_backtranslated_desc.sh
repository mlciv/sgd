#!/bin/bash
# extract the descriptions and adding . at the end of the sentence.

cat $1 | grep description | cut -d':' -f 2 | cut -d'"' -f2 | awk '{print $0"." }' > $1.ori_desc
BACK_TRANSLATION_HOME=/home/ubuntu/git-workspace/Backtranslation/

eval "$(pyenv init -)"
pyenv activate tf_bt
pushd $BACK_TRANSLATION_HOME
./paraphrase.sh $1.ori_desc cs 5
popd
