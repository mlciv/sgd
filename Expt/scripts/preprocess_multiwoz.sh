#!/bin/bash

. env.sh

task_name="multiwoz21_all"
WORK_DIR=$BASE_WORK_DIR/$task_name

### CHECK WORK & DATA DIR
if [ -e ${WORK_DIR} ]; then
  today=`date +%m-%d.%H:%M`
  mv ${WORK_DIR} ${WORK_DIR%?}_${today}
  echo "rename original training folder to "${WORK_DIR%?}_${today}
fi

mkdir -p $WORK_DIR/models/

pargs="
--input_data_dir=$MULTIWOZ21_DATA_DIR \
--output_dir=$CONV_MULTIWOZ21_DATA_DIR \
"

pushd $GOOGLE_ROOT_DIR
python -m schema_guided_dst.multiwoz.create_data_from_multiwoz  $pargs > ${WORK_DIR}/preprocess_multiwoz.log 2>&1
popd
