# https://stackoverflow.com/questions/965053/extract-filename-and-extension-in-bash
config_name=`basename "$1"`

EXP_NAME="${config_name%.*}"
echo $EXP_NAME
TASK_NAME=multiwoz21_all
EXP_DIR=$SGD_WORK_DIR/$TASK_NAME/$EXP_NAME/
EXP_MODELS=$EXP_DIR/models/
EXP_SUMMARY=$EXP_DIR/summary/
EXP_RESULTS=$EXP_DIR/results/

# model_type, some model name to initialize or load pretrained model
MODEL_TYPE=noncat_slots_cls_match
# encoder config name for the task
ENCODER_CONFIG_NAME=
# encoder_model_name_path, whether a name or a path for the model
ENCODER_MODEL_NAME_PATH=$SGD_JSON_CONFIG_DIR/encoders/bert-base-cased.json
# config name for the task
CONFIG_NAME=$SGD_JSON_CONFIG_DIR/models/noncat_slots_cls_match.json
# model_name_path, whether a name or a path for the model
MODEL_NAME_PATH=
# cache_dir, the cache_dir for store the mebdding, exampls.
CACHE_DIR=$SGD_CACHE_DIR
# data_dir, the data_dir for the splits
DATA_DIR=$CONV_MULTIWOZ21_DATA_DIR
# train_file, the file for training
TRAIN_FILE=train
# dev_file, the file for eval
DEV_FILE=dev
# test_file, the file for eval
TEST_FILE=test
# per_gpu_eval_batch_size
PER_GPU_EVAL_BATCH_SIZE=32
# per_gpu_train_batch_size
PER_GPU_TRAIN_BATCH_SIZE=32
# num_train_epochs
NUM_TRAIN_EPOCHS=100
# learning_rate
LEARNING_RATE=2e-5
# gradient_accumulation_steps
GRADIENT_ACCUMULATION_STEPS=4
# logging_steps
LOGGING_STEPS=3000
# save_steps 
SAVE_STEPS=1000000
# JOINT_ACC_ACROSS_TURN
JOINT_ACC_ACROSS_TURN=x
# USE_FUZZY_MATCH
USE_FUZZY_MATCH=
# MAX_SEQ_LENGTH
MAX_SEQ_LENGTH=80
# warmup_step
WARMUP_PORTION=0.1
# whether finetuning the encoder
ENCODER_FINETUNING=x
