Schema-guided Dialogue
==================

This repo implements a family of neural components for schema-guided dialog, inclduing encoder architectures, description styles, supplementary training and so on.

# Part I. Usage
*******************

## Required Software

   - Checkout this project.

   `src`, `modules`, `utils` are main python(pytorch) source code folders

   `Expt` folder is a folder for experiment managing, which includes all the schemas descriptions, and commands(```Expt/sgd-scripts/commands```), mmain config files(```Expt/sgd-scripts/configs```) to launch the experiments.

    In this repo, except `Expt/sgd_scirpts/commands/env.sh` contains the global variables, all model hyperparameters and reltaed configurations will be assigned in the config files in ```Expt/psyc_scripts/configs```, each of them is corresponding to a model with different encoder architectures, pretrained BERT on different datesets.

   - Install pyenv or other python environment manager

   In our case, we use pyenv and its plugin pyenv-virtualenv to set up
   the python environment. Please follow the detailed steps in
   https://github.com/pyenv/pyenv-virtualenv for details. Alternative
   environments management such as conda will be fine.

   - Install required packages

   ```bash
   pyenv install 3.7.7
   # in our default setting, we use `pyenv activate py2.7_tf1.4` to
   # activate the envivronment, please change this according to your preference.

   pyenv virtualenv 3.7.7 py-sgd
   pyenv activate py-sgd
   pip install -r requirements.txt
  ```

## Data Preprocessing

### Dataset Download
Please download the schema-guided dataset and multiwoz 2.2 dataset from the following repos
```
# for schema-guided dialog
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
# for MulwiWOZ 2.2
git clone https://github.com/budzianowski/multiwoz/
cd multiwoz/tree/master/data/MultiWOZ_2.2
```

### Remix MultiWOZ 2.2 for zero-shot evaluation

To support both seen and unseen service evalution, we follow the
schema-guided dialog settings.  We remix the MultiWOZ 2.2 datasets to
include as seen services dialogs related to `restaurant`, `attraction`
and `train` during training, and eliminate slots from other
domains/services from training split.  For dev, we add two new domains
`hotel` and `taxi` as unseen services. For test, we add all remaining
domains as unseen, including those that have minimum overlap with seen
services, such as `hospital`, `police`, `bus`.

```
# all commands scripts are in the Expt/sgd-scripts/commands/
./zero_multiwoz.sh $your_multiwoz_dataset_folder $your_output_folder
```

## Training/Resume/Evaluate
all trainig/evaluting commands simply follow a single config file arguments as follows,
the <config_file> are in the ```Expt/sgd-scripts/configs/```

```bash
# for cross encoder model, use scripts with flat keyword
./train_flat_fp16.sh <config_file>
# training from saved checkpoint, matched by model file name
./train_restore_flat_fp16_from_ckpt.sh <config_file> $checkpoint_path
# evaluate with the saved model on test
./eval_test_flat_fp16.sh <config_file> $checkpoint_path

# for dual-encoder and fusion encoder baselines, use train_fp16
./train_fp16.sh <config_file>
./train_restore_fp16_from_ckpt.sh <config_file> $checkpoint_path
./eval_test_fp16.sh <config_file> $checkpoint_path
```

# Part II. Experiment Desgining and Configurations
The commands used for all experiments follow the previous ```./command.sh <config_file>``` pattern.Here we first introduce the configuration files, and then we list all the configuration files used for each experiment.

## Configuration
The main configurations files are located in ```/Expt/sgd-scripts/configs```.
Besides that, the configuration file are compositional, which includes other two basic configurations files, encoder config and model config.

### Encoder Config.
```encoder config``` tries to seperate out the configuration for the pretrained BERT.
For example, if we want to use different pretrained model from huggingface model hub.
We can simply create an encoder config in ```Expt/sgd-scripts/json-configs/encoders/```

```enc_model_type``` is just an alias of the model name, which will be used to identify different cached files, saved checkpoints and so on.

```enc_checkpoint``` is to point out the checkpoint location, it can be the path in Huggingface model hub, or any local or remote path.

```json
{
  "enc_model_type": "bert-squad2",
  "enc_checkpoint": "deepset/bert-base-cased-squad2"
}
```

### Model Config
We try to seperate out the model bussiness into this model config.
Model config mainly focus on the model parameters and model architectures for each dialog tasks in our model.

```json
{
  "utterance_dropout": 0.3,
  "token_dropout": 0.3,
  # cached schema embedding file for each task
  "schema_embedding_file_name": "flat_noncat_seq2_features.npy",
  "schema_max_seq_length": 80,
  # max dialg context length
  "dialog_cxt_length": 15,
  # this type will decide how to prepare the schema embedding and input for the model, more options are in modules/schema_embedding_generator.py
  "schema_embedding_type": "flat_seq2_feature",
  "schema_embedding_dim": 768,
  # finetuning_type, means whether finetuning the schema BERT.
  "schema_finetuning_type": "bert_snt_pair",
  # the key used to retrieve features, it used for different compositions for different subtasks, "service_desc", and "desc_only". More options are in modules/schema_embedding_generator.py
  "noncat_slot_seq2_key": "noncat_slot_desc_only",
  "utterance_embedding_dim": 768
}
```

Please see the detailed comments in the following configuration files.
Take file ```Expt/sgd-scripts/configs/flat_noncat_slots_bert_snt_pair_dstc8_question_rich_squad_bert_desc_only.sh``` as an example.

```bash
# https://stackoverflow.com/questions/965053/extract-filename-and-extension-in-bash
config_name=`basename "$1"`

EXP_NAME="${config_name%.*}"
echo $EXP_NAME
TASK_NAME=dstc8_question_rich
EXP_DIR=$SGD_WORK_DIR/$TASK_NAME/$EXP_NAME/
EXP_MODELS=$EXP_DIR/models/
EXP_SUMMARY=$EXP_DIR/summary/
EXP_RESULTS=$EXP_DIR/results/

# model_type, some model name to initialize or load pretrained model
MODEL_TYPE=flat_noncat_slots_bert_snt_pair_match
# encoder config name for the task
ENCODER_CONFIG_NAME=
# encoder_model_name_path, whether a name or a path for the model
ENCODER_MODEL_NAME_PATH=$SGD_JSON_CONFIG_DIR/encoders/bert-base-cased-squad2.json
# config name for the task
CONFIG_NAME=$SGD_JSON_CONFIG_DIR/models/flat_noncat_slots_bert_snt_pair_match_desc_only.json
# model_name_path, whether a name or a path for the model
MODEL_NAME_PATH=
# cache_dir, the cache_dir for store the mebdding, exampls.
CACHE_DIR=$SGD_CACHE_DIR
# data_dir, the data_dir for the splits
DATA_DIR=$DSTC8_DATA_DIR
# train_file, the file for training
TRAIN_FILE=train
# dev_file, the file for eval
DEV_FILE=dev
# test_file, the file for eval
TEST_FILE=test
# per_gpu_eval_batch_size
PER_GPU_EVAL_BATCH_SIZE=16
# per_gpu_train_batch_size
PER_GPU_TRAIN_BATCH_SIZE=16
# num_train_epochs
NUM_TRAIN_EPOCHS=10
# learning_rate
LEARNING_RATE=5e-5
# gradient_accumulation_steps
GRADIENT_ACCUMULATION_STEPS=8
# logging_steps
LOGGING_STEPS=2000
# save_steps 
SAVE_STEPS=1000000
# JOINT_ACC_ACROSS_TURN
JOINT_ACC_ACROSS_TURN=
# USE_FUZZY_MATCH
USE_FUZZY_MATCH=x
# MAX_SEQ_LENGTH
MAX_SEQ_LENGTH=512
# warmup_step
WARMUP_PORTION=0.1
# whether finetuning the encoder
ENCODER_FINETUNING=x

```

## Encoder Architectures
### Dual-Encoders
Joint Dual Encoder model: ```dstc8baseline_dstc8_all_bert.sh```

### Fusion-Encoders

Joint Fusion-Encoder model: ```dstc8baseline_toptrans_dstc8_all_bert_2_2_768_1024.json```

### Cross-Encoders
Intent: ```flat_active_intent_bert_snt_pair_dstc8_all_bert.sh```

ReqSlot: ```flat_requested_slots_bert_snt_pair_dstc8_all_bert.sh```

CatSlot: ```flat_cat_slot_value_bert_snt_pair_dstc8_all_bert.sh```

NonCatSlot: ```flat_noncat_slots_bert_snt_pair_dstc8_all_bert.sh```

## Supplementary Training
For supplementary training, we compare the model results before and after applying supplementary training on SNLI and SQUAD2
### SNLI
Gain on Intent: ```flat_active_intent_bert_snt_pair_dstc8_all_bert_uncased_snli_desc_only.sh``` and ```flat_active_intent_bert_snt_pair_dstc8_all_bert_uncased_desc_only.sh```

Gain on ReqSlot: ```flat_requested_slots_bert_snt_pair_dstc8_all_bert_uncased_snli_desc_only.sh``` and ```flat_requested_slots_bert_snt_pair_dstc8_all_bert_uncased_desc_only.sh```

Gain on CatSlot: ```flat_cat_slot_value_bert_snt_pair_dstc8_all_bert_uncased_snli_desc_only_128.sh```  and ```flat_cat_slot_value_bert_snt_pair_dstc8_all_bert_uncased_desc_only_128.sh```

Gain on NonCatSlot: ```flat_noncat_slots_bert_snt_pair_dstc8_all_uncased_snli_bert_desc_only_128.sh``` and ```flat_noncat_slots_bert_snt_pair_dstc8_all_uncased_bert_desc_only_128.sh```

### SQUAD
Gain on Intent: ```flat_active_intent_bert_snt_pair_dstc8_all_bert_squad2_desc_only.sh``` and ```flat_active_intent_bert_snt_pair_dstc8_all_bert_desc_only.sh```

Gain on ReqSlot: ```flat_requested_slots_bert_snt_pair_dstc8_all_squad2_desc_only.sh``` and ```flat_requested_slots_bert_snt_pair_dstc8_all_bert_desc_only.sh```

Gain on CatSlot: ```flat_cat_slot_value_bert_snt_pair_dstc8_all_bert_squad2_desc_only_128.sh``` and ```flat_cat_slot_value_bert_snt_pair_dstc8_all_bert_bert_desc_only_128.sh```

Gain on NoncatSlot: ```flat_noncat_slots_bert_snt_pair_dstc8_all_bert_squad2_desc_only_128.sh``` and ```flat_noncat_slots_bert_snt_pair_dstc8_all_bert_desc_only_128.sh```


## Impact of Description Styles
For different descriptions styles, we have create different corresponding schema files.
More details are in ```utils/schema_dataset_config.py```
Each description style will be corresponding to a dataset name, such as ```dstc8_name_only``` and ```dstc8_question_rich```
Hence, to train and evaluate on different dataset, we just need to specify the dataset to work on.

### Train on a description style
```bash
# you can find the corresponding configuration for different styles on sgd and multiwoz 2.2 dataset in the configs folder. The filenames are in similar template as following
./train_flat_fp16.sh ../configs/flat_noncat_slots_bert_snt_pair_dstc8_name_only_bert_desc_only.sh
```

### Evalution on description style

```bash
# homogenous evaluation
./eval_test_flat_fp16.sh ../configs/flat_noncat_slots_bert_snt_pair_dstc8_name_only_bert_desc_only.sh $some_model_trained_nameonly

# hetergenuous evaluation
./eval_test_flat_fp16.sh ../configs/flat_noncat_slots_bert_snt_pair_dstc8_name_only_bert_desc_only.sh $some_model_trained_on_other_styles
```




