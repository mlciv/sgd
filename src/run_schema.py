# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SGD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup
)

from transformers.configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP

from modules.core.schemadst_configuration import SchemaDSTConfig
from modules.core.encoder_configuration import EncoderConfig
from modules.core.encoder_utils import EncoderUtils
from modules.active_intent_cls_match_model import ActiveIntentCLSMatchModel
from modules.requested_slots_cls_match_model import RequestedSlotsCLSMatchModel
from modules.cat_slots_cls_match_model import CatSlotsCLSMatchModel
from modules.noncat_slots_cls_match_model import NonCatSlotsCLSMatchModel
from modules.modelling_dstc8baseline import DSTC8BaselineModel
from modules.modelling_dstc8baseline_toptrans import DSTC8BaselineTopTransModel
from modules.modelling_toptransformer import TopTransformerModel
from modules.schema_embedding_generator import SchemaEmbeddingGenerator
from utils import schema_dataset_config
from utils import data_utils
from utils import pred_utils
from utils import evaluate_utils
from utils import torch_ext
from src import utils_schema

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = SchemaDSTConfig.pretrained_config_archive_map.keys()

ALL_ENCODER_MODELS = ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()


# model classes to use
MODEL_CLASSES = {
    "dstc8baseline": (DSTC8BaselineModel),
    "active_intent_cls_match": (ActiveIntentCLSMatchModel),
    "requested_slots_cls_match": (RequestedSlotsCLSMatchModel),
    "cat_slots_cls_match": (CatSlotsCLSMatchModel),
    "noncat_slots_cls_match": (NonCatSlotsCLSMatchModel),
    "toptrans": (TopTransformerModel),
    "dstc8baseline_toptrans": (DSTC8BaselineTopTransModel),
    }

MODEL_TYPES = MODEL_CLASSES.keys()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, config, train_dataset, model, processor):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.summary_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.warmup_steps = args.warmup_portion * t_total
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = torch_ext.DataParallelPassthrough(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup portion = %f, warmup steps = %d", args.warmup_portion, args.warmup_steps)
    logger.info("  Model Type = %s", args.model_type)
    if args.model_type in ["dstc8baseline", "active_intent_cls_match",
                           "requested_slots_cls_match", "cat_slots_cls_match", "noncat_slots_cls_match",
                           "dstc8baseline_toptrans", "toptrans"]:
        schema_tensors = SchemaEmbeddingGenerator.create_or_load_schema_embedding(
            args,
            config,
            "train"
        )
        logger.info("  When Model Type = %s, create or load the schema embedding:", args.model_type)
        for key, tensor in schema_tensors.items():
            logger.info("%s: %s", key, tensor.size())

    logger.info("  Model Architecture = %s", model)

    best_metrics = {}
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            # "checkpoing-xxxx" or "best-model-xxxx"
            checkpoint_suffix = os.path.basename(os.path.normpath(args.model_name_or_path)).split("-")[-1]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", position=0, leave=True, disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    epoch = 0
    for _ in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True, disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            all_examples_ids = batch[0]
            all_service_ids = batch[1]
            all_input_ids = batch[2]
            all_attention_masks = batch[3]
            all_token_type_ids = batch[4]
            all_cat_slot_num = batch[5]
            all_cat_slot_value_num = batch[6]
            all_noncat_slot_num = batch[7]
            all_noncat_alignment_start = batch[8]
            all_noncat_alignment_end = batch[9]
            all_req_slot_num = batch[10]
            all_intent_num = batch[11]

            inputs = {
                "example_id": all_examples_ids,
                "service_id": all_service_ids,
                "utt": all_input_ids,
                "utt_mask": all_attention_masks,
                "utt_seg": all_token_type_ids,
                "cat_slot_num": all_cat_slot_num,
                "cat_slot_value_num": all_cat_slot_value_num,
                "noncat_slot_num": all_noncat_slot_num,
                "noncat_alignment_start": all_noncat_alignment_start,
                "noncat_alignment_end": all_noncat_alignment_end,
                "req_slot_num": all_req_slot_num,
                "intent_num": all_intent_num,
            }

            # Add schema tensor into features
            for key, tensor in schema_tensors.items():
                inputs[key] = tensor.to(args.device).index_select(0, inputs["service_id"])

            # results
            all_cat_slot_status = batch[12]
            all_cat_slot_values = batch[13]
            all_noncat_slot_status = batch[14]
            all_noncat_slot_start = batch[15]
            all_noncat_slot_end = batch[16]
            all_req_slot_status = batch[17]
            all_intent_status = batch[18]

            labels = {
                "cat_slot_status": all_cat_slot_status,
                "cat_slot_value": all_cat_slot_values,
                "noncat_slot_status": all_noncat_slot_status,
                "noncat_slot_value_start": all_noncat_slot_start,
                "noncat_slot_value_end": all_noncat_slot_end,
                "req_slot_status": all_req_slot_status,
                "intent_status": all_intent_status
            }

            if args.enc_model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                inputs["utt_seg"] = None

#            if args.model_type in ["xlnet", "xlm"]:
#                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
#                if args.version_2_with_negative:
#                    inputs.update({"is_impossible": batch[7]})
#                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
#                    inputs.update(
#                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
#                    )

            outputs = model(inputs, labels)
            # for outputs_name, l in outputs[0].items():
            #    logger.info("outputs_name:{}, l:{}".format(outputs_name, l.size()))


            losses = outputs[1]
            tmp_loss_dict = {}
            for loss_name, loss in losses.items():
                if isinstance(loss, torch.Tensor):
                    loss = loss.sum()
                tb_writer.add_scalar(loss_name, loss, global_step)
                try:
                    tmp_loss_dict[loss_name] = loss.item()
                except:
                    logger.error("loss_name:{}, loss:{}".format(loss_name, loss))

#            logger.info("Epoch = {}, Global_step = {}, losses = {}".format(
#                epoch, global_step, tmp_loss_dict
#                )
#            )

            loss = sum(losses.values())
            # loss = losses["span_start_loss"] + losses["span_end_loss"] + losses["noncat_slot_status_loss"]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # we need a metric here to track and save the checkpoints.
                        results, _ = evaluate(args, config, model, processor, "dev", step=global_step, tb_writer=tb_writer)

                        for key, value in results.items():
                            # tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            # We only save the metrics in the special list
                            metrics_for_key = best_metrics.setdefault(key, {})
                            # all the value is larger the better.
                            for v_key, v_value in value.items():
                                joint_key = "-".join([key, v_key])
                                tb_writer.add_scalar("eval_{}".format(joint_key), v_value, global_step)
                                update_metric = False
                                if v_key in metrics_for_key:
                                    if v_value > metrics_for_key[v_key][0]:
                                        update_metric = True
                                else:
                                    update_metric = True

                                if update_metric:
                                    logger.info("Epoch = {}, Global_step = {}, Update metric {} = {}".format(
                                        epoch, global_step, joint_key, v_value))
                                    # Save model checkpoint
                                    if v_key in metrics_for_key:
                                        old_global_step = metrics_for_key[v_key][1]
                                        old_name = "best-{}-{}".format(joint_key, old_global_step)
                                        old_path = os.path.join(args.models_dir, old_name)
                                    else:
                                        old_path = None
                                    metrics_for_key[v_key] = (v_value, global_step)
                                    # We only save the model for core metrics
                                    if key in evaluate_utils.CORE_METRIC_KEYS and \
                                       v_key in evaluate_utils.IMPORTANT_METRIC_SUBKEYS:
                                        if old_path:
                                            shutil.rmtree(old_path)
                                        save_checkpoint(args, model, processor._tokenizer,
                                                        optimizer, scheduler, "best-{}-{}".format(joint_key, global_step))

                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("Epoch = {}, Global_step = {}, lr = {}, losses = {}".format(
                        epoch, global_step, scheduler.get_last_lr()[0], tmp_loss_dict)
                    )
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, model, processor._tokenizer, optimizer, scheduler, "checkpoint-{}".format(global_step))


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    logger.info("Best metrics on dev after {} training:{}".format(global_step, best_metrics))

    return global_step, tr_loss / global_step


def evaluate(args, config, model, processor, mode, step="", tb_writer=None):
    # In this task, we simply use the split name as the train, dev, test files.
    if mode == 'train':
        file_split = args.train_file
    elif mode == "dev":
        file_split = args.dev_file
    else:
        file_split = args.test_file

    dataset, examples, features, dials, schemas = load_and_cache_examples(args, processor, mode, output_examples=True)

    if not os.path.exists(args.models_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.models_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        # model = torch.nn.DataParallel(model)
        model = torch_ext.DataParallelPassthrough(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(step))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # schame embededing should be done before the training
    logger.info("  Model Type = %s", args.model_type)
    if args.model_type in ["dstc8baseline", "active_intent_cls_match",
                           "requested_slots_cls_match", "cat_slots_cls_match", "noncat_slots_cls_match",
                           "dstc8baseline_toptrans", "toptrans"]:
        schema_tensors = SchemaEmbeddingGenerator.create_or_load_schema_embedding(
            args,
            config,
            file_split
        )
        logger.info("  When Model Type = %s, create or load the schema embedding:", args.model_type)
        for key, tensor in schema_tensors.items():
            logger.info("%s: %s", key, tensor.size())

    all_predictions = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        all_examples_ids = batch[0]
        all_service_ids = batch[1]
        all_input_ids = batch[2]
        all_attention_masks = batch[3]
        all_token_type_ids = batch[4]
        all_cat_slot_num = batch[5]
        all_cat_slot_value_num = batch[6]
        all_noncat_slot_num = batch[7]
        all_noncat_alignment_start = batch[8]
        all_noncat_alignment_end = batch[9]
        all_req_slot_num = batch[10]
        all_intent_num = batch[11]

        # no results
        inputs = {
            "example_id": all_examples_ids,
            "service_id": all_service_ids,
            "utt": all_input_ids,
            "utt_mask": all_attention_masks,
            "utt_seg": all_token_type_ids,
            "cat_slot_num": all_cat_slot_num,
            "cat_slot_value_num": all_cat_slot_value_num,
            "noncat_slot_num": all_noncat_slot_num,
            "noncat_alignment_start": all_noncat_alignment_start,
            "noncat_alignment_end": all_noncat_alignment_end,
            "req_slot_num": all_req_slot_num,
            "intent_num": all_intent_num,
        }

        # Add schema tensor into features
        for key, tensor in schema_tensors.items():
            inputs[key] = tensor.to(args.device).index_select(0, inputs["service_id"])

        # not adding token_type_ids for roberta
        if args.enc_model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            inputs["utt_seg"] = None

        if len(batch) > 12:
            # results
            all_cat_slot_status = batch[12]
            all_cat_slot_values = batch[13]
            all_noncat_slot_status = batch[14]
            all_noncat_slot_start = batch[15]
            all_noncat_slot_end = batch[16]
            all_req_slot_status = batch[17]
            all_intent_status = batch[18]

            labels = {
                "cat_slot_status": all_cat_slot_status,
                "cat_slot_value": all_cat_slot_values,
                "noncat_slot_status": all_noncat_slot_status,
                "noncat_slot_value_start": all_noncat_slot_start,
                "noncat_slot_value_end": all_noncat_slot_end,
                "req_slot_status": all_req_slot_status,
                "intent_status": all_intent_status
            }
        else:
            labels = None

        with torch.no_grad():
            outputs = model(inputs, labels)
            if labels:
                losses = outputs[1]
                tmp_loss_dict = {}
                for loss_name, loss in losses.items():
                    if isinstance(loss, torch.Tensor):
                        loss = loss.sum()
                    if tb_writer:
                        tb_writer.add_scalar(mode + "_" + loss_name, loss, step)
                    try:
                        tmp_loss_dict[loss_name] = loss.item()
                    except:
                        logger.error("loss_name:{}, loss:{}".format(loss_name, loss))

            # dict: batch_size
            # dict of list => list of dict
            predictions = model.define_predictions(inputs, outputs[0])
            for i in range(len(all_examples_ids)):
                prediction = {}
                for key in predictions.keys():
                    prediction[key] = predictions[key][i]
                all_predictions.append(prediction)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                eval_time, eval_time / len(dataset))

    # Compute predictions
    start_time = timeit.default_timer()
    # indexed dialogue, key is (dialogue_id, turn_idx, service_name)
    indexed_predictions = pred_utils.get_predictions_index_dict(all_predictions)
    # Here dials will be modified and return
    all_predicted_dialogues = pred_utils.get_all_prediction_dialogues(dials, indexed_predictions, schemas)
    # has ground truth
    ref_dialogs = processor.get_whole_dialogs(args.data_dir, file_split)
    ref_dialogue_dict = evaluate_utils.get_dialogue_dict(ref_dialogs)
    predicted_dialogue_dict = evaluate_utils.get_dialogue_dict(all_predicted_dialogues)
    all_metric_aggregate = evaluate_utils.get_metrics_result(
        ref_dialogue_dict, predicted_dialogue_dict,
        args.data_dir, file_split,
        args.use_fuzzy_match,
        args.joint_acc_across_turn)

    logger.info("Model_step: {}, Dialog metrics: {}, {}, {}".format(
        step,
        str(all_metric_aggregate[evaluate_utils.ALL_SERVICES]),
        str(all_metric_aggregate[evaluate_utils.SEEN_SERVICES]),
        str(all_metric_aggregate[evaluate_utils.UNSEEN_SERVICES]))
    )
    metric_time = timeit.default_timer() - start_time
    logger.info("  Metrics done in total %f secs (%f sec per example)",
                metric_time, metric_time / len(dataset))
    return all_metric_aggregate, all_predictions


def load_and_cache_examples(args, processor, mode, output_examples=False):
    """
    load and cache sgd dialogue examples
    """
    if args.local_rank not in [-1, 0] and mode == "train":
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    # In this task, we simply use the split name as the train, dev, test files.
    if mode == 'train':
        file_split = args.train_file
    elif mode == "dev":
        file_split = args.dev_file
    else:
        file_split = args.test_file

    # Here are also return the labels for the dev set
    is_eval = True if mode in ["test"] else False

    sub_dir = os.path.join(args.cache_dir, args.task_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    cached_features_file = os.path.join(
        sub_dir,
        "cached_{}_{}_{}_{}".format(
            list(filter(None, file_split.split("/"))).pop(),
            args.model_type,
            args.enc_model_type,
            args.max_seq_length
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples, dials, schemas = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
            features_and_dataset["dials"],
            features_and_dataset["schemas"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (
                (mode == "dev" and not args.dev_file) or
                (mode == "test" and not args.test_file) or
                (mode == "train" and not args.train_file)):
            try:
                # by default, usng tensorflow_datasets
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            # the schema dataset is not in tensorflow_datasets right now
            # tfds_examples = tfds.load("schema_guided_dataset")
            # examples = SquadV1Processor().get_examples_from_dataset(tfds_examples,  evaluate=evaluate)
            raise NotImplementedError("schema_guided_dialogue is not in tensorflow_datasets")
        else:
            examples = processor.get_dialog_examples(args.data_dir, file_split)
            dials = processor.get_whole_dialogs(args.data_dir, file_split)
            schemas = processor.get_schemas(args.data_dir, file_split)

        # return_dataset = pt, is pytorch TensorDataset, tf is tf.data.Dataset
        features, dataset = utils_schema.convert_examples_to_features(
            examples=examples,
            dataset_config=args.dataset_config,
            max_seq_length=args.max_seq_length,
            is_training=not is_eval,
            return_dataset="pt"
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(
                {
                    "features": features, "dataset": dataset,
                    "examples": examples, "dials": dials, "schemas": schemas
                },
                cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset
        # and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features, dials, schemas
    return dataset

def save_checkpoint(args, model, tokenizer, optimizer, scheduler, name):
    # Save model checkpoint
    output_dir = os.path.join(args.models_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # model
    model_to_save.save_pretrained(output_dir)
    # tokenizer
    tokenizer.save_pretrained(output_dir)
    # args
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    # optimizer and scheduler
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

def main():
    parser = argparse.ArgumentParser()

    # task related arguments
    # Input and output paths and other flags.
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        choices=schema_dataset_config.DATASET_CONFIG.keys(),
        required=True,
        help="The name of the task to train.")

    parser.add_argument(
        "--output_metric_file",
        default=None,
        type=str,
        help="Single JSON output file containing aggregated evaluation metrics results"
        " for all predictions files in args.prediction_dir.")

    parser.add_argument(
        "--joint_acc_across_turn",
        default=False,
        type=bool,
        help="Whether to compute joint accuracy across turn instead of across service. "
        "Should be set to True when conducting multiwoz style evaluation.")

    parser.add_argument(
        "--use_fuzzy_match",
        default=True,
        type=bool,
        help="Whether to use fuzzy string matching when comparing non-categorical slot "
        "values. Should be set to False when conducting multiwoz style evaluation.")

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier",
    )

    parser.add_argument(
        "--intent_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained intent model or model identifier",
    )

    parser.add_argument(
        "--req_slots_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained requested_slots model or model identifier",
    )

    parser.add_argument(
        "--cat_slots_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained cat_slots model or model identifier",
    )

    parser.add_argument(
        "--noncat_slots_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained noncat_slots model or model identifier",
    )

    parser.add_argument(
        "--encoder_model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained encoder_model or shortcut name selected in the list: " + ", ".join(ALL_ENCODER_MODELS)
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input evaluation file . If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="The input evaluation file for test. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--encoder_config_name", default="", type=str,
        help="Pretrained encoder_config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_portion", default=0.0, type=float, help="Linear warmup over warmup_portion.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--log_data_warnings", type=bool, default=True, help="Log the warnings when handling data.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()


    # for debuging only
    #np.random.seed(0)
    #random.seed(0)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    args.models_dir = os.path.join(args.output_dir, "models")
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)
    args.results_dir = os.path.join(args.output_dir, "results")
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    args.summary_dir = os.path.join(args.output_dir, "summary")
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # setup datasetconfig
    if args.task_name not in schema_dataset_config.DATASET_CONFIG:
        raise ValueError("Task not found: %s" % (args.task_name))
    else:
        args.dataset_config = schema_dataset_config.DATASET_CONFIG[args.task_name]

    # build different model type for different options.
    args.model_type = args.model_type.lower()
    # encoder config
    encoder_config = EncoderConfig.from_pretrained(
        args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    args.enc_model_type = encoder_config.enc_model_type.lower()

    # tokenizer
    tokenizer = EncoderUtils.create_tokenizer(encoder_config)
    tokenizer_class = tokenizer.__class__

    # write a data processor for schema_guided dialogue
    # For each model, we write different processor, examples, features, and model
    # Here is the baseline model from the google
    processor = utils_schema.SchemaDSTC8Processor(
        args.dataset_config,
        tokenizer,
        args.max_seq_length,
        args.log_data_warnings)

    # model config
    config = SchemaDSTConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.update_encoder_config(encoder_config)
    # model class
    # Here, we need to build differe model class for different models
    model_class = MODEL_CLASSES[args.model_type]

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            encoder=None,
            config=config,
            args=args
        )
    else:
        logger.info("{} is not existed, training model from scratch".format(args.model_name_or_path))
        model = model_class(encoder=None, config=config, args=args)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, processor, mode="train", output_examples=False)
        # with torch.autograd.detect_anomaly():
        global_step, tr_loss = train(args, config, train_dataset, model, processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.models_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.models_dir)

        logger.info("Saving model checkpoint to %s", args.models_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.models_dir)
        tokenizer.save_pretrained(args.models_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.models_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.models_dir, args=args)  # , force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.models_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.models_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.models_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        else:
            # not training, but we still want to evaluate on all checkpoints
            if args.eval_all_checkpoints and not args.model_name_or_path:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.models_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
            else:
                # we only want to evaluate on specific checkpoint
                logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
                checkpoints = [args.model_name_or_path]

        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            if checkpoint == args.models_dir:
                global_step = -1
            else:
                global_step = os.path.basename(os.path.normpath(checkpoint)).split("-")[-1]
            model = model_class.from_pretrained(checkpoint, args=args)  # , force_download=True)
            model.eval()
            logger.info("Model's state_dict:")
            for param_tensor in model.state_dict():
                logger.info("{}\t{}".format(param_tensor, model.state_dict()[param_tensor].size()))
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
            model.to(args.device)

            # write a data processor for schema_guided dialogue
            # For each model, we write different processor, examples, features, and model
            # Here is the baseline model from the google
            processor = utils_schema.SchemaDSTC8Processor(
                args.dataset_config,
                tokenizer,
                args.max_seq_length,
                args.log_data_warnings)

            # Evaluate
            metrics, all_predictions = evaluate(args, config, model, processor, "test", step=global_step)
            # Write predictions to file in DSTC8 format.
            dataset_mark = os.path.basename(args.data_dir)
            prediction_dir = os.path.join(
                args.results_dir, "pred_res_{}_{}_{}_{}".format(
                    int(global_step), args.test_file, args.task_name, dataset_mark))
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)

            input_json_files = processor.get_input_dialog_files(args.data_dir, args.test_file)
            schema_json_file = processor.get_schema_file(args.data_dir, args.test_file)
            # Here, we write the evaluation files, and then get the metrics
            pred_utils.write_predictions_to_file(all_predictions, input_json_files,
                                                 schema_json_file, prediction_dir)

            output_metric_file = os.path.join(prediction_dir, "eval_metrics.json")
            evaluate_utils.write_metrics_to_file(output_metric_file, metrics)
            logger.info("metrics for checkpoint:{} is :{}".format(checkpoint, metrics))
            # TODO: do some error analysis

if __name__ == "__main__":
    main()
