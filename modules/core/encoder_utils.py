# Time-stamp: <2020-06-03>
# --------------------------------------------------------------------
# File Name          : encoder_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : encoder utils for initializing a model or load a
# models from a enc_checkpoint
# --------------------------------------------------------------------


import torch
import logging
import os

from utils import bert_utils
from transformers.configuration_auto import CONFIG_MAPPING
logger = logging.getLogger(__name__)


class EncoderUtils(object):
    """
    Pretraining Utils that can be used to create the encoder for finetuning
    """

    @staticmethod
    def create_encoder(config):
        """
        config is object of PretrainedConfig, which hasattr enc_model_type and enc_checkpoint
        """
        logger.info("create encoder, load encoder model from encoder config: {}".format(config))
        #if config.enc_model_type in CONFIG_MAPPING:
        model = bert_utils.get_bert_model(config.enc_checkpoint)
        #else:
        #    raise NotImplementedError("{} is not supported".format(config.enc_model_type))
        return model

    @staticmethod
    def set_encoder_finetuning_status(encoder, finetuning):
        if finetuning:
            # It’s a bit confusing, but model.train(False) doesn’t change param.requires_grad. It only changes the behavior of nn.Dropout and nn.BatchNorm (and maybe a few other modules) to use the inference-mode behavior.
            # it is not setting with model.train()
            pass
        else:
            for param in encoder.parameters():
                param.requires_grad = False
        logger.info("set encoder finetuning status: {}".format(finetuning))


    @staticmethod
    def create_tokenizer(config):
        """
        config is object of PretrainedConfig, which hasattr enc_model_type and enc_checkpoint
        """
        logger.info("create tokenizer, load encoder tokenizer from encoder config: {}".format(config))
        #if config.enc_model_type in CONFIG_MAPPING:
        tokenizer = bert_utils.get_bert_tokenizer(config.enc_checkpoint)
        #else:
        #    raise NotImplementedError("{} is not supported".format(config.enc_model_type))
        return tokenizer

    @staticmethod
    def add_special_tokens(tokenizer, model, special_tokens_dict):
        previous_tokens = len(tokenizer)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        if model:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("adding special tokens to the dict {}, num_added_toks:{}, total_tokens:{} -> {}".format(
            special_tokens_dict, num_added_toks, previous_tokens, len(tokenizer)))
