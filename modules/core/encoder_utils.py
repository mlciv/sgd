# Time-stamp: <2020-06-03>
# --------------------------------------------------------------------
# File Name          : pretrain_utils.py
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
        if config.enc_model_type in CONFIG_MAPPING:
            model = bert_utils.get_bert_model(config.enc_checkpoint)
        else:
            raise NotImplementedError("{} is not supported".format(config.enc_model_type))
        return model

    @staticmethod
    def create_tokenizer(config):
        """
        config is object of PretrainedConfig, which hasattr enc_model_type and enc_checkpoint
        """
        logger.info("create tokenizer, load encoder tokenizer from encoder config: {}".format(config))
        if config.enc_model_type in CONFIG_MAPPING:
            tokenizer = bert_utils.get_bert_tokenizer(config.enc_checkpoint)
        else:
            raise NotImplementedError("{} is not supported".format(config.enc_model_type))
        return tokenizer
