# Time-stamp: <2020-05-23>
# --------------------------------------------------------------------
# File Name          : pretrain_utils.py
# Original Author    : jiessie.cao@gmail.com
# Description        : pretrain_utils for retrieve the models from the
# pretrained models
# --------------------------------------------------------------------

import logging

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel
)

logger = logging.getLogger(__name__)

def get_bert(bert_model_name_or_path):
    """
    use pytorch transformer to get all kinds of bert,
    luckily, they follow the same input and output, tokenization interface.
    Hence, you can easily support that for various bert by AutoModels
    """
    # When path is a folder, it use get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs)
    # to get a config.json file
    config = get_bert_config(bert_model_name_or_path)
    tokenizer = get_bert_tokenizer(bert_model_name_or_path)
    model = get_bert_model(bert_model_name_or_path)
    return config, tokenizer, model


def get_bert_model(bert_model_name_or_path):
    logger.info("get model by AutoModel %s",
                bert_model_name_or_path)
    # When path is a folder, it use get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs)
    # to get a config.json file
    model = AutoModel.from_pretrained(bert_model_name_or_path)
    return model


def get_bert_tokenizer(bert_model_name_or_path):
    logger.info("get tokenizer by AutoModel %s",
                bert_model_name_or_path)
    # When path is a folder, it use get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs)
    # to get a config.json file
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name_or_path)
    return tokenizer


def get_bert_config(bert_model_name_or_path):
    logger.info("get bert config by AutoModel %s",
                bert_model_name_or_path)
    # When path is a folder, it use get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs)
    # to get a config.json file
    config = AutoConfig.from_pretrained(bert_model_name_or_path)
    return config
