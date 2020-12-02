# Time-stamp: <2020-05-23>
# --------------------------------------------------------------------
# File Name          : encoder_configuration.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Configuration for training or loading model
# --------------------------------------------------------------------

import sys
from transformers.configuration_utils import PretrainedConfig
from transformers.configuration_roberta import RobertaConfig

import logging

# A encoder configuration, it support lstm, transformer, other
# pretrained transformers
ENCODER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
}


class EncoderConfig(PretrainedConfig):
    """configuration class for AutoEncoding Model, which can be used for
    training and reloading for finetuning
    Using PretrainedConfig will help to load a json config file, or parse the kwargs
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
    then, it will initial the EncoderConfig with those key values in the config dict,
    and merge with other kwargs
    """
    pretrained_config_archive_map = ENCODER_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, **kwargs):
        # **kwargs will be set into the config attributes
        super(EncoderConfig, self).__init__(**kwargs)
        # enc_checkpoint must be specified for loading pretrained encoder
        # All the bert model actually can be pretrained and stored in another
        # folder rather than those stored in S3. It can be empty
        assert "enc_model_type" in kwargs, "missing enc_checkpoint"
        assert "enc_checkpoint" in kwargs, "missing enc_checkpoint"
