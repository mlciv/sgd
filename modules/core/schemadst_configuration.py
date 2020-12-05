# Time-stamp: <>
# --------------------------------------------------------------------
# File Name          : configuration_classify.py
<<<<<<< HEAD
# Original Author    : cajie@amazon.com
=======
# Original Author    : jiessie.cao@gmail.com
>>>>>>> 4131baf55e48139fdc95ab20c573a529d9982b3d
# Description        : The configuration class for ClassifierModel
# --------------------------------------------------------------------

import sys
import json

from transformers.configuration_utils import PretrainedConfig

# Now we use the same json config
CLS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
}


class SchemaDSTConfig(PretrainedConfig):
    """configuration class for ClassifierModel, which can be used for
    training and reloading
    We seperate those encoder-related config out of the model config,
    to make the configuration is more scalable.
    """
    pretrained_config_archive_map = CLS_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, **kwargs):
        super(SchemaDSTConfig, self).__init__(**kwargs)

    def update_encoder_config(self, encoder_config):
        """
        update the encoder_config key values in the model config
        """
        for key, value in encoder_config.__dict__.items():
            self.__dict__[key] = value
