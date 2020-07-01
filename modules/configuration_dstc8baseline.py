# Time-stamp: <11/27/2019>
# --------------------------------------------------------------------
# File Name          : configuration_ae.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Configuration for training or loading model
# --------------------------------------------------------------------

import sys
from transformers.configuration_utils import PretrainedConfig
from transformers.configuration_roberta import RobertaConfig

# Now we use the same json config
RobertaTree_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-bfs": "",
    "roberta-dfs": "",
    "roberta-dfsb": ""
}

class RobertaTreeConfig(PretrainedConfig):
    """
    configuration class for AutoEncoding Model, which can be used for training and reloading for finetuning
    All RobertaTree, sparseRobertaTree, sparseklKE will use this.
    Arguments:
    input_size: Size of last dimension for FNN layer in AUTOENCODER.
    embed_size: Size of the middle layer which compresss the information in the AUTOENCODER
    """
    pretrained_config_archive_map = RobertaTree_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, **kwargs):
        # **kwargs will be set into the config attributes
        super(RobertaTreeConfig, self).__init__(**kwargs)
        if "enc_checkpoint" in kwargs:
            self.enc_model_type = "roberta"
        else:
            raise ValueError("missing enc_checkpoint")
