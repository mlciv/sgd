# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : modelling_dstc8baseline.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict active_intent, requested_slots, slot goals
# --------------------------------------------------------------------

import logging
import collections
import re
import os
import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from modules.core.encoder_utils import EncoderUtils
from modules.core.schemadst_configuration import SchemaDSTConfig
from modules.modelling_toptransformer import TopTransformerModel
from modules.dstc8baseline_output_interface import DSTC8BaselineOutputInterface
from modules.schema_embedding_generator import SchemaInputFeatures
from src import utils_schema
from utils import (
    torch_ext,
    data_utils,
    schema
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)

_NL_SEPARATOR = "|||"
# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class DSTC8BaselineTopTransModel(TopTransformerModel, DSTC8BaselineOutputInterface):
    """
    Main entry of Classifier Model
    """

    config_class = SchemaDSTConfig
    base_model_prefix = ""
    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config=None, args=None, encoder=None):
        super(DSTC8BaselineTopTransModel, self).__init__(config=config, args=args)

    @classmethod
    def _encode_schema(cls, tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training):
        """Directly read from precomputed schema embedding."""
        # scala_mix is not used here, since we only cache the last layer of token schema embedding
        # batch_size, max_intent_num, max_seq_length, dim
        cached_schema_tokens = features[SchemaInputFeatures.get_tok_embedding_tensor_name(schema_type)]
        # batch_size, max_intent_num, max_seq_length
        cached_schema_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(schema_type)]
        if is_training:
            cached_schema_tokens = dropout_layer(cached_schema_tokens)
        return None, cached_schema_tokens, cached_schema_mask
