# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : fixed_schema_cache.py
# Original Author    : jiessie.cao@gmail.com
# Description        : an interface for encoding schema with fixed cache
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

class FixedSchemaCacheEncoder(object):
    """
    Main entry of Classifier Model
    """
    @classmethod
    def _encode_schema(cls, tokenizer, encoder, features, dropout_layer, _scalar_mix, schema_type, is_training):
        """Directly read from precomputed schema embedding."""
        # The cached also inclduing the cls token.
        # scala_mix is not used here, since we only cache the last layer of token schema embedding
        # batch_size, max_intent_num, max_seq_length, dim
        # batch_size, max_slot_num, max_seq_length, dim
        # batch_size, max_slot_num, max_value_num, max_seq_length, dim
        # for tokens , it is encoded with sentence or single sentence, all will be ok
        # because make will include both special token cls, sep tokens, but making those paddings are 0
        cached_schema_tokens = features[SchemaInputFeatures.get_embedding_tensor_name(schema_type)]
        # batch_size, max_intent_num, max_seq_length
        cached_schema_mask = features[SchemaInputFeatures.get_input_mask_tensor_name(schema_type)]
        logger.info("shape for fixed_schema: {}, {}".format(cached_schema_mask.size(), cached_schema_mask.size()))
        size_length = len(list(cached_schema_mask.size()))
        if size_length == 4:
            cls_embeddings = cached_schema_tokens[:, :, 0, :]
        elif size_length == 5:
            cls_embeddings = cached_schema_tokens[:, :, :, 0, :]

        if is_training:
            cached_schema_tokens = dropout_layer(cached_schema_tokens)
        return cls_embeddings, cached_schema_tokens, cached_schema_mask
