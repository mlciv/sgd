# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : flat_active_intent_fusion.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A baseline model for schema-guided dialogyem given the input,
# to predict active_intent, flatten examples
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
from modules.flat_dst_output_interface import FlatDSTOutputInterface
from modules.fixed_schema_cache import FixedSchemaCacheEncoder
from modules.flat_active_intent_toptrans import FlatActiveIntentTopTransModel
from modules.core.encode_sep_utterance_schema_interface import EncodeSepUttSchemaInterface
from modules.schema_embedding_generator import SchemaInputFeatures
import modules.core.schema_constants as schema_constants
from src import utils_schema
from utils import (
    torch_ext,
    data_utils,
    schema,
    scalar_mix
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)

# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class FlatActiveIntentFusionModel(FlatActiveIntentTopTransModel, FixedSchemaCacheEncoder):
    """
    Main entry of Classifier Model
    """
    config_class = SchemaDSTConfig
    base_model_prefix = ""
    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config=None, args=None, encoder=None, utt_encoder=None, schema_encoder=None):
        super(FlatActiveIntentFusionModel, self).__init__(config=config, args=args, utt_encoder=utt_encoder, schema_encoder=None)
