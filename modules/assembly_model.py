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

# Now we use the same json config
CLS_PRETRAINED_MODEL_ARCHIVE_MAP = {

}

class AssemblyModel(PreTrainedModel, DSTC8BaselineOutputInterface):
    """
    Main entry of DSTC model, this model is based on official baseline, for cls matching model
    """

    config_class = SchemaDSTConfig
    base_model_prefix = ""
    # see the trick in https://github.com/huggingface/transformers/blob/cae641ff269478f74b5895d36fbc686f7074f2fb/transformers/modeling_utils.py#L457
    # 1.To support the base model part from a model
    # 1.1. should only load the basemodel part of a different deveried model
    # 1.2. the original BertModel has base_model_prefix="", and no attr called base_model_prefix; A derived model will have a base_model_prefix="bert/roberta", and have parameters started with $base_model_prefix
    # Assumption 1: A derived model has an attr called $bas_model_prefix$, also its state_dict has parameter started with $base_model_prefix$. Then it will not meet any of the condition in the code. staret_prefix and model_to_load will not change. But if the source model is not eaxactly the same as the target model, some parameters will show warning.
    # Assumption 2: A base model has no attr or parameters started with base_model_str.
    # 2. When loading, we will check base the target model and source state_dict
    # if the model has attribute called $base_model_prefix$, but state_dict has no $base_model_prefix$(the base mode itself). Then we only load the base model part.
    # 2.1. Target: A derived model, Source: A derived model
    # condition 1 and 2 will failed. no change to source and target
    # 2.2  Target: A derived model, Source: base model part.
    # condition 1 will fail: all state_dict will be used, no prefix will change
    # condition 2 will succeed. only the base_model part in the target model will be fill
    # 2.3  Target: A based model. Source: A base model
    # condition 1 will fail: all state_dict will be used, no prefix will change
    # condition 2 will fail. all target model will be filled
    # 2.4 Target: Base model. Source: Derived model
    # condition 1 will succeed, only the basemodel part of source derived model  will be used
    # condition 2 will fail. No change for the source model part.


    pretrained_model_archieve_map = CLS_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config=None, args=None):
        super(DSTC8BaselineModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        setattr(self, self.base_model_prefix, torch.nn.Sequential())
        self.active_intent_model = None
        self.requested_slots_model = None
        self.categorical_slots_model = None
        self.noncategorical_slots_model = None
        # for non-categorical span value
        if not args.no_cuda:
            self.cuda(args.device)

    def forward(self, features, labels=None):
        """
        given input, output probibilities of each selection
        input: (input1_ids, input2_ids)
        In the sentence pair of Bert, token_type_ids indices to indicate first and second portions of the inputs.
        Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        corresponds to a `sentence B` token
        But in our case, we encode each sentence seperately.
        plan2 is the same, for AE, only input_ids is useful.
        """
        outputs = {}
        active_intent_outputs = self.active_intent_model(features, labels)
        requested_slots_outputs = self.requested_slots_model(features, labels)
        cat_slots_outputs = self.categorical_slots_model(features, labels)
        noncat_slots_outputs = self.noncategorical_slot_model(features, labels)
        outputs.update(active_intent_outputs[0])
        outputs.update(requested_slots_outputs[0])
        outputs.update(cat_slots_outputs[0])
        outputs.update(noncat_slots_outputs[0])
        # when it is dataparallel, the output will keep the tuple, but the content are gathered from different GPUS.
        if labels:
            losses = {}
            losses.update(active_intent_outputs[1])
            losses.update(requested_slots_outputs[1])
            losses.update(cat_slots_outputs[1])
            losses.update(noncat_slots_outputs[1])
            return (outputs, losses)
        else:
            return (outputs, )
