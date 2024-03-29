# Time-stamp: <>
# --------------------------------------------------------------------
# File Name          : schema_embedding_geneator.py
# Original Author    : jiessie.cao@gmail.com
# Description        : preprcoess for snt, used for bert like model
# --------------------------------------------------------------------
import os

import collections
import json
import os
import re
import numpy as np
import logging

import torch
import torch.nn as nn
from utils import torch_ext
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from modules.core.encoder_utils import EncoderUtils
from modules.core.schema_input_features import SchemaInputFeatures
from utils import data_utils
import src.utils_schema as utils_schema
import  modules.core.schema_constants as schema_constants

logger = logging.getLogger(__name__)
_NL_SEPARATOR = "|||"
CAT_SLOT_VALUE_SEPARATOR = ":"
SERVICE_INTENT_SLOT_SEPARATOR = "|"

class SchemaEmbeddingGenerator(nn.Module):
    """Generate embeddings different components of a service schema."""
    def __init__(self, tokenizer, enc_model_type, encoder, schema_embedding_dim, max_seq_length, device):
        """Generate the embeddings for a schema's elements.
        Args:
        tokenizer: BERT's wordpiece tokenizer.
        estimator: Estimator object of BERT model.
        max_seq_length: Sequence length used for BERT model.
        """
        super(SchemaEmbeddingGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.enc_model_type = enc_model_type
        self.encoder = encoder
        self.schema_embedding_dim = schema_embedding_dim
        self.max_seq_length = max_seq_length
        self.device = device
        self.cuda(device)

    @staticmethod
    def create_or_load_schema_embedding(args, config, split, processor):
        """
        create or load schema_embedding
        Pay attention : This method should be seperated from this model
        It should be done before loading the pretrained model when evaluating
        Args:
        cache_dir: the dir used to store the task-related global files
        schema_embedding_dir_name: dirname for storing schema embedding
        schema_embedding_file_name: filename of schema embedding
        """
        schema_embedding_dir = "{}_{}".format(args.enc_model_type, config.schema_max_seq_length)
        schema_embedding_dir = os.path.join(
            args.cache_dir, args.task_name,
            schema_embedding_dir)
        if not os.path.exists(schema_embedding_dir):
            os.makedirs(schema_embedding_dir)
        schema_embedding_file = os.path.join(
            schema_embedding_dir,
            ("{}_" + config.schema_embedding_file_name).format(split)
        )
        if os.path.exists(schema_embedding_file):
            with open(schema_embedding_file, "rb") as f:
                schema_data = np.load(f, allow_pickle=True)
        else:
            schemas = processor.get_schemas(args.data_dir, split)
            encoder = EncoderUtils.create_encoder(config)
            tokenizer = EncoderUtils.create_tokenizer(config)
            EncoderUtils.add_special_tokens(tokenizer, encoder, schema_constants.USER_AGENT_SPECIAL_TOKENS)
            with torch.no_grad():
                schema_emb_generator = SchemaEmbeddingGenerator(
                    tokenizer, args.enc_model_type, encoder,
                    config.schema_embedding_dim, config.schema_max_seq_length, args.device)
                if config.schema_embedding_type == "cls":
                    schema_data = schema_emb_generator.save_cls_embeddings(
                        schemas,
                        schema_embedding_file,
                        args.dataset_config)
                elif config.schema_embedding_type == "token":
                    schema_data = schema_emb_generator.save_token_embeddings(
                        schemas,
                        schema_embedding_file,
                        args.dataset_config)
                elif config.schema_embedding_type == "flat_token":
                    schema_data = schema_emb_generator.save_flat_token_embeddings(
                        schemas,
                        schema_embedding_file,
                        args.dataset_config)
                elif config.schema_embedding_type == "feature":
                    schema_data = schema_emb_generator.save_feature_tensors(
                        schemas,
                        schema_embedding_file,
                        args.dataset_config)
                elif config.schema_embedding_type == "seq2_feature":
                    schema_data = schema_emb_generator.save_seq2_feature_tensors(
                        schemas,
                        schema_embedding_file,
                        args.dataset_config)
                elif config.schema_embedding_type == "flat_seq2_feature":
                    schema_data = schema_emb_generator.save_flat_seq2_feature_tensors(
                        schemas,
                        schema_embedding_file,
                        args.dataset_config)
                else:
                    raise NotImplementedError(
                        "config.schema_embedding_type {} is not implemented".format(config.schema_embedding_type))
        # Convert from list of dict to dict of list
        # add without gradeint
        schema_data_dict = collections.defaultdict(list)
        for service in schema_data:
            for key, value in service.items():
                schema_data_dict[key].append(value)

        # if the schema_tensor is a dict, the dataparallel will split by service id for the schema
        # it cause the index error when running on multiple GPUs
        schema_tensors = {}
        for key, array in schema_data_dict.items():
            np_arr = np.asarray(array)
            if "ids" in key or "mask" in key:
                t_type = torch.long
            else:
                t_type = torch.float32
            schema_tensors[key] = torch.tensor(np_arr, dtype=t_type).cpu()
        return schema_tensors

    def _create_feature(self, input_line, schema_type, service_id,
                        intent_or_slot_id, value_id=-1):
        """Create a single InputFeatures instance., if containing  |||, then using sentence pair, otherwise, using single sentence"""
        line = input_line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)

        batch_encodings = self.tokenizer.encode_plus(
            text_a, text_b,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True
        )
        input_ids = batch_encodings["input_ids"]
        input_mask = batch_encodings["attention_mask"]
        input_type_ids = batch_encodings["token_type_ids"]

        #pad_len = self.max_seq_length - len(input_ids)
        #if pad_len > 0:
        #    input_ids = np.append(input_ids, np.zeros(pad_len))
        #    input_mask = np.append(input_mask, np.zeros(pad_len))
        #    input_type_ids = np.append(input_type_ids, np.zeros(pad_len))
        input_ids = input_ids.view(1, -1).long()
        input_mask = input_mask.view(1, -1).long()
        input_type_ids = input_type_ids.view(1, -1).long()

        return SchemaInputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            schema_type=schema_type,
            service_id=service_id,
            intent_or_slot_id=intent_or_slot_id,
            value_id=value_id)

    def _create_seq2_feature(self, input_line, schema_type, service_id, intent_or_slot_id, value_id=-1):
        """
        Create a single InputFeatures instance.
        Here, we mainly prepare for seq2
        For bert, we prepare X[SEP] here, hence, for single sentence encoder, bert model, we need to add extra CLS at the beginning
        For RoBERTa, we prepare [CLS] X [SEP] here.
        """
        line = input_line.strip()

        bert_tokens = data_utils._tokenize(line, self.tokenizer)
        # Construct the tokens, segment mask and valid token mask which will be
        # input to BERT, using the tokens for system utterance (sequence A) and
        # user utterance (sequence B).
        schema_subword = []
        schema_seg = []
        schema_mask = []

        if isinstance(self.tokenizer, RobertaTokenizer):
            schema_subword.append(self.tokenizer.cls_token)
            schema_seg.append(1)
            schema_mask.append(1)

        for subword_idx, subword in enumerate(bert_tokens):
            schema_subword.append(subword)
            schema_seg.append(1)
            schema_mask.append(1)

        schema_subword.append(self.tokenizer.sep_token)
        schema_seg.append(1)
        schema_mask.append(1)
        # convert subwords to ids
        schema_ids = self.tokenizer.convert_tokens_to_ids(schema_subword)

        # Zero-pad up to the BERT input sequence length.
        while len(schema_ids) < self.max_seq_length:
            schema_ids.append(self.tokenizer.pad_token_id)
            schema_seg.append(self.tokenizer.pad_token_type_id)
            schema_mask.append(0)

        return SchemaInputFeatures(
            input_ids=schema_ids,
            input_mask=schema_mask,
            input_type_ids=schema_seg,
            schema_type=schema_type,
            service_id=service_id,
            intent_or_slot_id=intent_or_slot_id,
            value_id=value_id)

    def _get_intents_input_features(self, service_schema):
        """Create features for BERT inference for all intents of a service.
        We use "[service description] ||| [intent name] [intent description]" as an
        intent's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of SchemaInputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        intent_descriptions = {
            i["name"]: i["description"]
            for i in service_schema.schema_json["intents"]
        }

        for intent_id, intent in enumerate(service_schema.intents):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, intent_descriptions[intent]])
            features.append(self._create_feature(
                nl_seq, "intent", service_schema.service_id, intent_id))
        return features

    def _get_intents_input_seq2_features(self, service_schema):
        """Create features for BERT inference for all intents of a service.
        We use "[service description] ||| [intent name] [intent description]" as an
        intent's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of SchemaInputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        intent_descriptions = {
            i["name"]: i["description"]
            for i in service_schema.schema_json["intents"]
        }

        for intent_id, intent in enumerate(service_schema.intents):
            nl_seq = " ".join(
                [service_des, intent, intent_descriptions[intent]])
            desc_only_nl_seq = " ".join([intent_descriptions[intent]])
            service_desc_nl_seq = " ".join([service_des, intent_descriptions[intent]])
            features.append(self._create_seq2_feature(
                nl_seq, "intent", service_schema.service_id, intent_id))
            features.append(self._create_seq2_feature(
                desc_only_nl_seq, "intent_desc_only", service_schema.service_id, intent_id))
            features.append(self._create_seq2_feature(
                service_desc_nl_seq, "intent_service_desc", service_schema.service_id, intent_id))
        return features

    def _get_req_slots_input_features(self, service_schema):
        """Create features for BERT inference for all requested slots of a service.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.

        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }
        features = []
        for slot_id, slot in enumerate(service_schema.slots):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, slot_descriptions[slot]])
            features.append(self._create_feature(
                nl_seq, "req_slot", service_schema.service_id, slot_id))
        return features

    def _get_req_slots_input_seq2_features(self, service_schema):
        """Create features for BERT inference for all requested slots of a service.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.

        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }
        features = []
        for slot_id, slot in enumerate(service_schema.slots):
            nl_seq = " ".join(
                [service_des, slot, slot_descriptions[slot]])
            desc_only_nl_seq = " ".join([slot_descriptions[slot]])
            service_desc_nl_seq = " ".join([service_des, slot_descriptions[slot]])
            features.append(self._create_seq2_feature(
                nl_seq, "req_slot", service_schema.service_id, slot_id))
            features.append(self._create_seq2_feature(
                desc_only_nl_seq, "req_slot_desc_only", service_schema.service_id, slot_id))
            features.append(self._create_seq2_feature(
                service_desc_nl_seq, "req_slot_service_desc", service_schema.service_id, slot_id))
        return features

    def _get_cat_slots_and_values_input_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.categorical_slots):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, slot_descriptions[slot]])
            features.append(self._create_feature(
                nl_seq, "cat_slot",
                service_schema.service_id, slot_id))
            # this is used for joint dstc8 model, no special value added.
            for value_id, value in enumerate(
                    service_schema.get_categorical_slot_values(slot)):
                nl_seq = " ".join([slot_descriptions[slot], _NL_SEPARATOR, value])
                features.append(self._create_feature(
                    nl_seq, "cat_slot_value",
                    service_schema.service_id, slot_id, value_id))
        return features

    def _get_cat_slots_and_values_flat_input_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.categorical_slots):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, slot_descriptions[slot]])
            features.append(self._create_feature(
                nl_seq, "cat_slot",
                service_schema.service_id, slot_id))
            # adding special features here.
            for value_id, value in enumerate(
                    [schema_constants.STR_DONTCARE, schema_constants.STR_UNKNOWN] + service_schema.get_categorical_slot_values(slot)):
                nl_seq = " ".join([slot_descriptions[slot], _NL_SEPARATOR, value])
                features.append(self._create_feature(
                    nl_seq, "cat_slot_value",
                    service_schema.service_id, slot_id, value_id))
        return features

    def _get_cat_slots_and_values_input_flat_seq2_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.categorical_slots):
            nl_seq = " ".join(
                [service_des, slot, slot_descriptions[slot]])
            desc_only_nl_seq = " ".join([slot_descriptions[slot]])
            service_desc_nl_seq = " ".join([service_des, slot_descriptions[slot]])
            features.append(self._create_seq2_feature(
                nl_seq, "cat_slot",
                service_schema.service_id, slot_id))
            features.append(self._create_seq2_feature(
                desc_only_nl_seq, "cat_slot_desc_only",
                service_schema.service_id, slot_id))
            features.append(self._create_seq2_feature(
                service_desc_nl_seq, "cat_slot_service_desc",
                service_schema.service_id, slot_id))
            # here, every value_id is shifted with special_cat_value_offset
            for value_id, value in enumerate(
                    [schema_constants.STR_DONTCARE, schema_constants.STR_UNKNOWN] + service_schema.get_categorical_slot_values(slot)):
                service_slot_desc_value_nl_seq2 = " ".join([service_des, slot, slot_descriptions[slot], CAT_SLOT_VALUE_SEPARATOR, value])
                service_slot_desc_value_nl_seq = " ".join([service_des, slot, slot_descriptions[slot], _NL_SEPARATOR, value])
                service_desc_value_nl_seq2 = " ".join([service_des, slot_descriptions[slot], CAT_SLOT_VALUE_SEPARATOR, value])
                service_desc_value_nl_seq = " ".join([service_des, slot_descriptions[slot], _NL_SEPARATOR, value])
                slot_desc_value_nl_seq2 = " ".join([slot, slot_descriptions[slot], CAT_SLOT_VALUE_SEPARATOR, value])
                slot_desc_value_nl_seq = " ".join([slot, slot_descriptions[slot], _NL_SEPARATOR, value])
                desc_value_nl_seq2 = " ".join([slot_descriptions[slot], CAT_SLOT_VALUE_SEPARATOR, value])
                desc_value_nl_seq = " ".join([slot_descriptions[slot], _NL_SEPARATOR, value])
                slot_value_nl_seq2 = " ".join([slot, CAT_SLOT_VALUE_SEPARATOR, value])
                slot_value_nl_seq = " ".join([slot, _NL_SEPARATOR, value])
                value_nl_seq2 = " ".join([value])
                value_nl_seq = " ".join([value, _NL_SEPARATOR])
                features.append(self._create_seq2_feature(
                    service_slot_desc_value_nl_seq2, "cat_service_slot_desc_value_seq2",
                    service_schema.service_id, slot_id, value_id))
                features.append(self._create_feature(
                    service_slot_desc_value_nl_seq, "cat_service_slot_desc_value_seq",
                    service_schema.service_id, slot_id, value_id))

                features.append(self._create_seq2_feature(
                    service_desc_value_nl_seq2, "cat_service_desc_value_seq2",
                    service_schema.service_id, slot_id, value_id))
                features.append(self._create_feature(
                    service_desc_value_nl_seq, "cat_service_desc_value_seq",
                    service_schema.service_id, slot_id, value_id))

                features.append(self._create_seq2_feature(
                    slot_desc_value_nl_seq2, "cat_slot_desc_value_seq2",
                    service_schema.service_id, slot_id, value_id))
                features.append(self._create_feature(
                    slot_desc_value_nl_seq, "cat_slot_desc_value_seq",
                    service_schema.service_id, slot_id, value_id))

                features.append(self._create_seq2_feature(
                    desc_value_nl_seq2, "cat_desc_value_seq2",
                    service_schema.service_id, slot_id, value_id))
                features.append(self._create_feature(
                    desc_value_nl_seq, "cat_desc_value_seq",
                    service_schema.service_id, slot_id, value_id))

                features.append(self._create_seq2_feature(
                    slot_value_nl_seq2, "cat_slot_value_seq2",
                    service_schema.service_id, slot_id, value_id))

                features.append(self._create_feature(
                    slot_value_nl_seq, "cat_slot_value_seq",
                    service_schema.service_id, slot_id, value_id))

                features.append(self._create_seq2_feature(
                    value_nl_seq2, "cat_value_seq2",
                    service_schema.service_id, slot_id, value_id))
                features.append(self._create_feature(
                    value_nl_seq, "cat_value_seq",
                    service_schema.service_id, slot_id, value_id))
        return features

    def _get_cat_slots_and_values_input_seq2_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.categorical_slots):
            nl_seq = " ".join(
                [service_des, slot, slot_descriptions[slot]])
            desc_only_nl_seq = " ".join([slot_descriptions[slot]])
            features.append(self._create_seq2_feature(
                nl_seq, "cat_slot",
                service_schema.service_id, slot_id))
            features.append(self._create_seq2_feature(
                desc_only_nl_seq, "cat_slot_desc_only",
                service_schema.service_id, slot_id))
            for value_id, value in enumerate(
                    service_schema.get_categorical_slot_values(slot)):
                nl_seq = " ".join([slot, CAT_SLOT_VALUE_SEPARATOR, value])
                features.append(self._create_seq2_feature(
                    nl_seq, "cat_slot_value",
                    service_schema.service_id, slot_id, value_id))
        return features

    def _get_noncat_slots_input_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.non_categorical_slots):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, slot_descriptions[slot]])
            features.append(self._create_feature(
                nl_seq, "noncat_slot",
                service_schema.service_id, slot_id))
        return features

    def _get_noncat_slots_input_seq2_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.non_categorical_slots):
            nl_seq = " ".join(
                [service_des, slot, slot_descriptions[slot]])
            features.append(self._create_seq2_feature(
                nl_seq, "noncat_slot",
                service_schema.service_id, slot_id))
            desc_only_nl_seq = " ".join([slot_descriptions[slot]])
            service_desc_nl_seq = " ".join([service_des, slot_descriptions[slot]])
            features.append(self._create_seq2_feature(
                desc_only_nl_seq, "noncat_slot_desc_only",
                service_schema.service_id, slot_id))
            features.append(self._create_seq2_feature(
                service_desc_nl_seq, "noncat_slot_service_desc",
                service_schema.service_id, slot_id))
        return features

    def _get_goal_slots_and_values_flat_input_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        cat_slot_features = self._get_cat_slots_and_values_flat_input_features(service_schema)
        noncat_slot_features = self._get_noncat_slots_input_features(service_schema)
        return cat_slot_features + noncat_slot_features

    def _get_goal_slots_and_values_input_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        cat_slot_features = self._get_cat_slots_and_values_input_features(service_schema)
        noncat_slot_features = self._get_noncat_slots_input_features(service_schema)
        return cat_slot_features + noncat_slot_features

    def _get_goal_slots_and_values_input_flat_seq2_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        cat_slot_features = self._get_cat_slots_and_values_input_flat_seq2_features(service_schema)
        noncat_slot_features = self._get_noncat_slots_input_seq2_features(service_schema)
        return cat_slot_features + noncat_slot_features

    def _get_goal_slots_and_values_input_seq2_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.
        We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
        We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.
        Args:
        service_schema: A ServiceSchema object containing the schema for the
        corresponding service.
        Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
        """
        cat_slot_features = self._get_cat_slots_and_values_input_seq2_features(service_schema)
        noncat_slot_features = self._get_noncat_slots_input_seq2_features(service_schema)
        return cat_slot_features + noncat_slot_features

    def _get_input_flat_schema_features(self, schemas):
        """Get the input function to compute schema element embeddings.
        Args:
        schemas: A wrapper for all service schemas in the dataset to be embedded.
        Returns:
        The input_fn to be passed to the estimator.
        """
        # Obtain all the features.
        features = []
        for service in schemas.services:
            service_schema = schemas.get_service_schema(service)
            # for intent and req slot, no different with flat or not
            features.extend(self._get_intents_input_features(service_schema))
            features.extend(self._get_req_slots_input_features(service_schema))
            # for cat_value, when it flat, we need two extra value
            features.extend(
                self._get_goal_slots_and_values_flat_input_features(service_schema))
        return features

    def _get_input_schema_features(self, schemas):
        """Get the input function to compute schema element embeddings.
        Args:
        schemas: A wrapper for all service schemas in the dataset to be embedded.
        Returns:
        The input_fn to be passed to the estimator.
        """
        # Obtain all the features.
        features = []
        for service in schemas.services:
            service_schema = schemas.get_service_schema(service)
            features.extend(self._get_intents_input_features(service_schema))
            features.extend(self._get_req_slots_input_features(service_schema))
            features.extend(
                self._get_goal_slots_and_values_input_features(service_schema))
        return features

    def _get_input_schema_seq2_features(self, schemas):
        """Get the input function to compute schema element embeddings.
        Args:
        schemas: A wrapper for all service schemas in the dataset to be embedded.
        Returns:
        The input_fn to be passed to the estimator.
        """
        # Obtain all the features.
        features = []
        for service in schemas.services:
            service_schema = schemas.get_service_schema(service)
            features.extend(self._get_intents_input_seq2_features(service_schema))
            features.extend(self._get_req_slots_input_seq2_features(service_schema))
            features.extend(
                self._get_goal_slots_and_values_input_seq2_features(service_schema))
        return features

    def _get_input_schema_flat_seq2_features(self, schemas):
        """Get the input function to compute schema element embeddings.
        Args:
        schemas: A wrapper for all service schemas in the dataset to be embedded.
        Returns:
        The input_fn to be passed to the estimator.
        """
        # Obtain all the features.
        features = []
        for service in schemas.services:
            service_schema = schemas.get_service_schema(service)
            features.extend(self._get_intents_input_seq2_features(service_schema))
            features.extend(self._get_req_slots_input_seq2_features(service_schema))
            features.extend(
                self._get_goal_slots_and_values_input_flat_seq2_features(service_schema))
        return features

    def _populate_schema_token_embeddings(self, schemas, schema_embeddings):
        """Run the BERT estimator and populate all schema embeddings."""
        features = self._get_input_schema_features(schemas)
        # prepare features into tensor dataset
        completed_services = set()
        # not batch or sampler
        for feature in features:
            if self.enc_model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                token_type_ids = None
            else:
                token_type_ids = feature.input_type_ids.to(self.device)
            output = self.encoder(
                input_ids=feature.input_ids.to(self.device),
                attention_mask=feature.input_mask.to(self.device),
                token_type_ids=token_type_ids)
            service = schemas.get_service_from_id(feature.service_id)
            if service not in completed_services:
                logger.info("Generating token embeddings for service %s.", service)
                completed_services.add(service)
            ids_mat = schema_embeddings[feature.service_id][feature.input_ids_tensor_name]
            emb_mat = schema_embeddings[feature.service_id][feature.tok_embedding_tensor_name]
            mask_mat = schema_embeddings[feature.service_id][feature.input_mask_tensor_name]
            type_mat = schema_embeddings[feature.service_id][feature.input_type_ids_tensor_name]
            # Obtain the whole token encodings
            embedding = output[0][0, :, :].cpu().numpy()
            mask = feature.input_mask.cpu().numpy()
            schema_type = feature.schema_type
            # logger.info("max_seq_length:{}, input_ids:{}, embedding:{}, mask:{}".format(self.max_seq_length, feature.input_ids, embedding.shape, mask.shape))
            if schema_type in ["cat_slot_value"]:
                ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_ids
                emb_mat[feature.intent_or_slot_id, feature.value_id] = embedding
                mask_mat[feature.intent_or_slot_id, feature.value_id] = mask
                type_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_type_ids
            else:
                ids_mat[feature.intent_or_slot_id] = feature.input_ids
                emb_mat[feature.intent_or_slot_id] = embedding
                mask_mat[feature.intent_or_slot_id] = mask
                type_mat[feature.intent_or_slot_id] = feature.input_type_ids

    def _populate_schema_flat_token_embeddings(self, schemas, schema_embeddings):
        """Run the BERT estimator and populate all schema embeddings."""
        features = self._get_input_flat_schema_features(schemas)
        # prepare features into tensor dataset
        completed_services = set()
        # not batch or sampler
        for feature in features:
            if self.enc_model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                token_type_ids = None
            else:
                token_type_ids = feature.input_type_ids.to(self.device)
            output = self.encoder(
                input_ids=feature.input_ids.to(self.device),
                attention_mask=feature.input_mask.to(self.device),
                token_type_ids=token_type_ids)
            service = schemas.get_service_from_id(feature.service_id)
            if service not in completed_services:
                logger.info("Generating flat token embeddings for service %s.", service)
                completed_services.add(service)
            ids_mat = schema_embeddings[feature.service_id][feature.input_ids_tensor_name]
            emb_mat = schema_embeddings[feature.service_id][feature.tok_embedding_tensor_name]
            mask_mat = schema_embeddings[feature.service_id][feature.input_mask_tensor_name]
            type_mat = schema_embeddings[feature.service_id][feature.input_type_ids_tensor_name]
            # Obtain the whole token encodings
            embedding = output[0][0, :, :].cpu().numpy()
            mask = feature.input_mask.cpu().numpy()
            schema_type = feature.schema_type
            # logger.info("max_seq_length:{}, input_ids:{}, embedding:{}, mask:{}".format(self.max_seq_length, feature.input_ids, embedding.shape, mask.shape))
            if schema_type in ["cat_slot_value"]:
                ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_ids
                emb_mat[feature.intent_or_slot_id, feature.value_id] = embedding
                mask_mat[feature.intent_or_slot_id, feature.value_id] = mask
                type_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_type_ids
            else:
                ids_mat[feature.intent_or_slot_id] = feature.input_ids
                emb_mat[feature.intent_or_slot_id] = embedding
                mask_mat[feature.intent_or_slot_id] = mask
                type_mat[feature.intent_or_slot_id] = feature.input_type_ids

    def save_flat_token_embeddings(self, schemas, output_file, dataset_config):
        """Generate schema element embeddings and save it as a numpy file."""
        schema_embs = []
        max_num_intent = dataset_config.max_num_intent
        max_num_cat_slot = dataset_config.max_num_cat_slot
        max_num_noncat_slot = dataset_config.max_num_noncat_slot
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = dataset_config.max_num_value_per_cat_slot
        # the first cat value is 0, all other value id shift by 2
        self.dontcare_value_id = schema_constants.VALUE_DONTCARE_ID
        self.unknown_value_id = schema_constants.VALUE_UNKNOWN_ID
        self.special_cat_value_offset = schema_constants.SPECIAL_CAT_VALUE_OFFSET
        max_aug_num_value = max_num_value + self.special_cat_value_offset
        max_seq_length = self.max_seq_length
        embedding_dim = self.schema_embedding_dim
        for _ in schemas.services:
            # follow the naming for embedding_tensor_name and mask_tensor_name
            schema_embs.append({
                "intent_tok_emb": np.zeros([max_num_intent, max_seq_length, embedding_dim]),
                "intent_input_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_mask": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_type_ids": np.zeros([max_num_intent, max_seq_length]),
                "req_slot_tok_emb": np.zeros([max_num_slot, max_seq_length, embedding_dim]),
                "req_slot_input_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_mask": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_type_ids": np.zeros([max_num_slot, max_seq_length]),
                "cat_slot_tok_emb": np.zeros([max_num_cat_slot, max_seq_length, embedding_dim]),
                "cat_slot_input_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_mask": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_type_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "noncat_slot_tok_emb": np.zeros([max_num_noncat_slot, max_seq_length, embedding_dim]),
                "noncat_slot_input_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_type_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_mask": np.zeros([max_num_noncat_slot, max_seq_length]),
                "cat_slot_value_tok_emb": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length, embedding_dim]),
                "cat_slot_value_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_value_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_value_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
            })
        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_flat_token_embeddings(schemas, schema_embs)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_embs)
        return schema_embs

    def save_token_embeddings(self, schemas, output_file, dataset_config):
        """Generate schema element embeddings and save it as a numpy file."""
        schema_embs = []
        max_num_intent = dataset_config.max_num_intent
        max_num_cat_slot = dataset_config.max_num_cat_slot
        max_num_noncat_slot = dataset_config.max_num_noncat_slot
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = dataset_config.max_num_value_per_cat_slot
        embedding_dim = self.schema_embedding_dim
        for _ in schemas.services:
            # follow the naming for embedding_tensor_name and mask_tensor_name
            schema_embs.append({
                "intent_tok_emb": np.zeros([max_num_intent, self.max_seq_length, embedding_dim]),
                "intent_input_ids": np.zeros([max_num_intent, self.max_seq_length]),
                "intent_input_mask": np.zeros([max_num_intent, self.max_seq_length]),
                "intent_input_type_ids": np.zeros([max_num_intent, self.max_seq_length]),
                "req_slot_tok_emb": np.zeros([max_num_slot, self.max_seq_length, embedding_dim]),
                "req_slot_input_ids": np.zeros([max_num_slot, self.max_seq_length]),
                "req_slot_input_mask": np.zeros([max_num_slot, self.max_seq_length]),
                "req_slot_input_type_ids": np.zeros([max_num_slot, self.max_seq_length]),
                "cat_slot_tok_emb": np.zeros([max_num_cat_slot, self.max_seq_length, embedding_dim]),
                "cat_slot_input_ids": np.zeros([max_num_cat_slot, self.max_seq_length]),
                "cat_slot_input_mask": np.zeros([max_num_cat_slot, self.max_seq_length]),
                "cat_slot_input_type_ids": np.zeros([max_num_cat_slot, self.max_seq_length]),
                "noncat_slot_tok_emb": np.zeros([max_num_noncat_slot, self.max_seq_length, embedding_dim]),
                "noncat_slot_input_ids": np.zeros([max_num_noncat_slot, self.max_seq_length]),
                "noncat_slot_input_type_ids": np.zeros([max_num_noncat_slot, self.max_seq_length]),
                "noncat_slot_input_mask": np.zeros([max_num_noncat_slot, self.max_seq_length]),
                "cat_slot_value_tok_emb": np.zeros([max_num_cat_slot, max_num_value, self.max_seq_length, embedding_dim]),
                "cat_slot_value_input_ids": np.zeros([max_num_cat_slot, max_num_value, self.max_seq_length]),
                "cat_slot_value_input_mask": np.zeros([max_num_cat_slot, max_num_value, self.max_seq_length]),
                "cat_slot_value_input_type_ids": np.zeros([max_num_cat_slot, max_num_value, self.max_seq_length]),
            })
        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_token_embeddings(schemas, schema_embs)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_embs)
        return schema_embs

    def _populate_schema_cls_embeddings(self, schemas, schema_embeddings):
        """Run the BERT estimator and populate all schema embeddings."""
        features = self._get_input_schema_features(schemas)
        # prepare features into tensor dataset
        completed_services = set()
        # not batch or sampler
        for feature in features:
            if self.enc_model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                token_type_ids = None
            else:
                token_type_ids = feature.input_type_ids.to(self.device)
            output = self.encoder(
                input_ids=feature.input_ids.to(self.device),
                attention_mask=feature.input_mask.to(self.device),
                token_type_ids=token_type_ids)
            service = schemas.get_service_from_id(feature.service_id)
            if service not in completed_services:
                logger.info("Generating cls embeddings for service %s.", service)
                completed_services.add(service)
            tensor_name = feature.embedding_tensor_name
            emb_mat = schema_embeddings[feature.service_id][tensor_name]
            # Obtain the encoding of the [CLS] token.
            embedding = [round(float(x), 6) for x in output[0][0, 0, :].cpu().numpy()]
            if tensor_name == SchemaInputFeatures.get_embedding_tensor_name("cat_slot_value"):
                emb_mat[feature.intent_or_slot_id, feature.value_id] = embedding
            else:
                emb_mat[feature.intent_or_slot_id] = embedding

    def save_cls_embeddings(self, schemas, output_file, dataset_config):
        """Generate schema element embeddings and save it as a numpy file."""
        schema_embs = []
        max_num_intent = dataset_config.max_num_intent
        max_num_cat_slot = dataset_config.max_num_cat_slot
        max_num_noncat_slot = dataset_config.max_num_noncat_slot
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = dataset_config.max_num_value_per_cat_slot
        embedding_dim = self.schema_embedding_dim
        for _ in schemas.services:
            schema_embs.append({
                "intent_emb": np.zeros([max_num_intent, embedding_dim]),
                "req_slot_emb": np.zeros([max_num_slot, embedding_dim]),
                "cat_slot_emb": np.zeros([max_num_cat_slot, embedding_dim]),
                "noncat_slot_emb": np.zeros([max_num_noncat_slot, embedding_dim]),
                "cat_slot_value_emb":
                np.zeros([max_num_cat_slot, max_num_value, embedding_dim]),
            })
        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_cls_embeddings(schemas, schema_embs)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_embs)
        logger.info("Schema CLS Embedding saved into {}".format(output_file))
        return schema_embs

    def _populate_schema_feature_tensors(self, schemas, schema_features):
        """Run the BERT estimator and populate all schema embeddings."""
        features = self._get_input_schema_features(schemas)
        # prepare features into tensor dataset
        completed_services = set()
        # not batch or sampler
        for feature in features:
            service = schemas.get_service_from_id(feature.service_id)
            if service not in completed_services:
                logger.info("Generating schema feature for service %s.", service)
                completed_services.add(service)
            schema_type = feature.schema_type
            input_ids_mat = schema_features[feature.service_id][feature.input_ids_tensor_name]
            input_mask_mat = schema_features[feature.service_id][feature.input_mask_tensor_name]
            input_type_ids_mat = schema_features[feature.service_id][feature.input_type_ids_tensor_name]
            if schema_type == "cat_slot_value":
                input_ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_ids
                input_mask_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_mask
                input_type_ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_type_ids
            else:
                input_ids_mat[feature.intent_or_slot_id] = feature.input_ids
                input_mask_mat[feature.intent_or_slot_id] = feature.input_mask
                input_type_ids_mat[feature.intent_or_slot_id] = feature.input_type_ids

    def save_feature_tensors(self, schemas, output_file, dataset_config):
        """Generate schema input tensors and save it as a numpy file."""
        schema_input_features = []
        max_num_intent = dataset_config.max_num_intent
        max_num_cat_slot = dataset_config.max_num_cat_slot
        max_num_noncat_slot = dataset_config.max_num_noncat_slot
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = dataset_config.max_num_value_per_cat_slot
        max_seq_length = self.max_seq_length
        for _ in schemas.services:
            schema_input_features.append({
                "intent_input_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_mask": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_type_ids": np.zeros([max_num_intent, max_seq_length]),
                "req_slot_input_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_mask": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_type_ids": np.zeros([max_num_slot, max_seq_length]),
                "cat_slot_input_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_mask": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_type_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_value_input_ids": np.zeros([max_num_cat_slot, max_num_value, max_seq_length]),
                "cat_slot_value_input_mask": np.zeros([max_num_cat_slot, max_num_value, max_seq_length]),
                "cat_slot_value_input_type_ids": np.zeros([max_num_cat_slot, max_num_value, max_seq_length]),
                "noncat_slot_input_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_mask": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_type_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
            })
        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_feature_tensors(schemas, schema_input_features)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_input_features)
        logger.info("Schema Feature Tensors saved into {}".format(output_file))
        return schema_input_features

    def _populate_schema_flat_seq2_feature_tensors(self, schemas, schema_features):
        """Run the BERT estimator and populate all schema embeddings."""
        features = self._get_input_schema_flat_seq2_features(schemas)
        # prepare features into tensor dataset
        completed_services = set()
        # not batch or sampler
        for feature in features:
            service = schemas.get_service_from_id(feature.service_id)
            if service not in completed_services:
                logger.info("Generating schema feature for service %s.", service)
                completed_services.add(service)
            schema_type = feature.schema_type
            if schema_type.endswith("_value_seq2"):
                input_ids_mat = schema_features[feature.service_id][feature.input_ids_tensor_name]
                input_mask_mat = schema_features[feature.service_id][feature.input_mask_tensor_name]
                input_type_ids_mat = schema_features[feature.service_id][feature.input_type_ids_tensor_name]
                input_ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_ids
                input_mask_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_mask
                input_type_ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_type_ids
            elif schema_type.endswith("_value_seq"):
                # precomputing the cls
                emb_mat = schema_features[feature.service_id][feature.embedding_tensor_name]
                if self.enc_model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                    token_type_ids = None
                else:
                    token_type_ids = feature.input_type_ids.to(self.device)
                output = self.encoder(
                    input_ids=feature.input_ids.to(self.device),
                    attention_mask=feature.input_mask.to(self.device),
                    token_type_ids=token_type_ids)
                # Obtain the encoding of the [CLS] token.
                embedding = [round(float(x), 6) for x in output[0][0, 0, :].cpu().numpy()]
                emb_mat[feature.intent_or_slot_id, feature.value_id] = embedding
            else:
                try:
                    input_ids_mat = schema_features[feature.service_id][feature.input_ids_tensor_name]
                    input_mask_mat = schema_features[feature.service_id][feature.input_mask_tensor_name]
                    input_type_ids_mat = schema_features[feature.service_id][feature.input_type_ids_tensor_name]
                    input_ids_mat[feature.intent_or_slot_id] = feature.input_ids
                    input_mask_mat[feature.intent_or_slot_id] = feature.input_mask
                    input_type_ids_mat[feature.intent_or_slot_id] = feature.input_type_ids
                except:
                    logger.warning("feature is {}".format(feature))

    def _populate_schema_seq2_feature_tensors(self, schemas, schema_features):
        """Run the BERT estimator and populate all schema embeddings."""
        features = self._get_input_schema_seq2_features(schemas)
        # prepare features into tensor dataset
        completed_services = set()
        # not batch or sampler
        for feature in features:
            service = schemas.get_service_from_id(feature.service_id)
            if service not in completed_services:
                logger.info("Generating schema feature for service %s.", service)
                completed_services.add(service)
            schema_type = feature.schema_type
            input_ids_mat = schema_features[feature.service_id][feature.input_ids_tensor_name]
            input_mask_mat = schema_features[feature.service_id][feature.input_mask_tensor_name]
            input_type_ids_mat = schema_features[feature.service_id][feature.input_type_ids_tensor_name]
            if schema_type == "cat_slot_value":
                input_ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_ids
                input_mask_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_mask
                input_type_ids_mat[feature.intent_or_slot_id, feature.value_id] = feature.input_type_ids
            else:
                input_ids_mat[feature.intent_or_slot_id] = feature.input_ids
                input_mask_mat[feature.intent_or_slot_id] = feature.input_mask
                input_type_ids_mat[feature.intent_or_slot_id] = feature.input_type_ids

    def save_seq2_feature_tensors(self, schemas, output_file, dataset_config):
        """Generate schema input tensors and save it as a numpy file."""
        schema_input_features = []
        max_num_intent = dataset_config.max_num_intent
        max_num_cat_slot = dataset_config.max_num_cat_slot
        max_num_noncat_slot = dataset_config.max_num_noncat_slot
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = dataset_config.max_num_value_per_cat_slot
        max_seq_length = self.max_seq_length
        for _ in schemas.services:
            schema_input_features.append({
                "intent_input_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_mask": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_type_ids": np.zeros([max_num_intent, max_seq_length]),
                "req_slot_input_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_mask": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_type_ids": np.zeros([max_num_slot, max_seq_length]),
                "cat_slot_input_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_mask": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_type_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_value_input_ids": np.zeros([max_num_cat_slot, max_num_value, max_seq_length]),
                "cat_slot_value_input_mask": np.zeros([max_num_cat_slot, max_num_value, max_seq_length]),
                "cat_slot_value_input_type_ids": np.zeros([max_num_cat_slot, max_num_value, max_seq_length]),
                "noncat_slot_input_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_mask": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_type_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
            })
        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_seq2_feature_tensors(schemas, schema_input_features)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_input_features)
        logger.info("Schema Feature Seq2 Tensors saved into {}".format(output_file))
        return schema_input_features

    def save_flat_seq2_feature_tensors(self, schemas, output_file, dataset_config):
        """Generate schema input tensors and save it as a numpy file."""
        schema_input_features = []
        max_num_intent = dataset_config.max_num_intent
        max_num_cat_slot = dataset_config.max_num_cat_slot
        max_num_noncat_slot = dataset_config.max_num_noncat_slot
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = dataset_config.max_num_value_per_cat_slot
        # the first cat value is 0, all other value id shift by 2
        self.dontcare_value_id = schema_constants.VALUE_DONTCARE_ID
        self.unknown_value_id = schema_constants.VALUE_UNKNOWN_ID
        self.special_cat_value_offset = schema_constants.SPECIAL_CAT_VALUE_OFFSET
        max_aug_num_value = max_num_value + self.special_cat_value_offset
        max_seq_length = self.max_seq_length
        embedding_dim = self.schema_embedding_dim
        for _ in schemas.services:
            schema_input_features.append({
                # [service_desc intent intent_desc] -> it also support empty service desc, and empty intent desc
                # => [service_dsc intent intent_desc], impact of the service_desc
                # => [intent intent_desc], impact of the intent_desc
                # => [intent], is intent name enough
                # => [intent_desc], is intent desc enough
                # => [intent_index_name], is intent name enough
                "intent_input_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_mask": np.zeros([max_num_intent, max_seq_length]),
                "intent_input_type_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_desc_only_input_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_desc_only_input_mask": np.zeros([max_num_intent, max_seq_length]),
                "intent_desc_only_input_type_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_service_desc_input_ids": np.zeros([max_num_intent, max_seq_length]),
                "intent_service_desc_input_mask": np.zeros([max_num_intent, max_seq_length]),
                "intent_service_desc_input_type_ids": np.zeros([max_num_intent, max_seq_length]),
                # [service_desc slot slot_desc] -> it also support empty service desc, and empty intent desc
                # => [service_dsc slot slot_desc], impact of the service_desc
                # => [slot slot_desc], impact of the intent_desc
                # => [slot], is intent name enough
                # => [slot_desc], is intent desc enough
                # => [slot_index_name], is intent name enough
                "req_slot_input_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_mask": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_input_type_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_desc_only_input_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_desc_only_input_mask": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_desc_only_input_type_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_service_desc_input_ids": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_service_desc_input_mask": np.zeros([max_num_slot, max_seq_length]),
                "req_slot_service_desc_input_type_ids": np.zeros([max_num_slot, max_seq_length]),
                # [service_desc slot slot_desc] -> it also support empty service desc, and empty intent desc
                # => [service_dsc slot slot_desc], impact of the service_desc
                # => [slot slot_desc], impact of the intent_desc
                # => [slot], is intent name enough
                # => [slot_desc], is intent desc enough
                # => [slot_index_name], is intent name enough
                "cat_slot_input_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_mask": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_input_type_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_desc_only_input_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_desc_only_input_mask": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_desc_only_input_type_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_service_desc_input_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_service_desc_input_mask": np.zeros([max_num_cat_slot, max_seq_length]),
                "cat_slot_service_desc_input_type_ids": np.zeros([max_num_cat_slot, max_seq_length]),
                # [service, slot slot_desc value]
                "cat_service_slot_desc_value_seq2_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_service_slot_desc_value_seq2_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_service_slot_desc_value_seq2_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_service_slot_desc_value_seq_emb": np.zeros([max_num_cat_slot, max_aug_num_value, embedding_dim]),
                # [service_desc, slot_desc, value]
                "cat_service_desc_value_seq2_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_service_desc_value_seq2_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_service_desc_value_seq2_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_service_desc_value_seq_emb": np.zeros([max_num_cat_slot, max_aug_num_value, embedding_dim]),
                # [slot slot_desc value]
                "cat_slot_desc_value_seq2_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_desc_value_seq2_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_desc_value_seq2_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_desc_value_seq_emb": np.zeros([max_num_cat_slot, max_aug_num_value, embedding_dim]),
                # [slot_desc value]
                "cat_desc_value_seq2_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_desc_value_seq2_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_desc_value_seq2_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_desc_value_seq_emb": np.zeros([max_num_cat_slot, max_aug_num_value, embedding_dim]),
                # [slot value]
                "cat_slot_value_seq2_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_value_seq2_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_value_seq2_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_slot_value_seq_emb": np.zeros([max_num_cat_slot, max_aug_num_value, embedding_dim]),
                # [value]
                "cat_value_seq2_input_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_value_seq2_input_mask": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_value_seq2_input_type_ids": np.zeros([max_num_cat_slot, max_aug_num_value, max_seq_length]),
                "cat_value_seq_emb": np.zeros([max_num_cat_slot, max_aug_num_value, embedding_dim]),
                # noncat
                "noncat_slot_input_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_mask": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_input_type_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_desc_only_input_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_desc_only_input_mask": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_desc_only_input_type_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_service_desc_input_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_service_desc_input_mask": np.zeros([max_num_noncat_slot, max_seq_length]),
                "noncat_slot_service_desc_input_type_ids": np.zeros([max_num_noncat_slot, max_seq_length]),
            })
        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_flat_seq2_feature_tensors(schemas, schema_input_features)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_input_features)
        logger.info("Schema Feature Flat Seq2 Tensors saved into {}".format(output_file))
        return schema_input_features
