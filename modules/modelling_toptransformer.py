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

class TopTransformerModel(PreTrainedModel):
    """
    Main entry of Classifier Model
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
        super(TopTransformerModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
        # one encoder used for both utterance and schema
        self.encoder = EncoderUtils.create_encoder(self.config)
        setattr(self, self.base_model_prefix, self.encoder)
        self.embedding_dim = self.config.schema_embedding_dim
        self.utterance_embedding_dim = self.config.utterance_embedding_dim
        self.utterance_dropout = torch.nn.Dropout(self.config.utterance_dropout)
        self.token_dropout = torch.nn.Dropout(self.config.token_dropout)
        self.intent_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.requested_slots_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.categorical_slots_status_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.categorical_slots_values_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        self.noncategorical_slots_status_utterance_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.utterance_embedding_dim, self.embedding_dim),
            torch.nn.GELU()
        )
        # Project the combined embeddings to obtain logits.
        # for intent, one logits
        self.intent_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )

        # for requested slots, one logits
        self.requested_slots_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )

        # for categorical_slots, 3 logits
        self.categorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 3)
        )

        # for categorical_slot_values,
        self.categorical_slots_values_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 1)
        )

        # for non-categorical_slots, 3 logits
        self.noncategorical_slots_status_final_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim, 3)
        )
        self.noncat_layer_1 = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.GELU()
        )
        self.noncat_layer_2 = nn.Sequential(
            nn.Linear(self.embedding_dim, 2),
        )

        # for non-categorical span value
        if not args.no_cuda:
            self.cuda(args.device)

    def _encode_utterances(self, features, is_training):
        """Encode system and user utterances using BERT."""
        # Optain the embedded representation of system and user utterances in the
        # turn and the corresponding token level representations.
        output = self.encoder(
            input_ids=features["utt"],
            attention_mask=features["utt_mask"],
            token_type_ids=features["utt_seg"])

        encoded_utterance = output[0][:, 0, :]
        encoded_tokens = output[0]
        # Apply dropout in training mode.
        if is_training:
            encoded_utterance = self.utterance_dropout(encoded_utterance)
            encoded_tokens = self.utterance_dropout(encoded_tokens)
        return encoded_utterance, encoded_tokens

    def _get_logits(self, element_embeddings, _encoded_utterance, utterance_proj, final_proj):
        """Get logits for elements by conditioning on utterance embedding.
        Args:
        element_embeddings: A tensor of shape (batch_size, num_elements,
        embedding_dim).
        num_classes: An int containing the number of classes for which logits are
        to be generated.
        Returns:
        A tensor of shape (batch_size, num_elements, num_classes) containing the
        logits.
        """
        _, num_elements, _ = element_embeddings.size()
        # Project the utterance embeddings.
        utterance_embedding = utterance_proj(_encoded_utterance)
        # logger.info("element_embeddings:{}, utterance_embeddings:{}".format(element_embeddings.size(), utterance_embedding.size()))
        # Combine the utterance and element embeddings.
        repeat_utterance_embeddings = utterance_embedding.unsqueeze(1).expand(-1, num_elements, -1)
        utterance_element_emb = torch.cat((repeat_utterance_embeddings, element_embeddings), dim=2)
        return final_proj(utterance_element_emb)

    def _get_intents(self, features, _encoded_utterance):
        """Obtain logits for intents."""
        # service_id is the index of emb value
        # [service_num, max_intentnum, dim]
        intent_embeddings = features["intent_emb"].index_select(0, features["service_id"])
        # Add a trainable vector for the NONE intent.
        _, max_num_intents, embedding_dim = intent_embeddings.size()
        # init a matrix
        null_intent_embedding = torch.empty(1, 1, embedding_dim, device=self.device)
        torch.nn.init.normal_(null_intent_embedding, std=0.02)
        batch_size = intent_embeddings.size()[0]
        repeated_null_intent_embedding = null_intent_embedding.expand(batch_size, 1, -1)
        intent_embeddings = torch.cat(
            (repeated_null_intent_embedding, intent_embeddings), dim=1)

        logits = self._get_logits(
            intent_embeddings, _encoded_utterance,
            self.intent_utterance_proj, self.intent_final_proj)
        # Shape: (batch_size, max_intents + 1)
        logits = logits.squeeze(-1)
        # Mask out logits for padded intents. 1 is added to account for NONE intent.
        # [batch_size, max_intent+1]
        mask = torch_ext.sequence_mask(
            features["intent_num"] + 1,
            maxlen=max_num_intents + 1, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(logits) * torch.finfo().max
        return torch.where(mask, logits, negative_logits)

    def _get_requested_slots(self, features, _encoded_utterance):
        """Obtain logits for requested slots."""
        slot_embeddings = features["req_slot_emb"].index_select(0, features["service_id"])
        logits = self._get_logits(
            slot_embeddings, _encoded_utterance,
            self.requested_slots_utterance_proj, self.requested_slots_final_proj)
        return torch.squeeze(logits, dim=-1)

    def _get_categorical_slots_goals(self, features, _encoded_utterance):
        """Obtain logits for status and values for categorical slots."""
        # Predict the status of all categorical slots.
        slot_embeddings = features["cat_slot_emb"].index_select(0, features["service_id"])
        status_logits = self._get_logits(slot_embeddings,
                                         _encoded_utterance,
                                         self.categorical_slots_status_utterance_proj,
                                         self.categorical_slots_status_final_proj)
        # Predict the goal value.
        # Shape: (batch_size, max_categorical_slots, max_categorical_values,
        # embedding_dim).
        value_embeddings = features["cat_slot_value_emb"].index_select(0, features["service_id"])
        _, max_num_slots, max_num_values, embedding_dim = (
            value_embeddings.size())
        value_embeddings_reshaped = value_embeddings.view(-1, max_num_slots * max_num_values, embedding_dim)
        value_logits = self._get_logits(value_embeddings_reshaped,
                                        _encoded_utterance,
                                        self.categorical_slots_values_utterance_proj,
                                        self.categorical_slots_values_final_proj)
        # Reshape to obtain the logits for all slots.
        value_logits = value_logits.view(-1, max_num_slots, max_num_values)
        # Mask out logits for padded slots and values because they will be
        # softmaxed.
        mask = torch_ext.sequence_mask(
            features["cat_slot_value_num"],
            maxlen=max_num_values, device=self.device, dtype=torch.bool)
        negative_logits = -0.7 * torch.ones_like(value_logits) * torch.finfo().max
        value_logits = torch.where(mask, value_logits, negative_logits)
        return status_logits, value_logits

    def _get_noncategorical_slots_goals(self, features, _encoded_utterance, _encoded_tokens):
        """Obtain logits for status and slot spans for non-categorical slots."""
        # Predict the status of all non-categorical slots.
        slot_embeddings = features["noncat_slot_emb"].index_select(0, features["service_id"])
        max_num_slots = slot_embeddings.size()[1]
        status_logits = self._get_logits(slot_embeddings,
                                         _encoded_utterance,
                                         self.noncategorical_slots_status_utterance_proj,
                                         self.noncategorical_slots_status_final_proj)

        # Predict the distribution for span indices.
        token_embeddings = _encoded_tokens
        max_num_tokens = token_embeddings.size()[1]
        tiled_token_embeddings = token_embeddings.unsqueeze(1).expand(-1, max_num_slots, -1, -1)
        tiled_slot_embeddings = slot_embeddings.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
        # Shape: (batch_size, max_num_slots, max_num_tokens, 2 * embedding_dim).
        slot_token_embeddings = torch.cat(
            [tiled_slot_embeddings, tiled_token_embeddings], dim=3)

        # Shape: (batch_size, max_num_slots, max_num_tokens, 2)
        span_logits = self.noncat_layer_2(self.noncat_layer_1(slot_token_embeddings))
        # Mask out invalid logits for padded tokens.
        token_mask = features["utt_mask"]  # Shape: (batch_size, max_num_tokens).
        tiled_token_mask = token_mask.unsqueeze(1).unsqueeze(3).expand(-1, max_num_slots, -1, 2)
        negative_logits = -0.7 * torch.ones_like(span_logits) * torch.finfo().max
        #logger.info("span_logits:{}, token_mask: {} , titled_token_mask:{}".format(
        #    span_logits.size(), token_mask.size(), tiled_token_mask.size()))
        span_logits = torch.where(tiled_token_mask.bool(), span_logits, negative_logits)
        # Shape of both tensors: (batch_size, max_num_slots, max_num_tokens).
        span_start_logits = span_logits[:, :, :, 0]
        span_end_logits = span_logits[:, :, :, 1]
        return status_logits, span_start_logits, span_end_logits

class SchemaInputFeatures(object):
    """A single set of features for BERT inference."""
    def __init__(self, input_ids, input_mask, input_type_ids,
                 embedding_tensor_name, service_id, intent_or_slot_id, value_id):
        # The ids in the vocabulary for input tokens.
        self.input_ids = input_ids
        # A boolean mask indicating which tokens in the input_ids are valid.
        self.input_mask = input_mask
        # Denotes the sequence each input token belongs to.
        self.input_type_ids = input_type_ids
        # The name of the embedding tensor corresponding to this example.
        self.embedding_tensor_name = embedding_tensor_name
        # The id of the service corresponding to this example.
        self.service_id = service_id
        # The id of the intent (for intent embeddings) or slot (for slot or slot
        # value embeddings) corresponding to this example.
        self.intent_or_slot_id = intent_or_slot_id
        # The id of the value corresponding to this example. Only set if slot value
        # embeddings are being calculated.
        self.value_id = value_id


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

    def _create_feature(self, input_line, embedding_tensor_name, service_id,
                        intent_or_slot_id, value_id=-1):
        """Create a single InputFeatures instance."""
        seq_length = self.max_seq_length
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
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True
        )
        input_ids = batch_encodings["input_ids"]
        input_mask = batch_encodings["attention_mask"]
        input_type_ids = batch_encodings["token_type_ids"]

        return SchemaInputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            embedding_tensor_name=embedding_tensor_name,
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
                [service_des, _NL_SEPARATOR, intent, intent_descriptions[intent]])
            features.append(self._create_feature(
                nl_seq, "intent_emb", service_schema.service_id, intent_id))
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
                [service_des, _NL_SEPARATOR, slot, slot_descriptions[slot]])
            features.append(self._create_feature(
                nl_seq, "req_slot_emb", service_schema.service_id, slot_id))
        return features

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
        service_des = service_schema.description
        features = []
        slot_descriptions = {
            s["name"]: s["description"] for s in service_schema.schema_json["slots"]
        }

        for slot_id, slot in enumerate(service_schema.non_categorical_slots):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, slot, slot_descriptions[slot]])
            features.append(self._create_feature(nl_seq, "noncat_slot_emb",
                                                 service_schema.service_id, slot_id))

        for slot_id, slot in enumerate(service_schema.categorical_slots):
            nl_seq = " ".join(
                [service_des, _NL_SEPARATOR, slot, slot_descriptions[slot]])
            features.append(self._create_feature(nl_seq, "cat_slot_emb",
                                                 service_schema.service_id, slot_id))
            for value_id, value in enumerate(
                    service_schema.get_categorical_slot_values(slot)):
                nl_seq = " ".join([slot, slot_descriptions[slot], _NL_SEPARATOR, value])
                features.append(self._create_feature(
                    nl_seq, "cat_slot_value_emb", service_schema.service_id, slot_id,
                    value_id))
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

    def _populate_schema_embeddings(self, schemas, schema_embeddings):
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
                logger.info("Generating embeddings for service %s.", service)
                completed_services.add(service)
            tensor_name = feature.embedding_tensor_name
            emb_mat = schema_embeddings[feature.service_id][tensor_name]
            # Obtain the encoding of the [CLS] token.
            embedding = [round(float(x), 6) for x in output[0][0, 0, :].cpu().numpy()]
            if tensor_name == "cat_slot_value_emb":
                emb_mat[feature.intent_or_slot_id, feature.value_id] = embedding
            else:
                emb_mat[feature.intent_or_slot_id] = embedding
