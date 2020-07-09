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

class DSTC8BaselineModel(PreTrainedModel):
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
        super(DSTC8BaselineModel, self).__init__(config=config)
        # config is the configuration for pretrained model
        self.config = config
        self.tokenizer = EncoderUtils.create_tokenizer(self.config)
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
        self.noncat_span_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, 2)
        )

        # for non-categorical span value
        if not args.no_cuda:
            self.cuda(args.device)

    def create_or_load_schema_embedding(self, args, split):
        """
        create or load schema_embedding
        Args:
        cache_dir: the dir used to store the task-related global files
        schema_embedding_dir_name: dirname for storing schema embedding
        schema_embedding_file_name: filename of schema embedding
        # TODO: schema embdding type
        """
        schema_embedding_dir = "{}_{}".format(args.enc_model_type, self.config.schema_max_seq_length)
        schema_embedding_dir = os.path.join(
            args.cache_dir, args.task_name,
            schema_embedding_dir)
        if not os.path.exists(schema_embedding_dir):
            os.makedirs(schema_embedding_dir)
        schema_embedding_file = os.path.join(
            schema_embedding_dir,
            ("{}_" + self.config.schema_embedding_file_name).format(split)
        )
        if os.path.exists(schema_embedding_file):
            with open(schema_embedding_file, "rb") as f:
                schema_data = np.load(f, allow_pickle=True)
        else:
            schemas = utils_schema.SchemaDSTC8Processor.get_schemas(args.data_dir, split)
            with torch.no_grad():
                schema_emb_generator = SchemaEmbeddingGenerator(
                    self.tokenizer, args.enc_model_type, self.encoder,
                    self.embedding_dim, self.config.schema_max_seq_length, self.device)
                schema_data = schema_emb_generator.save_embeddings(
                    schemas,
                    schema_embedding_file,
                    args.dataset_config)
            # Shapes for reference: (all have type tf.float32)
            # "cat_slot_emb": [max_num_cat_slot, hidden_dim]
            # "cat_slot_value_emb": [max_num_cat_slot, max_num_value, hidden_dim]
            # "noncat_slot_emb": [max_num_noncat_slot, hidden_dim]
            # "req_slot_emb": [max_num_total_slot, hidden_dim]
            # "intent_emb": [max_num_intent, hidden_dim]

        # Convert from list of dict to dict of list
        # add without gradeint
        schema_data_dict = collections.defaultdict(list)
        for service in schema_data:
            schema_data_dict["cat_slot_emb"].append(service["cat_slot_emb"])
            schema_data_dict["cat_slot_value_emb"].append(service["cat_slot_value_emb"])
            schema_data_dict["noncat_slot_emb"].append(service["noncat_slot_emb"])
            schema_data_dict["req_slot_emb"].append(service["req_slot_emb"])
            schema_data_dict["intent_emb"].append(service["intent_emb"])

        schema_tensors = {}
        for key, array in schema_data_dict.items():
            schema_tensors[key] = torch.tensor(np.asarray(array, np.float32)).cpu()
        return schema_tensors

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
        span_logits = self.noncat_span_layer(slot_token_embeddings)
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
        is_training = (labels is not None)
        _encoded_utterance, _encoded_tokens = self._encode_utterances(features, is_training)
        outputs = {}
        outputs["logit_intent_status"] = self._get_intents(features, _encoded_utterance)
        outputs["logit_req_slot_status"] = self._get_requested_slots(features, _encoded_utterance)
        cat_slot_status, cat_slot_value = self._get_categorical_slots_goals(features, _encoded_utterance)
        outputs["logit_cat_slot_status"] = cat_slot_status
        outputs["logit_cat_slot_value"] = cat_slot_value
        noncat_slot_status, noncat_span_start, noncat_span_end = (
            self._get_noncategorical_slots_goals(features, _encoded_utterance, _encoded_tokens))
        outputs["logit_noncat_slot_status"] = noncat_slot_status
        outputs["logit_noncat_slot_start"] = noncat_span_start
        outputs["logit_noncat_slot_end"] = noncat_span_end

        if labels:
            losses = self.define_loss(features, labels, outputs)
            return (outputs, losses)
        else:
            return (outputs, )

    def define_loss(self, features, labels, outputs):
        """Obtain the loss of the model."""
        # Intents.
        # Shape: (batch_size, max_num_intents + 1).
        intent_logits = outputs["logit_intent_status"]
        max_intent_num = intent_logits.size()[-1]
        # Shape: (batch_size, max_num_intents).
        intent_labels = labels["intent_status"]
        # Add label corresponding to NONE intent.
        # [batch_size, 1]
        num_active_intents = intent_labels.sum(dim=1).unsqueeze(1)
        # for intent, it only have 0 and 1 two value, if all is 0, then NONE intent is 1.
        none_intent_label = torch.ones_like(num_active_intents) - num_active_intents
        # Shape: (batch_size, max_num_intents + 1).
        onehot_intent_labels = torch.cat([none_intent_label, intent_labels], dim=1).type_as(intent_logits)
        intent_weights = torch_ext.sequence_mask(
            features["intent_num"],
            maxlen=max_intent_num, device=self.device, dtype=torch.float)
        # the weight expect for a (batch_size, x ) tensor
        # real_examples_mask_for_bce = features["is_real_example"].unsqueeze(1).expand_as(intent_logits)
        # https://pytorch.org/docs/stable/nn.functional.html?highlight=binary%20cross%20entropy#torch.nn.functional.binary_cross_entropy_with_logits
        # we split N intent classification into N binary classification
        # A directy way is to use the binary_cross_entropy
        intent_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            intent_logits,
            onehot_intent_labels,
            weight=intent_weights,
            reduction="sum"
        )

        # Requested slots.
        # Shape: (batch_size, max_num_slots).
        requested_slot_logits = outputs["logit_req_slot_status"]
        # batch_size, num_max_slot
        requested_slot_labels = labels["req_slot_status"].type_as(requested_slot_logits)
        max_num_requested_slots = requested_slot_labels.size()[-1]
        requested_slots_weights = torch_ext.sequence_mask(
            features["req_slot_num"],
            maxlen=max_num_requested_slots, device=self.device, dtype=torch.float)
        # Sigmoid cross entropy is used because more than one slots can be requested
        # in a single utterance.
        # We use sum because it naturely add a weight for each loss
        requested_slot_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            requested_slot_logits,
            requested_slot_labels,
            weight=requested_slots_weights,
            reduction="sum"
        )

        # Categorical slot status.
        # Shape: (batch_size, max_num_cat_slots, 3).
        cat_slot_status_logits = outputs["logit_cat_slot_status"]
        # (batch_size, max_num_cat_slots)
        cat_slot_status_labels = labels["cat_slot_status"]
        max_num_cat_slots = cat_slot_status_labels.size()[-1]
        # (batch_size,  max_num_cat_slots)
        cat_weights = torch_ext.sequence_mask(
            features["cat_slot_num"],
            maxlen=max_num_cat_slots, device=self.device, dtype=torch.float32).view(-1)
        # (batch_size x max_num_cat)
        cat_slot_status_losses = torch.nn.functional.cross_entropy(
            cat_slot_status_logits.view(-1, 3),
            cat_slot_status_labels.view(-1).long(), reduction='none')
        cat_slot_status_loss = (cat_slot_status_losses * cat_weights).sum()
        # Categorical slot values.
        # Shape: (batch_size, max_num_cat_slots, max_num_slot_values).
        cat_slot_value_logits = outputs["logit_cat_slot_value"]
        # (batch_size, max_num_cat_slot)
        cat_slot_value_labels = labels["cat_slot_value"]
        max_num_slot_values = cat_slot_value_logits.size()[-1]
        # Zero out losses for categorical slot value when the slot status is not
        # active.
        # (batch_size, max_num_cat)
        cat_loss_weight = (cat_slot_status_labels == utils_schema.STATUS_ACTIVE).type(torch.float32).view(-1)
        cat_slot_value_losses = torch.nn.functional.cross_entropy(
            cat_slot_value_logits.view(-1, max_num_slot_values),
            cat_slot_value_labels.view(-1).long(), reduction='none')

        # batch_size, max_num_cat
        active_cat_value_weights = cat_weights * cat_loss_weight
        cat_slot_value_loss = (cat_slot_value_losses * active_cat_value_weights).sum()
        # Non-categorical slot status.
        # Shape: (batch_size, max_num_noncat_slots, 3).
        noncat_slot_status_logits = outputs["logit_noncat_slot_status"]
        noncat_slot_status_labels = labels["noncat_slot_status"]
        max_num_noncat_slots = noncat_slot_status_labels.size()[-1]
        noncat_weights = torch_ext.sequence_mask(
            features["noncat_slot_num"],
            maxlen=max_num_noncat_slots,
            device=self.device,
            dtype=torch.float32).view(-1)
        # Logits for padded (invalid) values are already masked.
        noncat_slot_status_losses = torch.nn.functional.cross_entropy(
            noncat_slot_status_logits.view(-1, 3),
            noncat_slot_status_labels.view(-1).long(),
            reduction='none'
        )
        noncat_slot_status_loss = (noncat_slot_status_losses * noncat_weights).sum()
        # Non-categorical slot spans.
        # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
        span_start_logits = outputs["logit_noncat_slot_start"]
        span_start_labels = labels["noncat_slot_value_start"]
        max_num_tokens = span_start_logits.size()[-1]
        # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
        span_end_logits = outputs["logit_noncat_slot_end"]
        span_end_labels = labels["noncat_slot_value_end"]
        # Zero out losses for non-categorical slot spans when the slot status is not
        # active.
        # (batch_size, max_num_noncat_slots)
        noncat_loss_weight = (noncat_slot_status_labels == utils_schema.STATUS_ACTIVE).type(torch.float32).view(-1)
        active_noncat_value_weights = noncat_weights * noncat_loss_weight
        span_start_losses = torch.nn.functional.cross_entropy(
            span_start_logits.view(-1, max_num_tokens),
            span_start_labels.view(-1).long(), reduction='none')
        span_end_losses = torch.nn.functional.cross_entropy(
            span_end_logits.view(-1, max_num_tokens),
            span_end_labels.view(-1).long(), reduction='none')
        span_start_loss = (span_start_losses * active_noncat_value_weights).sum()
        span_end_loss = (span_end_losses * active_noncat_value_weights).sum()

        losses = {
            "loss_intent": intent_loss,
            "loss_requested_slot": requested_slot_loss,
            "loss_cat_slot_status": cat_slot_status_loss,
            "loss_cat_slot_value": cat_slot_value_loss,
            "loss_noncat_slot_status": noncat_slot_status_loss,
            "loss_span_start": span_start_loss,
            "loss_span_end": span_end_loss,
        }
        return losses

    def define_predictions(self, features, outputs):
        """Define model predictions."""
        predictions = {
            "example_id": features["example_id"].cpu().numpy(),
            "service_id": features["service_id"].cpu().numpy(),
            "is_real_example": features["is_real_example"].cpu().numpy(),
        }
        # Scores are output for each intent.
        # Note that the intent indices are shifted by 1 to account for NONE intent.
        predictions["intent_status"] = torch.argmax(
            outputs["logit_intent_status"], dim=-1)

        # Scores are output for each requested slot.
        predictions["req_slot_status"] = torch.sigmoid(
            outputs["logit_req_slot_status"])

        # For categorical slots, the status of each slot and the predicted value are
        # output.
        predictions["cat_slot_status"] = torch.argmax(
            outputs["logit_cat_slot_status"], dim=-1)
        predictions["cat_slot_value"] = torch.argmax(
            outputs["logit_cat_slot_value"], dim=-1)

        # For non-categorical slots, the status of each slot and the indices for
        # spans are output.
        predictions["noncat_slot_status"] = torch.argmax(
            outputs["logit_noncat_slot_status"], dim=-1)
        start_scores = torch.nn.functional.softmax(outputs["logit_noncat_slot_start"], dim=-1)
        end_scores = torch.nn.functional.softmax(outputs["logit_noncat_slot_end"], dim=-1)
        _, max_num_slots, max_num_tokens = end_scores.size()
        batch_size = end_scores.size()[0]
        # Find the span with the maximum sum of scores for start and end indices.
        total_scores = (
            start_scores.unsqueeze(3) +
            end_scores.unsqueeze(2))
        # Mask out scores where start_index > end_index.
        # exclusive
        start_idx = torch.arange(0, max_num_tokens, device=self.device).view(1, 1, -1, 1)
        end_idx = torch.arange(0, max_num_tokens, device=self.device).view(1, 1, 1, -1)
        invalid_index_mask = (start_idx > end_idx).expand(batch_size, max_num_slots, -1, -1)
        # logger.info("invalid_index_mask:{}, total_scores:{}".format(invalid_index_mask.size(), total_scores.size()))
        total_scores = torch.where(invalid_index_mask, torch.zeros_like(total_scores), total_scores)
        max_span_index = torch.argmax(total_scores.view(-1, max_num_slots, max_num_tokens**2), dim=-1)
        span_start_index = (max_span_index.float() / max_num_tokens).floor().long()
        span_end_index = torch.fmod(max_span_index.float(), max_num_tokens).floor().long()
        predictions["noncat_slot_start"] = span_start_index
        predictions["noncat_slot_end"] = span_end_index
        # Add inverse alignments.
        predictions["noncat_alignment_start"] = features["noncat_alignment_start"]
        predictions["noncat_alignment_end"] = features["noncat_alignment_end"]
        return predictions


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
        self.encoder= encoder
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

    def save_embeddings(self, schemas, output_file, dataset_config):
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
        self._populate_schema_embeddings(schemas, schema_embs)
        with open(output_file, "wb") as f_s:
            np.save(f_s, schema_embs)
        return schema_embs
