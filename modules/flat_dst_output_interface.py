# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : flat_dst_output_interface.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A output interface for schema-guided dialogyem given the input,
# to predict flatten examples for active_intent, requested_slots, slot goals
# --------------------------------------------------------------------

import logging
import torch
import torch.nn.functional as F
from src import utils_schema
import modules.core.schema_constants as schema_constants
from utils import (
    torch_ext
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)


class FlatDSTOutputInterface(object):
    @classmethod
    def define_loss(cls, features, labels, outputs):
        """Obtain the loss of the model."""
        losses = {}
        if "logit_intent_status" in outputs:
            # Intents.
            # Shape: (batch_size,) STATUS_ACTIVE or STATUS_OFF
            intent_logits = outputs["logit_intent_status"]
            # batch_size
            # if no p_active > 0.5, it is NONE
            # argmax(p_active)
            intent_labels = labels["intent_status"]
            intent_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                intent_logits,
                intent_labels.type_as(intent_logits),
                reduction="mean"
            )
            losses["loss_intent"] = intent_loss

        # Requested slots.
        if "logit_req_slot_status" in outputs:
            # Shape: (batch_size, ).
            # all p_active > 0.5 will be selected
            requested_slot_logits = outputs["logit_req_slot_status"]
            # (batch_size)
            requested_slot_labels = labels["req_slot_status"]
            requested_slot_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                requested_slot_logits,
                requested_slot_labels.type_as(requested_slot_logits),
                reduction="mean"
            )
            losses["loss_requested_slot"] = requested_slot_loss

        # Categorical slot
        if "logit_cat_slot_status" in outputs and "logit_cat_slot_value" in outputs:
            # Categorical slot status.
            # Shape: (batch_size, 3).
            cat_slot_status_logits = outputs["logit_cat_slot_status"]
            # (batch_size)
            cat_slot_status_labels = labels["cat_slot_status"]
            cat_slot_status_loss = torch.nn.functional.cross_entropy(
                cat_slot_status_logits,
                cat_slot_status_labels.long(), reduction='sum')

            # Categorical slot values.
            # Shape: (batch_size, max_num_slot_values).
            cat_slot_value_logits = outputs["logit_cat_slot_value"]
            # (batch_size)
            cat_slot_value_labels = labels["cat_slot_value"]
            # Zero out losses for categorical slot value when the slot status is not
            # active.
            # (batch_size)
            cat_loss_weight = (cat_slot_status_labels == schema_constants.STATUS_ACTIVE).type(torch.float32).view(-1)

            # _, dialog_id, turn_id, service_name = [x.rstrip() for x in bytes(features["example_id"][0].tolist()).decode("utf-8").split("-")]
            # logger.info("examples_id:{}, cat_name:{}, cat_slot_value_logits:{}, cat_slot_value_labels:{}".format(
            #    " ".join([dialog_id, turn_id, service_name]), features["cat_slot_id"][0], cat_slot_value_logits.size(), cat_slot_status_labels))
            # (batch_size, max_num_cat)
            cat_slot_value_losses = torch.nn.functional.cross_entropy(
                cat_slot_value_logits,
                cat_slot_value_labels.long(), reduction='none')

            # batch_size, max_num_cat
            cat_slot_value_loss = (cat_slot_value_losses * cat_loss_weight).sum()

            losses["loss_cat_slot_status"] = cat_slot_status_loss
            losses["loss_cat_slot_value"] = cat_slot_value_loss

            argmax_cat_slot_status = torch.argmax(cat_slot_status_logits, dim=-1)
            cat_slot_status_error = (argmax_cat_slot_status != cat_slot_status_labels).sum()

            # every batch is for a single slot
            # (batch_size, max_cat_value_num) -> (batch_size)
            argmax_cat_slot_value = torch.argmax(cat_slot_value_logits, dim=-1)
            # batch_size
            cat_slot_value_error = ((argmax_cat_slot_value != cat_slot_value_labels) * cat_loss_weight).sum()
            losses["error_cat_slot_status"] = cat_slot_status_error
            losses["error_cat_slot_value"] = cat_slot_value_error

        # Categorical slot full state
        if "logit_cat_slot_value_full_state" in outputs:
            # Categorical slot status.
            # Categorical slot values.
            # Shape: (batch_size, max_num_slot_values).
            cat_slot_value_logits = outputs["logit_cat_slot_value_full_state"]
            # (batch_size)
            cat_slot_value_labels = labels["cat_slot_value_full_state"]
            # (batch_size, max_num_cat)
            cat_slot_value_loss = torch.nn.functional.cross_entropy(
                cat_slot_value_logits,
                cat_slot_value_labels.long(), reduction='mean')

            # batch_size, max_num_cat
            losses["loss_cat_slot_value_full_state"] = cat_slot_value_loss

            # every batch is for a single slot
            # (batch_size, max_cat_value_num) -> (batch_size)
            argmax_cat_slot_value = torch.argmax(cat_slot_value_logits, dim=-1)
            # batch_size
            cat_slot_value_error = (argmax_cat_slot_value != cat_slot_value_labels).sum()
            losses["error_cat_slot_value_full_state"] = cat_slot_value_error

        if "logit_cat_slot_value_status" in outputs:
            # Categorical slot status.
            # adding doncare values
            # if no value p_active > 0.5, then this slot status is non-active
            # otherwise, doncare and any value can be the argmax
            # Shape: (batch_size, 1).
            cat_slot_value_status_logits = outputs["logit_cat_slot_value_status"]
            # (batch_size)
            cat_slot_value_status_labels = labels["cat_slot_value_status"]
            cat_slot_value_status_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                cat_slot_value_status_logits,
                cat_slot_value_status_labels.type_as(cat_slot_value_status_logits),
                reduction='mean'
            )
            losses["loss_cat_slot_value"] = cat_slot_value_status_losses

        # Non-categorical
        if "logit_noncat_slot_status" in outputs:
            # Non-categorical slot status.
            # Shape: (batch_size, 3).
            noncat_slot_status_logits = outputs["logit_noncat_slot_status"]
            noncat_slot_status_labels = labels["noncat_slot_status"]
            noncat_slot_status_losses = torch.nn.functional.cross_entropy(
                noncat_slot_status_logits,
                noncat_slot_status_labels.long(),
                reduction='mean'
            )
            # Non-categorical slot spans.
            # Shape: (batch_size, max_num_tokens).
            span_start_logits = outputs["logit_noncat_slot_start"]
            span_start_labels = labels["noncat_slot_value_start"]
            # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).
            span_end_logits = outputs["logit_noncat_slot_end"]
            span_end_labels = labels["noncat_slot_value_end"]
            span_start_losses = torch.nn.functional.cross_entropy(
                span_start_logits,
                span_start_labels.long(), reduction='mean')
            span_end_losses = torch.nn.functional.cross_entropy(
                span_end_logits,
                span_end_labels.long(), reduction='mean')
            losses["loss_noncat_slot_status"] = noncat_slot_status_losses
            losses["loss_span_start"] = span_start_losses
            losses["loss_span_end"] = span_end_losses

        return losses

    @classmethod
    def define_predictions(cls, features, outputs):
        """Define model predictions."""
        predictions = {
            "example_id": features["example_id"].cpu().numpy(),
            "service_id": features["service_id"].cpu().numpy(),
        }

        # Scores are output for each intent.
        # Note that the intent indices are shifted by 1 to account for NONE intent.
        # [batch_size, 1]
        if "logit_intent_status" in outputs:
            predictions["intent_id"] = features["intent_id"].cpu().numpy()
            predictions["intent_status"] = torch.sigmoid(
                outputs["logit_intent_status"])

        # Scores are output for each requested slot.
        if "logit_req_slot_status" in outputs:
            predictions["req_slot_id"] = features["req_slot_id"].cpu().numpy()
            predictions["req_slot_status"] = torch.sigmoid(
                outputs["logit_req_slot_status"])

        # For categorical slots, the status of each slot and the predicted value are
        # output.
        if "logit_cat_slot_status" in outputs:
            predictions["cat_slot_id"] = features["cat_slot_id"].cpu().numpy()
            predictions["cat_slot_status"] = torch.argmax(
                outputs["logit_cat_slot_status"], dim=-1)

        # For categorical slots, the status of each slot and the predicted value are
        # output.
        if "logit_cat_slot_value_full_state" in outputs:
            predictions["cat_slot_id"] = features["cat_slot_id"].cpu().numpy()
            predictions["cat_slot_value_full_state"] = torch.argmax(
                outputs["logit_cat_slot_value_full_state"], dim=-1)

        if "logit_cat_slot_value" in outputs:
            predictions["cat_slot_value"] = torch.argmax(
                outputs["logit_cat_slot_value"], dim=-1)
            ## debug for makring cat_slot_value as groud truth
            #if labels and "cat_slot_value" in labels:
            #    predictions["cat_slot_value"] = labels["cat_slot_value"]
            #else:
            #    predictions["cat_slot_value"] = predicted_cat_slot_value

        if "logit_cat_slot_value_status" in outputs:
            predictions["cat_slot_id"] = features["cat_slot_id"].cpu().numpy()
            predictions["cat_slot_value_id"] = features["cat_slot_value_id"].cpu().numpy()
            predictions["cat_slot_value_status"] = torch.sigmoid(
                outputs["logit_cat_slot_value_status"])

        # For non-categorical slots, the status of each slot and the indices for
        # spans are output.
        if "logit_noncat_slot_status" in outputs:
            predictions["noncat_slot_id"] = features["noncat_slot_id"].cpu().numpy()
            predictions["noncat_slot_status"] = torch.argmax(
                outputs["logit_noncat_slot_status"], dim=-1)

        if "logit_noncat_slot_start" in outputs:
            predictions["noncat_slot_id"] = features["noncat_slot_id"].cpu().numpy()
            # batch_size, max_length
            start_scores = torch.nn.functional.softmax(outputs["logit_noncat_slot_start"], dim=-1)
            # batch_size, max_length
            end_scores = torch.nn.functional.softmax(outputs["logit_noncat_slot_end"], dim=-1)
            batch_size, max_num_tokens = end_scores.size()
            # Find the span with the maximum sum of scores for start and end indices.
            total_scores = (
                start_scores.unsqueeze(2) +
                end_scores.unsqueeze(1))
            # Mask out scores where start_index > end_index.
            # exclusive
            start_idx = torch.arange(0, max_num_tokens, device=total_scores.device).view(1, -1, 1)
            end_idx = torch.arange(0, max_num_tokens, device=total_scores.device).view(1, 1, -1)
            invalid_index_mask = (start_idx > end_idx).expand(batch_size, -1, -1)
            total_scores = torch.where(invalid_index_mask, torch.zeros_like(total_scores), total_scores)
            max_span_index = torch.argmax(total_scores.view(batch_size, max_num_tokens**2), dim=-1)
            span_start_index = (max_span_index.float() / max_num_tokens).floor().long()
            span_end_index = torch.fmod(max_span_index.float(), max_num_tokens).floor().long()
            predictions["noncat_slot_start"] = span_start_index
            predictions["noncat_slot_end"] = span_end_index
            # Add inverse alignments.
            predictions["noncat_alignment_start"] = features["noncat_alignment_start"]
            predictions["noncat_alignment_end"] = features["noncat_alignment_end"]
        return predictions

    @classmethod
    def define_oracle_predictions(cls, features, labels):
        """Define model predictions."""
        predictions = {
            "example_id": features["example_id"].cpu().numpy(),
            "service_id": features["service_id"].cpu().numpy(),
        }

        # Scores are output for each intent.
        # Note that the intent indices are shifted by 1 to account for NONE intent.
        # [batch_size, num_intent + 1]
        if "intent_status" in labels:
            predictions["intent_id"] = features["intent_id"].cpu().numpy()
            predictions["intent_status"] = labels["intent_status"]

        # Scores are output for each requested slot.
        if "req_slot_status" in labels:
            predictions["req_slot_id"] = features["req_slot_id"].cpu().numpy()
            predictions["req_slot_status"] = labels["req_slot_status"]

        # For categorical slots, the status of each slot and the predicted value are
        # output.
        if "cat_slot_status" in labels:
            predictions["cat_slot_id"] = features["cat_slot_id"].cpu().numpy()
            predictions["cat_slot_status"] = labels["cat_slot_status"]

        if "cat_slot_value_full_state" in labels:
            predictions["cat_slot_id"] = features["cat_slot_id"].cpu().numpy()
            predictions["cat_slot_value_full_state"] = labels["cat_slot_value_full_state"]

        if "cat_slot_value" in labels:
            predictions["cat_slot_value"] = labels["cat_slot_value"]

        if "cat_slot_value_status" in labels:
            predictions["cat_slot_id"] = features["cat_slot_id"].cpu().numpy()
            predictions["cat_slot_value_id"] = features["cat_slot_value_id"].cpu().numpy()
            predictions["cat_slot_value_status"] = labels["cat_slot_value_status"]

        # For non-categorical slots, the status of each slot and the indices for
        # spans are output.
        if "noncat_slot_status" in labels:
            predictions["noncat_slot_id"] = features["noncat_slot_id"].cpu().numpy()
            predictions["noncat_slot_status"] = labels["noncat_slot_status"]
            predictions["noncat_slot_start"] = labels["noncat_slot_value_start"]
            predictions["noncat_slot_end"] = labels["noncat_slot_value_end"]
            # Add inverse alignments
            predictions["noncat_alignment_start"] = features["noncat_alignment_start"]
            predictions["noncat_alignment_end"] = features["noncat_alignment_end"]
        return predictions
