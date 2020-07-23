# Time-stamp: <2020-06-06>
# --------------------------------------------------------------------
# File Name          : dstc8baseline_output_interface.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A output interface for schema-guided dialogyem given the input,
# to predict active_intent, requested_slots, slot goals
# --------------------------------------------------------------------

import logging
import torch
from src import utils_schema
from utils import (
    torch_ext
)

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
logger = logging.getLogger(__name__)


class DSTC8BaselineOutputInterface(object):
    @classmethod
    def define_loss(cls, features, labels, outputs):
        """Obtain the loss of the model."""
        losses = {}
        if "logit_intent_status" in outputs:
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
            # (batch_size, max_intent_num +1)
            intent_weights = torch_ext.sequence_mask(
                features["intent_num"] + 1,
                maxlen=max_intent_num, device=onehot_intent_labels.device, dtype=torch.float)
            # the weight expect for a (batch_size, x ) tensor
            # real_examples_mask_for_bce = features["is_real_example"].unsqueeze(1).expand_as(intent_logits)
            # https://pytorch.org/docs/stable/nn.functional.html?highlight=binary%20cross%20entropy#torch.nn.functional.binary_cross_entropy_with_logits
            # we split N intent classification into N binary classification
            # A directy way is to use the binary_cross_entropy
            # logger.info("intent_logits:{}, onehot_intent_labels:{}, weight:{}".format(intent_logits, onehot_intent_labels, intent_weights))
            intent_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                intent_logits,
                onehot_intent_labels,
                weight=intent_weights,
                reduction="sum"
            )
            losses["loss_intent"] = intent_loss

        # Requested slots.
        if "logit_req_slot_status" in outputs:
            # Shape: (batch_size, max_num_slots).
            requested_slot_logits = outputs["logit_req_slot_status"]
            # batch_size, num_max_slot
            requested_slot_labels = labels["req_slot_status"].type_as(requested_slot_logits)
            max_num_requested_slots = requested_slot_labels.size()[-1]
            requested_slots_weights = torch_ext.sequence_mask(
                features["req_slot_num"],
                maxlen=max_num_requested_slots, device=requested_slot_logits.device, dtype=torch.float)
            # Sigmoid cross entropy is used because more than one slots can be requested
            # in a single utterance.
            # We use sum because it naturely add a weight for each loss
            requested_slot_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                requested_slot_logits,
                requested_slot_labels,
                weight=requested_slots_weights,
                reduction="sum"
            )
            losses["loss_requested_slot"] = requested_slot_loss

        # Categorical slot
        if "logit_cat_slot_status" in outputs and "logit_cat_slot_value" in outputs:
            # Categorical slot status.
            # Shape: (batch_size, max_num_cat_slots, 3).
            cat_slot_status_logits = outputs["logit_cat_slot_status"]
            # (batch_size, max_num_cat_slots)
            cat_slot_status_labels = labels["cat_slot_status"]
            max_num_cat_slots = cat_slot_status_labels.size()[-1]
            # (batch_size,  max_num_cat_slots)
            cat_weights = torch_ext.sequence_mask(
                features["cat_slot_num"],
                maxlen=max_num_cat_slots, device=cat_slot_status_logits.device, dtype=torch.float32).view(-1)
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
            losses["loss_cat_slot_status"] = cat_slot_status_loss
            losses["loss_cat_slot_value"] = cat_slot_value_loss

        # Non-categorical
        if "logit_noncat_slot_status" in outputs:
            # Non-categorical slot status.
            # Shape: (batch_size, max_num_noncat_slots, 3).
            noncat_slot_status_logits = outputs["logit_noncat_slot_status"]
            noncat_slot_status_labels = labels["noncat_slot_status"]
            max_num_noncat_slots = noncat_slot_status_labels.size()[-1]
            noncat_weights = torch_ext.sequence_mask(
                features["noncat_slot_num"],
                maxlen=max_num_noncat_slots,
                device=noncat_slot_status_logits.device,
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
            losses["loss_noncat_slot_status"] = noncat_slot_status_loss
            losses["loss_span_start"] = span_start_loss
            losses["loss_span_end"] = span_end_loss

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
        # [batch_size, num_intent + 1]
        if "logit_intent_status" in outputs:
            predictions["intent_status"] = torch.argmax(
                outputs["logit_intent_status"], dim=-1)

        # Scores are output for each requested slot.
        if "logit_req_slot_status" in outputs:
            predictions["req_slot_status"] = torch.sigmoid(
                outputs["logit_req_slot_status"])

        # For categorical slots, the status of each slot and the predicted value are
        # output.
        if "logit_cat_slot_status" in outputs:
            predictions["cat_slot_status"] = torch.argmax(
                outputs["logit_cat_slot_status"], dim=-1)

        if "logit_cat_slot_value" in outputs:
            predictions["cat_slot_value"] = torch.argmax(
                outputs["logit_cat_slot_value"], dim=-1)

        # For non-categorical slots, the status of each slot and the indices for
        # spans are output.
        if "noncat_slot_status" in outputs:
            predictions["noncat_slot_status"] = torch.argmax(
                outputs["logit_noncat_slot_status"], dim=-1)

        if "logit_noncat_slot_start" in outputs:
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
            start_idx = torch.arange(0, max_num_tokens, device=total_scores.device).view(1, 1, -1, 1)
            end_idx = torch.arange(0, max_num_tokens, device=total_scores.device).view(1, 1, 1, -1)
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
