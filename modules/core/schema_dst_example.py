import modules.core.schema_constants as schema_constants

from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
import logging
from utils import schema
import copy
from utils import data_utils

logger = logging.getLogger(__name__)

class BaseBertExample(object):
    """
    input_ids
    input_masks
    input_token_type_ids
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg):
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_seg = input_seg

class ActiveIntentExample(BaseBertExample):
    """
    Exmple for active intent classfication
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg,
                 intent_id,
                 intent_status):
        super(ActiveIntentExample, self).__init__(example_id, service_id, input_ids, input_mask, input_seg)
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_seg = input_seg
        self.intent_id = intent_id
        self.intent_status = intent_status

class RequestedSlotExample(BaseBertExample):
    """
    Example for requested slot classfication
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg,
                 requested_slot_id,
                 requested_slot_status):
        super(RequestedSlotExample, self).__init__(example_id, service_id, input_ids, input_mask, input_seg)
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_token_type_ids = input_seg
        self.requested_slot_id = requested_slot_id
        self.requested_slot_status = requested_slot_status

class CatSlotValueExample(BaseBertExample):
    """
    Example for cat slot classification
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg,
                 cat_slot_id,
                 cat_slot_value_id,
                 cat_slot_value_status):
        super(CatSlotValueExample, self).__init__(example_id, service_id, input_ids, input_mask, input_seg)
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_seg = input_seg
        self.cat_slot_id = cat_slot_id
        self.cat_slot_value_id = cat_slot_value_id
        self.cat_slot_value_status = cat_slot_value_status
        # adding doncare and off two value, change it to binary check

class CatSlotFullStateExample(BaseBertExample):
    """
    Example for cat slot classification
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg,
                 cat_slot_id,
                 cat_slot_value,
                 num_cat_slot_values
    ):
        super(CatSlotFullStateExample, self).__init__(example_id, service_id, input_ids, input_mask, input_seg)
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_seg = input_seg
        self.cat_slot_id = cat_slot_id
        self.cat_slot_value = cat_slot_value
        self.num_cat_slot_values = num_cat_slot_values

class CatSlotExample(BaseBertExample):
    """
    Example for cat slot classification
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg,
                 cat_slot_id,
                 cat_slot_status,
                 cat_slot_value,
                 num_cat_slot_values
    ):
        super(CatSlotExample, self).__init__(example_id, service_id, input_ids, input_mask, input_seg)
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_seg = input_seg
        self.cat_slot_id = cat_slot_id
        self.cat_slot_status = cat_slot_status
        # predict the cat slot status and value for this cat slot value
        self.cat_slot_value = cat_slot_value
        self.num_cat_slot_values = num_cat_slot_values


class NonCatSlotExample(BaseBertExample):
    """
    Example for cat slot classification
    """
    def __init__(self,
                 example_id,
                 service_id,
                 input_ids,
                 input_mask,
                 input_seg,
                 noncat_slot_id,
                 noncat_start_char_idx,
                 noncat_end_char_idx,
                 noncat_slot_status,
                 noncat_slot_value_start,
                 noncat_slot_value_end):
        super(NonCatSlotExample, self).__init__(example_id, service_id, input_ids, input_mask, input_seg)
        self.example_id = example_id
        self.service_id = service_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_seg = input_seg
        self.noncat_slot_id = noncat_slot_id
        self.noncat_start_char_idx = noncat_start_char_idx
        self.noncat_end_char_idx = noncat_end_char_idx
        self.noncat_slot_status = noncat_slot_status
        self.noncat_slot_value_start = noncat_slot_value_start
        self.noncat_slot_value_end = noncat_slot_value_end


class SchemaDSTExample(object):
    """
    An example for training/inference.
    """
    def __init__(self, dataset_config,
                 max_seq_length=schema_constants.DEFAULT_MAX_SEQ_LENGTH,
                 max_schema_seq_length=schema_constants.MAX_SCHEMA_SEQ_LENGTH,
                 service_schema=None, example_id="NONE",
                 service_id="NONE",
                 tokenizer=None,
                 log_data_warnings=False,
                 dial_cxt_length=2):
        """Constructs an SchemaDSTExample.
        Args:
        dataset_config: DataConfig object denoting the config of the dataset.
        max_seq_length: The maximum length of the sequence. Sequences longer than
        this value will be truncated.
        service_schema: A ServiceSchema object wrapping the schema for the service
        corresponding to this example.
        example_id: Unique identifier for the example.
        tokenizer: A tokenizer object that has convert_tokens_to_ids and
        convert_ids_to_tokens methods. It must be non-None when
        log_data_warnings: If True, warnings generted while processing data are
        logged. This is useful for debugging data processing.
        """
        # The corresponding service schme object, which can be used to obatin the intent ans slots
        self.service_schema = service_schema
        # example_id - global_turn_id + service_name
        # global_turn_id = split-dialogue_id-turn_idx
        self.example_id = example_id
        self.service_id = service_id
        # max seq_length for bert sequence encoding
        self._max_seq_length = max_seq_length
        # the schema length, which is in the seq2 of bert encoding
        self._max_schema_seq_length = max_schema_seq_length
        # dial_cxt_length, maximum number of utterances we can use as dialogue history
        self._dial_cxt_length = dial_cxt_length
        # tokenizer used token the utterance
        self._tokenizer = tokenizer
        # whether log data warning
        self._log_data_warnings = log_data_warnings
        # the dataset config, which contains the range of dataset for single or multiple domain
        self._dataset_config = dataset_config
        # prepare for bert input
        # The id of each subword in the vocabulary for BERT.
        self.utterance_ids = [0] * self._max_seq_length
        self.unpadded_utterance_ids = None
        # Denotes the identity of the sequence. Takes values 0 (system utterance)
        # and 1 (user utterance).
        self.utterance_segment = [0] * self._max_seq_length
        self.unpadded_utterance_segment = None
        # Mask which takes the value 0 for padded tokens and 1 otherwise.
        self.utterance_mask = [0] * self._max_seq_length
        self.unpadded_utterance_mask = None


        # Start and inclusive end character indices in the original utterance
        # corresponding to the tokens. This is used to obtain the character indices
        # from the predicted subword indices during inference.
        # For dstc baseline model:
        # NOTE: A positive value indicates the character indices in the user
        # utterance whereas a negative value indicates the character indices in the
        # system utterance. The indices are offset by 1 to prevent ambiguity in the
        # 0 index, which could be in either the user or system utterance by the
        # above convention. Now the 0 index corresponds to padded tokens.
        # For new MRC baseline:
        # all start and end index are positive, shifted by 1
        self.start_char_idx = [0] * self._max_seq_length
        self.unpadded_start_char_idx = [0] * self._max_seq_length
        self.end_char_idx = [0] * self._max_seq_length
        self.unpadded_end_char_idx = [0] * self._max_seq_length

        # Number of categorical slots present in the service.
        self.num_categorical_slots = 0
        # The status of each categorical slot in the service.
        # Each slot has thress slot status: off, dontcare, active
        # off , means no new assignment for this slot, keeping unknown
        # doncare, means no preference for the slot, hence, a special slot value doncare for it
        # active, means this slot will predict a new value and get assigned in the next stage.
        self.categorical_slot_status = [schema_constants.STATUS_OFF] * dataset_config.max_num_cat_slot
        # Number of values taken by each categorical slot. This is to check each availible slot
        self.num_categorical_slot_values = [0] * dataset_config.max_num_cat_slot
        # The index of the correct value for each categorical slot.
        self.categorical_slot_values = [0] * dataset_config.max_num_cat_slot

        # Number of non-categorical slots present in the service.
        self.num_noncategorical_slots = 0
        # The status of each non-categorical slot in the service.
        self.noncategorical_slot_status = [schema_constants.STATUS_OFF] * dataset_config.max_num_noncat_slot
        # The index of the starting subword corresponding to the slot span for a
        # non-categorical slot value.
        self.noncategorical_slot_value_start = [0] * dataset_config.max_num_noncat_slot
        # The index of the ending (inclusive) subword corresponding to the slot span
        # for a non-categorical slot value.
        self.noncategorical_slot_value_end = [0] * dataset_config.max_num_noncat_slot

        # Total number of slots present in the service. All slots are included here
        # since every slot can be requested
        self.num_slots = 0
        # Takes value 1 if the corresponding slot is requested, 0 otherwise.
        self.requested_slot_status = [schema_constants.STATUS_OFF] * (
            dataset_config.max_num_cat_slot + dataset_config.max_num_noncat_slot)

        # Total number of intents present in the service.
        self.num_intents = 0
        # Takes value 1 if the intent is active, 0 otherwise.
        self.intent_status = [schema_constants.STATUS_OFF] * dataset_config.max_num_intent

    def __str__(self):
        """
        return the to string
        """
        return self.__repr__()

    def __repr__(self):
        """
        more rich to string
        """
        summary_dict = self.readable_summary
        return json.dumps(summary_dict, sorted_keys=True)

    @property
    def readable_summary(self):
        """
        Get a readable dict that summarizes the attributes of an SchemaDSTExample.
        """
        seq_length = sum(self.utterance_mask)
        utt_toks = self._tokenizer.convert_ids_to_tokens(
            self.utterance_ids[:seq_length])
        utt_tok_mask_pairs = list(
            zip(utt_toks, self.utterance_segment[:seq_length]))
        active_intents = [
            self.service_schema.get_intent_from_id(idx)
            for idx, s in enumerate(self.intent_status)
            if s == schema_constants.STATUS_ACTIVE
        ]
        if len(active_intents) > 1:
            raise ValueError(
                "Should not have multiple active intents in a single service.")
        active_intent = active_intents[0] if active_intents else ""
        slot_values_in_state = {}
        for idx, status in enumerate(self.categorical_slot_status):
            if status == schema_constants.STATUS_ACTIVE:
                value_id = self.categorical_slot_values[idx]
                cat_slot = self.service_schema.get_categorical_slot_from_id(idx)
                slot_values_in_state[cat_slot] = self.service_schema.get_categorical_slot_value_from_id(
                    idx, value_id)
            elif status == schema_constants.STATUS_DONTCARE:
                slot_values_in_state[cat_slot] = schema_constants.STR_DONTCARE

        for idx, status in enumerate(self.noncategorical_slot_status):
            if status == schema_constants.STATUS_ACTIVE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                start_id = self.noncategorical_slot_value_start[idx]
                end_id = self.noncategorical_slot_value_end[idx]
                # Token list is consisted of the subwords that may start with "##". We
                # remove "##" to reconstruct the original value. Note that it's not a
                # strict restoration of the original string. It's primarily used for
                # debugging.
                # ex. ["san", "j", "##ose"] --> "san jose"
                readable_value = " ".join(utt_toks[start_id:end_id + 1]).replace(" ##", "")
                slot_values_in_state[slot] = readable_value
            elif status == schema_constants.STATUS_DONTCARE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                slot_values_in_state[slot] = schema_constants.STR_DONTCARE

        summary_dict = {
            "utt_tok_mask_pairs": utt_tok_mask_pairs,
            "utt_len": seq_length,
            "num_categorical_slots": self.num_categorical_slots,
            "num_categorical_slot_values": self.num_categorical_slot_values,
            "num_noncategorical_slots": self.num_noncategorical_slots,
            "service_name": self.service_schema.service_name,
            "active_intent": active_intent,
            "slot_values_in_state": slot_values_in_state
        }
        return summary_dict

    def add_dial_history_features(self, utterances):
        """
        utterances : an array of (utterance, tokens, alignements, frames)
        [CLS] X [SEP], which can be directly used for single utterance encoding for bert and roberta
        """
        # In this case, max_seq_length is only counting the seq1 (dialog history part)
        offsets = []
        # Modify lengths of sys & usr utterance so that length of total utt
        # (including [CLS], ) is no more than max_utt_len,  not incluing special tokens for sequence2 here.
        # For Roberta, we need to change this.
        if isinstance(self._tokenizer, RobertaTokenizer):
            special_token_num = 2
        else:
            special_token_num = 1

        max_utt_len = self._max_seq_length - self._max_schema_seq_length - special_token_num
        # logger.info("max_seq_length:{}, max_schema_seq_length:{}".format(
        #    self._max_seq_length, self._max_schema_seq_length))
        # logger.info("utterances:{}".format(utterances))
        # for cls
        current_len = 1
        # adding user and system utterance from the end to the beginning
        total_utterances = len(utterances)
        # the min start turn is -1
        min_start_turn = max(-1, total_utterances - self._dial_cxt_length - 1)
        # last turn of the utterances
        start_turn = total_utterances - 1
        # the last utt, it is not enough to put the whole sentence in, we put part of it starting form this offset
        last_utt_offset = 0
        # last we can at least add one token and one user agent token from the utterance, otherwise, we stop
        while max_utt_len - current_len > 1:
            # adding one user turn, with a user tag
            if start_turn <= min_start_turn:
                break
            # using the tokens after tokenizion of each utt
            utt_len = len(utterances[start_turn][2]) + 1
            if current_len + utt_len > max_utt_len:
                # when only partially fit in(at leat one token)
                start_turn = start_turn - 1
                last_utt_offset = utt_len - (max_utt_len - current_len)
                # after filling the utt from the offset, current_len is max_utt_len
                current_len = max_utt_len
                break
            else:
                # if the current utt can fully fit in.
                current_len = current_len + utt_len
                start_turn = start_turn - 1
                last_utt_offset = 0

        # after the counting, start_turn is the turn right before the added utterances
        # here the charindex is agnostic to the bert encoding, which is just the concatenaing of all snts
        # only add globa_char_offeset to each utterance boundaries.
        global_char_offset = sum([len(utt[1]) for utt in utterances[:start_turn + 1]])
        #logger.info("skip {} turns/ {} with last_utt_offset={}, global_offset = {}, current_len = {}".format(
        #    start_turn + 1, total_utterances, last_utt_offset, global_char_offset, current_len))
        # here we only consider seq 1 in bert
        utt_subword = []
        utt_seg = []
        utt_mask = []
        start_char_idx = []
        end_char_idx = []

        utt_subword.append(self._tokenizer.cls_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for i in range(start_turn + 1, total_utterances):
            # adding utterance one by one
            agent_token, utt_utterance, utt_tokens, utt_alignments, utt_inv_alignments, utt_frames = utterances[i]
            # make sure the agent token added to bert
            utt_subword.append(agent_token)
            utt_seg.append(0)
            utt_mask.append(1)
            start_char_idx.append(0)
            end_char_idx.append(0)
            # add the offset before adding the first subtoken
            offsets.append(len(utt_subword))
            # if it is the first adding utterance
            if i == start_turn + 1:
                utt_offset = last_utt_offset
            else:
                utt_offset = 0
            for subword_idx, subword in enumerate(utt_tokens[utt_offset:]):
                utt_subword.append(subword)
                utt_seg.append(0)
                utt_mask.append(1)
                # for each subword, add its start and end char index in its local utt
                start, end = utt_inv_alignments[subword_idx]
                # transform the local start and end to a global start and end respect to the whole dialog
                # here, we shift the char index by 1
                start_char_idx.append(start + 1 + global_char_offset)
                end_char_idx.append(end + 1 + global_char_offset)

            # only add globa_char_offeset to each utterance boundaries.
            global_char_offset += len(utt_utterance)

        # convert subwords to ids
        utterance_ids = self._tokenizer.convert_tokens_to_ids(utt_subword)
        unpadded_utterance_ids = copy.deepcopy(utterance_ids)
        unpadded_utt_seg = copy.deepcopy(utt_seg)
        unpadded_utt_mask = copy.deepcopy(utt_mask)
        unpadded_start_char_idx = copy.deepcopy(start_char_idx)
        unpadded_end_char_idx = copy.deepcopy(end_char_idx)

        # Zero-pad up to the BERT input sequence length before adding sep
        while len(utterance_ids) < max_utt_len + special_token_num - 1:
            utterance_ids.append(self._tokenizer.pad_token_id)
            utt_seg.append(self._tokenizer.pad_token_type_id)
            utt_mask.append(0)
            start_char_idx.append(0)
            end_char_idx.append(0)

        utterance_ids.append(self._tokenizer.sep_token_id)
        unpadded_utterance_ids.append(self._tokenizer.pad_token_type_id)
        # For Roberta, token_type_ids has an all zeor embedding layer,
        # hence, token_type_ids are ignored in Roberta
        # https://github.com/huggingface/transformers/issues/1114
        # https://github.com/huggingface/transformers/blob/54f9fbeff822ec0547fd23d0338654456925f6b7/src/transformers/tokenization_bert.py#L292
        # for bert,token_type_id for first sep is 0
        utt_seg.append(0)
        unpadded_utt_seg.append(0)
        utt_mask.append(1)
        unpadded_utt_mask.append(1)
        start_char_idx.append(0)
        unpadded_start_char_idx.append(0)
        end_char_idx.append(0)
        unpadded_end_char_idx.append(0)

        # after padding , all utt_features are in the same length, but we use the attention mask to mask those paddings
        self.utterance_ids = utterance_ids
        self.utterance_segment = utt_seg
        self.utterance_mask = utt_mask
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx

        # unpadded ids are used for singlete instance construction, which is no need for prepadded.
        self.unpadded_utterance_ids = unpadded_utterance_ids
        self.unpadded_utterance_segment = unpadded_utt_seg
        self.unpadded_utterance_mask = unpadded_utt_mask
        self.unpadded_start_char_idx = unpadded_start_char_idx
        self.unpadded_end_cahr_idx = unpadded_end_char_idx

        return start_turn + 1, last_utt_offset, offsets

    def add_utterance_features(self, system_tokens, system_inv_alignments,
                               user_tokens, user_inv_alignments):
        """Add utterance related features input to bert.  Note: this method
        modifies the system tokens and user_tokens in place to make
        their total length <= the maximum input length for BERT model.

        Args:
        system_tokens: a list of strings which represents system
        utterance.
        system_inv_alignments: a list of tuples which
        denotes the start and end charater of the tpken that a bert
        token originates from in the original system utterance.
        user_tokens: a list of strings which represents user
        utterance.
        user_inv_alignments: a list of tuples which
        denotes the start and end charater of the token that a bert
        token originates from in the original user utterance.
        return offsets for all the utterances
        """
        # Make user-system utterance input (in BERT format)
        # Input sequence length for utterance BERT encoder
        max_utt_len = self._max_seq_length
        # Modify lengths of sys & usr utterance so that length of total utt
        # (including [CLS], [SEP], [SEP]) is no more than max_utt_len
        # For Roberta, we need to change this.
        if isinstance(self._tokenizer, RobertaTokenizer):
            special_token_num = 4
        else:
            special_token_num = 3
        is_too_long = data_utils.truncate_seq_pair(system_tokens, user_tokens, max_utt_len - special_token_num)
        if is_too_long and self._log_data_warnings:
            logger.info("Utterance sequence truncated in example id - %s.",
                        self.example_id)

        # Construct the tokens, segment mask and valid token mask which will be
        # input to BERT, using the tokens for system utterance (sequence A) and
        # user utterance (sequence B).
        utt_subword = []
        utt_seg = []
        utt_mask = []
        start_char_idx = []
        end_char_idx = []

        utt_subword.append(self._tokenizer.cls_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(system_tokens):
            utt_subword.append(subword)
            utt_seg.append(0)
            utt_mask.append(1)
            st, en = system_inv_alignments[subword_idx]
            start_char_idx.append(-(st + 1))
            end_char_idx.append(-(en + 1))

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        if isinstance(self._tokenizer, RobertaTokenizer):
            utt_subword.append(self._tokenizer.cls_token)
            utt_seg.append(1)
            utt_mask.append(1)
            start_char_idx.append(0)
            end_char_idx.append(0)

        for subword_idx, subword in enumerate(user_tokens):
            utt_subword.append(subword)
            utt_seg.append(1)
            utt_mask.append(1)
            st, en = user_inv_alignments[subword_idx]
            start_char_idx.append(st + 1)
            end_char_idx.append(en + 1)

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(1)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)
        # convert subwords to ids
        utterance_ids = self._tokenizer.convert_tokens_to_ids(utt_subword)

        unpadded_utterance_ids = copy.deepcopy(utterance_ids)
        unpadded_utt_seg = copy.deepcopy(utt_seg)
        unpadded_utt_mask = copy.deepcopy(utt_mask)
        unpadded_start_char_idx = copy.deepcopy(start_char_idx)
        unpadded_end_char_idx = copy.deepcopy(end_char_idx)
        # Zero-pad up to the BERT input sequence length.
        while len(utterance_ids) < max_utt_len:
            utterance_ids.append(self._tokenizer.pad_token_id)
            utt_seg.append(self._tokenizer.pad_token_type_id)
            utt_mask.append(0)
            start_char_idx.append(0)
            end_char_idx.append(0)

        self.utterance_ids = utterance_ids
        self.utterance_segment = utt_seg
        self.utterance_mask = utt_mask
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx

        self.unpadded_utterance_ids = unpadded_utterance_ids
        self.unpadded_utterance_segment = unpadded_utt_seg
        self.unpadded_utterance_mask = unpadded_utt_mask
        self.unpadded_start_char_idx = unpadded_start_char_idx
        self.unpadded_end_cahr_idx = unpadded_end_char_idx


    def make_copy_with_utterance_features(self):
        """
        Make a copy of the current example with utterance features.
        """
        new_example = SchemaDSTExample(
            dataset_config=self._dataset_config,
            max_seq_length=self._max_seq_length,
            max_schema_seq_length=self._max_schema_seq_length,
            service_schema=self.service_schema,
            example_id=self.example_id,
            service_id=self.service_id,
            tokenizer=self._tokenizer,
            log_data_warnings=self._log_data_warnings)

        new_example.utterance_ids = list(self.utterance_ids)
        new_example.utterance_segment = list(self.utterance_segment)
        new_example.utterance_mask = list(self.utterance_mask)
        new_example.start_char_idx = list(self.start_char_idx)
        new_example.end_char_idx = list(self.end_char_idx)
        new_example.unpadded_utterance_ids = list(self.unpadded_utterance_ids)
        new_example.unpadded_utterance_segment = list(self.unpadded_utterance_segment)
        new_example.unpadded_utterance_mask = list(self.unpadded_utterance_mask)
        new_example.unpadded_start_char_idx = list(self.unpadded_start_char_idx)
        new_example.unpadded_end_char_idx = list(self.unpadded_end_char_idx)

        return new_example

    def add_categorical_slot_values_full_state(self, state):
        """
        For every state update, Add features and labels for categorical slots.
        """
        categorical_slots = self.service_schema.categorical_slots
        cat_slot_all_value_examples = []
        for slot_idx, slot in enumerate(categorical_slots):
            values = state.get(slot, [])
            # Add categorical slot value features.
            slot_values = self.service_schema.get_categorical_slot_values(slot)
            if not values:
                for v in slot_values:
                    # skip doncare and off
                    v_id = self.service_schema.get_categorical_slot_value_id(slot, v) + schema_constants.SPECIAL_CAT_VALUE_OFFSET
                    tmp_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, v_id, schema_constants.STATUS_OFF)
                    cat_slot_all_value_examples.append(tmp_cat_slot_value_example)
                # make doncare value as OFF
                donotcare_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_DONTCARE_ID, schema_constants.STATUS_OFF)
                cat_slot_all_value_examples.append(donotcare_cat_slot_value_example)
                # make the unknown vlaue as ACTIVE
                unknown_cat_slot_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_UNKNOWN_ID, schema_constants.STATUS_ACTIVE)
                cat_slot_all_value_examples.append(unknown_cat_slot_example)
            elif values[0] == schema_constants.STR_DONTCARE:
                # use a spaecial value dontcare
                for v in slot_values:
                    # skip doncare and off
                    v_id = self.service_schema.get_categorical_slot_value_id(slot, v) + schema_constants.SPECIAL_CAT_VALUE_OFFSET
                    tmp_cat_slot_value_example = CatSlotValueExample(
                        self.example_id, self.service_id, self.utterance_ids,
                        self.utterance_mask, self.utterance_segment, slot_idx, v_id, schema_constants.STATUS_OFF)
                    cat_slot_all_value_examples.append(tmp_cat_slot_value_example)
                # make doncare value as OFF
                donotcare_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_DONTCARE_ID, schema_constants.STATUS_ACTIVE)
                cat_slot_all_value_examples.append(donotcare_cat_slot_value_example)
                # make the unchange vlaue as OFF
                unknown_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_UNKNOWN_ID, schema_constants.STATUS_OFF)
                cat_slot_all_value_examples.append(unknown_cat_slot_value_example)
            else:
                # all slot value is off,  except value[0]
                for v in slot_values:
                    # skip doncare and off
                    v_id = self.service_schema.get_categorical_slot_value_id(slot, v) + schema_constants.SPECIAL_CAT_VALUE_OFFSET
                    if v != values[0]:
                        tmp_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, v_id, schema_constants.STATUS_OFF)
                        cat_slot_all_value_examples.append(tmp_cat_slot_value_example)
                    else:
                        tmp_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, v_id, schema_constants.STATUS_ACTIVE)
                        cat_slot_all_value_examples.append(tmp_cat_slot_value_example)
                # make doncare value as OFF
                donotcare_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_DONTCARE_ID, schema_constants.STATUS_OFF)
                cat_slot_all_value_examples.append(donotcare_cat_slot_value_example)
                # make the unchange vlaue as OFF
                unknown_cat_slot_value_example = CatSlotValueExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_UNKNOWN_ID, schema_constants.STATUS_OFF)
                cat_slot_all_value_examples.append(unknown_cat_slot_value_example)

        return cat_slot_all_value_examples

    def add_categorical_slots_full_state(self, state):
        """
        For every state update, Add features and labels for categorical slots.
        """
        categorical_slots = self.service_schema.categorical_slots
        self.num_categorical_slots = len(categorical_slots)
        cat_slot_examples = []
        for slot_idx, slot in enumerate(categorical_slots):
            values = state.get(slot, [])
            # Add categorical slot value features.
            slot_values = self.service_schema.get_categorical_slot_values(slot)
            self.num_categorical_slot_values[slot_idx] = len(slot_values)
            total_slot_value = len(slot_values) + schema_constants.SPECIAL_CAT_VALUE_OFFSET
            if not values:
                cat_slot_example = CatSlotFullStateExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_UNKNOWN_ID, total_slot_value)
                cat_slot_examples.append(cat_slot_example)
            elif values[0] == schema_constants.STR_DONTCARE:
                # use a spaecial value dontcare
                cat_slot_example = CatSlotFullStateExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, schema_constants.VALUE_DONTCARE_ID, total_slot_value)
                cat_slot_examples.append(cat_slot_example)
            else:
                value_id = self.service_schema.get_categorical_slot_value_id(slot, values[0]) + schema_constants.SPECIAL_CAT_VALUE_OFFSET
                cat_slot_example = CatSlotFullStateExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, value_id, len(slot_values))
                cat_slot_examples.append(cat_slot_example)
        return cat_slot_examples

    def add_categorical_slots(self, state_update):
        """
        For every state update, Add features and labels for categorical slots.
        """
        categorical_slots = self.service_schema.categorical_slots
        self.num_categorical_slots = len(categorical_slots)
        cat_slot_examples = []
        for slot_idx, slot in enumerate(categorical_slots):
            values = state_update.get(slot, [])
            # Add categorical slot value features.
            slot_values = self.service_schema.get_categorical_slot_values(slot)
            self.num_categorical_slot_values[slot_idx] = len(slot_values)
            if not values:
                # the status is off, it means no new assignment for the slot, the id has no shift
                self.categorical_slot_status[slot_idx] = schema_constants.STATUS_OFF
                cat_slot_status = schema_constants.STATUS_OFF
                # here, only predict the increments
                cat_slot_example = CatSlotExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, cat_slot_status, 0, len(slot_values))
                cat_slot_examples.append(cat_slot_example)
            elif values[0] == schema_constants.STR_DONTCARE:
                # use a spaecial value dontcare
                self.categorical_slot_status[slot_idx] = schema_constants.STATUS_DONTCARE
                cat_slot_status = schema_constants.STATUS_DONTCARE
                # here, only predict the increments, here we only make the slot value as 0, it will not paticipate the loss
                cat_slot_example = CatSlotExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, cat_slot_status, 0, len(slot_values))
                cat_slot_examples.append(cat_slot_example)
            else:
                self.categorical_slot_status[slot_idx] = schema_constants.STATUS_ACTIVE
                cat_slot_status = schema_constants.STATUS_ACTIVE
                # this value id is startin from 0, no special value ids
                value_id = self.service_schema.get_categorical_slot_value_id(slot, values[0])
                # here it only use the first values
                self.categorical_slot_values[slot_idx] = value_id
                # here, only predict the increments, here we only make the slot value as 0, it will not paticipate the loss
                cat_slot_example = CatSlotExample(self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, cat_slot_status, value_id, len(slot_values))
                cat_slot_examples.append(cat_slot_example)
        return cat_slot_examples

    def add_noncategorical_slots_old(self, state_update, system_span_boundaries, user_span_boundaries):
        """
        Add features for non-categorical slots.
        Here only consider the spans in the last user and system turns
        """
        noncategorical_slots = self.service_schema.non_categorical_slots
        self.num_noncategorical_slots = len(noncategorical_slots)
        for slot_idx, slot in enumerate(noncategorical_slots):
            values = state_update.get(slot, [])
            if not values:
                self.noncategorical_slot_status[slot_idx] = schema_constants.STATUS_OFF
            elif values[0] == schema_constants.STR_DONTCARE:
                self.noncategorical_slot_status[slot_idx] = schema_constants.STATUS_DONTCARE
            else:
                self.noncategorical_slot_status[slot_idx] = schema_constants.STATUS_ACTIVE
                # Add indices of the start and end tokens for the first encountered
                # value. Spans in user utterance are prioritized over the system
                # utterance. If a span is not found, the slot value is ignored.
                if slot in user_span_boundaries:
                    start, end = user_span_boundaries[slot]
                elif slot in system_span_boundaries:
                    start, end = system_span_boundaries[slot]
                else:
                    # A span may not be found because the value was cropped out or because
                    # the value was mentioned earlier in the dialogue. Since this model
                    # only makes use of the last two utterances to predict state updates,
                    # it will fail in such cases.
                    if self._log_data_warnings:
                        logger.info(
                            "Slot values %s not found in user or system utterance in "
                            + "example with id - %s, service_id: %s .",
                            str(values), self.example_id, self.service_id)
                    continue
                self.noncategorical_slot_value_start[slot_idx] = start
                self.noncategorical_slot_value_end[slot_idx] = end

    def add_noncategorical_slots(self, state_update, system_span_boundaries,
                                 user_span_boundaries, utterances, start_turn, start_turn_subtoken_offset, global_subtoken_offsets):
        """
        Add features for non-categorical slots.
        Here only consider the spans in the last user and system turns
        """
        noncategorical_slots = self.service_schema.non_categorical_slots
        self.num_noncategorical_slots = len(noncategorical_slots)
        noncat_slot_examples = []
        for slot_idx, slot in enumerate(noncategorical_slots):
            values = state_update.get(slot, [])
            if not values:
                self.noncategorical_slot_status[slot_idx] = schema_constants.STATUS_OFF
                noncat_slot_status = schema_constants.STATUS_OFF
                start = end = 0
            elif values[0] == schema_constants.STR_DONTCARE:
                self.noncategorical_slot_status[slot_idx] = schema_constants.STATUS_DONTCARE
                noncat_slot_status = schema_constants.STATUS_DONTCARE
                start = end = 0
            else:
                self.noncategorical_slot_status[slot_idx] = schema_constants.STATUS_ACTIVE
                # Add indices of the start and end tokens for the first encountered
                # value. Spans in user utterance are prioritized over the system
                # utterance. If a span is not found, the slot value is ignored.
                if slot in user_span_boundaries:
                    start, end = user_span_boundaries[slot]
                elif slot in system_span_boundaries:
                    start, end = system_span_boundaries[slot]
                else:
                    # A span may not be found because the value was cropped out or because
                    # the value was mentioned earlier in the dialogue. Since this model
                    # only makes use of the last two utterances to predict state updates,
                    # it will fail in such cases.
                    if self._log_data_warnings:
                        logger.info(
                            "Slot values %s not found in user or system utterance in "
                            + "example with id - %s, service_id: %s, try to search on previous utterance, start_turn : %d, user_boundaries :%s, system_boundaries: %s",
                            str(values), self.example_id, self.service_id, start_turn, user_span_boundaries, system_span_boundaries)
                    # try to find them in the longer history, start_turn is inclusive
                    # we serch for each utterance from the close to the furthest
                    if len(utterances) < 2:
                        logger.info("Slot values %s cannot be found, no previous utt in example {}, service_id = {}".format(
                            str(values), self.example_id, self.service_id))
                        continue

                    not_last_two_turn_idx = max(start_turn, len(utterances) - 3)
                    start = 0
                    end = 0
                    found_utt_idx = -1
                    for i in range(not_last_two_turn_idx, start_turn - 1, -1):
                        agent_token, utt_utterance, utt_tokens, utt_alignments, utt_inv_alignments, utt_frames = utterances[i]
                        current_global_subtoken_offset = global_subtoken_offsets[i - start_turn]
                        if i == start_turn:
                            # the first offset may be the partial utt
                            local_char_offset = utt_inv_alignments[start_turn_subtoken_offset][0]
                        else:
                            local_char_offset = 0
                        for v in values:
                            # check v in all the utterances other than user ans system utt
                            found_char_idx = utt_utterance.find(v, local_char_offset, len(utt_utterance))
                            if found_char_idx >= 0:
                                # found the char idx, then find the corrsponding local subword idxs
                                end_char_idx = found_char_idx + len(v) - 1
                                if found_char_idx in utt_alignments and end_char_idx in utt_alignments:
                                    found_start_subtoken_idx = utt_alignments[found_char_idx]
                                    found_end_subtoken_idx = utt_alignments[end_char_idx]
                                    start = current_global_subtoken_offset + found_start_subtoken_idx
                                    end = current_global_subtoken_offset + found_end_subtoken_idx
                                    found_utt_idx = i
                                    # once found, exit
                                    break
                                else:
                                    # end char is not in sep token, maybe a brief, continue to search
                                    continue
                    if start == 0 and end == 0:
                        logger.info("Still not found value {} in previous utt in example {}, service_id = {}".format(
                            str(values), self.example_id, self.service_id))
                        continue
                    else:
                        logger.info("Found value {} in previous utt {} in example {}, service_id = {}".format(
                            str(values), not_last_two_turn_idx - found_utt_idx + 2, self.example_id, self.service_id))
                noncat_slot_status = schema_constants.STATUS_ACTIVE
                self.noncategorical_slot_value_start[slot_idx] = start
                self.noncategorical_slot_value_end[slot_idx] = end

            noncat_slot_example = NonCatSlotExample(
                self.example_id, self.service_id,
                self.utterance_ids, self.utterance_mask, self.utterance_segment,
                slot_idx, self.start_char_idx, self.end_char_idx,
                noncat_slot_status, start, end
            )
            noncat_slot_examples.append(noncat_slot_example)
        return noncat_slot_examples

    def add_requested_slots(self, frame):
        """
        requested_slots can be only slot in the schema definitions
        """
        all_slots = self.service_schema.slots
        self.num_slots = len(all_slots)
        req_slot_examples = []
        for slot_idx, slot in enumerate(all_slots):
            if slot in frame["state"]["requested_slots"]:
                self.requested_slot_status[slot_idx] = schema_constants.STATUS_ACTIVE
                req_slot_status = schema_constants.STATUS_ACTIVE
            else:
                req_slot_status = schema_constants.STATUS_OFF
            req_slot_example = RequestedSlotExample(
                self.example_id, self.service_id,
                self.utterance_ids, self.utterance_mask, self.utterance_segment, slot_idx, req_slot_status)
            req_slot_examples.append(req_slot_example)
        return req_slot_examples

    def add_intents(self, frame):
        """
        active intents for this turn frame examples
        """
        all_intents = self.service_schema.intents
        self.num_intents = len(all_intents)
        active_intent_examples = []
        for intent_idx, intent in enumerate(all_intents):
            if intent == frame["state"]["active_intent"]:
                self.intent_status[intent_idx] = schema_constants.STATUS_ACTIVE
                intent_status = schema_constants.STATUS_ACTIVE
            else:
                intent_status = schema_constants.STATUS_OFF
            active_intent_example = ActiveIntentExample(
                self.example_id, self.service_id, self.utterance_ids, self.utterance_mask, self.utterance_segment, intent_idx, intent_status)
            active_intent_examples.append(active_intent_example)
        return active_intent_examples

#    def create_context_schema_features(self, nl_seq):
#        """
#        create seq2 schema feature first, then concatenate it with unpadded untterance features.
#        """
#        schema_ids, schema_seg, schema_mask = self.create_schema_seq2_features(nl_seq)
#        paired_input_ids = []
#        paired_input_ids.extend(self.unpadded_utterance_ids)
#        paired_input_ids.extend(schema_ids)
#        paired_mask = []
#        paired_mask.extend(self.unpadded_utterance_mask)
#        paired_mask.extend(schema_mask)
#        paired_seg = []
#        paired_seg.extend(self.unpadded_utterance_segment)
#        paired_seg.extend(schema_seg)
#
#        # Zero-pad up to the BERT input sequence length.
#        while len(paired_input_ids) < self._max_seq_length:
#            paired_input_ids.append(self._tokenizer.pad_token_id)
#            paired_seg.append(self._tokenizer.pad_token_type_id)
#            paired_mask.append(0)
#
#        return paired_input_ids, paired_seg, paired_mask
#
#    def create_schema_seq2_features(self, input_line):
#        """
#        given a schema description line, create the features.
#        """
#        line = input_line.strip()
#        bert_tokens = data_utils._tokenize(line, self._tokenizer)
#        # Construct the tokens, segment mask and valid token mask which will be
#        # input to BERT, using the tokens for system utterance (sequence A) and
#        # user utterance (sequence B).
#        schema_subword = []
#        schema_seg = []
#        schema_mask = []
#
#        if isinstance(self._tokenizer, RobertaTokenizer):
#            schema_subword.append(self._tokenizer.cls_token)
#            schema_seg.append(1)
#            schema_mask.append(1)
#
#        for subword_idx, subword in enumerate(bert_tokens):
#            schema_subword.append(subword)
#            schema_seg.append(1)
#            schema_mask.append(1)
#
#        schema_subword.append(self._tokenizer.sep_token)
#        schema_seg.append(1)
#        schema_mask.append(1)
#        # convert subwords to ids
#        schema_ids = self._tokenizer.convert_tokens_to_ids(schema_subword)
#        return schema_ids, schema_seg, schema_mask
