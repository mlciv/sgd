# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset reader and tokenization-related utilities for baseline model."""

from __future__ import absolute_import
from __future__ import division

import re
import logging
import json


logger = logging.getLogger(__name__)


def load_dialogues(dialog_json_filepaths):
    """Obtain the list of all dialogues from specified json files."""
    dialogs = []
    for dialog_json_filepath in sorted(dialog_json_filepaths):
        with open(dialog_json_filepath) as f:
            dialogs.extend(json.load(f))
    return dialogs

def normalize_list_length(input_list, target_len, padding_unit):
    """Post truncate or pad the input list in place to be of target length.
    Args:
    input_list: the list whose length will be normalized to `target_len` by post
      truncation or padding.
    target_len: the target length which `input_list` should be.
    padding_unit: when the length of `input_list` is smaller than target_len, we
      append a sequence of `padding_unit`s at the end of the input_list so that
      the length of input_list will be `target_len`.
    """
    if len(input_list) < target_len:
        input_list.extend(
            [padding_unit for _ in range(target_len - len(input_list))])
    elif len(input_list) > target_len:
        del input_list[target_len:]
        assert len(input_list) == target_len

def _tokenize(utterance, bert_tokenizer):
    """Tokenize the utterance using word-piece tokenization used by BERT.
       Args:
       utterance: A string containing the utterance to be tokenized.
       Returns:
       bert_tokens: A list of tokens obtained by word-piece tokenization of the
       utterance.
       alignments: A dict mapping indices of characters corresponding to start
       and end positions of words (not subwords) to corresponding indices in
       bert_tokens list.
       inverse_alignments: A list of size equal to bert_tokens. Each element is a
       tuple containing the index of the starting and inclusive ending
       character of the word corresponding to the subword. This list is used
       during inference to map word-piece indices to spans in the original
       utterance.
       """
    # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
    # direct concatenation of all the tokens in the sequence will be the
    # original string.
    tokens = _naive_tokenize(utterance)
    # Filter out empty tokens and obtain aligned character index for each token.
    bert_tokens = []
    for token in tokens:
        if token.strip():
            subwords = bert_tokenizer.tokenize(token)
            bert_tokens.extend(subwords)
            # The inclusive ending character index corresponding to the word.
    return bert_tokens

def _naive_tokenize(s):
    """Tokenize a string, separating words, spaces and punctuations."""
    # Spaces and punctuation marks are all retained, i.e. direct concatenation
    # of all the tokens in the sequence will be the original string.
    seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
    return seq_tok


def _get_token_char_range(utt_tok):
    """Get starting and end character positions of each token in utt_tok."""
    char_pos = 0
    # List of (start_char_pos, end_char_pos) for each token in utt_tok.
    utt_char_range = []
    for tok in utt_tok:
        start = char_pos
        end = start + len(tok) - 1
        utt_char_range.append((start, end))
        char_pos = end + 1
    return utt_char_range


def _get_token_label(utt_char_range, start_char_pos, exclusive_end_char_pos):
    """Get position of token according to char range of each tokens."""
    end_char_pos = exclusive_end_char_pos - 1
    label_at_boundary = True
    for idx, (start, end) in enumerate(utt_char_range):
        if start <= start_char_pos <= end:
            if start != start_char_pos:
                label_at_boundary = False
            start_tok_pos = idx
        if start <= end_char_pos <= end:
            if end != end_char_pos:
                label_at_boundary = False
            end_tok_pos = idx
    assert start_tok_pos <= end_tok_pos
    return start_tok_pos, end_tok_pos, label_at_boundary


# Modified from run_classifier._truncate_seq_pair in the public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate a seq pair in place so that their total length <= max_length."""
    is_too_long = False
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        is_too_long = True
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return is_too_long
