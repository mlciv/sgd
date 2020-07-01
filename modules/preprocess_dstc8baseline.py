# Time-stamp: <>
# --------------------------------------------------------------------
# File Name          : preprocess_snt.py
# Original Author    : jiessie.cao@gmail.com
# Description        : preprcoess for snt, used for bert like model
# --------------------------------------------------------------------


from models.structure.pretraining.pretrained_preprocessor import PreTrainedPreprocessor

from models.structure.pretraining.RobertaDBTokenizer import * 


import logging
from models.structure.utils.mrp_tools import *
logger = logging.getLogger(__name__)

class SNTPreprocessor(PreTrainedPreprocessor):

    def __init__(self, tokenizer_class=RobertaDBTokenizer, config=None, **kwargs):
        super(SNTPreprocessor, self).__init__()
        # for roberta, it needs a vocabfilename in a folder pretrained_model_name_or_path
        self.config = config
        self.tokenizer = tokenizer_class.from_pretrained(config.enc_checkpoint, **kwargs)

    def preprocess(self, graph):
        if "prep_linearize_type" in self.config.__dict__:
            linearize_type = self.config.__dict__["prep_linearize_type"]
            if linearize_type == 'bfs2s':
                snt, count = bfs_linearize_str(graph)
                # TODO, add tokenizer for the sentence
                input_dict = self.tokenizer.encode_plus(snt, max_length=self.config.__dict__["max_length"] if "max_length" in self.config.__dict__ else 256, pad_to_max_length=True)
                # tokenizer snt into tokenized sentence
                graph_input = (input_dict, count)
            elif linearize_type == 'dfs2s':
                snt, count = dfs_linearize_str(graph)
                """
                A Dictionary of shape::
                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }
                """
                input_dict = self.tokenizer.encode_plus(snt, max_length=self.config.__dict__["max_length"] if "max_length" in self.config.__dict__ else 512, pad_to_max_length=True)
                # tokenizer snt into tokenized sentence
                graph_input = (input_dict, count)
            elif linearize_type == 'dfs2br_fair':
                snt, count = dfs_linearize_with_bracket(graph)
                """
                A Dictionary of shape::
                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }
                """
                input_dict = self.tokenizer.encode_plus(snt, max_length=self.config.__dict__["max_length"] if "max_length" in self.config.__dict__ else 512, pad_to_max_length=True)
                # tokenizer snt into tokenized sentence
                graph_input = (input_dict, count)
            elif linearize_type == 'dfs2br':
                snt, count = dfs_linearze_with_bracket(mrp)
                input_dict = self.tokenizer.encode_plus(snt, max_length=self.config.__dict__["max_length"] if "max_length" in self.config.__dict__ else 256, pad_to_max_length=True)
                graph_input = (input_dict, count)
            else:
                raise NotImplementedError("Not supported for {}".format(linearize_type))
        else:
            # using graph encoding
            raise NotImplementedError("Not supported for other transformation with graph")

        return graph_input


