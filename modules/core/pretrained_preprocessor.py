# Time-stamp: <>
# --------------------------------------------------------------------
# File Name          : pretrained_preprocessor.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A base class for all preprocessor for pretraining the graph
# --------------------------------------------------------------------

from transformers.file_utils import cached_path, CONFIG_NAME
class PreTrainedPreprocessor(object):
    """
    load from pretrained preprocessor, to make a graph into specific input for a model
    """
    def __init__(self, **kwargs):
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, preprocessor_name,  *init_inputs, **kwargs):
        # some init
        # Prepare preprocessor initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        if preprocessor_name is not None:
            #init_kwargs = json.load(open(preprocessor_name, encoding="utf-8"))
            #saved_init_inputs = init_kwargs.pop('init_inputs', ())
            #if not init_inputs:
            #    init_inputs = saved_init_inputs
            init_kwargs = {}
        else:
            init_kwargs = {}

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        preprocessor = cls(*init_inputs, **init_kwargs)
        return preprocessor

    def preprocess(self, graph, **kwargs):
        raise NotImplementedError





