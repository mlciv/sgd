# Time-stamp: <>
# --------------------------------------------------------------------
# File Name          : schema_input_features.py
# Original Author    : jiessie.cao@gmail.com
# Description        : preprcoess for snt, used for bert like model
# --------------------------------------------------------------------

class SchemaInputFeatures(object):
    """A single set of features for BERT inference."""
    def __init__(self, input_ids, input_mask, input_type_ids,
                 schema_type, service_id, intent_or_slot_id, value_id):
        # The ids in the vocabulary for input tokens.
        self.input_ids = input_ids
        # A boolean mask indicating which tokens in the input_ids are valid.
        self.input_mask = input_mask
        # Denotes the sequence each input token belongs to.
        self.input_type_ids = input_type_ids
        # schema type : intent, req_slot, cat_slot, noncat_slot, cat_slot_value
        self.schema_type = schema_type
        # The name of the embedding tensor corresponding to this example.
        self.embedding_tensor_name = SchemaInputFeatures.get_embedding_tensor_name(schema_type)
        # The name of the mask tensor corresponding to this example.
        self.input_ids_tensor_name = SchemaInputFeatures.get_input_ids_tensor_name(schema_type)
        # The name of the mask tensor corresponding to this example.
        self.input_mask_tensor_name = SchemaInputFeatures.get_input_mask_tensor_name(schema_type)
        # The name of the mask tensor corresponding to this example.
        self.input_type_ids_tensor_name = SchemaInputFeatures.get_input_type_ids_tensor_name(schema_type)
        # The id of the service corresponding to this example.
        self.service_id = service_id
        # The id of the intent (for intent embeddings) or slot (for slot or slot
        # value embeddings) corresponding to this example.
        self.intent_or_slot_id = intent_or_slot_id
        # The id of the value corresponding to this example. Only set if slot value
        # embeddings are being calculated.
        self.value_id = value_id

    @staticmethod
    def get_tok_embedding_tensor_name(schema_type):
        return "{}_tok_emb".format(schema_type)

    @staticmethod
    def get_embedding_tensor_name(schema_type):
        return "{}_emb".format(schema_type)

    @staticmethod
    def get_input_ids_tensor_name(schema_type):
        return "{}_input_ids".format(schema_type)

    @staticmethod
    def get_input_mask_tensor_name(schema_type):
        return "{}_input_mask".format(schema_type)

    @staticmethod
    def get_input_type_ids_tensor_name(schema_type):
        return "{}_input_type_ids".format(schema_type)
