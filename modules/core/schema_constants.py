"""
all the constant values in the schema-guided dialogue
"""
STR_DONTCARE = "dontcare"
STR_UNKNOWN = "unknown"
VALUE_DONTCARE_ID = 0
VALUE_UNKNOWN_ID = 1
SPECIAL_CAT_VALUE_OFFSET = 2

# The maximum total input sequence length after WordPiece tokenization.
DEFAULT_MAX_SEQ_LENGTH = 128

MAX_SCHEMA_SEQ_LENGTH = 128

USER_SPECIAL_TOKEN = '[user:]'
SYSTEM_SPECIAL_TOKEN = '[system:]'

USER_AGENT_SPECIAL_TOKENS = {"additional_special_tokens": [USER_SPECIAL_TOKEN, SYSTEM_SPECIAL_TOKEN]}

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2
