

"""Vocab."""

# A shared vocab among tasks and its structure -
# Special tokens: [0, 99).
# Class tokens: [100, coord_vocab_shift). Total coord_vocab_shift - 100 classes.
# Coordinate tokens: [coord_vocab_shift, text_vocab_shift).
# Text tokens: [text_vocab_shift, ...].

PADDING_TOKEN = 20

# 10-29 reserved for task id.

FAKE_CLASS_TOKEN = 30
FAKE_TEXT_TOKEN = 30  # Same token to represent fake class and fake text.
SEPARATOR_TOKEN = 40
INVISIBLE_TOKEN = 41

BASE_VOCAB_SHIFT = 100

