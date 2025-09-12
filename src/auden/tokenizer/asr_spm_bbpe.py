"""Byte-BPE flavored SentencePiece tokenizer for ASR with CJK handling.

Pipeline:
- Insert spaces between CJK characters; uppercase non-CJK text.
- Byte-encode the text so each character maps to byte tokens (fake words).
- Feed the result to the SentencePiece unigram model.
"""

import re
from typing import List

from ..utils.byte_utils import byte_encode, smart_byte_decode
from .asr_spm import AsrSpmTokenizer


def tokenize_by_CJK_char(line: str) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return characters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])


class AsrSpmBbpeTokenizer(AsrSpmTokenizer):
    """Icefall-like SPmBbpeTokenizer wrapper with CJK-aware preprocessing."""

    def __init__(self, config, dir):
        super().__init__(config, dir)

    def encode(self, texts: List[str]) -> List[List[int]]:
        texts = [byte_encode(tokenize_by_CJK_char(text)) for text in texts]
        return self._tokenizer.encode(texts, out_type=int)

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        texts = self._tokenizer.decode(token_ids)
        return [smart_byte_decode(text) for text in texts]
