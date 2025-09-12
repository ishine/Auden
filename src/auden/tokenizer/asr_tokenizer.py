import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import sentencepiece as spm


class AbstractAsrTokenizer(ABC):
    """Abstract class for ASR tokenizer.

    Subclasses must implement: vocab_size, unk_id, add_tokens, encode, decode.
    """

    def __init__(self, name, config=None, dir=None):
        super().__init__()
        self.name = name
        self.config = config or {}
        self.dir = dir

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def unk_id(self):
        pass

    @abstractmethod
    def add_tokens(self, tokens):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, token_ids):
        pass

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        # best-effort copy for SPM models
        try:
            if hasattr(self, "_tokenizer") and isinstance(
                self._tokenizer, spm.SentencePieceProcessor
            ):
                if self.dir is not None:
                    input_spm_path = os.path.join(self.dir, "bpe.model")
                    if os.path.isfile(input_spm_path):
                        output_spm_path = os.path.join(save_dir, "bpe.model")
                        shutil.copy(input_spm_path, output_spm_path)
                        logging.info(f"Saved SentencePiece model to {output_spm_path}")
        except Exception as e:
            logging.warning(f"Skipping SentencePiece model copy: {e}")


class AsrIcefallLexiconCharTokenizer(AbstractAsrTokenizer):
    def __init__(self, lang_dir):
        name = "IcefallLexiconTokenizer"
        super().__init__(
            name=name, config={"type": name, "lang_dir": str(lang_dir)}, dir=lang_dir
        )

        from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
        from icefall.lexicon import Lexicon

        self._lexicon = Lexicon(lang_dir)
        self._graph_compiler = CharCtcTrainingGraphCompiler(
            lexicon=self._lexicon, device="cpu"
        )

    def add_tokens(self, tokens):
        raise NotImplementedError(
            "add_tokens is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def vocab_size(self):
        try:
            # icefall Lexicon usually has a list `tokens` where index is id
            return len(self._lexicon.tokens)
        except Exception:
            # fallback to token_table (str->int)
            return max(self._lexicon.token_table.values()) + 1

    @property
    def unk_id(self):
        return self._lexicon.token_table["<unk>"]

    @property
    def blank_id(self):
        return self._lexicon.token_table["<blk>"]

    def encode(self, text):
        # Accept a single string or a list of strings; return list[list[int]]
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        return self._graph_compiler.texts_to_ids(texts)

    def decode(self, token_ids, use_space=False):
        split_char = " " if use_space else ""

        # id -> token mapping
        def id_to_token(i):
            try:
                return self._lexicon.tokens[i]
            except Exception:
                # invert token_table (token -> id)
                if not hasattr(self, "_id2token"):
                    table = self._lexicon.token_table
                    self._id2token = {v: k for k, v in table.items()}
                return self._id2token[i]

        if token_ids and isinstance(token_ids[0], list):
            texts = []
            for seq in token_ids:
                text = split_char.join([id_to_token(i) for i in seq])
                texts.append(text)
            return texts
        else:
            return split_char.join([id_to_token(i) for i in token_ids])
