"""SentencePiece-based tokenizer for ASR.

Provides a thin wrapper around sentencepiece to unify encode/decode behavior:
- encode accepts str or List[str] and always returns List[List[int]].
- decode accepts List[List[int]] and returns List[str].
Also supports removing a leading space token ("▁") per item when configured.
"""

import logging
from pathlib import Path
from typing import List, Union

import sentencepiece as spm

from .asr_tokenizer import AbstractAsrTokenizer


class AsrSpmTokenizer(AbstractAsrTokenizer):
    def __init__(self, config, dir):
        super().__init__(name="asr-spm", config=config, dir=dir)
        model_file = Path(dir) / "bpe.model"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self._tokenizer = spm.SentencePieceProcessor(model_file=str(model_file))
        self._vocab = {
            self._tokenizer.id_to_piece(i): i
            for i in range(self._tokenizer.get_piece_size())
        }
        if "<blk>" not in self._vocab:
            self.add_tokens(["<blk>"])
            logging.warning(
                "<blk> not included in the current spm. Now added <blk> to the end."
            )
        self.blank_id = self._vocab["<blk>"]
        logging.info(
            f"[Tokenizer] Loaded {self.__class__.__name__} (type=sentencepiece) "
            f"from {model_file} | vocab size: {len(self._vocab)} | blank_id: {self.blank_id} | unk_id: {self.unk_id}"
        )

        self.remove_start_space = self.config.get("remove_start_space", False)
        if self.remove_start_space:
            self.space_token = self._tokenizer.piece_to_id(
                "▁"
            )  # be careful! it's LOWER ONE EIGHTH BLOCK U+2581
            logging.info(
                f"Starting single space ('▁') of id {self.space_token} will be removed after encode"
            )

    def add_tokens(self, tokens: List[str]) -> None:
        assert isinstance(tokens, list)
        for t in tokens:
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def unk_id(self) -> int:
        return self._tokenizer.piece_to_id("<unk>")

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.piece_to_id("<sos/eos>")

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.piece_to_id("<sos/eos>")

    def encode(self, text: Union[str, List[str]]) -> List[List[int]]:
        # Wrap single string to list for consistent List[List[int]] output
        if isinstance(text, str):
            texts: List[str] = [text]
        else:
            texts = text
        text_ids: List[List[int]] = self._tokenizer.encode(texts, out_type=int)  # type: ignore[assignment]
        if self.remove_start_space:
            text_ids = [
                ids[1:] if ids and ids[0] == self.space_token else ids
                for ids in text_ids
            ]
        return text_ids

    def decode(
        self, token_ids: List[List[int]], *, skip_bos_eos: bool = False
    ) -> List[str]:
        """Decode token id sequences to strings.

        Args:
            token_ids: List of token id sequences.
            skip_bos_eos: If True, remove BOS/EOS token ids before decoding.

        Returns:
            List of decoded strings.
        """
        if skip_bos_eos:
            bos_id = self.bos_token_id
            eos_id = self.eos_token_id
            remove_ids = {bos_id, eos_id}
            token_ids = [[t for t in seq if t not in remove_ids] for seq in token_ids]
        return self._tokenizer.decode(token_ids)
