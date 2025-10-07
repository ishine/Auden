from __future__ import annotations

import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer as HFTokenizer

from ...auto.auto_model import AutoConfig, AutoModel
from ..zipformer.utils.padding import make_pad_mask
from .model_config import AudioCaptionConfig

SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
}


def ensure_special_tokens(tokenizer, required_special_tokens=SPECIAL_TOKENS):
    to_add = {
        k: v
        for k, v in required_special_tokens.items()
        if getattr(tokenizer, k) is None
    }
    if to_add:
        tokenizer.add_special_tokens(to_add)


def causal_mask(length: int, device: Optional[torch.device] = None) -> torch.BoolTensor:
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[: x.size(1)].unsqueeze(0).to(x.device)


class DecoderModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.vocab_size = len(tokenizer)
        self.d_model = config.d_model
        self.decoder_nhead = config.decoder_nhead
        self.text_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if config.decoder_shared_emb:
            self.output_proj.weight = self.text_embed.weight

        self.dim_feedforward = config.dim_feedforward
        self.dropout = config.decoder_dropout
        self.activation = config.decoder_activation
        self.norm_first = config.decoder_norm_first
        self.bias = config.decoder_bias
        self.num_layers = config.num_decoder_layers

        self.positional_encoding = SinusoidalPositionalEncoding(self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.decoder_nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
            bias=self.bias,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt_emb = self.text_embed(tgt) + self.positional_encoding(tgt)
        return self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class AudioCaptionModel(nn.Module):
    """Audio captioning with a custom Transformer decoder over encoder frames.

    Components:
      - Audio encoder (AutoModel)
      - Tokenizer + Transformer decoder (teacher forcing + greedy generation)
      - Optional linear projection to match encoder/decoder dims
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        # Support HF Hub repo IDs
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)
        if os.path.isdir(model_path):
            model_dir = model_path
            weight_path = None
            for ext in (".safetensors", ".pt"):
                for name in ("pretrained", "model"):
                    p = os.path.join(model_dir, f"{name}{ext}")
                    if os.path.exists(p):
                        weight_path = p
                        break
                if weight_path is not None:
                    break
            if weight_path is None:
                raise FileNotFoundError(
                    f"Expected one of ['pretrained.safetensors','model.safetensors','pretrained.pt','model.pt'] under {model_dir}"
                )
        else:
            weight_path = model_path
            model_dir, _ = os.path.split(model_path)

        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = HFTokenizer.from_pretrained(model_dir)
        model = cls(config, tokenizer)

        if weight_path and os.path.exists(weight_path):
            ext = os.path.splitext(weight_path)[1].lower()
            if ext == ".safetensors":
                from safetensors.torch import load_file as safe_load_file

                device_arg = (
                    str(map_location)
                    if isinstance(map_location, torch.device)
                    else map_location
                )
                state_obj = safe_load_file(weight_path, device=device_arg)
            else:
                state_obj = torch.load(weight_path, map_location=map_location)
            state_dict = (
                state_obj["state_dict"]
                if isinstance(state_obj, dict) and "state_dict" in state_obj
                else state_obj
            )
            model.load_state_dict(state_dict, strict=strict)

        model.eval()
        return model

    def __init__(self, config: AudioCaptionConfig, tokenizer: HFTokenizer):
        super().__init__()
        self.config = config

        # Audio encoder (Zipformer expected, but any encoder returning frame features works)
        self.audio_encoder = AutoModel.from_config(config.audio_encoder_config)
        self.audio_encoder_dim = self.audio_encoder.encoder_out_dim

        # Tokenizer and custom Transformer decoder
        self.tokenizer = tokenizer
        ensure_special_tokens(self.tokenizer)
        self.id_bos = self.tokenizer.bos_token_id
        self.id_eos = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id
        self.id_mask = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)

        self.text_decoder = DecoderModel(config, self.tokenizer)

        d_model = self.text_decoder.d_model
        self.enc_dec_proj = (
            nn.Linear(self.audio_encoder_dim, d_model, bias=False)
            if self.audio_encoder_dim != d_model
            else None
        )
        label_smoothing = getattr(config, "label_smoothing", 0.1)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.id_pad, label_smoothing=label_smoothing
        )

    def encode_audio(self, x, x_lens):
        outputs = self.audio_encoder(x, x_lens)
        encoder_out = outputs.encoder_out  # [B, S, D]
        padding_mask = make_pad_mask(
            outputs.encoder_out_lens, max_len=encoder_out.size(1)
        ).to(encoder_out.device)
        return encoder_out, padding_mask

    def forward(
        self,
        x,
        x_lens,
        text: Optional[List[str]] = None,
        parallel_decoding_prob: float = 0.0,
        max_length: int = 128,
    ):
        device = next(self.parameters()).device
        x = x.to(device)
        x_lens = x_lens.to(device)

        # Encode audio frames and masks
        encoder_hidden, padding_mask = self.encode_audio(x, x_lens)
        if self.enc_dec_proj is not None:
            encoder_hidden = self.enc_dec_proj(encoder_hidden)

        if self.training:
            assert text is not None, "Text must be provided during training"
        tgt_in, tgt_out, tgt_pad_mask, tgt_mask, _ = self._prepare_tgt(
            text, parallel_decoding_prob=parallel_decoding_prob, max_length=max_length
        )

        decoder_out = self.text_decoder(
            tgt=tgt_in,
            memory=encoder_hidden,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=padding_mask,
        )
        logits = self.text_decoder.output_proj(decoder_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), tgt_out.view(-1))
        return loss

    def _tokenise(self, text: List[str], *, max_length: int = 128):
        ids = self.tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_length
        ).input_ids
        return ids

    def _prepare_tgt(
        self, text: List[str], parallel_decoding_prob: float, max_length: int = 128
    ):
        """
        Prepare target input for decoder.
        Args:
            text: List[str]
        Returns:
            tgt_in: Tensor [B, T], input to decoder
            tgt_out: Tensor [B, T], target output for loss computation
            tgt_pad_mask: Tensor [B, T], padding mask for decoder input
            tgt_mask: Tensor [T, T], causal mask for decoder input
        """
        PAD, BOS, EOS, MASK = self.id_pad, self.id_bos, self.id_eos, self.id_mask
        device = next(self.parameters()).device  # always valid

        ids = self._tokenise(text, max_length=max_length)  # tokenize text to ids
        lengths = [len(s) for s in ids]
        B = len(ids)

        n_parallel = int(round(B * parallel_decoding_prob))
        perm = torch.randperm(B, device=device)
        parallel_ids = perm[:n_parallel]
        causal_ids = perm[n_parallel:]

        max_causal_len = max(
            (lengths[idx] + 1 for idx in causal_ids), default=0
        )  # +1 for EOS
        max_parallel_len = max((lengths[idx] for idx in parallel_ids), default=0)
        max_len = max(max_causal_len, max_parallel_len)

        tgt_in = torch.full(
            (B, max_len), PAD, dtype=torch.long, device=device
        )  # [B, max_len]
        tgt_out = torch.full_like(tgt_in, PAD)  # [B, max_len]
        tgt_pad_mask = torch.zeros_like(tgt_in, dtype=torch.bool)  # [B, max_len]

        if causal_ids.numel():
            tgt_in_causal = [
                torch.tensor([BOS] + ids[idx], dtype=torch.long) for idx in causal_ids
            ]  # add BOS token
            tgt_out_causal = [
                torch.tensor(ids[idx] + [EOS], dtype=torch.long) for idx in causal_ids
            ]
            tgt_in_causal = nn.utils.rnn.pad_sequence(
                tgt_in_causal, batch_first=True, padding_value=PAD
            ).to(device)
            tgt_out_causal = nn.utils.rnn.pad_sequence(
                tgt_out_causal, batch_first=True, padding_value=PAD
            ).to(device)

            T_causal = tgt_in_causal.size(1)
            tgt_in[causal_ids, :T_causal] = tgt_in_causal
            tgt_out[causal_ids, :T_causal] = tgt_out_causal

        if parallel_ids.numel():
            tgt_out_parallel = [
                torch.tensor(ids[idx], dtype=torch.long) for idx in parallel_ids
            ]  # no EOS token
            tgt_in_parallel = [
                torch.full_like(t, MASK) for t in tgt_out_parallel
            ]  # use MASK token
            tgt_out_parallel = nn.utils.rnn.pad_sequence(
                tgt_out_parallel, batch_first=True, padding_value=PAD
            ).to(device)
            tgt_in_parallel = nn.utils.rnn.pad_sequence(
                tgt_in_parallel, batch_first=True, padding_value=PAD
            ).to(device)

            T_parallel = tgt_out_parallel.size(1)
            tgt_in[parallel_ids, :T_parallel] = tgt_in_parallel
            tgt_out[parallel_ids, :T_parallel] = tgt_out_parallel

        tgt_pad_mask = tgt_in.eq(PAD)  # True → ignore position

        per_sample_mask = torch.zeros(
            (B, max_len, max_len), dtype=torch.bool, device=device
        )  # [B, T, T]
        if causal_ids.numel():
            per_sample_mask[causal_ids] = causal_mask(
                max_len, device=device
            )  # causal mask for causal_ids

        num_heads = self.text_decoder.decoder_nhead
        tgt_mask = per_sample_mask[:, None, :, :].repeat_interleave(
            num_heads, dim=1
        )  # [B, num_heads, T, T]
        tgt_mask = tgt_mask.view(
            B * num_heads, max_len, max_len
        )  # [B, num_heads, T, T]

        return tgt_in, tgt_out, tgt_pad_mask, tgt_mask, lengths

    @torch.inference_mode()
    def generate(self, input, max_length: int = 128):
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        else:
            x, x_lens = self.audio_encoder.extract_feature(input)

        device = next(self.parameters()).device
        x = x.to(device)
        x_lens = x_lens.to(device)

        encoder_hidden, padding_mask = self.encode_audio(x, x_lens)
        if self.enc_dec_proj is not None:
            encoder_hidden = self.enc_dec_proj(encoder_hidden)

        B = encoder_hidden.size(0)
        tokens = torch.full((B, 1), self.id_bos, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length):
            tgt_mask = causal_mask(tokens.size(1), device=device)
            dec_out = self.text_decoder(
                tgt=tokens,
                memory=encoder_hidden,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=padding_mask,
            )
            next_tok = self.text_decoder.output_proj(dec_out[:, -1]).argmax(-1)
            tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
            finished |= next_tok.eq(self.id_eos)
            if finished.all():
                break

        for b in range(B):
            eos_pos = (tokens[b] == self.id_eos).nonzero(as_tuple=True)[0]
            if len(eos_pos):
                tokens[b, eos_pos[0] :] = self.id_pad

        text_ids = tokens[:, 1:].detach().cpu()
        text = self.tokenizer.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return text
