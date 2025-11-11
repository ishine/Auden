from __future__ import annotations

import os
import logging
from typing import List, Dict, Union, Optional

import torch
import torch.nn as nn
from transformers import AutoModel as HFModel
from transformers import AutoTokenizer as HFTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer as HFTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from .projector.base_projector import EncoderProjector
from .utils import (
    replace_whisper_encoder_forward,
    compute_accuracy,
)
from ...auto.auto_config import AutoConfig
from ...auto.auto_model import AutoModel

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEFAULT_AUDIO_TOKEN = "<|AUDIO|>"
DEFAULT_PLACE_HOLDER = "<|PLACEHOLDER|>"
CHAT_TEMPLATE = """{% for message in messages -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}"""

class AzerosModel(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> "AzerosModel":
        # Resolve model_dir and checkpoint
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

        # Load config and tokenizers
        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = HFTokenizer.from_pretrained(model_dir)

        model = cls(config, tokenizer)

        # Load weights (pt or safetensors)
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

        # load excluded submodules saved separately under model_dir
        for module_name in ('llm', 'speech_encoder', 'speech_encoder_projector',
                            'paraling_encoder', 'paraling_encoder_projector'):
            subdir = os.path.join(model_dir, module_name)
            if os.path.isdir(subdir):
                module_type = type(getattr(model, module_name))
                loaded_module = module_type.from_pretrained(subdir).state_dict()
                getattr(model, module_name).load_state_dict(loaded_module, strict=True)
                logging.info(f"Load {module_name} from {subdir}")

        model.eval()
        return model

    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.use_flash_attn = config.use_flash_attn
        self.exclude_from_checkpoint = config.exclude_from_checkpoint

        self.audio_token = DEFAULT_AUDIO_TOKEN
        self.audio_token_wrapped = self.audio_token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.audio_token]}
        )
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_TOKEN)
        self.pad_token_id = self.tokenizer.pad_token_id

        if self.use_flash_attn:
            self.attn_implementation = "flash_attention_2"
        else:
            self.attn_implementation = "sdpa"

        # from accelerate import init_empty_weights
        # with init_empty_weights(): # this will cause error when pytorch < 2.4
        from transformers.modeling_utils import no_init_weights
        with no_init_weights(): # this won't do any random weights initialization to save time
            self.llm = AutoModelForCausalLM.from_config(
                self.config.llm_config,
                attn_implementation=self.attn_implementation,
                torch_dtype=torch.float16
            )

        self.speech_encoder, self.speech_encoder_projector = self.set_speech_encoder(
            config.speech_encoder_config,
            config.speech_encoder_projector_ds_rate,
        )
        self.paraling_encoder, self.paraling_encoder_projector = self.set_speech_encoder(
            config.paraling_encoder_config,
            config.paraling_encoder_projector_ds_rate,
        )
        if self.paraling_encoder is not None:
            self.audio_token_wrapped = f"<audio><meta>{self.audio_token}</meta><text>{self.audio_token}</text></audio>"

        # optional pretrained model
        if self.config.get('pretrained_model'):
            pretrained = torch.load(self.config.get('pretrained_model'))
            self.load_state_dict(pretrained, strict=False)

    def set_speech_encoder(self, config, downsample_rate):
        if config is None:
            return None, None
        model_type = config.model_type
        if "whisper" in model_type:
            replace_whisper_encoder_forward() # this will handle when audio input is not 30s
            from transformers.modeling_utils import no_init_weights
            with no_init_weights():
                speech_model = HFModel.from_config(config)
            speech_encoder = speech_model.encoder
            speech_encoder_dim = config.d_model
        else:
            speech_encoder = AutoModel.from_config(config)
            speech_encoder_dim = speech_encoder.encoder_out_dim

        encoder_projector = EncoderProjector(
            speech_encoder_dim,
            self.llm.config.hidden_size,
            downsample_rate
        )
        return speech_encoder, encoder_projector

    def preprocess_text_and_audio(
        self,
        messages: List[Dict[str, str]],
        audio_features: Optional[torch.Tensor] = None,
        audio_feature_lens: Optional[torch.Tensor] = None,
        max_length: int = None,
        is_training: bool = False,
    ):
        """
        Combine the text messages with audio features. This is done by inserting audio tokens of the non-padded length
        into the text messages, and then padding them. Modified from
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py.
        """
        placeholder = DEFAULT_PLACE_HOLDER
        audio_token = self.audio_token
        audio_token_id = self.audio_token_id
        batch_size = audio_features.size(0)

        if is_training:
            add_generation_prompt = False
            fix_chat_template = True
            prepare_label = True
        else:
            add_generation_prompt = True
            fix_chat_template = False
            prepare_label = False

        n_audio_token = 0
        input_ids = []
        max_input_len = 0

        # step 1: expand all the audio tokens to their target length
        for message in messages:
            expanded_audio_tokens = []
            assert isinstance(message, list), message
            message = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                chat_template=CHAT_TEMPLATE if fix_chat_template else None,
                add_generation_prompt=add_generation_prompt,
                padding='do_not_pad',
                truncation=False,
            )

            # NOTE: since the pre-generated responses were truncated at 256 tokens,
            # we have to do roughly the same truncation here to avoid UNEXPECTED <im_end> tokens 
            _tokens = self.tokenizer.encode(message, add_special_tokens=False)
            message = self.tokenizer.decode(_tokens[:256])

            while audio_token in message:
                audio_len = audio_feature_lens[n_audio_token].item()
                n_audio_token += 1
                message = message.replace(audio_token, placeholder, 1)
                expanded_audio_tokens.append(audio_token * audio_len)

            while placeholder in message:
                message = message.replace(placeholder, expanded_audio_tokens.pop(0), 1)

            tokens = self.tokenizer.encode(message, add_special_tokens=False)
            if max_length is not None:
                tokens = tokens[:max_length]
            input_ids.append(tokens)
            max_input_len = max(max_input_len, len(tokens))

        # step 2: get input embeddings and insert audio features
        if self.tokenizer.padding_side == "right":
            input_ids = [
                ids + [self.pad_token_id] * (max_input_len - len(ids))
                for ids in input_ids
            ]
        else:
            input_ids = [
                [self.pad_token_id] * (max_input_len - len(ids)) + ids
                for ids in input_ids
            ]
        input_ids = torch.Tensor(input_ids).to(torch.int).to(audio_features.device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        padding_mask = (input_ids == self.pad_token_id)

        audio_token_mask = (input_ids == audio_token_id)
        audio_non_padding_mask = torch.arange(audio_features.size(1), device=audio_features.device)
        audio_non_padding_mask = (audio_non_padding_mask[None, :] < audio_feature_lens[:, None])
        audio_features = audio_features[audio_non_padding_mask]
        num_audio_tokens = audio_token_mask.sum().item()
        num_audio_features = audio_features.size(0)
        assert num_audio_tokens == num_audio_features, \
            f"Expect {num_audio_tokens} audio frames, get {num_audio_features} instead."

        audio_token_mask = audio_token_mask[..., None].expand_as(input_embeds)
        input_embeds = input_embeds.masked_scatter(audio_token_mask, audio_features)

        # step 3: (Optional) deal with labels and attention mask for training
        if prepare_label:
            target_ids = input_ids.clone().to(torch.long)
            target_ids[padding_mask] = IGNORE_TOKEN_ID
            assistant_token_id = self.tokenizer.convert_tokens_to_ids('assistant')
            response_start_x, response_start_y = torch.where(input_ids == assistant_token_id)
            for row, col in zip(response_start_x, response_start_y):
                if target_ids[row, col - 1] == self.tokenizer.convert_tokens_to_ids('<|im_start|>'):
                    target_ids[row, : col + 2] = IGNORE_TOKEN_ID
        else:
            target_ids = None
        attention_mask = ~padding_mask

        return input_ids, input_embeds, attention_mask, target_ids

    @staticmethod
    def forward_speech_encoder(x, x_lens, model, projector, model_type):
        if "whisper" in model_type:
            x = x.transpose(1, 2) # (N, C, T)
            x = x[:, :, :3000]
            encoder_outs = model(x)[0]
            # # (choice 1) fix length as 30s with no padding mask
            # feature_lens = torch.ones_like(x_lens) * 1500
            # (choice 2) use the actual length with padding mask (recommend)
            feature_lens = ((x_lens - 1) // 2 + 1)
            feature_lens = torch.where(feature_lens > 1500, 1500, feature_lens)
            encoder_outs = encoder_outs[:, :feature_lens.max()]
        else:
            x_lens = torch.where(x_lens > x.size(1), x.size(1), x_lens)
            encoder_output = model(x, x_lens)
            encoder_outs = encoder_output.encoder_out
            feature_lens = encoder_output.encoder_out_lens

        encoder_outs, feature_lens = projector(encoder_outs, feature_lens)
        encoder_outs = encoder_outs.to(torch.float16)
        return encoder_outs, feature_lens

    def forward_audio_features(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ):
        encoder_outs, feature_lens = self.forward_speech_encoder(
            x,
            x_lens,
            self.speech_encoder,
            self.speech_encoder_projector,
            self.config.speech_encoder_config.model_type,
        )
        if self.paraling_encoder is not None:
            paraling_outs, paraling_lens = self.forward_speech_encoder(
                x,
                x_lens,
                self.paraling_encoder,
                self.paraling_encoder_projector,
                self.config.paraling_encoder_config.model_type,
            )
            # merge two encoders as interleaved audio tokens
            B, L1, D = encoder_outs.shape
            _, L2, _ = paraling_outs.shape
            audio_features = torch.zeros(B * 2, max(L1, L2), D, 
                                        dtype=encoder_outs.dtype, device=encoder_outs.device)
            _feature_lens = torch.zeros(B * 2, dtype=feature_lens.dtype, device=feature_lens.device)
            for i in range(B):
                audio_features[i * 2, :L2] = paraling_outs[i]
                audio_features[i * 2 + 1, :L1] = encoder_outs[i]
                _feature_lens[i * 2] = paraling_lens[i]
                _feature_lens[i * 2 + 1] = feature_lens[i]
            feature_lens = _feature_lens
            encoder_outs = audio_features

        return encoder_outs, feature_lens

    def forward(self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        messages: List[Union[List[Dict], str]],
        texts: Optional[List[str]] = None,
    ):
        audio_features, feature_lens = self.forward_audio_features(x, x_lens)

        input_ids, inputs_embeds, attention_mask, labels = self.preprocess_text_and_audio(
            messages,
            audio_features=audio_features,
            audio_feature_lens=feature_lens,
            max_length=512,
            is_training=True,
        )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return model_outputs, acc

    def generate(
        self,
        input,
        messages,
        **kwargs,
    ):
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        elif hasattr(self.speech_encoder, 'extract_feature'):
            x, x_lens = self.speech_encoder.extract_feature(input)
            x = x.to(self.llm.device)
            x_lens = x_lens.to(self.llm.device)
        elif 'whisper' in self.config.speech_encoder['model_type']:
            from whisper import log_mel_spectrogram
            import numpy as np
            x_lens = [wav.shape[0] // 160 for wav in input]
            x = [log_mel_spectrogram(wav.astype(np.float32), 80, padding=480000)[:, :3000] for wav in input]
            x = torch.stack(x, dim=0).transpose(1, 2).to(self.llm.device)
            x_lens = torch.LongTensor(x_lens).to(self.llm.device)

        audio_features, feature_lens = self.forward_audio_features(x, x_lens)

        # enforce left padding for auto-regressive batch decoding
        self.tokenizer.padding_side = 'left'

        input_ids, inputs_embeds, attention_mask, _ = self.preprocess_text_and_audio(
            messages,
            audio_features=audio_features,
            audio_feature_lens=feature_lens,
            max_length=None,
            is_training=False,
        )

        generated_ids = self.llm.generate(
            # input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.pad_token_id,
            **kwargs,
        )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return response
