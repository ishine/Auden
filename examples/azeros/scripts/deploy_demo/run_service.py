from __future__ import annotations

import os
from typing import Tuple, Optional, Union, Generator

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

import torch
import librosa
import numpy as np
import gradio as gr
import soundfile as sf
from auden.auto.auto_model import AutoModel


TITLE = "AZeroS WebUI (Mic + WAV)"
DESC = """
Record or upload audio, then the model responds.
"""
CHAT_TEMPLATE = """{% for message in messages -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}
"""


class ModelService:
    def __init__(self, args):
        self.tts_model = CosyVoice2(args.cosyvoice_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self.speech_llm = AutoModel.from_pretrained(args.model_path, strict=False)
        self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        self.audio_token_wrapped = self.speech_llm.audio_token_wrapped
        self.audio_sr = 16000
        self.speech_llm.tokenizer.padding_side = "left"
        self.speech_llm.to(self.device)
        self.speech_llm.eval()
        self.generate_config = {
            "max_new_tokens": 200,
            "num_beams": 1,
            "min_length": 1,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "top_p": None,
            "top_k": None,
        }
        self.prompt_speech_16k = load_wav(args.zeroshot_prompt, 16000)
        self.out_sr = self.tts_model.sample_rate

    def load_audio_mono(self, path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file as mono float32 and resample to target_sr if needed.
        Returns (waveform, sr).
        """
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        # shape handling
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            # high-quality resample via librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        # normalize to [-1, 1] safety
        max_abs = np.max(np.abs(audio)) or 1.0
        if max_abs > 1.0:
            audio = audio / max_abs
        return audio.astype(np.float32), sr

    def run(
        self,
        audio_path: Optional[str],
        temperature: float = 0.0,
        messages: Optional[list] = None,
        past_inputs_embeds=None,
        past_key_values=None,
    ) -> Union[str, Generator[str, None, None]]:
        """Return a string (non-stream)
        """
        do_sample = bool(temperature and temperature > 0.0)
        
        if audio_path is not None:
            audio, _ = self.load_audio_mono(audio_path, target_sr=self.audio_sr)
            x, x_lens = self.speech_llm.speech_encoder.extract_feature([audio])
            x = x.to(self.device)
            x_lens = x_lens.to(self.device)
            audio_features, feature_lens = self.speech_llm.forward_audio_features(x, x_lens)
            _, inputs_embeds, _, _ = self.speech_llm.preprocess_text_and_audio(
                [messages],
                audio_features=audio_features,
                audio_feature_lens=feature_lens,
                max_length=400,
                is_training=False,
            )
        else:
            input_ids = self.speech_llm.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                chat_template=CHAT_TEMPLATE,
                add_generation_prompt=True,
                padding="do_not_pad",
                return_tensors="pt",
                max_length=400
            )
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.speech_llm.llm.get_input_embeddings()(input_ids)

        if past_key_values is not None:
            inputs_embeds = torch.cat((past_inputs_embeds, inputs_embeds), dim=1)
        
        outputs = self.speech_llm.llm.generate(
            inputs_embeds=inputs_embeds,
            bos_token_id=self.speech_llm.llm.config.bos_token_id,
            eos_token_id=self.speech_llm.llm.config.eos_token_id,
            pad_token_id=self.speech_llm.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            temperature=temperature,
            do_sample=do_sample,
            **self.generate_config,
        )

        # 5) Decode + return updated cache
        sequences = outputs.sequences
        response = self.speech_llm.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        past_key_values_new = getattr(outputs, "past_key_values", None)
        output_embeds = self.speech_llm.llm.get_input_embeddings()(sequences)
        inputs_embeds = torch.cat((inputs_embeds, output_embeds), dim=1)
        print('=> [Response]', response)
        try:
            instruct, content = response[0].split("|eop|")
        except:
            instruct = "正常"
            content = response[0]
        gen = self.tts_model.inference_instruct2(
            content, instruct, self.prompt_speech_16k, stream=False
        )
        chunks = []
        for out in gen:
            x = out["tts_speech"]                    # [T] 或 [1, T]
            x = x.squeeze(0) if isinstance(x, torch.Tensor) and x.ndim == 2 else x
            x = torch.as_tensor(x).detach().cpu().float()
            chunks.append(x)
        out_speech = torch.cat(chunks, dim=-1)           # [T]
        return response, past_key_values_new, inputs_embeds, out_speech


# Singleton for the app
model_service: Optional[ModelService] = None
args: Optional[argparse.Namespace] = None
def get_service() -> ModelService:
    global model_service
    global args
    if model_service is None:
        model_service = ModelService(args=args)
    return model_service


def conversation(audio_path, sys_prompt_active, temperature, user_text, chat_messages, inputs_state_dict, kv_state_dict):
    service = get_service()
    # normalize chat_messages (users cannot edit; we control all writes)
    if not isinstance(chat_messages, list) or (chat_messages and not isinstance(chat_messages[0], dict)):
        chat_messages = []

    if audio_path:
       chat_messages.append({"role": "user", "content": (audio_path, )})

    if user_text:
        chat_messages = chat_messages + [{"role": "user", "content": user_text}]

    # Build the **new turn only** for the model (PKV holds prior context)
    new_messages = []
    if not sys_prompt_active:
        sys_prompt_active = (
            "你是由腾讯AILAB研发的语音助手，请根据用户的音频或文本输入，决定回复的语气和风格，并以'|eop|'作为分隔符，按照以下格式输出："
            "<语气和风格><|eop|><回复内容>。其中<语气和风格>只能从这个列表中选择："
            "[正常, 高兴, 悲伤, 惊讶, 愤怒, 恐惧, 厌恶, 冷静, 严肃, 快速, 非常快速, 慢速, 非常慢速, 粤语, 四川话, 上海话, 郑州话, 长沙话, 天津话，神秘, 凶猛, 好奇, 优雅, 孤独, 机器人, 小猪佩奇]。"
            "例如：'愤怒|eop|我不愿意做奴隶'"
        )
    if kv_state_dict.get("past") is None and sys_prompt_active:
        new_messages.append({"role": "system", "content": sys_prompt_active}) 
    if audio_path:
        new_messages.append({"role": "user", "content": f"{service.audio_token_wrapped} {user_text}"})
    else:
        new_messages.append({"role": "user", "content": user_text})
    
    print('=> [Messages]', new_messages)
    yield (
        None, chat_messages, "", inputs_state_dict, kv_state_dict
    )

    out, kv_out, input_out, out_speech = service.run(
        audio_path=audio_path,
        temperature=temperature,
        messages=new_messages,                         # <-- ONLY the new turn (and system if first)
        past_inputs_embeds=inputs_state_dict.get("past"),
        past_key_values=kv_state_dict.get("past"),     # <-- pass PKV only
    )

    chat_messages = chat_messages + [{"role": "assistant", "content": ""}]
    acc = str(out[0])
    chat_messages[-1]["content"] = acc
    kv_state_dict["past"] = kv_out
    inputs_state_dict["past"] = input_out
    yield None, chat_messages, "", inputs_state_dict, kv_state_dict
    
    audio_msg = gr.Audio(value=(service.out_sr, out_speech.numpy()), format="wav")  # <- 直接数组
    chat_messages = chat_messages + [
        {"role": "assistant", "content": audio_msg},
    ]
    yield None, chat_messages, "", inputs_state_dict, kv_state_dict


def _hard_reset(inputs_state_dict, kv_state_dict):
    # 取出对象并清空 state 中的引用
    emb = None
    pkv = None
    try:
        emb = inputs_state_dict.get("past", None)
        inputs_state_dict["past"] = None
    except Exception:
        pass
    try:
        pkv = kv_state_dict.get("past", None)
        kv_state_dict["past"] = None
    except Exception:
        pass

    # 释放对象的最后引用
    del emb
    del pkv

    # 触发 GC + 释放 CUDA 缓存
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 返回 UI 组件的复位值（顺序要和 outputs 对齐）
    return (None, "", 0, "", [], {"past": None}, {"past": None})


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESC)

    # session states
    kv_state = gr.State({"past": None})
    inputs_state = gr.State({"past": None})
    sys_prompt_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                label="Microphone or Upload (wav/mp3/flac)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            user_text = gr.Textbox(
                label="Message",
                placeholder="在此输入附加文本（可选）…",
                lines=2,
                submit_btn="Respond",
            )
            with gr.Accordion("Advanced", open=False):
                system_prompt = gr.Textbox(  # used only to SET, not read every turn
                    label="System prompt (optional)",
                    placeholder="e.g., 'You are a helpful assistant'",
                    lines=2,
                )
                set_sys_btn = gr.Button("Set system prompt & reset", variant="secondary")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
                
            run_btn = gr.Button("Respond", variant="primary")
            clear_btn = gr.Button("Clear")

            # out_text = gr.Textbox(label="Result", lines=14, show_copy_button=True)
        
        with gr.Column(scale=2): 
            chat = gr.Chatbot(label="Conversation", 
                            type="messages",
                            height=700,)  # history is programmatic-only

    # Apply the system prompt and wipe the session
    def _apply_sys_prompt_and_reset(sp):
        return (
            sp,                             # sys_prompt_state
            None, sp, 0.0, "",              # audio, system_prompt textbox, temperature, user_text
            [],                         # chat, out_text
            {"past": None},                 # kv_state (only PKV)
            {"past": None},
        )

    set_sys_btn.click(
        _apply_sys_prompt_and_reset,
        inputs=[system_prompt],
        outputs=[sys_prompt_state, audio, system_prompt, temperature, user_text, chat, inputs_state, kv_state],
        show_progress=False,
    )

    # Hook up events — use sys_prompt_state instead of system_prompt textbox
    run_btn.click(
        conversation,
        inputs=[audio, sys_prompt_state, temperature, user_text, chat, inputs_state, kv_state],
        outputs=[audio, chat, user_text, inputs_state, kv_state],
        api_name="Response",
        concurrency_limit=4,
        show_progress=True,
    )

    user_text.submit(
        conversation,
        inputs=[audio, sys_prompt_state, temperature, user_text, chat, inputs_state, kv_state],
        outputs=[audio, chat, user_text, inputs_state, kv_state],
        concurrency_limit=4,
        show_progress=True,
    )

    # Clear wipes everything (soft reset, keeps sys_prompt_state)
    clear_btn.click(
        _hard_reset,
        inputs=[inputs_state, kv_state],
        outputs=[audio, system_prompt, temperature, user_text, chat, inputs_state, kv_state],
    )

    gr.Markdown("<sub>Tip: use HTTPS for microphone.</sub>")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default='0.0.0.0', type=str, help='Gradio server_name')
    parser.add_argument("--server-port", default='8080', type=int, help='Gradio server_port')
    parser.add_argument("--share", action='store_true', help='Gradio share')
    parser.add_argument("--model-path", default='exp/stage2/pretrained.pt',
                        type=str, help='Path to AZeroS model')
    parser.add_argument("--cosyvoice-path", default='myfolder/CosyVoice2-0.5B',
                        type=str, help='Path to CosyVoice')
    parser.add_argument("--zeroshot-prompt", default='assets/zero_shot_prompt.wav',
                        type=str, help='Audio file as zero-shot prompt')
    args = parser.parse_args()

    demo.queue(max_size=64).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )
