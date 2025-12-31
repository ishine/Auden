# Demo Deployment of AZeroS

The scripts provide a demo to deploy a demo website based on a pretrained AZeroS model.

## Speech-To-Text Demo

The generic AZeroS model support `speech-in, text-out` interaction. 
Users can input by microphone, audio file, or text message, and the model will respond with texts.
Multi-turn dialog is supported by `kv_cache`.

Requirements: Run `pip install gradio` to install the Gradio tool.

Usage:

```bash
python scripts/deploy_demo/run_service.py \
    --model-path expdir/pretrained.pt \
    --server-port 8080
```

## Speech-To-Speech Demo

We provide a simple `thinker-talker` solution to construct a `speech-in, speech-out` system with `--enable-tts` mode:

- The AZeroS model serves as the `thinker`. A system meesage is added by default, which guides the model to respond with both the speaking tone/style and text content.
The system message is:
```
You are a helpful voice assistant created by Tencent AI Lab. Please reply to speech and text inputs from users, and decide your speaking tone and style accrodingly. Always reply with the format of '<SPEAKING TONE AND STYLE><|EOP|><RESPONSE CONTENT>', where the '|EOP|' is a fixed separator, and the <SPEAKING TONE AND STYLE> should only be chosen from the following list: [正常, 高兴, 悲伤, 惊讶, 愤怒, 恐惧, 厌恶, 冷静, 严肃, 快速, 非常快速, 慢速, 非常慢速, 粤语, 四川话, 上海话, 郑州话, 长沙话, 天津话，神秘, 凶猛, 好奇, 优雅, 孤独, 机器人, 小猪佩奇]. e.g. '愤怒|EOP|All men are created equal.'
```  
- An Instruct-TTS model plays the role of `talker`. We adopt `CosyVoice2-0.5B` here. It will synthesize voices with provided `instruct` (the speaking tone/style from thinker) and `content` (the response content from thinker).

Requirements:

- [CosyVoice2 Github](https://github.com/Render-AI/CosyVoice2). Follow instructions in the official repo to install the dependencies.
- [CosyVoice2-0.5B Huggingface](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B). Download the model and pass its local path as `--cosyvoice-path`.

Usage: 

```bash
python scripts/deploy_demo/run_service.py \
    --enable-tts \
    --model-path expdir/pretrained.pt \
    --cosyvoice-path myfolder/CosyVoice2-0.5B \
    --zeroshot-prompt assets/zero_shot_prompt.wav \
    --server-port 8080
```