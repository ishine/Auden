import os
import yaml
import argparse
from functools import partial

import torch
from lhotse import CutSet
from transformers import AutoModelForCausalLM, AutoTokenizer
from auden.utils.text_normalization import text_normalization

CHAT_TEMPLATE = """{% for message in messages -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}"""

simt_system_message = (
    "You are a powerful virtual human who is capable of perceiving both text and speech inputs and generate precise natural responses. "
    "Speech inputs will be wrapped by <audio> and </audio> tags, containing both the text transcription and paralinguistic information. "
    "You must always pretend that you can indeed hear the input audios."
    "NEVER mention that any metadata is provided through texts, and only use them in your response when necessary."
)

def age_to_group(age):
    try:
        age = int(age)
        if 0 < int(age) < 18:
            age_group = 'teenager'
        elif int(age) < 40:
            age_group = 'young adult'
        elif int(age) <= 60:
            age_group = 'middle-age adult'
        elif 60 < int(age) < 200:
            age_group = 'senior'
        else:
            raise ValueError(age)
    except:
        age_group = 'null'
    return age_group

def generate_input(meta, mode=None):
    if mode == 'sift_s':
        # SIFT_s: Self-generated Instruction-Free Tuning (semantic-only)
        system_message = ''
        user_prompt = ''
    elif mode == 'sift_sp':
        # SIFT_sp: Self-generated Instruction-Free Tuning (semantic + paralinguistic)
        system_message = ''
        user_prompt = ''
    elif mode == 'simt_sp_1':
        # SIMT_sp: Self-generated, System-prompt-enhanced Instruction Mixture Tuning
        # split-1: instruction-free tuning
        system_message = simt_system_message
        user_prompt = ''
    elif mode == 'simt_sp_2':
        # SIMT_sp: Self-generated, System-prompt-enhanced Instruction Mixture Tuning
        # split-2: instruction tuning
        system_message = simt_system_message
        user_prompt = {
            'en': 'Describe all information you can hear.',
            'zh': '描述你听到的所有信息。'
        }[meta['language']]
    else:
        raise RuntimeError(mode)

    para_text = ', '.join(f"{k}: {v}" for k, v in meta.items() if k not in ('text', 'language') and v != '?')
    input_text = f"<audio><meta>{para_text}</meta><text>{meta['text']}</text></audio>"
    return input_text, user_prompt, system_message

def get_metadata(c):
    sup = c.supervisions[0]
    metadata = {'age': '?', 'gender': '?', 'emotion': '?'}
    if getattr(sup, 'age_group', None):
        metadata['age'] = sup.age_group
    elif getattr(sup, 'age', None):
        metadata['age'] = age_to_group(sup.age)
    if getattr(sup, 'gender', None):
        metadata['gender'] = sup.gender
    if getattr(sup, 'emotion', None):
        metadata['emotion'] = sup.emotion
    # 1. remove all contents in brackets; 2. convert to lowercase;
    metadata['text'] = text_normalization(sup.text, case='lower', remove_symbols=False, simplified_chinese=True)
    if getattr(sup, 'language', None):
        metadata['language'] = sup.language
    metadata = {
        k: v.lower()
        if v.lower() not in ('na', '', 'null', 'unk', 'unknown')
        else '?' for k, v in metadata.items()
    }
    return metadata

def batch_cutset(cutset, bs=1):
    buffer = []
    for cut in cutset:
        if len(buffer) == bs:
            batch = buffer
            buffer = []
            yield batch
        else:
            buffer.append(cut)
    if len(buffer) > 0:
        yield buffer

@torch.no_grad()
def main(dset, args):
    name = dset['name']
    lang = dset.get('lang', 'en')
    output = f"{args.output_dir}/{name}"
    os.makedirs(output, exist_ok=True)

    results = []
    mode = args.mode
    nshards, shard = args.nshards, args.shard

    print(f"[Process Set:{name} Shard:{shard}]")
    if os.path.exists(f"{output}/{mode}_{shard}.jsonl.gz"):
        print('Already processed. Skip.')
        return

    def cuts_preprocess(c, lang):
        c.supervisions[0].language = lang
        if not 1.0 < c.duration <= 30.0:
            return False
        if not 0 < len(c.supervisions[0].text) < c.duration * 30:
            return False
        return True

    cuts = CutSet.from_file(dset['manifest'])
    cuts = cuts.filter(partial(cuts_preprocess, lang=lang))

    llm_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_id, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        torch_dtype=torch.float16,
        attn_implementation='sdpa',
        device_map="cpu"
    )
    device = "cuda"
    model.to(device)

    def process_cuts(cut_list):
        inputs = []
        instructions = []
        for cut in cut_list:
            metadata = get_metadata(cut)
            input_text, user_prompt, system_message = generate_input(metadata, mode)
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if user_prompt:
                messages.append({"role": "user", "content": f"{input_text} {user_prompt}"})
            else:
                messages.append({"role": "user", "content": f"{input_text}"})

            instructions.append(user_prompt)
            inputs.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    chat_template=CHAT_TEMPLATE,
                    add_generation_prompt=True,
                )
            )

        torch.cuda.empty_cache()
        model_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256, # allow more tokens if possible
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, cut in enumerate(cut_list):
            cut.supervisions[0].response = responses[i]
            cut.supervisions[0].instruction = instructions[i]
        return cut_list

    bs = args.bs * nshards
    total_cuts, valid_cuts = 0, 0
    for cut_list in batch_cutset(cuts, bs=bs):
        # split into nshards by inner-batch index modulus
        cut_list = [c for idx, c in enumerate(cut_list) if idx % nshards == shard]
        if not cut_list:
            continue
        cut_list = process_cuts(cut_list)
        for cut in cut_list:
            total_cuts += 1
            valid_cuts += 1
            results.append(cut)
        print(f"\rProcesss: {total_cuts}", end='')

    cuts = CutSet.from_cuts(results)
    cuts.to_file(f"{output}/{mode}_{shard}.jsonl.gz")
    print(f"[Finish Set:{name} Shard:{shard}] {valid_cuts} / {total_cuts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process manifests')
    parser.add_argument('--manifests', default='myfolder/configs/paralinguistic_raw.yaml',
                        type=str, help='Yaml of all manifests')
    parser.add_argument('--output-dir', default='./myfolder/manifests/tmp', type=str, help='Output dir.')
    parser.add_argument('--mode', choices=('sift_s', 'sift_sp', 'simt_sp_1', 'simt_sp_2'), required=True,
                        type=str, help='Mode for different type of data generation.')
    parser.add_argument('--nshards', default=1, type=int, help='Split each manifest into N shards.')
    parser.add_argument('--shard', default=0, type=int, help='Current shard to process.')
    parser.add_argument('--bs', default=200, type=int, help='Size of each batch.')
    args = parser.parse_args()

    with open(args.manifests, 'r') as f:
        dsets = yaml.load(f, Loader=yaml.FullLoader)
    dsets = sorted(dsets, key=lambda x: x['hours'], reverse=False)

    for dset in dsets:
        main(dset, args)
