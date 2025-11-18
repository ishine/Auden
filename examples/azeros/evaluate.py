"""Demo script for AZeroS decoding.
"""

import logging
import os
from functools import partial

import hydra
import torch
import yaml
from data_module import Speech2ResponseDataset
from model import AzerosModel
from lhotse import (
    CutSet, 
    Fbank, 
    FbankConfig,
    WhisperFbank,
    WhisperFbankConfig,
    set_audio_duration_mismatch_tolerance,
)
from lhotse.dataset import DynamicBucketingSampler, OnTheFlyFeatures
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints


def _remove_long_short_utterance(c):
    c.supervisions = [c.supervisions[0]]
    if c.duration < 1.0 or c.duration > 50.0:
        return False
    return True


def _unified_language_code(c, lang):
    if lang is not None:
        c.supervisions[0].language = lang
    return c


def get_test_dataloaders(cfg):
    test_dls = []
    test_names = []
    if cfg.data.whisper_fbank:
        input_strategy = OnTheFlyFeatures(WhisperFbank(WhisperFbankConfig(num_filters=80)))
    else:
        input_strategy = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(16000)
        cutset = cutset.filter(_remove_long_short_utterance)
        lang = test_set.get("lang", "zh")
        cutset = cutset.map(partial(_unified_language_code, lang=lang))
        if cfg.data.whisper_fbank:
            cutset = cutset.pad(num_samples=480000)
        test_name = test_set["name"]
        testset = Speech2ResponseDataset(
            input_strategy=input_strategy,
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cutset,
            max_duration=cfg.data.max_duration,
            max_cuts=cfg.data.max_cuts,
            shuffle=False,
        )
        test_dl = DataLoader(
            testset,
            batch_size=None,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
        )
        test_dls.append(test_dl)
        test_names.append(test_name)

    return test_names, test_dls


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
@torch.no_grad()
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # initialize dataloader
    set_audio_duration_mismatch_tolerance(0.1)
    test_sets, test_dls = get_test_dataloaders(cfg)

    # Resolve checkpoint path
    ckpt_cfg = cfg.checkpoint
    checkpoint_path = None
    filename = ckpt_cfg.get("filename", None)
    if filename:
        checkpoint_path = (
            filename if os.path.isabs(filename) else os.path.join(cfg.exp_dir, filename)
        )
    else:
        avg = ckpt_cfg.get("avg", 0)
        iters = ckpt_cfg.get("iter", 0)
        epoch = ckpt_cfg.get("epoch", 0)
        if iters > 0:
            model_name = f"averaged-iter-{iters}-avg-{avg}.pt"
        elif epoch > 0:
            model_name = f"averaged-epoch-{epoch}-avg-{avg}.pt"
        else:
            model_name = "pretrained.pt"
        checkpoint_path = os.path.join(cfg.exp_dir, model_name)
        if not os.path.exists(checkpoint_path) and (iters > 0 or epoch > 0):
            generate_model_checkpoint_from_trainer_checkpoints(
                model_dir=cfg.exp_dir,
                epochs=epoch or None,
                iters=iters or None,
                avg=avg,
                model_name=model_name,
            )

    # load model
    model = AzerosModel.from_pretrained(checkpoint_path, strict=False)
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)
    model.eval()
    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    generate_config = {
        "max_new_tokens": 200,
        "num_beams": 1,
        "do_sample": False,
        "min_length": 1,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "top_p": None,
        "top_k": None,
        "temperature": None
    }

    # run evaluation
    for test_set_name, test_dl in zip(test_sets, test_dls):
        num_cuts = 0

        for batch_idx, batch in enumerate(test_dl):
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
            num_cuts += len(cut_ids)
            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)

            # change this for other purposes
            messages = [
                [
                    {"role": "user", "content": f"{model.audio_token_wrapped} {instruction}"},
                ] for instruction in batch["supervisions"]["instruction"]
            ]
            hyps = model.generate((feature, feature_lens), messages, **generate_config)
            texts = batch["supervisions"]["response"]

            for hyp, ref in zip(hyps, texts):
                print('-' * 20)
                print(f"Ref-Text: {ref}")
                print(f"Hyp-Text: {hyp}")
                print()


if __name__ == "__main__":
    main()
