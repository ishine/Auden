"""AZeroS example datamodule and dataset built on BaseLhotseDatamodule.

This module provides a Speech2ResponseDataset emitting fields needed by
AZeroS, and a AzerosDatamodule that builds training/validation
loaders from YAML configs, mirroring other examples.

example lhotse MonoCut with LLM response:
    MonoCut(id='utt_0001', start=0.0, duration=3.21, channel=0,
    supervisions=[SupervisionSegment(id='utt_0001', recording_id='rec_0001',
    start=0.0, duration=3.21, channel=0, text='你好，世界！', language='zh',
    speaker=None, gender='female', custom={'age': 'fourties', 'emotion': 'happy',
    'response': '你好！很高兴见到你！🌍', 'instruction': ''}, alignment=None)], features=None,
    recording=Recording(id='rec_0001', sources=[AudioSource(type='file', channels=[0],
    source='/path/to/dummy.wav')], sampling_rate=16000, num_samples=51360,
    duration=3.21, channel_ids=[0], transforms=None), custom=None)
"""

import logging
from collections import defaultdict
from functools import partial
from typing import Dict, List, Union
from functools import cache

import numpy as np
import torch
import yaml
from lhotse import CutSet, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers
from auden.utils.text_normalization import text_normalization


class Speech2ResponseDataset(torch.utils.data.Dataset):
    """General speech-to-response dataset supporting SpeechLLM finetuning.

    Emits a dict with:
    - inputs: Tensor (B, T, F)
    - supervisions: dict with keys: target_text, target_prompt,
      plus supervision intervals and optional cuts.
    """

    def __init__(
        self,
        input_strategy,
        cut_transforms=None,
        input_transforms=None,
        return_cuts=False,
    ):
        self.input_strategy = input_strategy
        self.cut_transforms = cut_transforms
        self.input_transforms = input_transforms
        self.return_cuts = return_cuts
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts) -> Dict[str, Union[torch.Tensor, List[str]]]:
        self.hdf5_fix.update()
        cuts = cuts.sort_by_duration(ascending=False)

        if self.cut_transforms is not None:
            for tnfm in self.cut_transforms:
                cuts = tnfm(cuts)

        cuts = cuts.sort_by_duration(ascending=False)

        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            features, _, cuts = input_tpl
        else:
            features, _ = input_tpl

        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        if self.input_transforms is not None:
            for tnfm in self.input_transforms:
                features = tnfm(features, supervision_segments=segments)

        # assemble supervision info per cut
        from torch.utils.data.dataloader import default_collate

        infos: List[Dict[str, Union[str, List[str]]]] = []
        for sequence_idx, cut in enumerate(cuts):
            supervision = cut.supervisions[0]
            if not hasattr(supervision, 'response'):
                response = supervision.text
                instruction = 'Repeat the above texts.'
            else:
                response = supervision.response
                instruction = getattr(supervision, 'instruction', '')

            infos.append(
                {
                    "response": response,
                    "instruction": instruction
                }
            )

        batch = {
            "inputs": features,
            "supervisions": default_collate(infos),
        }
        # include intervals
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [cut for cut in cuts]
        return batch

    def __len__(self):
        return 0


class AzerosDatamodule(BaseLhotseDatamodule):
    """AZeroS datamodule specialized from BaseLhotseDatamodule.

    Mixes multiple training datasets, applies normalization/filtering, and
    constructs DataLoaders for training/validation.
    """

    def __init__(self, cfg):
        # relax minor duration mismatch
        set_audio_duration_mismatch_tolerance(0.1)
        super().__init__(cfg)

    def _filter_cutset(self, cutset, split="train"):
        def text_normalization_on_cut(c):
            text = c.supervisions[0].text
            text = text_normalization(
                text,
                case="lower",
                space_between_cjk=False,
                remove_diacritics=False,
                remove_symbols=False,
            )
            c.supervisions[0].text = text
            return c

        def remove_short_and_long_utt(c):
            # Keep only utterances with duration between 1 second and 30 seconds
            if c.duration < 1.0 or c.duration > 30.0:
                return False
            return True

        def remove_multiple_supervisions(c):
            # some audio may have multiple supervisions, just take the first one
            if len(c.supervisions) > 1:
                c.supervisions = [c.supervisions[0]]
            return c

        def cleanup_utt_with_wrong_text(c):
            # remove empty or abnormally long texts, should be done after text normalization
            text = c.supervisions[0].text
            if len(text) == 0 or len(text) > c.duration * 30:
                return False
            return True

        cutset = cutset.filter(remove_short_and_long_utt)
        cutset = cutset.map(remove_multiple_supervisions)
        if self.cfg.text_normalization:
            # recommed no text normalization for LLM responses
            cutset = cutset.map(text_normalization_on_cut)
        cutset = cutset.filter(cleanup_utt_with_wrong_text)

        if self.cfg.get("pad_to_30s", False):
            logging.info(
                "Padded all audio to 30s. This is usually used with `Whisper` encoder."
            )
            cutset = cutset.pad(duration=30)

        return cutset

    def _build_train_mux_cutset(self, train_data_config):
        cutset_list = []
        cutset_hours = []
        langs_hours = defaultdict(int)

        for train_set in train_data_config:
            logging.info(f"Getting {train_set['manifest']} cuts")
            cutset = CutSet.from_file(train_set["manifest"]).resample(
                self.sampling_rate
            )
            hours = train_set.get("hours", 1.0)
            weight = train_set.get("weights", 1)
            lang = train_set.get("lang", None)
            if self.cfg.get("use_infinite_dataset", True):
                cutset = cutset.repeat()
            else:
                cutset = cutset.repeat(weight)
            cutset[0].load_audio()
            langs_hours[lang] += weight * hours
            cutset_hours.append(weight * hours)
            # unify language code into supervision for downstream usage
            cutset = cutset.map(partial(self._unified_language_code, lang=lang))
            cutset_list.append(cutset)

        for lang in langs_hours:
            logging.info(
                f"Getting {langs_hours[lang]} hours of training data from {lang} language"
            )
        logging.info(
            f"Getting totally {sum(cutset_hours)} hours of training data from {len(cutset_hours)} manifests"
        )

        if len(cutset_list) > 1:
            logging.info("Muxing cuts")
            cutset_train = CutSet.mux(
                *cutset_list,
                weights=cutset_hours,
                stop_early=True,
            )
        else:
            cutset_train = cutset_list[0]

        return cutset_train

    @staticmethod
    def _unified_language_code(c, lang=None):
        if lang is not None:
            c.supervisions[0].language = lang
        return c

    def setup_train(self):
        with open(self.cfg.train_data_config, "r") as file:
            train_data_config = yaml.load(file, Loader=yaml.FullLoader)

        train_cutset = self._build_train_mux_cutset(train_data_config)
        train_cutset = self._filter_cutset(train_cutset, split="train")
        train_sampler = self._build_train_sampler(train_cutset)

        train_dataset = Speech2ResponseDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
        )
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        self.train_dl = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.cfg.get("num_workers", 8),
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def setup_valid(self):
        with open(self.cfg.valid_data_config, "r") as file:
            valid_data_config = yaml.load(file, Loader=yaml.FullLoader)

        self.valid_dls = []
        self.valid_names = []
        for valid_set in valid_data_config:
            logging.info(f"Getting validation cuts: {valid_set['manifest']}")
            cutset = CutSet.from_file(valid_set["manifest"]).resample(
                self.sampling_rate
            )
            # propagate language if configured per-valid set
            lang = valid_set.get("lang")
            if lang:
                cutset = cutset.map(partial(self._unified_language_code, lang=lang))
            cutset = self._filter_cutset(cutset, split="valid")
            valid_name = valid_set["name"]

            valid_dataset = Speech2ResponseDataset(
                input_strategy=self.input_strategy,
                return_cuts=True,
            )

            valid_sampler = DynamicBucketingSampler(
                cutset,
                max_duration=self.cfg.sampler.max_duration,
                max_cuts=getattr(self.cfg.sampler, 'max_cuts', None),
                shuffle=False,
            )
            valid_dl = DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                batch_size=None,
                num_workers=self.cfg.get("num_workers", 8),
                persistent_workers=False,
            )

            self.valid_names.append(valid_name)
            self.valid_dls.append(valid_dl)
