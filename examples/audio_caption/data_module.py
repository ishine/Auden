"""Audio Caption example datamodule built on BaseLhotseDatamodule. It has the same logic as
the CLAP example but might be different in the future.

This module provides train/valid DataLoaders for audio-caption tasks using Lhotse CutSets.
Captions are read from a configurable supervision field `label_field` (default: "caption").

Manifest expectations:
- Each cut should contain at least one supervision with a caption in `label_field`.
- If multiple supervisions exist, we keep only the first one.
- If a supervision's caption field is a list, we randomly pick one caption per supervision.
- Captions can be stored under `supervision.custom[label_field]` or as `supervision.<label_field>`.
"""

import random
from re import sub

import torch
import yaml
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers


class AudioCaptionDataset(torch.utils.data.Dataset):
    """Wraps a CutSet and emits (features, num_frames, caption_string)."""

    def __init__(
        self,
        input_strategy,
        cut_transforms=None,
        input_transforms=None,
        return_cuts=False,
        label_field: str = "caption",
    ):
        self.input_strategy = input_strategy
        self.cut_transforms = cut_transforms
        self.input_transforms = input_transforms
        self.return_cuts = return_cuts
        self.label_field = label_field
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def _text_preprocess(self, sentence):

        # transform to lower case
        sentence = sentence.lower()

        # remove any forgotten space before punctuation and double space
        sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")

        # remove punctuations
        # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
        sentence = sub('[(,.!?;:|")]', " ", sentence).replace("  ", " ")
        return sentence

    def __getitem__(self, cuts):
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

        # Collect captions from ALL supervisions; if multiple captions exist, pick one randomly
        captions = []
        for cut in cuts:
            has_caption = False
            for supervision in cut.supervisions:
                cap_val = None
                if hasattr(supervision, "custom") and isinstance(
                    supervision.custom, dict
                ):
                    cap_val = supervision.custom.get(self.label_field, None)
                if cap_val is None and hasattr(supervision, self.label_field):
                    cap_val = getattr(supervision, self.label_field)
                if cap_val is None:
                    continue
                has_caption = True
                if isinstance(cap_val, (list, tuple)):
                    chosen = random.choice(cap_val) if len(cap_val) > 0 else ""
                    captions.append(self._text_preprocess(str(chosen)))
                else:
                    captions.append(self._text_preprocess(str(cap_val)))
            if not has_caption:
                raise ValueError(
                    f"Missing label_field '{self.label_field}' in all supervisions for cut {cut.id}"
                )

        batch = {
            "inputs": features,
            "supervisions": {
                "caption": captions,
            },
        }
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for _ in cut.supervisions
            ]
        return batch

    def __len__(self):
        return 0


class AudioCaptionDatamodule(BaseLhotseDatamodule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _filter_cutset(self, cutset, split="train"):
        def keep(c):
            if c.duration < 1.0 or c.duration > 60.0:
                return False
            if len(c.supervisions) > 1:
                c.supervisions = [c.supervisions[0]]
            return True

        return cutset.filter(keep)

    def setup_train(self):
        with open(self.cfg.train_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        cutset = self._build_train_mux_cutset(cfg_list)
        cutset = self._filter_cutset(cutset, split="train")
        sampler = self._build_train_sampler(cutset)
        dataset = AudioCaptionDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
            label_field=self.cfg.get("label_field", "caption"),
        )
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        self.train_dl = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.cfg.get("num_workers", 8),
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def setup_valid(self):
        with open(self.cfg.valid_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        self.valid_dls = []
        self.valid_names = []
        for spec in cfg_list:
            cutset = CutSet.from_file(spec["manifest"]).resample(self.sampling_rate)
            cutset = self._filter_cutset(cutset, split="valid")
            dataset = AudioCaptionDataset(
                input_strategy=self.input_strategy,
                return_cuts=True,
                label_field=self.cfg.get("label_field", "caption"),
            )
            sampler = DynamicBucketingSampler(
                cutset, max_duration=self.cfg.sampler.max_duration, shuffle=False
            )
            dl = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=None,
                num_workers=self.cfg.get("num_workers", 8),
                persistent_workers=False,
            )
            self.valid_names.append(spec["name"])
            self.valid_dls.append(dl)
