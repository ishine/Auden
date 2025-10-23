"""Voice multitask datamodule for speaker ID, emotion, gender, and age classification."""

import logging

import torch
import yaml
from lhotse import CutSet, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers


class VoiceDataset(torch.utils.data.Dataset):
    """
    Dataset for voice multitask learning.

    Reads labels from supervision attributes (speaker, emotion, gender, age_group).
    Supports missing labels by using "Null" as placeholder.
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
        # Workaround for HDF5 memory growth when using precomputed features
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts):
        # Periodically close HDF5 handles to avoid memory growth
        self.hdf5_fix.update()
        # Sort by duration so the first determines batch dims
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms (e.g., speed perturb, padding)
        if self.cut_transforms is not None:
            for tnfm in self.cut_transforms:
                cuts = tnfm(cuts)

        # Sort again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Convert cuts to inputs via the input strategy
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            features, _, cuts = input_tpl
        else:
            features, _ = input_tpl

        # Supervision intervals (sequence_idx/start/num frames or samples)
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply input transforms if any
        if self.input_transforms is not None:
            for tnfm in self.input_transforms:
                features = tnfm(features)

        # Extract labels for all tasks from supervision
        MISSING_LABEL = "Null"

        def _get_label(cut, field_name):
            """Extract label from supervision, return MISSING_LABEL if not present."""
            sup = cut.supervisions[0]
            label = getattr(sup, field_name, MISSING_LABEL)
            return label or MISSING_LABEL

        id_labels = [_get_label(c, "speaker") for c in cuts]
        emotion_labels = [_get_label(c, "emotion") for c in cuts]
        age_labels = [_get_label(c, "age_group") for c in cuts]
        gender_labels = [_get_label(c, "gender") for c in cuts]

        batch = {
            "inputs": features,
            "supervisions": {
                "id": id_labels,
                "emotion": emotion_labels,
                "gender": gender_labels,
                "age": age_labels,
            },
        }
        # Merge supervision intervals into supervisions (includes num_frames/samples)
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for _ in cut.supervisions
            ]
        return batch

    def __len__(self):
        return 0  # unused with bucketing samplers


class VoiceDatamodule(BaseLhotseDatamodule):
    """
    Datamodule for voice multitask learning.

    Handles data loading for speaker ID, emotion, gender, and age classification tasks.
    """

    def __init__(self, cfg):
        # Set audio duration mismatch tolerance (same as other examples)
        set_audio_duration_mismatch_tolerance(0.1)
        super().__init__(cfg)

    def _filter_cutset(self, cutset, split="train"):
        """Filter cutset to keep only valid samples."""

        def keep(c):
            # Filter duration
            if c.duration < 1.0 or c.duration > 60.0:
                return False
            # Keep only first supervision if multiple exist
            if len(c.supervisions) > 1:
                c.supervisions = [c.supervisions[0]]
            return True

        return cutset.filter(keep)

    def _build_mux_cutset(self, cfg_list):
        """Build multiplexed cutset from config list."""
        cutset_list = []
        cutset_hours = []
        for spec in cfg_list:
            logging.info(f"Getting {spec['manifest']} cuts")
            cutset = CutSet.from_file(spec["manifest"]).resample(self.sampling_rate)
            hours = spec.get("hours", 0)
            weight = spec.get("weights", 1)
            if self.cfg.use_infinite_dataset:
                cutset = cutset.repeat()
            else:
                cutset = cutset.repeat(weight)
            cutset[0].load_audio()
            cutset_hours.append(weight * hours)
            cutset_list.append(cutset)

        logging.info(
            f"Total {sum(cutset_hours)} hours from {len(cutset_hours)} manifests"
        )

        if len(cutset_list) > 1:
            return CutSet.mux(*cutset_list, weights=cutset_hours, stop_early=True)
        return cutset_list[0]

    def setup_train(self):
        """Setup training dataloader."""
        with open(self.cfg.train_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        cutset = self._build_mux_cutset(cfg_list)
        cutset = self._filter_cutset(cutset, split="train")
        sampler = self._build_train_sampler(cutset)
        dataset = VoiceDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
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
        """Setup validation dataloaders."""
        with open(self.cfg.valid_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        self.valid_dls = []
        self.valid_names = []
        for spec in cfg_list:
            cutset = CutSet.from_file(spec["manifest"]).resample(self.sampling_rate)
            cutset = self._filter_cutset(cutset, split="valid")
            dataset = VoiceDataset(
                input_strategy=self.input_strategy,
                return_cuts=True,
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

