"""Voice multitask evaluation script.

Evaluates the model on test datasets for all 4 tasks:
speaker ID, emotion, gender, and age.

Usage:
    python evaluate.py \
        exp_dir=./exp/voice_exp \
        checkpoint.epoch=10 checkpoint.avg=5
"""

import logging
import os

import hydra
import torch
import yaml
from data_module import VoiceDataset
from model import VoiceMultitaskModel
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import DynamicBucketingSampler, OnTheFlyFeatures
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints


def get_test_dataloaders(cfg):
    """Create test dataloaders from config."""
    test_dls = []
    test_names = []
    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    def remove_short_utterance(cut):
        return cut.duration >= 1.0

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(
            getattr(cfg.data, "sampling_rate", 16000)
        )
        cutset = cutset.filter(remove_short_utterance)
        test_name = test_set["name"]
        testset = VoiceDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cutset,
            max_duration=cfg.data.max_duration,
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

    # Get dataloaders
    test_sets, test_dls = get_test_dataloaders(cfg)

    # Initialize model
    checkpoint_path = None
    ckpt_cfg = cfg.checkpoint
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
            raise ValueError(
                "When averaging, set either checkpoint.iter or checkpoint.epoch"
            )
        checkpoint_path = os.path.join(cfg.exp_dir, model_name)
        if not os.path.exists(checkpoint_path):
            generate_model_checkpoint_from_trainer_checkpoints(
                model_dir=cfg.exp_dir,
                epochs=epoch or None,
                iters=iters or None,
                avg=avg,
                model_name=model_name,
            )

    model = VoiceMultitaskModel.from_pretrained(checkpoint_path)

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)
    model.eval()
    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    # Evaluate on each test set
    for test_set_name, test_dl in zip(test_sets, test_dls):
        num_cuts = 0
        try:
            num_batches = len(test_dl)
        except TypeError:
            num_batches = "?"

        # Collect predictions for all tasks
        task_predictions = {
            "id": {"correct": 0, "total": 0},
            "emotion": {"correct": 0, "total": 0},
            "gender": {"correct": 0, "total": 0},
            "age": {"correct": 0, "total": 0},
        }

        result_str = f"\n{'='*60}\nEvaluation Results for {test_set_name}\n{'='*60}\n"

        for batch_idx, batch in enumerate(test_dl):
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
            num_cuts += len(cut_ids)

            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            supervisions = batch["supervisions"]

            # Evaluate each task
            for task in ["id", "emotion", "gender", "age"]:
                labels = supervisions[task]
                # Generate predictions for this task
                pred_labels, _, _ = model.generate((feature, feature_lens), task=task, topk=1)

                # Count correct predictions (skip missing labels)
                for pred_list, gt_label in zip(pred_labels, labels):
                    if gt_label not in [None, "None", "Null"]:
                        pred = pred_list[0]  # top-1 prediction
                        if pred == gt_label:
                            task_predictions[task]["correct"] += 1
                        task_predictions[task]["total"] += 1

            if batch_idx % 20 == 1:
                logging.info(f"Processed {num_cuts} cuts already.")

        logging.info("Finish evaluation")

        # Calculate and log accuracies
        for task in ["id", "emotion", "gender", "age"]:
            correct = task_predictions[task]["correct"]
            total = task_predictions[task]["total"]
            if total > 0:
                acc = 100.0 * correct / total
                result_str += f"{task.capitalize()} Accuracy: {acc:.2f}% ({correct}/{total})\n"
            else:
                result_str += f"{task.capitalize()} Accuracy: N/A (no samples)\n"

        logging.info(result_str)
        result_file = os.path.join(cfg.exp_dir, f"{test_set_name}_result.txt")
        logging.info(f"Save result to {result_file}")
        with open(result_file, "w") as f:
            f.write(result_str)
        logging.info("Done")


if __name__ == "__main__":
    main()

