"""
Checkpoint utilities for Auden.

This file includes code adapted from the Icefall project:
https://github.com/k2-fsa/icefall

Original Icefall license: Apache 2.0
Copyright  2021-2022  Xiaomi Corporation  (authors: Fangjun Kuang,
                                                    Zengwei Yao)

Substantial parts of the checkpoint loading and model averaging logic
were adapted from icefall/checkpoint.py, with modifications for Auden's model system.

Key functions adapted from Icefall:
- load_trainer_checkpoint, save_trainer_checkpoint
- update_averaged_model, average_state_dict
- average_checkpoints_with_averaged_model
- make_averaged_model_state_dict, find_checkpoints

The user-facing functions get_model_from_trainer_checkpoint and
get_averaged_model_from_trainer_checkpoints are Auden-specific additions.
"""

import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from lhotse.dataset.sampling.base import CutSampler
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

LRSchedulerType = object


def load_trainer_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load trainer checkpoint (for resuming training) from a checkpoint file.

    This loads a TRAINER CHECKPOINT which includes model + optimizer + scheduler + scaler + progress.
    This is different from MODEL CHECKPOINTS (for deployment/inference) which only contain model weights + config.

    Args:
        filename: Path to the trainer checkpoint file.
        model: Model to load state into (optional).
        model_avg: Averaged model to load state into (optional).
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        scaler: GradScaler to load state into (optional).
        sampler: Sampler to load state into (optional).
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function.

    Returns:
        Dictionary containing remaining checkpoint data (e.g., batch_idx_train).

    Note:
        - This loads a TRAINER CHECKPOINT for resuming training
        - For model deployment, use model.from_pretrained() instead
        - Trainer checkpoints contain: model + optimizer + scheduler + scaler + progress
        - Model checkpoints contain: model weights + config
    """
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    if model is not None and "model" in checkpoint:
        if next(iter(checkpoint["model"])).startswith("module."):
            logging.info("Loading checkpoint saved by DDP")

            dst_state_dict = model.state_dict()
            src_state_dict = checkpoint["model"]
            for key in dst_state_dict.keys():
                src_key = "{}.{}".format("module", key)
                dst_state_dict[key] = src_state_dict.pop(src_key)
            assert len(src_state_dict) == 0
            model.load_state_dict(dst_state_dict, strict=strict)
        else:
            if next(iter(model.state_dict())).startswith("module."):
                model.module.load_state_dict(checkpoint["model"], strict=strict)
            else:
                model.load_state_dict(checkpoint["model"], strict=strict)

        checkpoint.pop("model")

    if model_avg is not None and "model_avg" in checkpoint:
        logging.info("Loading averaged model")
        model_avg.load_state_dict(checkpoint["model_avg"], strict=strict)
        checkpoint.pop("model_avg")

    def load(name, obj):
        s = checkpoint.get(name, None)
        if obj and s:
            obj.load_state_dict(s)
            checkpoint.pop(name)

    load("optimizer", optimizer)
    load("scheduler", scheduler)
    load("grad_scaler", scaler)
    load("sampler", sampler)

    return checkpoint


def save_trainer_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    model_avg: Optional[nn.Module] = None,
    batch_idx_train: int = 0,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[Any] = None,
    rank: int = 0,
) -> None:
    """
    Save trainer checkpoint (for resuming training) to a checkpoint file (only on rank 0).

    This saves a TRAINER CHECKPOINT which includes model + optimizer + scheduler + scaler + progress.
    This is different from MODEL CHECKPOINTS (for deployment/inference) which only contain model weights + config.

    Args:
        filename: Path to save the trainer checkpoint.
        model: Current training model (can be wrapped in DDP).
        model_avg: Averaged model weights (should be CPU-side).
        batch_idx_train: Current global step.
        optimizer: Optimizer to save.
        scheduler: LR scheduler to save.
        scaler: AMP GradScaler to save.
        sampler: Optional sampler (e.g., CutSampler).
        rank: Only saves if rank == 0 (for DDP).

    Note:
        - This creates a TRAINER CHECKPOINT for resuming training
        - For model deployment, use model.save_pretrained() instead
        - Trainer checkpoints contain: model + optimizer + scheduler + scaler + progress
        - Model checkpoints contain: model weights + config
    """
    if rank != 0:
        return

    logging.info(f"Saving checkpoint to {filename}")

    # Unwrap DDP if needed
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model_avg, DDP):
        model_avg = model_avg.module

    def get_state_dict(m: Optional[nn.Module]) -> Optional[Dict[str, Tensor]]:
        if m is None:
            return None
        state = m.state_dict()
        ignore_keys = getattr(m, "exclude_from_checkpoint", None)
        if ignore_keys:
            state = {
                k: v
                for k, v in state.items()
                if not any(k == ign or k.startswith(ign + ".") for ign in ignore_keys)
            }
        return state

    checkpoint = {
        "model": get_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "grad_scaler": scaler.state_dict() if scaler else None,
        "sampler": sampler.state_dict() if sampler else None,
        "batch_idx_train": batch_idx_train,
    }

    if model_avg is not None:
        checkpoint["model_avg"] = get_state_dict(model_avg.to(torch.float32))

    torch.save(checkpoint, filename)


def average_state_dict(
    state_dict_1: Dict[str, Tensor],
    state_dict_2: Dict[str, Tensor],
    weight_1: float,
    weight_2: float,
    scaling_factor: float = 1.0,
) -> Dict[str, Tensor]:
    """Average two state_dict with given weights:
    state_dict_1 = (state_dict_1 * weight_1 + state_dict_2 * weight_2)
      * scaling_factor
    It is an in-place operation on state_dict_1 itself.
    """
    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()
    for k, v in state_dict_1.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())
    for k in uniqued_names:
        v = state_dict_1[k]
        if torch.is_floating_point(v):
            v *= weight_1
            v += state_dict_2[k].to(device=state_dict_1[k].device) * weight_2
            v *= scaling_factor


def update_averaged_model(
    average_period,
    batch_idx_train,
    model_cur: Union[nn.Module, DDP],
    model_avg: nn.Module,
) -> None:
    """Update the averaged model:
    model_avg = model_cur * (average_period / batch_idx_train)
      + model_avg * ((batch_idx_train - average_period) / batch_idx_train)

    Args:
      params:
        User defined parameters, e.g., epoch, loss.
      model_cur:
        The current model.
      model_avg:
        The averaged model to be updated.
    """
    weight_cur = average_period / batch_idx_train
    weight_avg = 1 - weight_cur

    if isinstance(model_cur, DDP):
        model_cur = model_cur.module

    cur = model_cur.state_dict()
    avg = model_avg.state_dict()

    average_state_dict(
        state_dict_1=avg,
        state_dict_2=cur,
        weight_1=weight_avg,
        weight_2=weight_cur,
    )


def find_checkpoints(out_dir: Path, iteration: int = 0) -> List[str]:
    """Find all available checkpoints in a directory.

    The checkpoint filenames have the form: `checkpoint-xxx.pt`
    where xxx is a numerical value.

    Assume you have the following checkpoints in the folder `foo`:

        - checkpoint-1.pt
        - checkpoint-20.pt
        - checkpoint-300.pt
        - checkpoint-4000.pt

    Case 1 (Return all checkpoints)::

      find_checkpoints(out_dir='foo')

    Case 2 (Return checkpoints newer than checkpoint-20.pt, i.e.,
    checkpoint-4000.pt, checkpoint-300.pt, and checkpoint-20.pt)

        find_checkpoints(out_dir='foo', iteration=20)

    Case 3 (Return checkpoints older than checkpoint-20.pt, i.e.,
    checkpoint-20.pt, checkpoint-1.pt)::

        find_checkpoints(out_dir='foo', iteration=-20)

    Args:
      out_dir:
        The directory where to search for checkpoints.
      iteration:
        If it is 0, return all available checkpoints.
        If it is positive, return the checkpoints whose iteration number is
        greater than or equal to `iteration`.
        If it is negative, return the checkpoints whose iteration number is
        less than or equal to `-iteration`.
    Returns:
      Return a list of checkpoint filenames, sorted in descending
      order by the numerical value in the filename.
    """
    checkpoints = list(glob.glob(f"{out_dir}/checkpoint-[0-9]*.pt"))
    pattern = re.compile(r"checkpoint-([0-9]+).pt")
    iter_checkpoints = []
    for c in checkpoints:
        result = pattern.search(c)
        if not result:
            logging.warn(f"Invalid checkpoint filename {c}")
            continue

        iter_checkpoints.append((int(result.group(1)), c))

    # iter_checkpoints is a list of tuples. Each tuple contains
    # two elements: (iteration_number, checkpoint-iteration_number.pt)

    iter_checkpoints = sorted(iter_checkpoints, reverse=True, key=lambda x: x[0])
    if iteration >= 0:
        ans = [ic[1] for ic in iter_checkpoints if ic[0] >= iteration]
    else:
        assert iter_checkpoints[0][0] >= -iteration
        ans = [ic[1] for ic in iter_checkpoints if ic[0] <= -iteration]

    return ans


def average_checkpoints_with_averaged_model(
    filename_start: str,
    filename_end: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    """Average model parameters over the range with given
    start model (excluded) and end model.

    Let start = batch_idx_train of model-start;
        end = batch_idx_train of model-end;
        interval = end - start.
    Then the average model over range from start (excluded) to end is
    (1) avg = (model_end * end - model_start * start) / interval.
    It can be written as
    (2) avg = model_end * weight_end + model_start * weight_start,
        where weight_end = end / interval,
              weight_start = -start / interval = 1 - weight_end.
    Since the terms `weight_end` and `weight_start` would be large
    if the model has been trained for lots of batches, which would cause
    overflow when multiplying the model parameters.
    To avoid this, we rewrite (2) as:
    (3) avg = (model_end + model_start * (weight_start / weight_end))
              * weight_end

    The model index could be epoch number or iteration number.

    Args:
      filename_start:
        Checkpoint filename of the start model. We assume it
        is saved by :func:`save_checkpoint`.
      filename_end:
        Checkpoint filename of the end model. We assume it
        is saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    """
    state_dict_start = torch.load(filename_start, map_location=device)
    state_dict_end = torch.load(filename_end, map_location=device)

    batch_idx_train_start = state_dict_start["batch_idx_train"]
    batch_idx_train_end = state_dict_end["batch_idx_train"]
    interval = batch_idx_train_end - batch_idx_train_start
    assert interval > 0, interval
    weight_end = batch_idx_train_end / interval
    weight_start = 1 - weight_end

    model_end = state_dict_end["model_avg"]
    model_start = state_dict_start["model_avg"]
    avg = model_end

    # scale the weight to avoid overflow
    average_state_dict(
        state_dict_1=avg,
        state_dict_2=model_start,
        weight_1=1.0,
        weight_2=weight_start / weight_end,
        scaling_factor=weight_end,
    )

    return avg


def make_averaged_model_state_dict(exp_dir, iter, epoch, avg):
    if iter > 0:
        filenames = find_checkpoints(exp_dir, iteration=-iter)[: avg + 1]
        if len(filenames) == 0:
            raise ValueError(f"No checkpoints found for" f" --iter {iter}, --avg {avg}")
        elif len(filenames) < avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {iter}, --avg {avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        logging.info(
            "Calculating the averaged model over iteration checkpoints"
            f" from {filename_start} (excluded) to {filename_end}"
        )
    else:
        assert avg > 0, avg
        start = epoch - avg
        assert start >= 1, start
        filename_start = f"{exp_dir}/epoch-{start}.pt"
        filename_end = f"{exp_dir}/epoch-{epoch}.pt"
        logging.info(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {epoch}"
        )

    state_dict = average_checkpoints_with_averaged_model(
        filename_start=filename_start,
        filename_end=filename_end,
    )

    return state_dict


def remove_trainer_checkpoints(
    out_dir: Path,
    topk: int,
    rank: int = 0,
):
    """Remove trainer checkpoints from the given directory.

    We assume that checkpoint filename has the form `checkpoint-xxx.pt`
    where xxx is a number, representing the number of processed batches
    when saving that checkpoint. We sort checkpoints by filename and keep
    only the `topk` trainer checkpoints with the highest `xxx`.

    Args:
      out_dir:
        The directory containing trainer checkpoints to be removed.
      topk:
        Number of checkpoints to keep.
      rank:
        If using DDP for training, it is the rank of the current node.
        Use 0 if no DDP is used for training.
    """
    assert topk >= 1, topk
    if rank != 0:
        return
    checkpoints = find_checkpoints(out_dir)

    if len(checkpoints) == 0:
        logging.warn(f"No trainer checkpoints found in {out_dir}")
        return

    if len(checkpoints) <= topk:
        return

    to_remove = checkpoints[topk:]
    for c in to_remove:
        os.remove(c)


def freeze_modules(model: nn.Module, frozen_modules: list[str]):
    """
    Freeze parameters of specified submodules in a model. Supports dot-separated paths like 'encoder.embed'.

    Args:
        model (torch.nn.Module): The model containing submodules to freeze.
        frozen_modules (list of str): List of module paths to freeze (e.g., ['encoder', 'decoder.attn']).

    Raises:
        ValueError: If any specified submodule path is invalid.
    """
    if not frozen_modules:
        logging.info("No modules specified for freezing, skipping freeze step")
        return

    for path in frozen_modules:
        submodule = model
        parts = path.split(".")
        for part in parts:
            if not hasattr(submodule, part):
                raise ValueError(
                    f"Model has no submodule '{path}' (failed at '{part}')"
                )
            submodule = getattr(submodule, part)

        logging.info(f"Freezing module: {path}")
        for param in submodule.parameters():
            param.requires_grad = False


def get_model_from_trainer_checkpoint(
    model_dir: Union[str, Path],
    trainer_checkpoint_path: Union[str, Path],
    model_type: str = "model_avg",
    model_name: str = "final_model.pt",
) -> None:
    """
    Extract a model from a trainer checkpoint and save it as a model checkpoint.

    This function extracts the model state dict from a trainer checkpoint and saves it
    to model_dir/model_name.pt.

    Args:
        model_dir: Directory to save the model checkpoint.
        trainer_checkpoint_path: Path to the trainer checkpoint file.
        model_type: Which model to extract ("model" or "model_avg").
        model_name: Name for the saved model file (should be xxx.pt).

    Example:
        ```python
        # Extract averaged model from trainer checkpoint
        get_model_from_trainer_checkpoint(
            model_dir="./models/",
            trainer_checkpoint_path="checkpoint-1000.pt",
            model_type="model_avg",
            model_name="final_model.pt"
        )
        ```
    """
    model_dir = Path(model_dir)
    trainer_checkpoint_path = Path(trainer_checkpoint_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"Extracting {model_type} from trainer checkpoint: {trainer_checkpoint_path}"
    )

    # Load checkpoint
    checkpoint = torch.load(trainer_checkpoint_path, map_location="cpu")

    # Validate model_type
    if model_type not in ["model", "model_avg"]:
        raise ValueError(
            f"model_type must be 'model' or 'model_avg', got '{model_type}'"
        )

    # Check if requested model exists in checkpoint
    if model_type not in checkpoint:
        available = list(checkpoint.keys())
        raise ValueError(
            f"Model type '{model_type}' not found in checkpoint. Available: {available}"
        )

    model_state_dict = checkpoint[model_type]

    # Save model state dict directly to model_dir/model_name.pt
    model_path = model_dir / model_name
    torch.save(model_state_dict, model_path)

    logging.info(f"Successfully saved {model_type} to: {model_path}")


def get_averaged_model_from_trainer_checkpoints(
    model_dir: Union[str, Path],
    epochs: Optional[int] = None,
    iters: Optional[int] = None,
    avg: int = 5,
    model_name: str = "averaged_model.pt",
) -> None:
    """
    Get an averaged model from trainer checkpoints and save it as a model checkpoint.

    This function averages multiple trainer checkpoints using Icefall logic and saves
    the result to model_dir/model_name.pt.

    Args:
        model_dir: Directory containing trainer checkpoints and where to save the averaged model.
        epochs: Epoch number (if provided, uses epoch-based averaging).
        iters: Iteration number (if provided, uses iteration-based averaging).
        avg: Number of checkpoints to average.
        model_name: Name for the saved model file (should be xxx.pt).

    Example:
        ```python
        # Average last 5 checkpoints from iteration 1000
        get_averaged_model_from_trainer_checkpoints(
            model_dir="./exp_dir/",
            iters=1000,
            avg=5,
            model_name="final_model.pt"
        )

        # Average last 3 epochs
        get_averaged_model_from_trainer_checkpoints(
            model_dir="./exp_dir/",
            epochs=10,
            avg=3,
            model_name="epoch_avg_model.pt"
        )
        ```
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Getting averaged model from trainer checkpoints in {model_dir}")

    # Use the legacy make_averaged_model_state_dict function
    averaged_state_dict = make_averaged_model_state_dict(
        exp_dir=model_dir, iter=iters or 0, epoch=epochs or 0, avg=avg
    )

    # Save averaged model state dict directly to model_dir/model_name.pt
    model_path = model_dir / model_name
    torch.save(averaged_state_dict, model_path)

    logging.info(f"Successfully saved averaged model to: {model_path}")
