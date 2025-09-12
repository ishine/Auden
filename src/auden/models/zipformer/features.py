import logging
import math
from typing import List

import torch


def construct_feature_extractor(
    *,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    device: str | torch.device = "cpu",
):
    try:
        import kaldifeat  # type: ignore
    except Exception as e:
        raise ImportError(
            "kaldifeat is required for feature extraction. Install with `pip install kaldifeat`."
        ) from e

    opts = kaldifeat.FbankOptions()
    opts.device = torch.device(device)
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = num_mel_bins
    opts.mel_opts.high_freq = -400
    return kaldifeat.Fbank(opts)


def _read_wavs_from_paths(paths: List[str], target_sample_rate: int):
    try:
        import torchaudio  # type: ignore
    except Exception as e:
        raise ImportError(
            "torchaudio is required to read audio files. Install with `pip install torchaudio`."
        ) from e

    wavs = []
    for f in paths:
        wave, sample_rate = torchaudio.load(f)  # shape: (1 or 2, num_samples)
        data = wave[0]  # take first channel
        if sample_rate != target_sample_rate:
            logging.warning(
                f"Sample rate for {f} is {sample_rate}Hz. Resampling to {target_sample_rate}Hz..."
            )
            data = torchaudio.functional.resample(
                data, orig_freq=sample_rate, new_freq=target_sample_rate
            )
        wavs.append(data)
    return wavs


def extract_features(
    input,
    feature_extractor,
    *,
    target_sample_rate: int = 16000,
    device: str | torch.device = "cpu",
):
    """Extract features from different input forms.

    Accepts:
      - (x, x_lens): precomputed features
      - List[str]: wav file paths
      - List[Tensor] or List[np.ndarray]: mono waveforms
    Returns: (features, feature_lens)
    """
    import numpy as np
    from torch.nn.utils.rnn import pad_sequence

    # Case 1: already precomputed
    if isinstance(input, tuple) and len(input) == 2:
        return input

    if isinstance(input, list) and input and isinstance(input[0], str):
        logging.info(f"Reading sound files: {input}")
        wavs = _read_wavs_from_paths(input, target_sample_rate)
    elif isinstance(input, list) and all(
        isinstance(x, (torch.Tensor, np.ndarray)) for x in input
    ):
        wavs = []
        for i, data in enumerate(input):
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            if data.ndim != 1:
                raise ValueError(
                    f"Waveform at index {i} must be 1D mono, but got shape {data.shape}"
                )
            wavs.append(data)
    else:
        raise ValueError(
            "Input must be either (x, x_lens), List[str] of wav paths, or List[1D waveform arrays]."
        )

    # Extract features
    features_list = feature_extractor(wavs)  # returns List[Tensor]
    feature_lens = [f.size(0) for f in features_list]
    features = pad_sequence(
        features_list, batch_first=True, padding_value=math.log(1e-10)
    )
    feature_lens = torch.tensor(feature_lens, dtype=torch.long)
    return features.to(device), feature_lens.to(device)
