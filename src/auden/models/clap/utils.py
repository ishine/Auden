from __future__ import annotations

import torch


@torch.no_grad()
def _retrieval_metrics(sim: torch.Tensor, k: int = 10):
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sim: Similarity matrix of shape (N, M) where rows are queries and
             columns are targets; larger is more similar.
        k:  Cutoff for mAP@k (MRR@k) and top-k recalls.

    Returns:
        r1, r5, r10, medr, meanr, mAPk – recalls in %, ranks are 1-based.

    Notes:
        - mAPk here uses MRR@k (reciprocal rank restricted to top-k), which
          matches common 1-to-1 retrieval practice.
    """
    ranks = torch.argsort(sim, dim=1, descending=True)
    correct = torch.arange(sim.size(0), device=sim.device).view(-1, 1)
    pos = (ranks == correct).nonzero(as_tuple=False)[:, 1] + 1
    r1 = (pos <= 1).float().mean().item() * 100.0
    r5 = (pos <= 5).float().mean().item() * 100.0
    r10 = (pos <= 10).float().mean().item() * 100.0
    medr = pos.median().item()
    meanr = pos.float().mean().item()
    # mAP@k as MRR@k (reciprocal rank restricted to top-k)
    in_topk = pos <= k
    mAPk = (1.0 / pos[in_topk].float()).mean().item() * 100.0 if in_topk.any() else 0.0
    return r1, r5, r10, medr, meanr, mAPk


@torch.no_grad()
def t2a_metric(text_embeds: torch.Tensor, audio_embeds: torch.Tensor):
    """Text→Audio retrieval metrics (1-to-1).

    Embeddings are L2-normalized in-function and cosine similarity is used.

    Args:
        text_embeds: (N, D) text embedding matrix
        audio_embeds: (M, D) audio embedding matrix

    Returns:
        r1, r5, r10, medr, meanr, mAPk (with k=10)
    """
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
    audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)
    sim = text_embeds @ audio_embeds.t()
    return _retrieval_metrics(sim)


@torch.no_grad()
def a2t_metric(audio_embeds: torch.Tensor, text_embeds: torch.Tensor):
    """Audio→Text retrieval metrics (1-to-1).

    Embeddings are L2-normalized in-function and cosine similarity is used.

    Args:
        audio_embeds: (N, D) audio embedding matrix
        text_embeds: (M, D) text embedding matrix

    Returns:
        r1, r5, r10, medr, meanr, mAPk (with k=10)
    """
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
    audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)
    sim = audio_embeds @ text_embeds.t()
    return _retrieval_metrics(sim)


@torch.no_grad()
def multi_a2t(
    audio_embed: torch.Tensor, text_embed: torch.Tensor, num_repeats: int, k: int = 10
):
    """Audio→Text retrieval for repeated-captions scheme (global negatives).

    Assumptions:
      - text_embed is laid out as groups of size `num_repeats` per audio in order
        [caps_of_audio0, caps_of_audio1, ...]
      - audio_embed contains repeated audio embeddings (aligned with text order);
        evaluation anchors on the first embedding of each group.

    Args:
      audio_embed: (A*num_repeats, D) audio embeddings
      text_embed:  (A*num_repeats, D) text embeddings (flattened groups)
      num_repeats: number of captions per audio (uniform across eval set)
      k: cutoff for AP@k normalization and recalls

    Returns:
      r1, r5, r10, medr, meanr, mAPk (percent for recalls/mAP; 1-based medr/meanr)
    """
    """Audio→Text retrieval for multi-caption per audio with global negatives.

    For each audio i with num_repeats captions, we rank ALL texts globally and:
      - R@1/5/10: uses the best (lowest) rank among that audio's positives
      - mAP@10: average precision at 10 across that audio's positives
    Returns percentages for recalls/mAP and 1-based medr/meanr.
    """
    # Normalize to cosine space to ensure fair similarity
    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

    num_audios = audio_embed.size(0) // num_repeats

    ranks = torch.zeros(num_audios, device=audio_embed.device)
    APk = torch.zeros(num_audios, device=audio_embed.device)

    for index in range(num_audios):
        audio = audio_embed[num_repeats * index]

        sim = torch.matmul(audio, text_embed.T)  # (N,)
        sorted_indices = sim.argsort(descending=True)

        rank = float("inf")
        inds_map = []
        for i in range(num_repeats * index, num_repeats * index + num_repeats):
            tmp = (sorted_indices == i).nonzero(as_tuple=False).item()
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)

        if len(inds_map) > 0:
            inds_map = torch.tensor(inds_map, device=audio_embed.device)
            denom = min(num_repeats, k)
            APk[index] = (
                torch.sum(
                    torch.arange(1, len(inds_map) + 1, device=audio_embed.device)
                    / inds_map
                )
                / denom
            )
        else:
            APk[index] = 0.0

        ranks[index] = rank

    r1 = (ranks < 1).float().mean().item() * 100.0
    r5 = (ranks < 5).float().mean().item() * 100.0
    r10 = (ranks < 10).float().mean().item() * 100.0
    mAPk = APk.mean().item() * 100
    medr = ranks.median().item() + 1
    meanr = ranks.mean().item() + 1

    return r1, r5, r10, medr, meanr, mAPk


@torch.no_grad()
def multi_t2a(
    text_embed: torch.Tensor, audio_embed: torch.Tensor, num_repeats: int, k: int = 10
):
    """Text→Audio retrieval for repeated-audio scheme (global negatives).

    Assumptions:
      - ``audio_embed`` contains repeated audio embeddings: for A unique audios and
        ``num_repeats`` captions per audio, the shape is ``(A*num_repeats, D)`` and
        rows are laid out in blocks per audio as
        ``[audio0]*num_repeats, [audio1]*num_repeats, ...``.
      - ``text_embed`` is flattened in the same order: captions of audio0 first, etc.

    Evaluation ranks each text query against UNIQUE audios (collapsing repeats) and
    the gold target for query index ``j`` is audio group ``j // num_repeats``.

    Args:
      text_embed:  (A*num_repeats, D) text embeddings (flattened groups)
      audio_embed: (A*num_repeats, D) repeated audio embeddings
      num_repeats: number of captions per audio (uniform across eval set)
      k:           cutoff for MRR@m (mAP@k here) and top-k recalls

    Returns:
      r1, r5, r10, medr, meanr, mAPk (percentages for recalls/mAP; ranks are 1-based)
    """
    # Normalize embeddings for cosine similarity
    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)

    # Collapse repeated audio to unique targets
    unique_audio = audio_embed[0::num_repeats]  # (A, D)

    # Similarity of each text against unique audios
    sim = text_embed @ unique_audio.t()  # (A*num_repeats, A)

    # Gold target index per text query
    N = text_embed.size(0)
    gold = torch.arange(N, device=text_embed.device) // num_repeats  # (N,)

    # Compute ranks and metrics with mapped gold indices
    ranks = torch.argsort(sim, dim=1, descending=True)  # (N, A)
    pos = (ranks == gold.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1  # 1-based

    r1 = (pos <= 1).float().mean().item() * 100.0
    r5 = (pos <= 5).float().mean().item() * 100.0
    r10 = (pos <= 10).float().mean().item() * 100.0
    medr = pos.median().item()
    meanr = pos.float().mean().item()
    in_topk = pos <= k
    mAPk = (1.0 / pos[in_topk].float()).mean().item() * 100.0 if in_topk.any() else 0.0

    return r1, r5, r10, medr, meanr, mAPk
