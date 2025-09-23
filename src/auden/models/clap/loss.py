import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric cross-entropy (softmax) contrastive loss for audio-text."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        sim_a2t: torch.Tensor,
        sim_t2a: torch.Tensor,
        sim_targets: torch.Tensor = None,
    ):
        """
        Args:
            sim_a2t: Tensor of shape (B, B), audio → text similarity
            sim_t2a: Tensor of shape (B, B), text → audio similarity
            sim_targets: Optional soft target matrix (B, B). Defaults to identity matrix.

        Returns:
            Scalar loss: sum-reduced audio-text contrastive loss
        """
        if sim_targets is None:
            sim_targets = torch.eye(sim_a2t.size(0), device=sim_a2t.device)

        log_probs_a2t = F.log_softmax(sim_a2t, dim=1)
        log_probs_t2a = F.log_softmax(sim_t2a, dim=1)

        loss_a2t = -(sim_targets * log_probs_a2t).sum()
        loss_t2a = -(sim_targets * log_probs_t2a).sum()

        return loss_a2t + loss_t2a


class SigLIPLoss(nn.Module):
    """Sigmoid-based contrastive loss inspired by SigLIP.

    Applies BCEWithLogits over pairwise similarities in both directions
    with positive targets on the diagonal.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        sim_a2t: torch.Tensor,
        sim_t2a: torch.Tensor,
        sim_targets: torch.Tensor = None,
    ):
        if sim_targets is None:
            sim_targets = torch.eye(sim_a2t.size(0), device=sim_a2t.device)
        loss_a2t = self.bce(sim_a2t, sim_targets)
        loss_t2a = self.bce(sim_t2a, sim_targets)
        return loss_a2t + loss_t2a
