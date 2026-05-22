"""BSARec + CL4SRec hybrid: frequency-domain attention + contrastive
augmentation in one model.

Idea:
    - Base encoder: BSARec (FRA = frequency rescaling attention) -- our
      exp_002 model already proven at public 0.0975.
    - Auxiliary objective: CL4SRec InfoNCE over crop/mask/reorder
      augmented views, motivated by data sparsity 99.96% + short
      sequence median 6 (regime where contrastive aug helps most).

Novel combination: original BSARec paper (AAAI 2024) does not include
contrastive aug; original CL4SRec paper (ICDE 2022) uses SASRec as the
backbone. We hybridise the two and let one model see both inductive
biases simultaneously.

Loss = rec_loss (BSARec CE on next item) + lmd * cl_loss (InfoNCE on
two augmented views' final hidden states).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from bsarec_model import BSARec
from aug import _crop, _mask, _reorder


class BSARecCL(BSARec):
    """BSARec backbone + CL4SRec-style contrastive auxiliary loss."""

    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)
        # CL4SRec hyperparameters -- RecBole Config 는 dict 아니므로 bracket notation 사용
        self.lmd = float(config["lmd"])             # contrastive weight
        self.tau = float(config["tau"])             # InfoNCE temperature
        self.crop_r = float(config["crop_ratio"])   # crop drop ratio
        self.mask_r = float(config["mask_ratio"])   # mask fraction
        self.reorder_r = float(config["reorder_ratio"])  # reorder fraction

    # --------------------------------------------------------------------
    # Augmentation: pick one of three per call. Length only changes for crop.
    # --------------------------------------------------------------------
    def _aug(
        self, seq: torch.Tensor, seq_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        choice = random.randint(0, 2)
        if choice == 0:
            return _crop(seq, self.crop_r)
        if choice == 1:
            return _mask(seq, self.mask_r, self.n_items), seq_len
        return _reorder(seq, self.reorder_r), seq_len

    # --------------------------------------------------------------------
    # Symmetric in-batch InfoNCE on two views' final hidden states.
    # --------------------------------------------------------------------
    def _infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.matmul(z1, z2.T) / self.tau                       # [B, B]
        labels = torch.arange(z1.size(0), device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    # --------------------------------------------------------------------
    # Combined loss: BSARec CE + lmd * InfoNCE
    # --------------------------------------------------------------------
    def calculate_loss(self, interaction) -> torch.Tensor:
        # 1. Standard BSARec rec loss (CE on next-item, full vocab).
        rec_loss = super().calculate_loss(interaction)

        # 2. Two augmented views -> final hidden states.
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        aug1, len1 = self._aug(item_seq, item_seq_len)
        aug2, len2 = self._aug(item_seq, item_seq_len)
        # forward gathers at index len-1; clamp to >=1 to avoid index -1.
        len1 = torch.clamp(len1, min=1)
        len2 = torch.clamp(len2, min=1)

        z1 = self.forward(aug1, len1)   # [B, H]
        z2 = self.forward(aug2, len2)
        cl_loss = self._infonce(z1, z2)

        return rec_loss + self.lmd * cl_loss
