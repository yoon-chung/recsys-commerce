"""CL4SRec augmentation primitives (crop / mask / reorder) for sequential
sequences, adapted to RecBole's LEFT-PADDED layout.

Original algorithm and InfoNCE pattern adapted from a team-internal CL4SRec
port written against a SASRec encoder with RIGHT-padded sequences. Padding
convention flipped here to match RecBole's `SequentialRecommender` (item
ids at positions [0, L), zero-pad at [L, max_seq)). Length tracking added
to `_crop` since cropping changes the effective sequence length and
RecBole's BSARec.forward consumes `item_seq_len` for gather_indexes.

Reference: Xie et al., "Contrastive Learning for Sequential Recommendation"
(CL4SRec), ICDE 2022.
"""

from __future__ import annotations

import random

import torch


def _crop(seq: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous crop of the non-pad prefix.

    Args:
        seq: [B, max_seq], left-padded. Non-pad item ids at positions
            [0, L_i), zero-pad at [L_i, max_seq).
        ratio: fraction of items dropped (e.g. 0.4 -> keep 60%).

    Returns:
        (cropped_seq, new_lengths). cropped_seq remains left-padded with
        the kept items moved to the front; new_lengths is the per-row
        non-pad count after cropping.
    """
    lengths = (seq != 0).sum(dim=1)
    out = torch.zeros_like(seq)
    new_lengths = lengths.clone()
    for i in range(seq.size(0)):
        L = int(lengths[i].item())
        if L <= 1:
            out[i] = seq[i]
            continue
        crop_len = max(1, int(L * (1 - ratio)))
        start = random.randint(0, L - crop_len)
        selected = seq[i, start : start + crop_len]
        out[i, :crop_len] = selected
        new_lengths[i] = crop_len
    return out, new_lengths


def _mask(seq: torch.Tensor, ratio: float, n_items: int) -> torch.Tensor:
    """Replace a random subset of non-pad positions with random item ids.

    Length is preserved -- only token identities change. Valid item ids
    in RecBole are [1, n_items); 0 is the pad token.
    """
    out = seq.clone()
    pad_mask = seq == 0
    rand = torch.rand_like(seq.float())
    rand[pad_mask] = 1.0
    mask_pos = rand < ratio
    n_mask = int(mask_pos.sum().item())
    if n_mask > 0:
        out[mask_pos] = torch.randint(
            low=1, high=max(2, n_items), size=(n_mask,), device=seq.device
        )
    return out


def _reorder(seq: torch.Tensor, ratio: float) -> torch.Tensor:
    """Shuffle a contiguous sub-sequence within the non-pad region.

    Length is preserved -- only the order of a contiguous slice changes.
    """
    lengths = (seq != 0).sum(dim=1)
    out = seq.clone()
    for i in range(seq.size(0)):
        L = int(lengths[i].item())
        if L <= 2:
            continue
        reorder_len = max(2, int(L * ratio))
        start = random.randint(0, L - reorder_len)
        s = start
        e = s + reorder_len
        perm = torch.randperm(e - s, device=seq.device)
        out[i, s:e] = out[i, s:e][perm]
    return out


if __name__ == "__main__":
    # ---- Quick sanity check on left-padded toy input -----------------------
    # 3 sequences, max_seq=5: lengths 4, 2, 5.
    seq = torch.tensor(
        [
            [11, 12, 13, 14, 0],
            [21, 22, 0,  0,  0],
            [31, 32, 33, 34, 35],
        ]
    )
    print("input:\n", seq)

    cropped, new_len = _crop(seq, ratio=0.4)
    print("\ncrop(ratio=0.4):\n", cropped)
    print("new lengths:", new_len.tolist())
    assert (cropped != 0).sum(dim=1).tolist() == new_len.tolist(), "length mismatch"

    masked = _mask(seq, ratio=0.5, n_items=100)
    print("\nmask(ratio=0.5, n_items=100):\n", masked)
    # Length preserved
    assert (masked != 0).sum(dim=1).tolist() == (seq != 0).sum(dim=1).tolist()
    # Pad positions untouched
    assert (masked[seq == 0] == 0).all()

    reordered = _reorder(seq, ratio=0.6)
    print("\nreorder(ratio=0.6):\n", reordered)
    # Same multiset per row, just permuted within non-pad
    for i in range(seq.size(0)):
        L = int((seq[i] != 0).sum().item())
        assert sorted(reordered[i, :L].tolist()) == sorted(seq[i, :L].tolist())

    print("\nleft-padded aug sanity OK")
