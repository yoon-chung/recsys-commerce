"""Frequency-Rescaling Attention component (BSARec contribution).

Ported (with minor style edits) from the official PyTorch implementation:
    Shin, Y., Choi, J., Wi, H., Park, N. (2024). "An Attentive Inductive
    Bias for Sequential Recommendation Beyond the Self-Attention." AAAI 2024.
    https://github.com/yehjin-shin/BSARec   (Apache-2.0)

Math (single layer, input X with shape [B, L, H]):
    X_freq    = rFFT(X, dim=L)                          # complex tensor
    low_pass  = irFFT( X_freq[:, :c]_zeros_above_c )    # low-frequency component
    high_pass = X - low_pass                            # residual = high-frequency
    out_fft   = low_pass + (sqrt_beta ** 2) * high_pass # learnable high-pass gain
    out       = LayerNorm( dropout(out_fft) + X )       # residual + norm

`sqrt_beta` is squared so the high-pass gain stays non-negative without
needing a separate clamp; this matches the paper.

Renames vs author's code:
    * `args.c`        -> constructor arg `c`
    * `args.hidden_dropout_prob` -> `hidden_dropout_prob`
    * uses torch.nn.LayerNorm directly instead of author's TF-style LayerNorm
      (numerically equivalent for our hidden sizes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyLayer(nn.Module):
    """Frequency-domain inductive bias for sequence embeddings.

    Args:
        hidden_size: model embedding dim.
        c: low-pass cutoff (in original frequency units, before halving for
            real-FFT). Smaller -> stronger smoothing.
        hidden_dropout_prob: dropout on the FFT-domain output.
        layer_norm_eps: LayerNorm eps (default matches RecBole).
    """

    def __init__(
        self,
        hidden_size: int,
        c: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        low_pass = x.clone()
        low_pass[:, self.c :, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm="ortho")
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = FrequencyLayer(hidden_size=16, c=5, hidden_dropout_prob=0.0)
    x = torch.randn(4, 20, 16)
    y = layer(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    print("FrequencyLayer forward OK -- in shape", tuple(x.shape), "out shape", tuple(y.shape))
