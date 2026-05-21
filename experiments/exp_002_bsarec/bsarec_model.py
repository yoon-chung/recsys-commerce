"""BSARec model -- SASRec backbone (RecBole) + Frequency-Rescaling Attention.

Hybrid implementation:
  - Data pipeline, item/position embedding, training loop, evaluation: RecBole
  - FRA contribution (FrequencyLayer + alpha blend): hand-implemented per
    Shin et al. AAAI 2024 (yehjin-shin/BSARec, Apache-2.0).

The only structural difference from RecBole's SASRec is the encoder: instead
of standard TransformerEncoder, each layer combines a frequency-domain branch
and a self-attention branch with weight alpha:

    hidden = alpha * FrequencyLayer(x)  +  (1 - alpha) * MultiHeadAttention(x)
    hidden = FeedForward(hidden)

Everything else (embedding, masking, loss, predict, full_sort_predict) is
identical to RecBole SASRec so existing training/eval/checkpointing works.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward, MultiHeadAttention
from recbole.model.loss import BPRLoss

from fra import FrequencyLayer


class BSARecLayer(nn.Module):
    """alpha * FrequencyLayer(x) + (1 - alpha) * MultiHeadAttention(x, mask)."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
        alpha: float,
        c: int,
    ) -> None:
        super().__init__()
        self.filter_layer = FrequencyLayer(
            hidden_size=hidden_size,
            c=c,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.attention_layer = MultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.alpha = alpha

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        return self.alpha * dsp + (1.0 - self.alpha) * gsp


class BSARecBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
        alpha: float,
        c: int,
    ) -> None:
        super().__init__()
        self.layer = BSARecLayer(
            n_heads=n_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            alpha=alpha,
            c=c,
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.layer(x, attn_mask))


class BSARecEncoder(nn.Module):
    """Drop-in replacement for RecBole's `TransformerEncoder`.

    Preserves forward(hidden_states, attention_mask, output_all_encoded_layers)
    so RecBole's SequentialRecommender forward() works unchanged.
    """

    def __init__(self, n_layers: int, **block_kwargs) -> None:
        super().__init__()
        block = BSARecBlock(**block_kwargs)
        self.layer = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = False,
    ) -> list[torch.Tensor]:
        all_layers: list[torch.Tensor] = []
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


class BSARec(SequentialRecommender):
    """RecBole-compatible BSARec.

    Reference:
        Shin, Y., Choi, J., Wi, H., Park, N. (2024). "An Attentive Inductive
        Bias for Sequential Recommendation beyond the Self-Attention."
        AAAI 2024. https://github.com/yehjin-shin/BSARec
    """

    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

        # ---- SASRec-identical params ----
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # ---- BSARec-specific params ----
        self.alpha = config["alpha"]
        self.c = config["c"]

        # ---- Embeddings (same as SASRec) ----
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # ---- BSARec encoder (the swap) ----
        self.trm_encoder = BSARecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            alpha=self.alpha,
            c=self.c,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be 'BPR' or 'CE'")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attn_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, attn_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embedding(pos_items)
            neg_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))
