"""BSARec backbone + multi-behavior input embedding (lightweight MB-STR variant).

Idea:
    paper 의 attention modification 까진 안 감 (구현 시간 + 추가 lift 작을 듯).
    가장 단순한 multi-behavior 신호:
        e_i = item_emb + behavior_emb + position_emb

Behavior IDs:
    0 = PAD
    1 = view
    2 = cart
    3 = purchase

RecBole 가 load_col 에 behavior_id 포함 시 sequence 변환 자동으로 해줌
(`interaction['behavior_id_list']` 으로 접근).

Inference 도 같은 behavior_id_list 를 받아 forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bsarec_model import BSARec


class MBSTR(BSARec):
    """BSARec + behavior embedding."""

    BEHAVIOR_FIELD = "behavior_id_list"   # RecBole 가 sequential variant 로 자동 생성

    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)
        n_behaviors = int(config["n_behaviors"])
        self.behavior_embedding = nn.Embedding(
            n_behaviors,
            self.hidden_size,
            padding_idx=0,
        )
        # paper init style
        self.behavior_embedding.weight.data.normal_(mean=0.0, std=config["initializer_range"])
        with torch.no_grad():
            self.behavior_embedding.weight[0].zero_()

    # ------------------------------------------------------------------
    # forward -- override BSARec.forward() 를 behavior_emb 추가하도록
    # BSARec.forward(item_seq, item_seq_len) -> add optional behavior_seq.
    # ------------------------------------------------------------------
    def forward(self, item_seq, item_seq_len, behavior_seq=None):
        """Mirrors BSARec.forward() but adds behavior_emb to input embedding."""
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)

        input_emb = item_emb + position_embedding
        if behavior_seq is not None:
            behavior_emb = self.behavior_embedding(behavior_seq)
            input_emb = input_emb + behavior_emb

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    # ------------------------------------------------------------------
    # calculate_loss -- BSARec parent 의 CE next-item loss + behavior 입력
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        behavior_seq = interaction.get(self.BEHAVIOR_FIELD, None)

        seq_output = self.forward(item_seq, item_seq_len, behavior_seq)

        # CE over full vocab (loss_type=CE, BSARec/SASRec 표준)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    # ------------------------------------------------------------------
    # predict -- inference 시 (interaction 객체 받음)
    # ------------------------------------------------------------------
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        behavior_seq = interaction.get(self.BEHAVIOR_FIELD, None)
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, behavior_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        behavior_seq = interaction.get(self.BEHAVIOR_FIELD, None)
        seq_output = self.forward(item_seq, item_seq_len, behavior_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
