"""TIFU-KNN -- Modeling Personalized Item Frequency Information for
Next-Basket Recommendation (He et al., SIGIR 2020).

Paper: https://arxiv.org/abs/2006.00556

Algorithm (adapted to our 1-week prediction):
    1. Per user, sort interactions by time, split into G equal-count groups.
    2. user_vec[u, i] = Σ_g (r_a^(G-1-g)) · Σ_t∈g (r_w^(L_g-1-pos)) · event_weight
       (oldest group has weight r_a^(G-1); within a group, oldest visit has
        weight r_w^(L_g-1). Newest = weight 1.)
    3. Row-L2-normalize for cosine sim. Compute top-K neighbors per user.
    4. score(u, i) = α · user_vec_raw[u, i] + (1-α) · mean(user_vec_raw[neighbors])

Sparse all the way. Memory profile on our data (638k users × 29.5k items):
    user_vec_raw ~8M nnz (~32 MB)
    Per-batch KNN: B × N_users dense slice ~2.6 GB for B=1024.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class TIFUKNN:
    def __init__(
        self,
        group_count: int = 7,
        decay_within: float = 0.9,
        decay_across: float = 0.7,
        knn_k: int = 300,
        alpha: float = 0.7,
    ) -> None:
        self.group_count = group_count
        self.decay_within = decay_within
        self.decay_across = decay_across
        self.knn_k = knn_k
        self.alpha = alpha

        self.user_vec_raw: sparse.csr_matrix | None = None    # for own/neighbor score
        self.user_vec_norm: sparse.csr_matrix | None = None   # row-L2-normalized, for cosine
        self.n_users: int = 0
        self.n_items: int = 0

    # ------------------------------------------------------------------
    # fit -- build user × item weighted frequency matrix
    # ------------------------------------------------------------------
    def fit(
        self,
        user_seqs: dict,        # user_idx -> list[(item_idx, event_weight)] (chronological)
        n_users: int,
        n_items: int,
    ) -> None:
        G = self.group_count
        r_w = self.decay_within
        r_a = self.decay_across

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for u_idx, seq in user_seqs.items():
            L = len(seq)
            if L == 0:
                continue
            actual_G = min(G, L)            # 짧은 user 는 group 수 줄임
            group_size = L / actual_G       # float -- 균등 분할

            for t, (item_idx, ev_w) in enumerate(seq):
                g = min(int(t / group_size), actual_G - 1)
                group_start = int(g * group_size)
                group_end = int((g + 1) * group_size) if g < actual_G - 1 else L
                group_len = max(1, group_end - group_start)
                pos_in_group = t - group_start
                w = (r_a ** (actual_G - 1 - g)) * (r_w ** (group_len - 1 - pos_in_group))
                rows.append(u_idx)
                cols.append(item_idx)
                vals.append(w * ev_w)

        coo = sparse.coo_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        self.user_vec_raw = coo.tocsr()
        self.user_vec_raw.sum_duplicates()   # same (u, i) -> weights add

        # L2 normalize rows for cosine
        norms_sq = self.user_vec_raw.multiply(self.user_vec_raw).sum(axis=1).A1
        norms = np.sqrt(norms_sq)
        inv = np.zeros_like(norms)
        nz = norms > 0
        inv[nz] = 1.0 / norms[nz]
        self.user_vec_norm = (sparse.diags(inv) @ self.user_vec_raw).tocsr()

        self.n_users = n_users
        self.n_items = n_items

        logger.info(
            "user_vec: %s x %s, %s nnz (avg %.1f items/user, %s active users)",
            f"{n_users:,}", f"{n_items:,}",
            f"{self.user_vec_raw.nnz:,}",
            self.user_vec_raw.nnz / max(1, nz.sum()),
            f"{int(nz.sum()):,}",
        )

    # ------------------------------------------------------------------
    # predict -- top-N items per query user
    # ------------------------------------------------------------------
    def predict_topn(
        self,
        query_user_idx: np.ndarray,    # int array
        top_n: int = 50,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (item_idx [Q, top_n], score [Q, top_n]) sorted desc by score."""
        assert self.user_vec_raw is not None and self.user_vec_norm is not None

        Q = len(query_user_idx)
        N = self.n_users
        V = self.n_items
        K = self.knn_k
        alpha = self.alpha

        out_idx = np.zeros((Q, top_n), dtype=np.int64)
        out_score = np.zeros((Q, top_n), dtype=np.float32)

        import time
        t0 = time.time()

        for bs in range(0, Q, batch_size):
            be = min(bs + batch_size, Q)
            q_idx = query_user_idx[bs:be]
            B = be - bs

            # cosine sim to all users via row matmul -- result CSR [B, N]
            sim = self.user_vec_norm[q_idx] @ self.user_vec_norm.T
            sim_dense = sim.toarray()                                # [B, N]

            # zero out self-similarity
            sim_dense[np.arange(B), q_idx] = 0.0

            # top-K neighbors (unsorted partition is enough for mean aggregation)
            topK = np.argpartition(-sim_dense, K, axis=1)[:, :K]     # [B, K]

            # vectorized neighbor mean: build [B, N] sparse selector M
            sel_rows = np.repeat(np.arange(B), K)
            sel_cols = topK.reshape(-1)
            sel_data = np.full(B * K, 1.0 / K, dtype=np.float32)
            M = sparse.csr_matrix(
                (sel_data, (sel_rows, sel_cols)),
                shape=(B, N),
            )
            neighbor_score = (M @ self.user_vec_raw).toarray()       # [B, V]

            own_score = self.user_vec_raw[q_idx].toarray()           # [B, V]
            final = alpha * own_score + (1.0 - alpha) * neighbor_score

            # top-N per row
            top_idx = np.argpartition(-final, top_n, axis=1)[:, :top_n]
            row_ar = np.arange(B)[:, None]
            top_score = final[row_ar, top_idx]
            sort_pos = np.argsort(-top_score, axis=1)
            top_idx = top_idx[row_ar, sort_pos]
            top_score = top_score[row_ar, sort_pos]

            out_idx[bs:be] = top_idx
            out_score[bs:be] = top_score

            if (bs // batch_size) % 20 == 0:
                logger.info("  batch %s/%s, %.1fs elapsed",
                            f"{be:,}", f"{Q:,}", time.time() - t0)

        logger.info("KNN+score done in %.1fs", time.time() - t0)
        return out_idx, out_score
