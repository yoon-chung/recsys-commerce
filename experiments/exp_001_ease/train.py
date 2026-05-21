"""exp_001_ease / train.py — Embarrassingly Shallow Autoencoder (Steck 2019).

Closed-form item-item: B = -P / diag(P) where P = (G + λI)^{-1}, G = X^T X,
diag(B) = 0. Single hyperparameter λ. Train time ~1-3 min on 29.5k items.

Artifacts written to ./saved/ (gitignored):
    B.npy              -- item × item weight matrix (float32 by default)
    interactions.npz   -- training CSR
    mappings/          -- user2idx.json + item2idx.json
    val_gt.parquet     -- held-out purchase events
    eval_users.json    -- users with val gt AND known in train
    wandb_run_id.txt   -- run id for inference.py to resume

Usage:
    python train.py
    python train.py --config path/to.yaml
    python train.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make `core/` importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, save_npz

from core.data_loader import (  # noqa: E402
    load_train_data,
    build_id_mappings,
    cache_mappings,
    add_idx_columns,
)
from core.validation import time_based_split, get_eval_users  # noqa: E402

logger = logging.getLogger(__name__)


def build_interaction_matrix(train_df, mappings: dict, event_weights: dict) -> csr_matrix:
    """Same pattern as exp_000 — event-type weighted CSR with sum_duplicates."""
    df = add_idx_columns(train_df, mappings)
    df = df.assign(_w=df["event_type"].map(event_weights).fillna(0.0))
    df = df[df["_w"] > 0]

    n_users = len(mappings["user2idx"])
    n_items = len(mappings["item2idx"])
    mat = csr_matrix(
        (
            df["_w"].to_numpy(dtype=np.float32),
            (df["user_idx"].to_numpy(), df["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    mat.eliminate_zeros()
    mat.sum_duplicates()
    return mat


def fit_ease(X: csr_matrix, lambda_reg: float, dtype_compute=np.float64) -> np.ndarray:
    """Fit EASE: B = -P / diag(P), P = (G + λI)^{-1}, G = X^T X, diag(B) = 0.

    Uses Cholesky factor + solve (G + λI is positive definite for λ > 0).
    Peak memory ≈ 2 × n_items² × dtype_size (during inversion).

    Args:
        X: sparse (n_users, n_items) interaction matrix.
        lambda_reg: ridge regularization strength.
        dtype_compute: dtype for the dense inversion (float64 recommended).

    Returns:
        B: dense (n_items, n_items) item-item weight matrix.
    """
    n_items = X.shape[1]

    logger.info("step 1: G = X^T X (sparse → dense)")
    t0 = time.time()
    G = (X.T @ X).toarray().astype(dtype_compute, copy=False)
    logger.info("  G shape=%s dtype=%s took %.1fs (%.1fGB)",
                G.shape, G.dtype, time.time() - t0, G.nbytes / 1024**3)

    logger.info("step 2: G += λI (λ=%.4f)", lambda_reg)
    diag_idx = np.arange(n_items)
    G[diag_idx, diag_idx] += lambda_reg

    logger.info("step 3: Cholesky factor (G is positive definite)")
    t0 = time.time()
    c, low = cho_factor(G, overwrite_a=True)
    logger.info("  factor took %.1fs", time.time() - t0)

    logger.info("step 4: solve P = (G+λI)^{-1} via cho_solve")
    t0 = time.time()
    P = cho_solve((c, low), np.eye(n_items, dtype=dtype_compute))
    logger.info("  solve took %.1fs (P shape=%s, %.1fGB)",
                time.time() - t0, P.shape, P.nbytes / 1024**3)
    del c, low

    logger.info("step 5: B = -P / diag(P)[None, :], fill_diagonal(B, 0)")
    diag_P = np.diag(P).copy()
    if np.any(diag_P <= 0):
        n_bad = int((diag_P <= 0).sum())
        raise ValueError(
            f"{n_bad}/{n_items} items have non-positive diag(P) — λ too small? "
            f"min(diag(P))={diag_P.min():.3e}"
        )
    # In-place to save memory: B = -P / diag(P)
    P /= -diag_P[None, :]  # P now equals B (with diagonal not yet zeroed)
    P[diag_idx, diag_idx] = 0.0
    return P  # this is B now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--train-data", default=None, help="override config train_data path")
    parser.add_argument("--saved-dir", default=str(Path(__file__).parent / "saved"))
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_data_path = args.train_data or cfg["train_data"]
    saved_dir = Path(args.saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg["seed"])

    # 1. Load + time-split
    df = load_train_data(train_data_path)
    train_df, val_gt_df = time_based_split(
        df,
        val_days=cfg["val_days"],
        gt_event_types=cfg["gt_event_types"],
    )
    eval_users = get_eval_users(val_gt_df, train_df)
    logger.info("eval_users: %s", f"{len(eval_users):,}")

    # 2. ID mappings -- built from train_df
    mappings = build_id_mappings(train_df)
    cache_mappings(mappings, str(saved_dir / "mappings"))

    # 3. Interaction matrix
    X = build_interaction_matrix(train_df, mappings, cfg["event_weights"])
    logger.info(
        "interactions: shape=%s, nnz=%s, density=%.4f%%",
        X.shape,
        f"{X.nnz:,}",
        100.0 * X.nnz / (X.shape[0] * X.shape[1]),
    )
    save_npz(str(saved_dir / "interactions.npz"), X)

    # 4. wandb (optional)
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                entity=cfg.get("wandb_entity"),
                project=cfg["wandb_project"],
                name=cfg["run_name"],
                config=cfg,
            )
        except ImportError:
            logger.warning("wandb not installed; continuing without it")

    # 5. Fit EASE
    dtype_compute = np.float64 if cfg["dtype_compute"] == "float64" else np.float32
    dtype_storage = np.float32 if cfg["dtype_storage"] == "float32" else np.float64

    logger.info("fitting EASE: λ=%.4f dtype_compute=%s",
                cfg["reg_lambda"], dtype_compute.__name__)
    B = fit_ease(X, lambda_reg=cfg["reg_lambda"], dtype_compute=dtype_compute)
    logger.info("B: shape=%s dtype=%s (%.1fGB)",
                B.shape, B.dtype, B.nbytes / 1024**3)

    # Stats — sparsity of B (most item pairs have small weight)
    abs_B = np.abs(B)
    logger.info("B stats: |B| min=%.3e p50=%.3e p99=%.3e max=%.3e",
                abs_B.min(), np.median(abs_B), np.quantile(abs_B, 0.99), abs_B.max())
    del abs_B

    # 6. Persist
    B = B.astype(dtype_storage, copy=False)
    b_path = saved_dir / "B.npy"
    np.save(b_path, B)
    logger.info("saved B.npy (%.1fGB on disk)", B.nbytes / 1024**3)

    val_gt_df.to_parquet(saved_dir / "val_gt.parquet")
    with open(saved_dir / "eval_users.json", "w", encoding="utf-8") as f:
        json.dump(sorted(eval_users), f)

    logger.info("saved: %s, %s, %s", b_path, saved_dir / "val_gt.parquet", saved_dir / "eval_users.json")

    # 7. wandb model artifact + record run id so inference.py can resume the same run
    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact(cfg["run_name"], type="model")
        artifact.add_file(str(b_path))
        artifact.add_file(str(saved_dir / "interactions.npz"))
        artifact.add_file(str(saved_dir / "mappings" / "user2idx.json"))
        artifact.add_file(str(saved_dir / "mappings" / "item2idx.json"))
        wandb.log_artifact(artifact)
        (saved_dir / "wandb_run_id.txt").write_text(wandb_run.id)
        wandb_run.finish()

    logger.info("train.py done")


if __name__ == "__main__":
    main()
