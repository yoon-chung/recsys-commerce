"""exp_000_als_baseline / train.py

Train implicit ALS on the time-split train portion of the competition data,
then save the model + supporting artifacts for inference.py to consume.

Artifacts written to ./saved/ (gitignored):
    als.npz            -- trained model (implicit.als.AlternatingLeastSquares.save)
    interactions.npz   -- training CSR (needed for recommend's filter_already_liked)
    mappings/          -- user2idx.json + item2idx.json (shared.data_loader format)
    val_gt.parquet     -- held-out purchase events for self-val NDCG@10
    eval_users.json    -- sorted list of users with val gt AND known in train

Usage:
    python train.py                           # uses ./config.yaml
    python train.py --config path/to.yaml
    python train.py --no-wandb                # skip wandb logging
    python train.py --train-data /alt/path    # override config train_data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make `shared/` importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from scipy.sparse import csr_matrix, save_npz

from implicit.als import AlternatingLeastSquares

from shared.data_loader import (  # noqa: E402
    load_train_data,
    build_id_mappings,
    cache_mappings,
    add_idx_columns,
)
from shared.validation import time_based_split, get_eval_users  # noqa: E402

logger = logging.getLogger(__name__)


def build_interaction_matrix(train_df, mappings: dict, event_weights: dict) -> csr_matrix:
    """train_df + per-event weights -> CSR (n_users, n_items).

    Multiple events on the same (user, item) are summed (CSR construction
    aggregates duplicate coordinates). Event types not in `event_weights`
    are dropped.
    """
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

    # 2. ID mappings -- built from train_df so the model's user/item universe
    # mirrors what it actually trains on. Cold-start users (in val only) get
    # popularity-fallback at submission time.
    mappings = build_id_mappings(train_df)
    cache_mappings(mappings, str(saved_dir / "mappings"))

    # 3. Interaction matrix
    interactions = build_interaction_matrix(train_df, mappings, cfg["event_weights"])
    logger.info(
        "interactions: shape=%s, nnz=%s, density=%.4f%%",
        interactions.shape,
        f"{interactions.nnz:,}",
        100.0 * interactions.nnz / (interactions.shape[0] * interactions.shape[1]),
    )
    save_npz(str(saved_dir / "interactions.npz"), interactions)

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
                job_type="train",
            )
        except ImportError:
            logger.warning("wandb not installed; continuing without it")

    # 5. Train ALS
    model = AlternatingLeastSquares(
        factors=cfg["factors"],
        regularization=cfg["regularization"],
        iterations=cfg["iterations"],
        alpha=cfg["alpha"],
        use_gpu=cfg["use_gpu"],
        random_state=cfg["seed"],
    )
    logger.info(
        "fit ALS: factors=%d iters=%d reg=%.4f alpha=%.1f gpu=%s",
        cfg["factors"], cfg["iterations"], cfg["regularization"],
        cfg["alpha"], cfg["use_gpu"],
    )
    model.fit(interactions, show_progress=True)

    # 6. Persist artifacts
    model_path = saved_dir / "als.npz"
    model.save(str(model_path))

    val_gt_df.to_parquet(saved_dir / "val_gt.parquet")
    with open(saved_dir / "eval_users.json", "w", encoding="utf-8") as f:
        json.dump(sorted(eval_users), f)

    logger.info("saved: %s, %s, %s", model_path, saved_dir / "val_gt.parquet", saved_dir / "eval_users.json")

    # 7. wandb model artifact
    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact(cfg["run_name"], type="model")
        artifact.add_file(str(model_path))
        artifact.add_file(str(saved_dir / "interactions.npz"))
        artifact.add_file(str(saved_dir / "mappings" / "user2idx.json"))
        artifact.add_file(str(saved_dir / "mappings" / "item2idx.json"))
        wandb.log_artifact(artifact)
        wandb_run.finish()

    logger.info("train.py done")


if __name__ == "__main__":
    main()
