"""exp_002_bsarec / inference.py -- score all 638,257 users, self-val, submission.

Tricky part: RecBole tokenizes our user/item UUIDs into its own internal int
IDs. We bridge two ID spaces:
    UUID (our exp_000/001 space)  <->  RecBole internal int

The bridge is the UUID string itself, present in both as
`dataset.field2token_id['item_id'][uuid_str]` (forward) and
`dataset.field2id_token['item_id'][int]` (backward).

For each user we build the sequence of their most recent items (from our own
time-based split of train.parquet) translated into RecBole internal IDs,
forward through BSARec, take top-N over all items, translate back to UUIDs.

Outputs mirror exp_000/001 exactly:
    predictions.parquet   user_id, item_id (UUID), score, rank   for known users
    output.csv            638,257 x 10, popularity fallback for cold-start

Usage:
    python inference.py
    python inference.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

from bsarec_model import BSARec  # noqa: E402

from core.data_loader import load_train_data  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


def build_user_sequences(train_df: pd.DataFrame) -> dict:
    """Group sorted item history per user.

    Returns: {user_uuid: [item_uuid, item_uuid, ...]} ordered by event_time asc.
    """
    sorted_df = train_df.sort_values(["user_id", "event_time"], kind="mergesort")
    return sorted_df.groupby("user_id", sort=False)["item_id"].apply(list).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--saved-dir", default=str(HERE / "saved"))
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        our_cfg = yaml.safe_load(f)

    saved_dir = Path(args.saved_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Rebuild RecBole dataset + model + load checkpoint ------------
    config = Config(
        model="BSARec",
        config_file_list=[args.config],
        dataset=our_cfg["dataset"],
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = BSARec(config, train_data.dataset).to(config["device"])

    ckpt_pointer = saved_dir / "best_ckpt_path.txt"
    if ckpt_pointer.exists():
        ckpt_path = Path(ckpt_pointer.read_text().strip())
    else:
        candidates = sorted(saved_dir.glob("BSARec-*.pth"))
        if not candidates:
            raise FileNotFoundError(f"no BSARec checkpoint under {saved_dir}")
        ckpt_path = candidates[-1]
    state = torch.load(ckpt_path, map_location=config["device"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    logger.info("loaded checkpoint %s (%.1f MB)",
                ckpt_path, ckpt_path.stat().st_size / 1024**2)

    # ---- 2. RecBole item-vocab bridge ------------------------------------
    item_field2id = dataset.field2token_id["item_id"]   # uuid_str -> int
    item_id2field = dataset.field2id_token["item_id"]   # array of uuid_str
    n_items_recbole = int(dataset.item_num)
    pad_token = item_field2id.get("[PAD]", 0)
    logger.info("RecBole item vocab: %s items (pad=%d)",
                f"{n_items_recbole:,}", pad_token)

    # ---- 3. Our train portion + per-user sequences -----------------------
    df_full = load_train_data(our_cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=our_cfg["val_days"],
        gt_event_types=our_cfg["gt_event_types"],
    )
    user_seqs = build_user_sequences(train_df)
    logger.info("user sequences built for %s users", f"{len(user_seqs):,}")

    popularity = compute_popularity(train_df, top_n=our_cfg["top_n"])

    # ---- 4. All users list (638,257) -------------------------------------
    sample = pd.read_csv(our_cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()
    logger.info("all_users from sample_submission: %s", f"{len(all_users):,}")

    # ---- 5. Split into known (sequence-ready) vs cold-start --------------
    max_len = int(config["MAX_ITEM_LIST_LENGTH"])
    top_n = int(our_cfg["top_n"])
    batch_size = int(our_cfg["inference_batch_size"])
    device = config["device"]

    known_users: list[str] = []
    known_seqs: list[list[int]] = []
    known_seq_lens: list[int] = []

    for u in all_users:
        if u not in user_seqs:
            continue
        items_uuid = user_seqs[u]
        items_id = [item_field2id[it] for it in items_uuid if it in item_field2id]
        if not items_id:
            continue
        items_id = items_id[-max_len:]
        L = len(items_id)
        padded = items_id + [pad_token] * (max_len - L)
        known_users.append(u)
        known_seqs.append(padded)
        known_seq_lens.append(L)

    cold_start = len(all_users) - len(known_users)
    logger.info("predicting: known=%s, cold-start (popularity-only)=%s",
                f"{len(known_users):,}", f"{cold_start:,}")

    # ---- 6. Batched scoring ----------------------------------------------
    n_known = len(known_users)
    all_ids = np.zeros((n_known, top_n), dtype=np.int64)
    all_scores = np.zeros((n_known, top_n), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_known, batch_size):
            end = min(start + batch_size, n_known)
            item_seq = torch.tensor(known_seqs[start:end], dtype=torch.long, device=device)
            item_seq_len = torch.tensor(known_seq_lens[start:end], dtype=torch.long, device=device)

            seq_output = model.forward(item_seq, item_seq_len)               # [B, H]
            all_item_emb = model.item_embedding.weight                       # [V, H]
            scores = torch.matmul(seq_output, all_item_emb.transpose(0, 1))  # [B, V]
            scores[:, pad_token] = -float("inf")                             # ignore PAD

            top_scores, top_idx = scores.topk(top_n, dim=1)
            all_ids[start:end] = top_idx.cpu().numpy()
            all_scores[start:end] = top_scores.cpu().numpy()

            if (start // batch_size) % 20 == 0:
                logger.info("batch %s/%s, %.1fs elapsed",
                            f"{end:,}", f"{n_known:,}", time.time() - t0)
    logger.info("inference done in %.1fs", time.time() - t0)

    # ---- 7. Translate item IDs back to UUID + build predictions.parquet --
    user_repeat = np.repeat(np.asarray(known_users, dtype=object), top_n)
    item_uuid_flat = np.array(
        [item_id2field[i] for i in all_ids.reshape(-1)], dtype=object
    )
    score_flat = all_scores.reshape(-1)
    rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), n_known)

    pred_df = pd.DataFrame(
        {
            "user_id": user_repeat,
            "item_id": item_uuid_flat,
            "score": score_flat.astype(np.float64, copy=False),
            "rank": rank_flat,
        }
    )
    pred_path = out_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("wrote %s (%s rows)", pred_path, f"{len(pred_df):,}")

    # ---- 8. Self-val NDCG@10 / recall@10 (reuse exp_001 val_gt) ----------
    ease_saved = (HERE / our_cfg["ease_saved"]).resolve()
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = set(json.load(f))
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(eval_users)]
    ndcg10 = ndcg_at_k_from_df(pred_df, val_gt_eval, k=10)
    recall10 = recall_at_k_from_df(pred_df, val_gt_eval, k=10)
    logger.info("self-val (last %d days, gt=%s, eval_users=%s):",
                our_cfg["val_days"], our_cfg["gt_event_types"], f"{len(eval_users):,}")
    logger.info("  NDCG@10   = %.6f", ndcg10)
    logger.info("  recall@10 = %.6f", recall10)

    # ---- 9. Submission CSV -----------------------------------------------
    # We need an item-UUID -> our internal idx mapping for popularity fallback.
    # Reuse exp_001's mapping (same UUID universe).
    from core.data_loader import load_mappings  # local import to keep top tidy

    mappings = load_mappings(str(ease_saved / "mappings"))

    output_csv = out_dir / "output.csv"
    predictions_to_submission(
        pred_path=str(pred_path),
        output_csv=str(output_csv),
        all_users=all_users,
        mappings=mappings,
        popularity_fallback=popularity,
        items_per_user=int(our_cfg["items_per_user"]),
    )
    ok = validate_submission(
        str(output_csv),
        expected_users=len(all_users),
        items_per_user=int(our_cfg["items_per_user"]),
    )
    if not ok:
        raise RuntimeError("validate_submission FAILED -- do not upload")
    logger.info("validate_submission OK")

    # ---- 10. wandb log (resume train run) --------------------------------
    if args.use_wandb:
        try:
            import wandb

            init_kwargs = dict(
                entity=our_cfg.get("wandb_entity"),
                project=our_cfg["wandb_project"],
                name=our_cfg["run_name"],
            )
            run_id_file = saved_dir / "wandb_run_id.txt"
            if run_id_file.exists():
                init_kwargs["id"] = run_id_file.read_text().strip()
                init_kwargs["resume"] = "must"
            else:
                init_kwargs["resume"] = "allow"
            run = wandb.init(**init_kwargs)
            wandb.log({
                "val_ndcg@10": ndcg10,
                "val_recall@10": recall10,
                "n_eval_users": len(eval_users),
                "n_known_users": len(known_users),
                "n_cold_start": cold_start,
            })
            pred_artifact = wandb.Artifact(f'{our_cfg["run_name"]}_predictions', type="prediction")
            pred_artifact.add_file(str(pred_path))
            pred_artifact.add_file(str(output_csv))
            wandb.log_artifact(pred_artifact)
            run.finish()
        except ImportError:
            logger.warning("wandb not installed -- skipping log")

    logger.info("inference.py done")


if __name__ == "__main__":
    main()
