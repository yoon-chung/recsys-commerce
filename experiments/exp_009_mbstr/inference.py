"""exp_009_mbstr / inference.py -- score all users, self-val, submission.

exp_006 BSARec+CL inference 와 같은 패턴. 차이점만:
    - 모델: MBSTR
    - per-user sequence 빌드 시 item + behavior 둘 다 RecBole int ID 로 변환
    - model.forward(item_seq, item_seq_len, behavior_seq) 호출

Behavior mapping (data_prep 과 동일):
    view -> 1, cart -> 2, purchase -> 3, PAD -> 0
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

from mbstr_model import MBSTR  # noqa: E402

from core.data_loader import load_train_data  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)

BEHAVIOR_MAP = {"view": 1, "cart": 2, "purchase": 3}


def build_user_sequences(train_df: pd.DataFrame) -> dict:
    """user_uuid -> [(item_uuid, behavior_int), ...] sorted by event_time asc."""
    df = train_df.copy()
    df["behavior_id"] = df["event_type"].map(BEHAVIOR_MAP)
    df = df.dropna(subset=["behavior_id"])
    df["behavior_id"] = df["behavior_id"].astype("int64")
    df = df.sort_values(["user_id", "event_time"], kind="mergesort")
    out: dict = {}
    for u, g in df.groupby("user_id", sort=False):
        out[u] = list(zip(g["item_id"].tolist(), g["behavior_id"].tolist()))
    return out


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

    # ---- 1. RecBole dataset + model + checkpoint -----------------------
    config = Config(
        model=MBSTR,
        config_file_list=[args.config],
        dataset=our_cfg["dataset"],
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = MBSTR(config, train_data.dataset).to(config["device"])

    ckpt_pointer = saved_dir / "best_ckpt_path.txt"
    if ckpt_pointer.exists():
        ckpt_path = Path(ckpt_pointer.read_text().strip())
    else:
        candidates = sorted(saved_dir.glob("MBSTR-*.pth"))
        if not candidates:
            raise FileNotFoundError(f"no MBSTR checkpoint under {saved_dir}")
        ckpt_path = candidates[-1]
    state = torch.load(ckpt_path, map_location=config["device"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    logger.info("loaded checkpoint %s (%.1f MB)",
                ckpt_path, ckpt_path.stat().st_size / 1024**2)

    # ---- 2. RecBole item-vocab bridge ----------------------------------
    item_field2id = dataset.field2token_id["item_id"]
    item_id2field = dataset.field2id_token["item_id"]
    n_items_recbole = int(dataset.item_num)
    pad_token = item_field2id.get("[PAD]", 0)
    logger.info("RecBole item vocab: %s items (pad=%d)",
                f"{n_items_recbole:,}", pad_token)

    # ---- 3. Our train portion + per-user seq (item + behavior) ---------
    df_full = load_train_data(our_cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=our_cfg["val_days"],
        gt_event_types=our_cfg["gt_event_types"],
    )
    user_seqs = build_user_sequences(train_df)
    logger.info("user sequences built for %s users", f"{len(user_seqs):,}")

    popularity = compute_popularity(train_df, top_n=our_cfg["top_n"])

    # ---- 4. All users (638k) -------------------------------------------
    sample = pd.read_csv(our_cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()
    logger.info("all_users from sample_submission: %s", f"{len(all_users):,}")

    # ---- 5. Split known vs cold-start ----------------------------------
    max_len = int(config["MAX_ITEM_LIST_LENGTH"])
    top_n = int(our_cfg["top_n"])
    batch_size = int(our_cfg["inference_batch_size"])
    device = config["device"]

    known_users: list[str] = []
    known_item_seqs: list[list[int]] = []
    known_beh_seqs: list[list[int]] = []
    known_seq_lens: list[int] = []

    for u in all_users:
        if u not in user_seqs:
            continue
        items_uuid_beh = user_seqs[u]
        # filter to items that are in RecBole vocab
        pairs = [(item_field2id[it], b) for (it, b) in items_uuid_beh if it in item_field2id]
        if not pairs:
            continue
        pairs = pairs[-max_len:]
        L = len(pairs)
        items_id, beh_id = zip(*pairs)
        item_padded = list(items_id) + [pad_token] * (max_len - L)
        beh_padded = list(beh_id) + [0] * (max_len - L)
        known_users.append(u)
        known_item_seqs.append(item_padded)
        known_beh_seqs.append(beh_padded)
        known_seq_lens.append(L)

    cold_start = len(all_users) - len(known_users)
    logger.info("predicting: known=%s, cold-start=%s",
                f"{len(known_users):,}", f"{cold_start:,}")

    # ---- 6. Batched scoring -------------------------------------------
    n_known = len(known_users)
    all_ids = np.zeros((n_known, top_n), dtype=np.int64)
    all_scores = np.zeros((n_known, top_n), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_known, batch_size):
            end = min(start + batch_size, n_known)
            item_seq = torch.tensor(known_item_seqs[start:end], dtype=torch.long, device=device)
            beh_seq = torch.tensor(known_beh_seqs[start:end], dtype=torch.long, device=device)
            item_seq_len = torch.tensor(known_seq_lens[start:end], dtype=torch.long, device=device)

            seq_output = model.forward(item_seq, item_seq_len, beh_seq)            # [B, H]
            all_item_emb = model.item_embedding.weight                              # [V, H]
            scores = torch.matmul(seq_output, all_item_emb.transpose(0, 1))         # [B, V]
            scores[:, pad_token] = -float("inf")

            top_scores, top_idx = scores.topk(top_n, dim=1)
            all_ids[start:end] = top_idx.cpu().numpy()
            all_scores[start:end] = top_scores.cpu().numpy()

            if (start // batch_size) % 20 == 0:
                logger.info("batch %s/%s, %.1fs elapsed",
                            f"{end:,}", f"{n_known:,}", time.time() - t0)
    logger.info("inference done in %.1fs", time.time() - t0)

    # ---- 7. Build predictions.parquet ----------------------------------
    user_repeat = np.repeat(np.asarray(known_users, dtype=object), top_n)
    item_uuid_flat = np.array(
        [item_id2field[i] for i in all_ids.reshape(-1)], dtype=object
    )
    score_flat = all_scores.reshape(-1)
    rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), n_known)

    pred_df = pd.DataFrame({
        "user_id": user_repeat,
        "item_id": item_uuid_flat,
        "score": score_flat.astype(np.float64, copy=False),
        "rank": rank_flat,
    })
    pred_path = out_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path)
    logger.info("wrote %s (%s rows)", pred_path, f"{len(pred_df):,}")

    # ---- 8. Self-val NDCG / recall (reuse exp_001 val_gt) --------------
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

    # ---- 9. Submission CSV ---------------------------------------------
    from core.data_loader import load_mappings
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

    # ---- 10. wandb -----------------------------------------------------
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
