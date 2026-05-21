"""exp_003_diffrec / inference.py -- score all users, self-val, submission.

Simpler than exp_002_bsarec/inference.py because DiffRec is a general
recommender (no per-user sequence construction needed). For each user we
just feed their RecBole user_id; DiffRec internally uses dataset's
interaction matrix.

Bridge between UUID space (exp_000/001) and RecBole int space mirrors
exp_002 -- same dataset.field2token_id pattern.

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
from recbole.data.interaction import Interaction  # noqa: E402
from recbole.model.general_recommender import DiffRec  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

from core.data_loader import load_train_data, load_mappings  # noqa: E402
from core.validation import time_based_split  # noqa: E402
from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402
from core.submission import (  # noqa: E402
    compute_popularity,
    predictions_to_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


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
        model=DiffRec,
        config_file_list=[args.config],
        dataset=our_cfg["dataset"],
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = DiffRec(config, train_data.dataset).to(config["device"])

    ckpt_pointer = saved_dir / "best_ckpt_path.txt"
    if ckpt_pointer.exists():
        ckpt_path = Path(ckpt_pointer.read_text().strip())
    else:
        candidates = sorted(saved_dir.glob("DiffRec-*.pth"))
        if not candidates:
            raise FileNotFoundError(f"no DiffRec checkpoint under {saved_dir}")
        ckpt_path = candidates[-1]
    state = torch.load(ckpt_path, map_location=config["device"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    logger.info("loaded checkpoint %s (%.1f MB)",
                ckpt_path, ckpt_path.stat().st_size / 1024**2)

    # ---- 2. RecBole user + item vocab bridges ----------------------------
    user_field2id = dataset.field2token_id["user_id"]   # uuid_str -> int
    item_id2field = dataset.field2id_token["item_id"]   # array of uuid_str
    n_items_recbole = int(dataset.item_num)
    item_pad_token = 0
    logger.info("RecBole vocab: %s users, %s items",
                f"{int(dataset.user_num):,}",
                f"{n_items_recbole:,}")

    # ---- 3. Our train portion (for popularity fallback + cold-start ID) --
    df_full = load_train_data(our_cfg["train_data"])
    train_df, _ = time_based_split(
        df_full,
        val_days=our_cfg["val_days"],
        gt_event_types=our_cfg["gt_event_types"],
    )
    popularity = compute_popularity(train_df, top_n=our_cfg["top_n"])

    # ---- 4. All users + split known/cold-start ---------------------------
    sample = pd.read_csv(our_cfg["sample_submission"])
    all_users = sample["user_id"].drop_duplicates().tolist()

    known_users: list[str] = []
    known_uids: list[int] = []
    for u in all_users:
        rb_id = user_field2id.get(u)
        if rb_id is not None and rb_id > 0:           # 0 = [PAD]
            known_users.append(u)
            known_uids.append(rb_id)

    cold_start = len(all_users) - len(known_users)
    logger.info("predicting: known=%s, cold-start (popularity-only)=%s",
                f"{len(known_users):,}", f"{cold_start:,}")

    # ---- 5. Batched full_sort_predict ------------------------------------
    user_id_field = config["USER_ID_FIELD"]
    top_n = int(our_cfg["top_n"])
    batch_size = int(our_cfg["inference_batch_size"])
    device = config["device"]

    n_known = len(known_users)
    all_ids = np.zeros((n_known, top_n), dtype=np.int64)
    all_scores = np.zeros((n_known, top_n), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_known, batch_size):
            end = min(start + batch_size, n_known)
            uid_batch = torch.tensor(
                known_uids[start:end], dtype=torch.long, device=device
            )
            interaction = Interaction({user_id_field: uid_batch}).to(device)

            scores = model.full_sort_predict(interaction)         # [B, n_items]
            if scores.dim() == 1:
                # Some general_recommenders flatten -- reshape defensively
                scores = scores.view(uid_batch.shape[0], n_items_recbole)
            scores[:, item_pad_token] = -float("inf")             # ignore PAD

            top_scores, top_idx = scores.topk(top_n, dim=1)
            all_ids[start:end] = top_idx.cpu().numpy()
            all_scores[start:end] = top_scores.cpu().numpy()

            if (start // batch_size) % 50 == 0:
                logger.info("batch %s/%s, %.1fs elapsed",
                            f"{end:,}", f"{n_known:,}", time.time() - t0)
    logger.info("inference done in %.1fs", time.time() - t0)

    # ---- 6. Build predictions.parquet (UUID space) -----------------------
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

    # ---- 7. Self-val (reuse exp_001 val_gt + eval_users) -----------------
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

    # ---- 8. Submission CSV -----------------------------------------------
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

    # ---- 9. wandb log ----------------------------------------------------
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
            pred_artifact = wandb.Artifact(
                f'{our_cfg["run_name"]}_predictions', type="prediction"
            )
            pred_artifact.add_file(str(pred_path))
            pred_artifact.add_file(str(output_csv))
            wandb.log_artifact(pred_artifact)
            run.finish()
        except ImportError:
            logger.warning("wandb not installed -- skipping log")

    logger.info("inference.py done")


if __name__ == "__main__":
    main()
