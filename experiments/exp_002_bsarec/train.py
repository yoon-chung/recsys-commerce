"""exp_002_bsarec / train.py -- BSARec via RecBole.

Reuses baseline's SASRec_dataset/ at /root/data/ (already RecBole-tokenized
from train.parquet). Drives training via RecBole's Trainer; wandb hooks
mirror exp_000/001 for consistent run management.

Artifacts written to ./saved/ (gitignored):
    BSARec-<timestamp>.pth     -- best checkpoint (RecBole default name)
    wandb_run_id.txt           -- so inference.py can resume the same wandb run

Usage:
    python train.py
    python train.py --config path/to.yaml
    python train.py --no-wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sibling modules (fra, bsarec_model) importable regardless of CWD.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.trainer import Trainer  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

from bsarec_model import BSARec  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Load our config (for wandb keys etc.) ---------------------------
    with open(args.config) as f:
        our_cfg = yaml.safe_load(f)
    saved_dir = HERE / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # ---- RecBole config (reads same YAML) --------------------------------
    config = Config(
        model="BSARec",
        config_file_list=[args.config],
        dataset=our_cfg["dataset"],
    )
    config["checkpoint_dir"] = str(saved_dir)
    init_seed(config["seed"], config["reproducibility"])
    logger.info("RecBole config loaded -- device=%s, dataset=%s",
                config["device"], config["dataset"])

    # ---- Dataset + dataloaders -------------------------------------------
    dataset = create_dataset(config)
    logger.info("dataset: %s users x %s items, %s interactions",
                f"{dataset.user_num:,}",
                f"{dataset.item_num:,}",
                f"{dataset.inter_num:,}")
    train_data, valid_data, _ = data_preparation(config, dataset)

    # ---- Model -----------------------------------------------------------
    model = BSARec(config, train_data.dataset).to(config["device"])
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("BSARec model -- %s params (%.2fM)", f"{n_params:,}", n_params / 1e6)
    logger.info("  alpha=%.2f c=%d n_layers=%d hidden=%d",
                config["alpha"], config["c"], config["n_layers"], config["hidden_size"])

    # ---- wandb (optional) ------------------------------------------------
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                entity=our_cfg.get("wandb_entity"),
                project=our_cfg["wandb_project"],
                name=our_cfg["run_name"],
                config={**dict(config.final_config_dict)},
            )
        except ImportError:
            logger.warning("wandb not installed -- continuing without it")

    # ---- Trainer + fit ---------------------------------------------------
    trainer = Trainer(config, model)
    t0 = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    train_time = time.time() - t0
    logger.info("training done in %.1f min", train_time / 60)
    logger.info("best valid score (RecBole, leave-one-out): %.6f", best_valid_score)
    logger.info("best valid result: %s", best_valid_result)

    # ---- Save wandb run id (so inference.py resumes the same run) --------
    if wandb_run is not None:
        import wandb

        wandb.log({
            "train_time_min": train_time / 60,
            "best_recbole_valid_score": best_valid_score,
            **{f"best_{k}": v for k, v in best_valid_result.items()},
        })
        (saved_dir / "wandb_run_id.txt").write_text(wandb_run.id)
        wandb_run.finish()

    # Echo the checkpoint path RecBole used so inference.py can find it.
    ckpt = Path(trainer.saved_model_file)
    logger.info("checkpoint saved: %s (%.1f MB)", ckpt, ckpt.stat().st_size / 1024**2)
    (saved_dir / "best_ckpt_path.txt").write_text(str(ckpt))


if __name__ == "__main__":
    main()
