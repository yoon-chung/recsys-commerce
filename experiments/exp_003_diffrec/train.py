"""exp_003_diffrec / train.py -- DiffRec via RecBole.

RecBole 1.2+ ships DiffRec out of the box (recbole.model.general_recommender).
We pass the class directly to Config to skip the get_model() lookup + the
lightgbm import that breaks otherwise (same trick as exp_002_bsarec).

Usage:
    python data_prep.py    # one-time
    python train.py
    python train.py --no-wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.model.general_recommender import DiffRec  # noqa: E402
from recbole.trainer import Trainer  # noqa: E402
from recbole.utils import init_seed  # noqa: E402

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

    with open(args.config) as f:
        our_cfg = yaml.safe_load(f)
    saved_dir = HERE / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # Pass the DiffRec class to skip get_model() lookup (avoids lightgbm).
    config = Config(
        model=DiffRec,
        config_file_list=[args.config],
        dataset=our_cfg["dataset"],
    )
    config["checkpoint_dir"] = str(saved_dir)
    init_seed(config["seed"], config["reproducibility"])
    logger.info("RecBole config loaded -- device=%s, dataset=%s",
                config["device"], config["dataset"])

    dataset = create_dataset(config)
    logger.info("dataset: %s users x %s items, %s interactions",
                f"{dataset.user_num:,}",
                f"{dataset.item_num:,}",
                f"{dataset.inter_num:,}")
    train_data, valid_data, _ = data_preparation(config, dataset)

    model = DiffRec(config, train_data.dataset).to(config["device"])
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("DiffRec model -- %s params (%.2fM)", f"{n_params:,}", n_params / 1e6)
    logger.info("  steps=%d, dims_dnn=%s, noise_scale=%.4f, mean_type=%s",
                config["steps"], config["dims_dnn"],
                config["noise_scale"], config["mean_type"])

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

    trainer = Trainer(config, model)
    t0 = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    train_time = time.time() - t0
    logger.info("training done in %.1f min", train_time / 60)
    logger.info("best valid score (RecBole): %.6f", best_valid_score)
    logger.info("best valid result: %s", best_valid_result)

    if wandb_run is not None:
        import wandb

        wandb.log({
            "train_time_min": train_time / 60,
            "best_recbole_valid_score": best_valid_score,
            **{f"best_{k}": v for k, v in best_valid_result.items()},
        })
        (saved_dir / "wandb_run_id.txt").write_text(wandb_run.id)
        wandb_run.finish()

    ckpt = Path(trainer.saved_model_file)
    logger.info("checkpoint saved: %s (%.1f MB)", ckpt, ckpt.stat().st_size / 1024**2)
    (saved_dir / "best_ckpt_path.txt").write_text(str(ckpt))


if __name__ == "__main__":
    main()
