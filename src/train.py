import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def training(cfg: DictConfig):
    try:
        cfg.task.corpus_args.data_folder = cfg.data_base_dir + cfg.task.corpus_args.data_folder
        cfg.task.corpus_args.data_folder = str((Path(get_original_cwd()) / cfg.task.corpus_args.data_folder).absolute())
    except Exception:
        pass

    task_id = cfg.task.task_name.replace("/", "_")

    run_marker = Path(f"{task_id}.run")
    done_marker = Path(f"{task_id}.done")

    # --- Early exit logic ---
    if done_marker.exists():
        logger.info(f"✅ Task '{task_id}' already finished — skipping training.")
        sys.exit(0)

    if run_marker.exists():
        mtime = datetime.fromtimestamp(run_marker.stat().st_mtime)
        age = datetime.now() - mtime
        if age < timedelta(hours=1):
            logger.info(f"⏸️ Task '{task_id}' already running or recently started (<24h ago) — skipping.", run_marker)
            return
        else:
            logger.warning(f"⚠️ Task '{task_id}' marker is stale (>24h). Rerunning training.")

    # --- Mark as started ---
    run_marker.touch()
    logger.info(OmegaConf.to_yaml(cfg))

    # --- Set random seeds, etc. ---
    import random, numpy as np, torch
    from transformers import set_seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    set_seed(cfg.seed)

    # --- Run training ---
    if cfg.task.framework == "flair":
        logger.info("Training with flair")
        from train_flair import training as train_flair
        train_flair(cfg)
    elif cfg.task.framework == "hf_qa":
        logger.info("Training with hf_qa")
        from train_hf_qa import training as train_hf_qa
        train_hf_qa(cfg)
    elif cfg.task.framework == "sentence_transformer":
        logger.info("Training with sentence_transformer")
        from train_st import training as train_st
        train_st(cfg)
    else:
        raise NotImplementedError(f"Unknown framework: {cfg.task.framework}")

    # --- Mark as finished ---
    done_marker.touch()
    logger.info(f"🏁 Task '{task_id}' completed successfully.")

    # --- Optional: remove .run marker after success ---
    try:
        run_marker.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    training()
