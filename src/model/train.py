import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

import wandb

PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(PROJECT_ROOT))
if os.path.exists(dotenv_path := PROJECT_ROOT / ".env"):
    load_dotenv(dotenv_path=dotenv_path)

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only

from src.model.callbacks import TrainingCallback
from src.model.data import MigDataset
from src.model.model import ContextGenModel
from src.utils.file_utils import get_config


def init_wandb(wandb_config, run_name, wandb_api_key=None):
    try:
        if wandb_api_key is None:
            wandb_api_key = os.getenv("WANDB_API_KEY", None)
        assert wandb_api_key is not None, "Please provide a valid WANDB_API_KEY"
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def save_config(config, run_name):
    save_path = config["save_path"]
    model_name = config["model_name"]
    os.makedirs(f"{save_path}/{model_name}/{run_name}", exist_ok=True)
    with open(f"{save_path}/{model_name}/{run_name}/config.yaml", "w") as f:
        yaml.dump(config, f)


def main():
    num_gpus = torch.cuda.device_count()
    config = get_config("./train/config/config.yaml")
    training_config = config["train"]
    run_name = f"{training_config['model_name']}_{time.strftime('%Y%m%d-%H%M%S')}"

    # load dataset metadata from json
    dataset_config = training_config.get("dataset", None)
    dataset = MigDataset(
        dataset_root=dataset_config["path"],
        text_drop_prob=dataset_config["text_drop_prob"],
        main_image_width=dataset_config["image_width"],
        reference_image_width=dataset_config["reference_image_width"],
        using_enhance_rate=dataset_config["using_enhance_rate"],
        data_mix_ratio=dataset_config.get("data_mix_ratio", None),
    )

    print("Dataset length:", len(dataset))

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name=run_name, training_config=training_config)] if rank_zero_only.rank == 0 else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=num_gpus if num_gpus > 0 else "auto",
        strategy="ddp" if num_gpus > 1 else "auto",
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    train_loader = dataset.get_dataloader(
        batch_size=training_config["batch_size"], num_workers=training_config["dataloader_workers"], shuffle=True
    )

    # Initialize model
    trainable_model = ContextGenModel(
        flux_pipe_id=config["flux_path"],
        save_dir=os.path.join(training_config["save_path"], training_config["model_name"]),
        training_config=training_config,
        dtype=getattr(torch, config["dtype"]),
        train_method=training_config.get("train_method", "sft"),
        process_visualize=training_config.get("process_visualize", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    setattr(trainer, "training_config", training_config)

    if rank_zero_only.rank == 0:
        # Initialize WanDB
        wandb_config = training_config.get("wandb", None)
        if wandb_config is not None:
            init_wandb(wandb_config, run_name, os.getenv("WANDB_API_KEY", None))

        # Save config
        save_config(training_config, run_name)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
