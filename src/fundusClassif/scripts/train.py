import argparse
import os

import torch
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from fundusClassif.data.data_factory import get_datamodule_from_config
from fundusClassif.my_lightning_module import TrainerModule
from fundusClassif.utils.logger import get_wandb_logger
from fundusClassif.callbacks.callback_factory import get_callbacks
torch.set_float32_matmul_precision("medium")


def train(arch: str):
    seed_everything(1234, workers=True)

    config = Config("configs/config.yaml")
    config["model"]["architecture"] = arch
    projet_name = config["logger"]["project"]

    
    wandb_logger = get_wandb_logger(projet_name, config.tracked_params, ('model.architecture', arch))
    datamodule = get_datamodule_from_config(config["datasets"], config["data"])
    
    test_dataloader = datamodule.test_dataloader()
    test_datasets_ids = [d.dataset.id for i, d in enumerate(test_dataloader)]
    model = TrainerModule(config["model"], config["training"], test_datasets_ids)

    training_callbacks = get_callbacks(config['training'])
    
    checkpoint_callback = ModelCheckpoint(
        monitor="Quadratic Kappa",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", os.environ["WANDB_RUN_NAME"]),
    )

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            *training_callbacks,
            checkpoint_callback,
            EarlyStopping(monitor="Quadratic Kappa", patience=25, mode="max"),
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    architecture = args.model
    train(architecture)
