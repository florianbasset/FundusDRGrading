import argparse
import os

import torch
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar

from fundusClassif.callbacks.callback_factory import get_callbacks
from fundusClassif.callbacks.result_saver import ResultSaver
from fundusClassif.data.data_factory import get_datamodule_from_config
from fundusClassif.my_lightning_module import TrainerModule
from fundusClassif.utils.logger import get_wandb_logger

torch.set_float32_matmul_precision("medium")

def train(config: Config):
    seed_everything(1234, workers=True)
    project_name = config["logger"]["project"]

    wandb_logger = get_wandb_logger(project_name, config.tracked_params, ('model/architecture', config["model"]["architecture"]))
    
    datamodule = get_datamodule_from_config(config["datasets"], config["data"])
    
    test_dataloader = datamodule.test_dataloader()
    test_datasets_ids = [d.dataset.id for i, d in enumerate(test_dataloader)]
    model = TrainerModule(config["model"], config["training"], test_datasets_ids)

    training_callbacks = get_callbacks(config['training'])
    
    checkpoint_callback = ModelCheckpoint(
        monitor="Validation Quadratic Kappa",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", project_name, os.environ["WANDB_RUN_NAME"]),
    )

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            *training_callbacks,
            ResultSaver(os.path.join("results", project_name)),
            #RichProgressBar(),
            checkpoint_callback,
            EarlyStopping(monitor="Validation Quadratic Kappa", patience=25, mode="max"),
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    #trainer.test(model, dataloaders=test_dataloader, ckpt_path="best", verbose=True)

if __name__ == "__main__":
    config = Config("configs/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=config["training"]["lr"])
    parser.add_argument("--optimizer", type=str, default=config["training"]["optimizer"]["name"])
    parser.add_argument("--ema", type=int, default=False)
    parser.add_argument("--swa", type=int, default=False)
    parser.add_argument("--as_regression", type=int, default=config["training"]["as_regression"])
    parser.add_argument("--data_augmentation_type", type=str, default=config["data"]["data_augmentation_type"])
    parser.add_argument("--mixup",type=int, default=True)
    parser.add_argument("--mixup_alpha", type=float, default=config["training"]["mixup"]["mixup_alpha"])
    parser.add_argument("--cutmix_alpha", type=float, default=config["training"]["mixup"]["cutmix_alpha"])

    args = parser.parse_args()
    lr = args.lr
    optimizer = args.optimizer
    ema = args.ema
    swa = args.swa
    as_regression = args.as_regression
    print(as_regression)
    data_augmentation_type = args.data_augmentation_type
    mixup = args.mixup
    print(mixup)
    mixup_alpha = args.mixup_alpha
    cutmix_alpha = args.cutmix_alpha

    if as_regression and mixup:
        # Regression and mixup are not compatible
        raise ValueError("Regression and mixup are not compatible")
    
    if not ema:
        del config["training"]["ema"]
    if not swa:
        del config["training"]["swa"]
    if not mixup:
        del config["training"]["mixup"]
    else:
        config["training"]["mixup"]["mixup_alpha"] = mixup_alpha
        config["training"]["mixup"]["cutmix_alpha"] = cutmix_alpha

    config["training"]["lr"] = lr
    config["training"]["optimizer"]["name"] = optimizer
    config["training"]["as_regression"] = as_regression
    config["data"]["data_augmentation_type"] = data_augmentation_type 

    train(config)