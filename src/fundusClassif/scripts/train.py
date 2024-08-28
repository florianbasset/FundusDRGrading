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
from fundusClassif.utils.images import get_preprocessing
from fundus_data_toolkit.datamodules import DataHookPosition

torch.set_float32_matmul_precision("medium")

def train(config: Config):
    seed_everything(1234, workers=True)
    project_name = config["logger"]["project"]
    
    if not config["trainer"]["fast_dev_run"]:
        wandb_logger = get_wandb_logger(project_name, config.tracked_params)

        checkpoint_callback = ModelCheckpoint(
        monitor="Validation Quadratic Kappa",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", project_name, os.environ["WANDB_RUN_NAME"]),
        )
    else:
        wandb_logger = None
        checkpoint_callback = None

    if config["data_preprocessing"]["name"] != "absent":
        config["data"]["cache_dir"] = config["data_preprocessing"]["name"]
        prepro_function = get_preprocessing(config["data_preprocessing"]["name"])

    datamodule = get_datamodule_from_config(config["datasets"], config["data"])

    if config["data_preprocessing"]["name"] != "absent":
        datamodule.set_data_pipeline_hook(prepro_function, position=DataHookPosition.POST_RESIZE_PRE_CACHE)

    test_dataloader = datamodule.test_dataloader()
    test_datasets_ids = [d.dataset.id for i, d in enumerate(test_dataloader)]
    model = TrainerModule(config["model"], config["training"], test_datasets_ids)

    training_callbacks = get_callbacks(config['training'])

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            *training_callbacks,
            #ResultSaver(os.path.join("results", project_name)),
            #RichProgressBar(),
            
            EarlyStopping(monitor="Validation Quadratic Kappa", patience=10, mode="max"),
            LearningRateMonitor(),
        ] + ([checkpoint_callback] if checkpoint_callback is not None else []),
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path="best", verbose=True)

if __name__ == "__main__":
    config = Config("configs/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=config["training"]["lr"])
    #parser.add_argument("--data_augmentation_type", type=str, default=config["data"]["data_augmentation_type"])
    parser.add_argument("--preprocessing", type=str, default=config["data_preprocessing"]["name"])
    parser.add_argument("--model", type=str, default=config["model"]["architecture"])

    #print(config)
    
    args = parser.parse_args()
    lr = args.lr
    model = args.model

    #if args.data_augmentation_type == "None":
    #    data_augmentation_type = None
    #else:   
    #    data_augmentation_type = args.data_augmentation_type

    preprocessing = args.preprocessing     
    
    config["training"]["lr"] = lr
    #config["data"]["data_augmentation_type"] = data_augmentation_type 
    config["data_preprocessing"]["name"] = preprocessing
    config["model"]["architecture"] = model

    #print(config)

    train(config)
