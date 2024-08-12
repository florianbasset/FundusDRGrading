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
        wandb_logger = get_wandb_logger(project_name, config.tracked_params, ('model/architecture', config["model"]["architecture"]), id="5q5uiz45")

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
            
            EarlyStopping(monitor="Validation Quadratic Kappa", patience=25, mode="max"),
            LearningRateMonitor(),
        ] + ([checkpoint_callback] if checkpoint_callback is not None else []),
    )

    #trainer.fit(model, datamodule=datamodule)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path="checkpoints/Grading-DiabeticRetinopathy-Comparisons-V3/polished-glade-174/epoch=64-step=61555.ckpt", verbose=True)

if __name__ == "__main__":
    config = Config("configs/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=config["training"]["lr"])
    parser.add_argument("--optimizer", type=str, default=config["training"]["optimizer"]["name"])
    parser.add_argument("--ema", type=int, default=False)
    parser.add_argument("--swa", type=int, default=False)
    parser.add_argument("--as_regression", type=int, default=config["training"]["as_regression"])
    parser.add_argument("--data_augmentation_type", type=str, default=config["data"]["data_augmentation_type"])
    parser.add_argument("--mixup",type=int, default=False)
    parser.add_argument("--mixup_alpha", type=float, default=config["training"]["mixup"]["mixup_alpha"])
    parser.add_argument("--cutmix_alpha", type=float, default=config["training"]["mixup"]["cutmix_alpha"])
    #parser.add_argument("--decay", type=float, default=config["training"]["ema"]["decay"])
    #parser.add_argument("--swa_lrs", type=float, default=config["training"]["swa"]["swa_lrs"])
    parser.add_argument("--preprocessing", type=str, default=config["data_preprocessing"]["name"])

    #print(config)
    
    args = parser.parse_args()
    lr = args.lr
    optimizer = args.optimizer
    ema = args.ema
    swa = args.swa
    as_regression = args.as_regression
    data_augmentation_type = args.data_augmentation_type
    mixup = args.mixup
    #decay = args.decay
    #swa_lrs = args.swa_lrs
    preprocessing = args.preprocessing

    if mixup:
        mixup_alpha = args.mixup_alpha
        cutmix_alpha = args.cutmix_alpha
        config["training"]["mixup"]["mixup_alpha"] = mixup_alpha
        config["training"]["mixup"]["cutmix_alpha"] = cutmix_alpha
    else:
        del config["training"]["mixup"]       
    
    if as_regression and mixup:
        # Regression and mixup are not compatible
        raise ValueError("Regression and mixup are not compatible")
    
    if not ema:
        del config["training"]["ema"]
    #else: 
        #config["training"]["ema"]["decay"] = decay

    if not swa:
        del config["training"]["swa"]
    #else:
        #config["training"]["swa"]["swa_lrs"] = swa_lrs

    config["training"]["lr"] = lr
    config["training"]["optimizer"]["name"] = optimizer
    config["training"]["as_regression"] = as_regression
    config["data"]["data_augmentation_type"] = data_augmentation_type 
    config["data_preprocessing"]["name"] = preprocessing
    #print(config["data_preprocessing"]["name"])

    #print(config)

    train(config)
