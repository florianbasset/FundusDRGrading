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


def train(arch: str):
    seed_everything(1234, workers=True)

    config = Config("configs/config.yaml")
    config["model"]["architecture"] = arch
    project_name = config["logger"]["project"]

    wandb_logger = get_wandb_logger(project_name, config.tracked_params, ('model/architecture', arch,),id = "ofjzmugj")
    datamodule = get_datamodule_from_config(config["datasets"], config["data"])
    
    test_dataloader = datamodule.test_dataloader()
    for i, d in enumerate(test_dataloader):
        print(len(d.dataset))
    return None    
    test_datasets_ids = [d.dataset.id for i, d in enumerate(test_dataloader)]
    model = TrainerModule(config["model"], config["training"], test_datasets_ids)

    training_callbacks = get_callbacks(config['training'])

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            *training_callbacks,
            #ResultSaver(os.path.join("results", project_name)),
            EarlyStopping(monitor="Validation Quadratic Kappa", patience=25, mode="max"),
            LearningRateMonitor(),
        ],
    )
    
    trainer.test(model, 
                 dataloaders=test_dataloader, 
                 ckpt_path="checkpoints/Grading-DiabeticRetinopathy-Comparisons-V3/stellar-pine-169/epoch=44-step=42615.ckpt", 
                 verbose=True,
                 )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    architecture = args.model
    train(architecture)
