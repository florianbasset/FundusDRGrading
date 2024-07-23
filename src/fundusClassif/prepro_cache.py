import os
import argparse

import torch
from nntools.utils import Config
from nntools.dataset import nntools_wrapper
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar

from fundusClassif.callbacks.callback_factory import get_callbacks
from fundusClassif.callbacks.result_saver import ResultSaver
from fundusClassif.data.data_factory import get_datamodule_from_config
from fundusClassif.my_lightning_module import TrainerModule

def train(arch: str):
    seed_everything(1234, workers=True)

    config = Config("configs/config.yaml")
    config["model"]["architecture"] = arch
    config['data']['cache_dir'] = os.path.join(config['data']['cache_dir'], '_seoud')

    datamodule = get_datamodule_from_config(config["datasets"], config["data"])

    seoud_preprocess = nntools_wrapper(seoud_preprocess)

    datamodule.post_resize_pre_cache.append(seoud_preprocess)
    
    datamodule.setup_all()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    architecture = args.model
    train(architecture)

