"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: A sample script to train Patent DB model. Training can be configured by overwriting the parameters in the config.
"""

from datetime import date
import logging
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from src.selfpad import SelfPAD
from utils_common.utils import set_dirs
from utils_common.arguments import get_arguments, get_config
from utils_pretrain.load_data import PADLoader
from utils_pretrain.model_utils import StopAtGlobalStepCallback

# Configure the logging level
logging.basicConfig(level=logging.INFO)


def train(
    data_loader: PADLoader, config: Dict[str, Any], save_weights: bool = True
) -> None:
    """
    Trains SelfPAD

    Parameters
    ----------
    data_loader : PADLoader
        PyTorch data loader.
    config : Dict[str, Any]
        Dictionary containing configuration options and arguments.
    save_weights: bool
        Whether to save weights
    """
    # Set the random seed to make the training deterministic
    pl.seed_everything(seed=config["seed"])

    # Initialize the model
    model = SelfPAD(config)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    # Determine the stopping criteria
    if config["hard_stop"] is not None:
        stop_callback = StopAtGlobalStepCallback(stop_at_step=config["hard_stop"])
        callbacks.append(stop_callback)

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=model._model_path,
            filename="pretrained_model.ckpt",
            save_top_k=1,
            verbose=True,
            every_n_epochs=100,
        )
        callbacks.append(checkpoint_callback)

    else:
        stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=model._model_path,
            filename="model-{val_loss:.2f}",
            save_top_k=2,
        )
        callbacks = callbacks + [stop_callback, checkpoint_callback]

    # Create a new instance of TensorBoardLogger with the desired save_dir
    if config["use_wandb"]:
        logger = WandbLogger(
            save_dir=config["results_dir"],
            project=config["experiment"],
            log_model="all",
        )
    else:
        today = date.today()
        logger = CSVLogger(save_dir=config["results_dir"], name="log_" + today.strftime('%Y-%m-%d'))

    val_dataloader = data_loader.validation_loader if config["use_validation"] else None

    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",  # "dp", Use data parallel
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        val_check_interval=config["val_check_interval"]
        if val_dataloader is not None
        else None,
        enable_checkpointing=True,
        limit_val_batches=0 if val_dataloader is None else 1.0,
    )

    # Fit the model
    trainer.fit(
        model,
        train_dataloaders=data_loader.train_loader,
        val_dataloaders=val_dataloader if val_dataloader is not None else None,
    )
    trainer.save_checkpoint(model._model_path + "/pretrained_model.ckpt")


def launch(config: Optional[Dict] = None) -> None:
    """
    Main function for evaluation.

    Parameters
    ----------
    config : Optional[Dict], optional
        Dictionary containing options and arguments. If None, retrieves configuration from
        command-line arguments and/or default values. Defaults to None.
    """

    if config is None:
        # Get parser / command line arguments
        args = get_arguments()

        # Get configuration file
        config = get_config(args)

    # Define the experiment name -- This is where the results will be saved
    config["experiment"] = config.get("experiment", config["dataset"])

    # Ser directories (or create if they don't exist)
    set_dirs(config)

    # Get data loader for first dataset.
    data_loader = PADLoader(config)

    # Start training.
    train(data_loader, config, save_weights=True)


def main():
    # Get parser / command line arguments
    args = get_arguments()

    # Get configuration file
    config = get_config(args)
    
    # Dataset is PAD and experiment name is set to pre-training. Comment them out if you want to define it as a command-line argument
    config["dataset"] = "pad"
    config["experiment"] = "pretraining"

    # Start training.
    launch(config=config)

    
if __name__ == "__main__":
    main()
