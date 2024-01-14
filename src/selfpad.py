"""
Author: Talip Ucar
email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: SelfPAD class, the framework used for self-supervised representation learning.
"""

import itertools
import json
import logging
import os
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import Tensor

from utils_pretrain.loss_functions import JointLoss
from utils_pretrain.model_utils import PADformer
from utils_common.utils import set_dirs, set_seed

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)


class SelfPAD(pl.LightningModule):
    """
    SelfPAD class for self-supervised representation learning.

    Attributes
    ----------
    config : dict
        Configuration dictionary.
    device : str
        Device type to be used for training ('cuda' or 'cpu').
    model_dict : dict
        Dictionary containing model instances.
    summary : dict
        Dictionary containing training and validation metrics.
    is_combination : bool
        Indicates whether combinations of projections should be used.
    transformer : PADformer
        The transformer model for self-supervised learning.
    ss_loss : JointLoss
        Joint loss object to calculate contrastive, and distance losses.
    optimizer_ae : torch.optim
        The optimizer for updating the transformer model.
    scheduler : torch.optim.lr_scheduler
        Optional learning rate scheduler for the optimizer.
    num_samples : int
        The number of samples in the dataset.
    epoch : int
        The current epoch during training.

    Methods
    -------
    set_transformer()
        Sets up the transformer model, optimizer, and loss.
    set_parallelism(model)
        Sets up parallelism in training for the given model.
    fit(data_loader)
        Trains and validates the model.
    on_train_epoch_end()
        Logs training metrics at the end of each epoch.
    on_validation_epoch_end()
        Logs validation metrics at the end of each epoch.
    on_train_end()
        Saves the final model and summary.
    save_models()
        Saves the model weights.
    save_summary()
        Saves the training and validation summary.
    """

    def __init__(self, config: Dict):
        """
        Initialize the SelfPAD class.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        super().__init__()
        # Get config
        self.config = config
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.config)
        # Set paths for results and initialize some arrays to collect data during training
        self._set_paths()
        # Set directories i.e. create ones that are missing.
        set_dirs(self.config)
        # Set the condition if we need to build combinations of 2 out of projections.
        self.is_combination = (
            self.config["contrastive_loss"] or self.config["distance_loss"]
        )
        # Instantiate networks
        logger.info(
            "Building the models for training and evaluation in SelfPAD framework..."
        )
        # Set transformers i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_transformer()
        self.save_hyperparameters()

    def set_transformer(self):
        """Sets up the transformer model, optimizer, and loss"""
        # Instantiate the model for the text transformer
        self.transformer = PADformer(self.config)
        # Set data parallelism
        # self.transformer = self.set_parallelism(self.transformer)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"transformer": self.transformer})
        # Assign transformer to a device
        for _, model in self.model_dict.items():
            model.to(self.config["device"])
        # Get model parameters
        parameters = [model.parameters() for _, model in self.model_dict.items()]
        # Joint loss including contrastive, reconstruction and distance losses
        self.ss_loss = JointLoss(self.config)
        # Set optimizer for transformer
        self.optimizer_ae = self._adam(parameters, lr=self.config["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"training_loss": [], "val_loss": []})

    def set_parallelism(self, model):
        """Sets up parallelism in training."""
        # If we are using GPU, and if there are multiple GPUs, parallelize training
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"{torch.cuda.device_count()} GPUs will be used!")
            model = torch.nn.DataParallel(model)
        return model

    def forward(self, seq, seq_aa, mask):
        """Forward pass through the transformer model."""
        return self.transformer(seq, seq_aa, mask=mask)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Performs a single training step.

        Parameters
        ----------
        batch : tuple
            A tuple containing the training data for this batch.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        Tensor
            The computed loss for this training step.
        """
        loss = self.step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Performs a single validation step.

        Parameters
        ----------
        batch : tuple
            A tuple containing the validation data for this batch.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        Tensor
            The computed loss for this validation step.
        """
        loss = self.step(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def step(self, batch: tuple) -> Tensor:
        """
        Computes the loss for the given batch.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input data for this batch.

        Returns
        -------
        Tensor
            The computed loss for the given batch.
        """
        x1, x2, x1aa, x2aa, y1, y2 = batch

        x1 = x1.reshape(-1, self.config["max_seq_length"])
        x2 = x2.reshape(-1, self.config["max_seq_length"])

        x1aa = x1aa.reshape(-1, self.config["max_seq_length"], 18)
        x2aa = x2aa.reshape(-1, self.config["max_seq_length"], 18)

        y1 = y1.reshape(-1, 1)
        y2 = y2.reshape(-1, 1)

        # Build the batch
        seq = torch.cat([x1, x2]).to(self.config["device"])
        seq_aa = torch.cat([x1aa, x2aa]).to(self.config["device"])

        # Generate a mask to mask out padded regions
        mask = seq.ne(self.config["pad_index"]).int()

        # Forwards pass
        out, _, _ = self(seq, seq_aa, mask=mask)

        # Reconstruction loss by using GAE's native function
        ss_loss = self.ss_loss(out)

        return ss_loss

    def configure_optimizers(self):
        """Configures the optimizer for the model."""
        return self.optimizer_ae

    def on_training_epoch_end(self, outputs: List[dict]) -> None:
        """
        Log training metrics at the end of each epoch.

        Parameters
        ----------
        outputs : List[dict]
            A list of dictionaries containing the outputs of each training step.
        """
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss", avg_train_loss, on_epoch=True, prog_bar=True, logger=True
        )

    def on_validation_epoch_end(self, outputs: List[dict]) -> None:
        """
        Log validation metrics at the end of each epoch.

        Parameters
        ----------
        outputs : List[dict]
            A list of dictionaries containing the outputs of each validation step.
        """
        if outputs:
            avg_val_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.log("val_loss", avg_val_loss, on_epoch=True)
        else:
            self.log("val_loss", 0, on_epoch=True)

    def on_train_end(self):
        """Save the final model and summary."""
        self.save_models()
        self.save_summary()

    def save_models(self):
        """Save the model weights."""
        for name, model in self.model_dict.items():
            torch.save(model.state_dict(), f"{name}_model.pt")
            logger.info(f"{name} model saved.")

    def _adam(self, params: List, lr: float = 1e-4) -> torch.optim.AdamW:
        """
        Sets up AdamW optimizer using model params.

        Parameters
        ----------
        params : List
            A list of model parameters.
        lr : float, optional
            The learning rate, by default 1e-4.

        Returns
        -------
        torch.optim.AdamW
            The configured AdamW optimizer.
        """
        return torch.optim.AdamW(
            itertools.chain(*params), lr=lr, betas=(0.9, 0.999), eps=1e-07
        )

    def _set_paths(self):
        """Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = os.path.join(
            self.config["results_dir"], self.config["experiment"]
        )
        # Directory to save model
        self._model_path = os.path.join(self.config["training_dir"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self.config["training_dir"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self.config["training_dir"], "loss")

    def save_summary(self):
        """Save the training and validation summary."""
        with open(self._loss_path + "/summary.json", "w") as f:
            json.dump(self.summary, f)
        logger.info("Summary saved.")
