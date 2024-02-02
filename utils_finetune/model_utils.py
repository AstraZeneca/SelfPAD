"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of models and related support functions.
"""

import logging
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from src.selfpad import SelfPAD
from pytorch_lightning import Callback

logging.basicConfig(level=logging.INFO)


class PADFT(nn.Module):
    def __init__(self, config: Dict[str, Any], reinit_n_layers: int = 1) -> None:
        """
        Fine-tunes a pre-trained an attention based model by adding a new classification head.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing model configuration.
        reinit_n_layers : int, optional
            Number of layers to reinitialize. Default is 1.

        Attributes
        ----------
        transformer : nn.Module
            A pre-trained model.
        mlp : MLP
            A multi-layer perceptron classification head.
        reinit_n_layers : int
            Number of layers to re-initialize.
        """
        super().__init__()
        self.config = config
        
        # Path to the pre-trained model
        checkpoint_path = config["pretrained_checkpoint_path"]

        self.transformer = SelfPAD.load_from_checkpoint(
            checkpoint_path=checkpoint_path, config=config
        )

        # Input dimension of MLP is the output dimension of pre-trained encoder
        input_dim = config["dim_o"]

        self.mlp_h = MLP(
            dim_i=input_dim,
            dim_o=config["mlp_output_dim"],
            softmax=False if config["use_regression"] else True,
        )

        self.reinit_n_layers = reinit_n_layers

        if reinit_n_layers > 0:
            self._do_reinit(model=self.transformer.transformer)

    def _init_weight_and_bias(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)

    def _do_reinit(self, model: nn.Module) -> None:
        """Reinitializes the final layers of the model."""
        # Re-init last n layers.
        for n in range(self.reinit_n_layers):
            model.layers[-(n + 1)].apply(self._init_weight_and_bias)

    def forward(
        self,
        x1: dict,
        x1aa,
        agg_dim: int = -1,
    ) -> tuple:
        """
        Forward pass of the RoBERTa model.

        Parameters
        ----------
        x1 : dict
            A dictionary containing input data and attention masks for heavy chain.
        x2 : dict
            A dictionary containing input data and attention masks for light chain.
        seqs_heavy: List[str]
            Heavy chain sequences
        seqs_light: List[str]
            Light chain sequences
        agg_dim : int, optional
            The dimension to average the logits over, by default -1.

        Returns
        -------
        tuple
            A tuple containing the model predictions and output from the final layer.
        """
        x1 = x1.reshape(-1, self.config["max_seq_length"]).to(self.config["device"])
        x1aa = x1aa.reshape(-1, self.config["max_seq_length"], 18).to(
            self.config["device"]
        )

        # Generate a mask to mask out padded regions
        mask1 = x1.ne(self.config["pad_index"]).int()

        # Get the embeddings
        heavy, hemb_l, mask1 = self.transformer(x1, x1aa, mask=mask1)
        h_emb = hemb_l[-1]

        h = torch.sum(h_emb * mask1, dim=1) / torch.sum(
            mask1[:, :, 0], dim=-1, keepdim=True
        )

        if self.training and self.config["add_noise"]:
            # Add noise to heavy and light chains.
            h = self.add_noise(h)

        preds, h = self.mlp_h(h)

        h_ext = torch.sum(hemb_l[-3], dim=2)

        return preds, h, h_ext

    
    def add_noise(self, h: torch.Tensor):
        """Adds noise to the embedding

        Parameters
        ----------
        h : torch.Tensor
            Embeddings to which the noise will be added

        Returns
        -------
        torch.Tensor
            Noisy embeddings
        """
        # Gaussian noise
        if self.config["noise_type"] == "normal":
            e = self.config["noise_level"] * torch.randn_like(h)
        
        # Swap noise
        else:
            no, dim = h.size()

            # Initialize corruption array
            e = torch.zeros_like(h)

            # Randomly (and column-wise) shuffle data
            for i in range(dim):
                idx = np.random.permutation(no)
                e[:, i] = h[idx, i]

        m = (
            torch.rand(h.size(), device=self.config["device"])
            < 1 - self.config["noise_ratio"]
        ).float()

        return m * h + (1 - m) * e


class Linear(nn.Module):
    """Multi-layer Perceptron (MLP) model.

    Parameters
    ----------
    dim_i : int
        Input feature dimension.
    dim_o : int
        Output feature dimension.
    """

    def __init__(self, dim_i: int = 150, dim_o: int = 3):
        super(Linear, self).__init__()

        self.linear = nn.Linear(dim_i, dim_o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """

        return self.linear(x)


class MLP(nn.Module):
    """Multi-layer Perceptron (MLP) model.

    Parameters
    ----------
    dim_i : int
        Input feature dimension.
    dim_o : int
        Output feature dimension.
    softmax : bool, optional
        Whether to apply softmax to the output or not. Default is False.
    """

    def __init__(self, dim_i: int = 128, dim_o: int = 3, softmax: bool = False):
        super(MLP, self).__init__()

        self.softmax = softmax
        self.res_block = Residual(Block(d_in=dim_i, d_out=dim_i))
        self.linear = nn.Linear(dim_i, dim_o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """

        h = self.res_block(x)
        x = self.linear(h)
        return (
            torch.softmax(x, dim=-1)
            if self.softmax
            else x.reshape(
                -1,
            ),
            h,
        )


class Residual(nn.Module):
    """Residual block that applies a function and adds the input to the output"""

    def __init__(self, fn: nn.Module):
        """
        Parameters
        ----------
        fn : nn.Module
            A module that applies some transformation to input
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the residual block and adds input to the output

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., hidden_size)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, ..., hidden_size)
        """
        return self.fn(x) + x


class Block(nn.Module):
    """The main building block of an MLP."""

    def __init__(self, d_in: int, d_out: int):
        super(Block, self).__init__()

        # Inner dimension
        dim = 4 * d_in

        self.head = nn.Sequential(
            nn.Linear(d_in, dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(p=0.3),
            nn.Linear(dim, d_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.head(x)


class StopAtGlobalStepCallback(Callback):
    """
    Callback that stops the training when a specified global step is reached.

    Parameters
    ----------
    stop_at_step : int
        The global step at which to stop the training.
    """

    def __init__(self, stop_at_step: int):
        super().__init__()
        self.stop_at_step = stop_at_step

    def on_train_batch_end(self, trainer, *args, **kwargs):
        """
        Callback function that is executed after every training batch.

        Parameters
        ----------
        trainer : Trainer
            The trainer object used for training.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """
        if trainer.global_step >= self.stop_at_step:
            logging.info(
                f"Reached global step {trainer.global_step}. Stopping training."
            )
            trainer.should_stop = True
