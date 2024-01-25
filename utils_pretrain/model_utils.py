"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

A library of models used in self-supervised learning framework.

"""


import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import Callback
from torch import Tensor, einsum, nn

logging.basicConfig(level=logging.INFO)


class PADformer(nn.Module):
    """
    PADformer class.

    Attributes
    ----------
    model_dim : int
        Model dimension.
    dim_i : int
        Input dimension.
    dim_o : int
        Output dimension.
    depth : int
        Number of layers in the model.

    Methods
    -------
    get_num_params(non_embedding=True)
        Get the number of parameters in the model.
    _init_weights(module)
        Initialize the weights of the given module.
    forward(x, mask=None)
        Perform a forward pass through the model.
    """

    def __init__(self, config: dict):
        """
        Initialize PADformer.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        super(PADformer, self).__init__()

        self.config = config
        self.model_dim: int = config["model_dim"]
        self.dim_o: int = config["dim_o"]
        self.depth: int = config["depth"]

        # PADformer layers
        self.wte = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.ab_emb = AbEmbeddings(config)

        self.layers = nn.ModuleList([])

        for _ in range(self.depth):
            layer = nn.ModuleList(
                [
                    Residual(PreNorm(self.config, Attention(self.config))),
                    Residual(PreNorm(self.config, FeedForward(self.config))),
                ]
            )
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(self.model_dim)

        self.dim_out = self.dim_o or self.model_dim
        mult = config["mult"]

        self.first_lin1 = nn.Linear(self.model_dim + 18, self.model_dim)
        self.final_lin1 = nn.Linear(self.model_dim, self.model_dim * mult)
        self.final_gelu = nn.GELU()
        self.final_lin2 = nn.Linear(self.model_dim * mult, self.dim_out)

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        logging.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get the number of parameters in the model.

        Parameters
        ----------
        non_embedding : bool, optional, default=True
            Exclude embedding parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module: nn.Module):
        """
        Initialize the weights of the given module.

        Parameters
        ----------
        module : torch.nn.Module
            The module to initialize weights for.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        x_aa: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x_aa : torch.Tensor
            Input tensor.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            Output tensor and list of embeddings.
        """
        emb_l = []

        x = self.ab_emb(x)

        # Combine the inputs over the sequence dimension: (B, S, d)
        x = torch.cat([x, x_aa.float()], dim=-1)

        if self.training:
            x = x + 0.01 * torch.randn(*x.size(), device=x.device)

        # Project from model_dim + 18 dimension to model_dim
        x = self.first_lin1(x)

        for (att, ff) in self.layers:
            x = ff(att(x, mask=mask))
            emb_l.append(x)

        x = self.norm_out(x)
        emb_l.append(x)

        # Project and record the embedding
        xlin = self.final_lin1(x)
        emb_l.append(xlin)

        # Apply nonlinearity and record the embedding
        xout_emb = self.final_gelu(xlin)
        emb_l.append(xout_emb)

        # Final linear layer
        xout = self.final_lin2(xout_emb)

        # Mask out the padded regions of the sequence
        if mask is not None:
            mask = mask[..., None]
            mask = torch.repeat_interleave(mask, xout.size(-1), dim=-1)
            xout = torch.sum(xout * mask, dim=1) / torch.sum(
                mask[:, :, 0], dim=-1, keepdim=True
            )
        else:
            xout = xout.reshape(-1, xout.size(1))

        # Normalization
        xout = F.normalize(xout, p=2, dim=-1)

        return xout, emb_l, mask

    def get_embeddings(self, x, emb_layer=-1):
        x = x.reshape(-1, self.config["max_seq_length"]).to(self.config["device"])
        mask = x.ne(self.config["pad_index"]).int()

        # Generate subsets
        out, embeddings = self(x, mask=mask)

        # Generate mask
        latent_emb = embeddings[emb_layer].cpu().detach().numpy()
        mask = mask[..., None]
        mask = np.repeat(mask.cpu().numpy(), latent_emb.shape[-1], axis=-1)

        return np.sum(latent_emb * mask, axis=1) / np.sum(
            mask[:, :, 0], axis=-1, keepdims=True
        )


class Attention(nn.Module):
    """
    Attention class.


    Methods
    -------
    forward(x, mask=None)
        Perform a forward pass through the attention layer.
    """

    def __init__(self, config: dict):
        """
        Initialize Attention.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        super(Attention, self).__init__()

        self.config = config

        # Parameters for PADformer layers
        self.model_dim: int = config["model_dim"]
        self.dim_head: int = config["dim_head"]
        self.heads: int = config["heads"]
        self.inner_dim = self.heads * self.dim_head
        self.scale = self.dim_head**-0.5

        # Linear layers
        self.to_qkv = nn.Linear(self.model_dim, 3 * self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, self.model_dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform a forward pass through the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        h = self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b n (h qkva d) -> b h n qkva d", h=h, qkva=3).unbind(
            dim=-2
        )
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if mask is not None:
            mask_value = 1e6  # torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]
            mask = mask.bool()

            # Mask attention
            dots.masked_fill_(~mask, -mask_value)

        att = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", att, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


class PreNorm(nn.Module):
    """
    PreNorm class.

    Attributes
    ----------
    config : dict
        Configuration dictionary.
    model_dim : int
        Model dimension.

    Methods
    -------
    forward(x, **kwargs)
        Perform a forward pass through the pre-normalization layer.
    """

    def __init__(self, config: dict, fn: nn.Module):
        """
        Initialize PreNorm.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        fn : torch.nn.Module
            Function to apply after normalization.
        """
        super(PreNorm, self).__init__()

        self.config = config
        self.model_dim: int = config["model_dim"]

        self.norm = nn.LayerNorm(self.model_dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Perform a forward pass through the pre-normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x


class Residual(nn.Module):
    """
    Residual class.

    Methods
    -------
    forward(x, **kwargs)
        Perform a forward pass through the residual layer.
    """

    def __init__(self, fn: nn.Module):
        """
        Initialize Residual.

        Parameters
        ----------
        fn : torch.nn.Module
            Function to apply in the residual connection.
        """
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Perform a forward pass through the residual layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return x + self.fn(x, **kwargs)


class FeedForward(nn.Module):
    """
    FeedForward class.

    Attributes
    ----------
    model_dim : int
        Model dimension.
    dim_out : int
        Output dimension.

    Methods
    -------
    forward(x)
        Perform a forward pass through the feedforward layer.
    """

    def __init__(self, config: dict, dim_o: Optional[int] = None):
        """
        Initialize FeedForward.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        dim_o : int, optional
            Output dimension.
        """
        super().__init__()

        self.model_dim: int = config["model_dim"]
        self.dim_out: int = dim_o or self.model_dim
        mult: int = config["mult"]

        self.net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * mult),
            nn.GELU(),
            nn.Linear(self.model_dim * mult, self.dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the feedforward layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.net(x)


class AbEmbeddings(torch.nn.Module):
    """
    AbEmbeddings class.

    Attributes
    ----------
    pad_token_id : int
        Padding token ID.
    AAEmbeddings : torch.nn.Embedding
        Amino acid embeddings.
    PositionEmbeddings : torch.nn.Embedding
        Position embeddings.
    LayerNorm : torch.nn.LayerNorm
        Layer normalization.
    Dropout : torch.nn.Dropout
        Dropout layer.

    Methods
    -------
    forward(src)
        Perform a forward pass through the embeddings layer.
    create_position_ids_from_input_ids(input_ids, padding_idx)
        Create position IDs from input IDs.
    """

    def __init__(self, config: dict):
        """
        Initialize AbEmbeddings.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        super().__init__()
        self.pad_token_id: int = config["pad_index"]

        vocab_size: int = config["vocab_size"]
        hidden_size: int = config["model_dim"]
        max_position_embeddings: int = config["max_seq_length"]

        self.AAEmbeddings = torch.nn.Embedding(
            vocab_size, hidden_size, padding_idx=self.pad_token_id
        )

        # Here padding_idx is always 0 since we use masking to make idx of paddings 0
        self.PositionEmbeddings = torch.nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=0
        )

        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-05)
        self.Dropout = torch.nn.Dropout(config["dropout_rate"])

    def forward(self, src: Tensor) -> Tensor:
        """
        Perform a forward pass through the embeddings layer.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        inputs_embeds = self.AAEmbeddings(src)

        position_ids = self.create_position_ids_from_input_ids(src, self.pad_token_id)
        position_embeddings = self.PositionEmbeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        return self.LayerNorm(embeddings)  # self.Dropout()

    def create_position_ids_from_input_ids(
        self, input_ids: Tensor, padding_idx: int
    ) -> Tensor:
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input tensor with IDs.
        padding_idx : int
            Padding index.

        Returns
        -------
        torch.Tensor
            Tensor containing position IDs.
        """
        mask = input_ids.ne(padding_idx).int()

        return torch.cumsum(mask, dim=1).long() * mask


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
