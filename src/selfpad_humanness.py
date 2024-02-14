"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Wrapper function for training routine.
"""
import copy
import gc
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from tqdm import tqdm
import itertools

from utils_common.utils import set_dirs, set_seed
from utils_finetune.model_utils import PADFT


class PADFintune(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize the PADFintune class with the given configuration.

        Parameters
        ----------
        config : Dict
            Configuration dictionary containing model and training parameters.
        """
        super().__init__()
        self.config = config
        self.device = config["device"]
        # Set random seed
        set_seed(self.config)
        # Set directories i.e. create ones that are missing.
        set_dirs(self.config)
        self._set_paths()

        # ------Network---------
        # Instantiate networks
        print("Building the models for training and evaluation in SelfPAD framework...")
        # Set Autoencoders i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder()
        # Print out model architecture
        self.print_model_summary()

    def set_autoencoder(self):
        """Sets up the autoencoder model, optimizer, and loss"""
        self.transformer = PADFT(
            self.config, reinit_n_layers=self.config["reinit_n_layers"]
        )
        self.transformer.train().to(self.config["device"])
        self.model_dict = {"transformer": self.transformer}
        
        # Set optimizer for autoencoder
        self.optimizer_ae = self.adjusted_AdamW()

        task = "binary" if self.config["mlp_output_dim"] <= 2 else "multiclass"
        self.roc_auc = AUROC(task=task)
        self.f1_score = F1Score(
            task=task, num_classes=self.config["mlp_output_dim"], average="macro"
        )
        self.acc_score = Accuracy(
            task=task, num_classes=self.config["mlp_output_dim"], average="macro"
        )
        self.precision_score = Precision(
            task=task, average="macro", num_classes=self.config["mlp_output_dim"]
        )
        self.recall_score = Recall(
            task=task, num_classes=self.config["mlp_output_dim"], average="macro"
        )
        self.acc_l, self.f1_l, self.recall_l = [], [], []
        self.vacc_l, self.vf1_l, self.vrecall_l, self.vprec_l = [], [], [], []

        self.val_f1_best = 0
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.5)
        self.agg_dim = -2 if self.config["aggregate_on_sequence_dim"] else -1

    def fit(self, data_loader):
        """Fits model to the data"""

        # Get data loaders
        train_loader = data_loader.train_loader
        validation_loader = data_loader.validation_loader
        test_loader = data_loader.test_loader

        # Placeholders to record losses per batch
        self.loss = {"tloss_b": [], "tloss_e": [], "vloss_e": []}

        # Turn on training mode for the model.
        self.set_mode(mode="training")

        # Compute total number of batches per epoch
        self.total_batches = len(train_loader)

        # Start joint training of Autoencoder with Projection network
        for epoch in range(self.config["epochs"]):
            self.epoch = epoch
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(
                enumerate(train_loader), total=self.total_batches, leave=True
            )

            # 0 - Update Autoencoder
            self.update_autoencoder()

            # 1 - Validate every nth epoch. n=1 by default, but it can be changed in the config file
            if self.config["validate"]:
                with torch.no_grad():
                    # Compute validation loss
                    _ = self.validate(validation_loader)
                # Get reconstruction loss for training per epoch
            self.loss["tloss_e"].append(
                sum(self.loss["tloss_b"][-self.total_batches : -1]) / self.total_batches
            )

        # Save plot of training and validation losses
        self.save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")

        with torch.no_grad():
            # Compute validation loss
            f1, recall, prec, auc, acc = self.validate(test_loader, split="test")
            return f1, recall, prec, auc, acc

    def validate(self, validation_loader, split="val"):
        # Attach progress bar to data_loader
        self.val_tqdm = tqdm(
            enumerate(validation_loader), total=self.total_batches, leave=True
        )
        val_loss = []
        preds_l = []
        labels_l = []

        self.transformer.eval()
        self.transformer.config["add_noise"] = False

        # pass data through model
        for batch in self.val_tqdm:

            x1, x1aa, y1 = batch[1]
            labels = y1.reshape(
                -1,
            )

            preds, h, _ = self.transformer(x1, x1aa, agg_dim=self.agg_dim)

            # Generate labels and predictions
            preds_idx = (preds.argmax(axis=-1).type(torch.LongTensor)).cpu()
            preds_l.append(preds_idx)
            labels_l.append(labels)

            loss = self.ce_loss(preds, labels.to(self.device))
            batch_loss = loss.item()
            val_loss.append(batch_loss)
            del loss

        preds_l = torch.cat(preds_l)
        labels_l = torch.cat(labels_l)

        # Compute metrics
        f1 = 100 * self.f1_score(preds_l, labels_l)
        acc = 100 * self.acc_score(preds_l, labels_l)
        recall = 100 * self.recall_score(preds_l, labels_l)
        auc = 100 * self.roc_auc(preds_l, labels_l)
        prec = 100 * self.precision_score(preds_l, labels_l)

        self.vf1_l.append(f1)
        self.vacc_l.append(acc)
        self.vrecall_l.append(recall)
        self.vprec_l.append(prec)
        self.loss["vloss_e"].append(sum(val_loss) / len(val_loss))

        gc.collect()

        if self.config["add_noise"]:
            self.transformer.config["add_noise"] = True

        self.transformer.train()

        if split == "test":
            return f1.numpy(), recall.numpy(), prec.numpy(), auc.numpy(), acc.numpy()

    def update_autoencoder(self):
        """Updates autoencoder model using subsets of features

        Args:
            x_tilde_list (list): A list that contains subsets in torch.tensor format
            Xorig (torch.tensor): Ground truth data used to generate subsets

        """

        # pass data through model
        for batch in self.train_tqdm:

            x1, x1aa, y1 = batch[1]
            labels = y1.reshape(
                -1,
            )

            preds, h, _ = self.transformer(x1, x1aa, agg_dim=self.agg_dim)

            loss = self.ce_loss(preds, labels.to(self.device))

            # Generate labels and predictions
            preds_idx = (preds.argmax(axis=-1).type(torch.LongTensor)).cpu()

            # Compute metrics
            f1 = 100 * self.f1_score(preds_idx, labels)
            acc = 100 * self.acc_score(preds_idx, labels)
            recall = 100 * self.recall_score(preds_idx, labels)

            self.f1_l.append(f1)
            self.acc_l.append(acc)
            self.recall_l.append(recall)
            self.loss["tloss_b"].append(loss.item())

            # Update Autoencoder params
            self._update_model(loss, self.optimizer_ae, retain_graph=True)
            if self.epoch > 0:
                self.update_log()
            # Delete loss and associated graph for efficient memory usage
            del loss
            gc.collect()

    def _update_model(self, loss, optimizer, retain_graph=True):
        """Does backprop, and updates the model parameters

        Args:
            loss (): Loss containing computational graph
            optimizer (torch.optim): Optimizer used during training
            retain_graph (bool): If True, retains graph. Otherwise, it does not.

        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def update_log(self):
        """Updates the messages displayed during training and evaluation"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch

        description = (
            f"Epoch-{self.epoch} Total training loss:{self.loss['tloss_e'][-1]:.4f}"
        )
        description += f", f1:{self.f1_l[-1]:.4f}"
        description += f", recall:{self.recall_l[-1]:.4f}"
        description += f", acc:{self.acc_l[-1]:.4f}"
        description += (
            f", val loss:{self.loss['vloss_e'][-1]:.4f}"
            if self.config["validate"]
            else ""
        )
        description += f", val f1:{self.vf1_l[-1]:.4f}"
        description += f", val recall:{self.vrecall_l[-1]:.4f}"
        description += f", val precision:{self.vprec_l[-1]:.4f}"
        description += f", val acc:{self.vacc_l[-1]:.4f}"

        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        """Sets the mode of the models, either as .train(), or .eval()"""
        for _, model in self.model_dict.items():
            model.train() if mode == "training" else model.eval()

    def save_weights(self):
        """Used to save weights."""
        for model_name in self.model_dict:
            torch.save(
                self.model_dict[model_name], self._model_path + "/" + model_name + ".pt"
            )
        print("Done with saving models.")

    def load_models(self):
        """Used to load weights saved at the end of the training."""
        for model_name in self.model_dict:
            model = torch.load(
                self._model_path + "/" + model_name + ".pt", map_location=self.device
            )
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def print_model_summary(self):
        """Displays model architectures as a sanity check to see if the models are constructed correctly."""
        # Summary of the model
        description = f"{40 * '-'}Summary of the models:{40 * '-'}\n"
        description += f"{self.transformer}\n"
        # Print model architecture
        print(description)

    def _adam(self, params, lr=1e-4):
        """Sets up AdamW optimizer using model params"""
        return torch.optim.AdamW(itertools.chain(*params), lr=lr, betas=(0.9, 0.999), eps=1e-07)

    def adjusted_AdamW(self, debug: bool = False) -> None:
        """
        Performs fine-tuning of the SelfPAD model with AdamW optimizer using custom learning rate decay.
        Only updates the parameters of the selected layers.

        Parameters
        ----------
        debug : bool, optional
            If True, sets the debug mode on, by default False.

        Returns
        -------
        params : list
            List of named parameters and their index in the model.
        """
        opt_params = []
        init_lr = self.config["learning_rate"]


        for name, params in self.transformer.named_parameters():
            print(name)

            lr = copy.deepcopy(init_lr)
            if name.endswith("bias") or name.endswith("norm.weight"):
                weight_decay = 0.0
            else:
                weight_decay = 0.01

            if name.startswith("transformer.transformer.layers.0"):
                lr = init_lr
            elif name.startswith("transformer.transformer.layers.1"):
                lr = 2 * init_lr
            elif name.startswith("transformer.transformer.layers.2"):
                lr = 3 * init_lr
            elif name.startswith("transformer.transformer.layers.3"):
                lr = 4 * init_lr
            elif name.startswith("transformer.transformer.final_lin"):
                lr = 5 * init_lr
            elif name.startswith("mlp_h"):
                lr = 10 * init_lr
            elif name.startswith("ab_1dconv"):
                lr = 10 * init_lr

            opt_params.append(
                {"params": params, "weight_decay": weight_decay, "lr": lr}
            )

        return torch.optim.AdamW(opt_params, lr=init_lr)

    def _set_paths(self):
        """Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = self.config["results_dir"]

        # Directory to save model
        self._model_path = os.path.join(
            self.config["training_dir"], "model"
        )
        # Directory to save plots as png files
        self._plots_path = os.path.join(
            self.config["training_dir"], "plots"
        )
        # Directory to save losses as csv file
        self._loss_path = os.path.join(
            self.config["training_dir"], "loss"
        )

    def save_loss_plot(self, losses, plots_path):
        """Saves loss plot

        Args:
            losses (dict): A dictionary contains list of losses
            plots_path (str): Path to use when saving loss plot

        """
        x_axis = list(range(len(losses["tloss_e"])))
        plt.plot(x_axis, losses["tloss_e"], c="r", label="Training")
        title = "Training"
        if len(losses["vloss_e"]) >= 1:
            # If validation loss is recorded less often, we need to adjust x-axis values by the factor of difference
            beta = len(losses["tloss_e"]) / len(losses["vloss_e"])
            x_axis = list(range(len(losses["vloss_e"])))
            # Adjust the values of x-axis by beta factor
            x_axis = [beta * i for i in x_axis]
            plt.plot(x_axis, losses["vloss_e"], c="b", label="Validation")
            title += " and Validation "
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title(title + " Loss", fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_path + "/loss.png")
