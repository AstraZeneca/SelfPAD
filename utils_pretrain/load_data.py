"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: A library for data loaders.
"""

import os
import json
import logging
import random
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset
from utils_common.tokenizer import ABtokenizer

logging.basicConfig(level=logging.INFO)

# Registry to store dataset loading functions
DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(dataset_name: str):
    """Decorator to register dataset loading functions."""

    def decorator(func: Callable):
        DATASET_REGISTRY[dataset_name.lower()] = func
        return func

    return decorator


class PADLoader:
    """Data loader"""

    def __init__(
        self,
        config: Dict[str, Any],
        drop_last: bool = True,
        is_training: bool = True,
        kwargs: Dict[str, Any] = {},
    ):
        """
        Pytorch data loader

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        drop_last : bool, optional, default: True
            True in training mode, False in evaluation.
        is_training : bool, optional, default: True
            True if in training mode, False otherwise.
        kwargs : Dict[str, Any], optional, default: {}
            Dictionary for additional parameters if needed.
        """

        dataset_name = config["dataset"]
        batch_size = config["batch_size"]
        self.is_training = is_training
        self.config = config

        train_dataset, test_dataset, validation_dataset = self.__get_dataset(
            dataset_name
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers= os.cpu_count(),
            **kwargs,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **kwargs,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            **kwargs,
        )

        self.labels_vocab = train_dataset.labels_vocab
        self.num_samples = train_dataset.num_samples

    def __get_dataset(self, dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Returns training, validation, and test datasets

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.

        Returns
        -------
        Tuple[Dataset, Dataset, Dataset]
            A tuple containing train, test, and validation datasets.
        """

        train_dataset = TabularDataset(
            self.config,
            dataset_name=dataset_name,
            mode="train",
            is_training=self.is_training,
        )
        test_dataset = TabularDataset(
            self.config,
            dataset_name=dataset_name,
            mode="test",
            is_training=False,
        )
        validation_dataset = TabularDataset(
            self.config,
            dataset_name=dataset_name,
            mode="validation",
            is_training=self.is_training,
        )

        return train_dataset, test_dataset, validation_dataset


class TabularDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        dataset_name: str,
        mode: str = "train",
        is_training: bool = True,
        load_data: bool = True,
    ):
        """
        Dataset class for tabular data format.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        dataset_name : str
            Name of the dataset to load
        mode : str, optional, default: "train"
            Defines whether the data is for Train, Validation, or Test mode
        is_training : bool, optional, default: True
            Indicates whether the dataset is used for training or not.
        load_data: bool, optional, default: True
            If True, loads the data during initialisation.
        """

        self.config = config
        self.is_training = is_training
        self.tokenizer = ABtokenizer(f"{config['data_path']}/vocab.json")
        self.mode = mode
        self.device = config["device"]
        self.dataset_name = dataset_name
        self.target_weights = []

        with open(f"{config['data_path']}/aa_idx_pc_features.json") as f_in:

            aa_data = json.load(f_in)
            self.aa_features = np.array([np.array(v) for v in aa_data["data"]])

            scaler = StandardScaler()
            self.aa_features = scaler.fit_transform(self.aa_features)
            self.aa_features = torch.from_numpy(self.aa_features)

        if load_data:
            self.data, self.labels = self._load_data(
                dataset_name=self.dataset_name, mode=self.mode, config=self.config
            )
            self.vocab_size = len(self.labels_vocab)
            self.num_samples = int(self.config["p_vocab"] * self.vocab_size)

            self.data_dict = {}
            for x, y in zip(self.data, self.labels):
                self.data_dict.setdefault(y, []).append(x)

            if self.is_training and self.data != []:
                for target in self.labels_vocab:
                    seqs = self.data_dict[target]
                    self.target_weights.append(self.vocab_size / len(seqs))

                # total_weight = sum(self.target_weights)
                # self.target_weights = [w / total_weight for w in self.target_weights]

                # Make it uniform distribution (as a test)
                total_weight = len(self.target_weights)
                self.target_weights = [1 / total_weight for i in self.target_weights]

    def __len__(self) -> int:
        """
        Returns number of samples in the data.

        Returns
        -------
        int
            Number of samples in the data.
        """
        num_samples = len(self.data)

        if self.mode == "train" and self.is_training:
            num_samples = self.config["num_labels"]

        return num_samples

    def encode_seq(self, seq: str) -> str:
        """
        Encode a sequence with padding.

        Parameters
        ----------
        seq : str
            Sequence to be encoded.

        Returns
        -------
        str
            Encoded and padded sequence.
        """
        seq = "#" + seq + "$" + "*" * self.config["max_seq_length"]
        seq = seq[: self.config["max_seq_length"]]

        return seq

    def encode_chem(self, tokenized_seq):
        aa_features = torch.stack(
            [
                self.aa_features[i, :]
                for i in tokenized_seq.reshape(
                    -1,
                ).tolist()
            ]
        )
        return aa_features.unsqueeze(0)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    ]:
        """
        Returns a batch.

        Parameters
        ----------
        idx : int
            Index of the batch.

        Returns
        -------
        Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]]
            Returns a tuple of tensors and arrays for the batch.
        """
        if self.is_training is False:
            seq = self.encode_seq(self.data[idx])
            seq_tokenized = self.tokenizer([seq], pad=True)
            encoded_seq = self.encode_chem(seq_tokenized)
            return seq_tokenized, encoded_seq, self.labels[idx]

        # label_subsets = random.sample(self.labels_vocab, self.num_samples)
        label_idx = (
            np.random.choice(
                self.vocab_size,
                size=self.num_samples,
                replace=False,
                p=self.target_weights,
            )
        ).tolist()
        label_subsets = [self.labels_vocab[i] for i in label_idx]

        x1, x1aa, y1 = [], [], []
        x2, x2aa, y2 = [], [], []

        for label in label_subsets:
            seqs = self.data_dict[label]
            random.shuffle(seqs)

            seq0 = self.encode_seq(seqs[0])
            seq1 = self.encode_seq(seqs[1])

            seq0_tokenized = self.tokenizer([seq0], pad=True)
            seq1_tokenized = self.tokenizer([seq1], pad=True)

            seq0_aa = self.encode_chem(seq0_tokenized)
            seq1_aa = self.encode_chem(seq1_tokenized)

            x1.append(seq0_tokenized)
            x2.append(seq1_tokenized)

            x1aa.append(seq0_aa)
            x2aa.append(seq1_aa)

        y1.append([self.voc2idx[label]])
        y2.append([self.voc2idx[label]])

        x1 = torch.cat(x1)
        x2 = torch.cat(x2)
        x1aa = torch.cat(x1aa)
        x2aa = torch.cat(x2aa)

        y1 = np.concatenate(y1, dtype=int)
        y2 = np.concatenate(y2, dtype=int)

        return x1, x2, x1aa, x2aa, y1, y2

    # Register the existing dataset loading functions
    @register_dataset("pad")
    def _load_pad(
        self, config: dict
    ) -> Tuple[
        List[str], List[int], List[str], List[int], List[str], List[int], List[str]
    ]:
        """
        Loads pad dataset.

        Returns
        -------
        Tuple[List[str], List[int], List[str], List[int], List[str], List[int], List[str]]
            Training data, training labels, validation data, validation labels, test data, test labels, and label vocabulary.
        """

        train_df = pd.read_csv("./data/train_df.csv")
        test_df = pd.read_csv("./data/test_df.csv")

        self.labels_vocab = train_df["targets"].value_counts().keys().tolist()
        self.voc2idx = {v: i for i, v in enumerate(self.labels_vocab)}
        self.idx2voc = {i: v for i, v in enumerate(self.labels_vocab)}

        x_train, y_train = (
            train_df["s"].tolist(),
            train_df[self.config["y_label"]].tolist(),
        )
        x_test, y_test = test_df["s"].tolist(), test_df[self.config["y_label"]].tolist()

        logging.info(
            f"Number of unique labels in {self.config['y_label']} for training set: {len(list(set(y_train)))}"
        )
        logging.info(
            f"Number of unique labels in {self.config['y_label']} for test set: {len(list(set(y_test)))}"
        )

        # Stratify only when we use targets or species as our labels
        is_stratify = (
            True if self.config["y_label"] in ["targets", "species"] else False
        )

        if config["use_validation"]:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=1 - self.config["training_data_ratio"],
                random_state=57,
                shuffle=True,
                stratify=y_train if is_stratify else None,
            )
        else:
            x_val = []
            y_val = []

        logging.info(
            f"Number of training, validation and test samples: {len(x_train), len(x_val), len(x_test)}"
        )

        return x_train, y_train, x_val, y_val, x_test, y_test

    
    def _load_data(
        self, dataset_name: str, mode: str, config: dict
    ) -> Tuple[List[str], List[int]]:
        """
        Loads one of many available datasets, and returns features and labels.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load.
        mode : str
            One of the following modes: "train", "validation", or "test".
        config : dict
            Configuration dictionary containing settings for data loading.

        Returns
        -------
        Tuple[List[str], List[int]]
            Features and labels.
        """
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower not in DATASET_REGISTRY:
            logging.info(
                f"Given dataset name is not found. Check for typos, or register the dataset loader "
                f"function for '{dataset_name}' using the register_dataset decorator."
            )
            exit()

        # Call the registered dataset loading function for the specified dataset
        x_train, y_train, x_val, y_val, x_test, y_test = DATASET_REGISTRY[
            dataset_name_lower
        ](self, config)

        if mode == "train":
            return x_train, y_train
        elif mode == "validation":
            return x_val, y_val
        elif mode == "test":
            return x_test, y_test
        else:
            logging.info(
                "Something is wrong with the data mode. Use one of three options: train, validation, and test."
            )
            exit()
