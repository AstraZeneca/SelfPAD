"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: A library for data loaders.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

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
            sampler=train_dataset.sampler,
            drop_last=drop_last,
            num_workers=config["num_workers"],
            **kwargs,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config["num_workers"],
            **kwargs,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=config["num_workers"],
            **kwargs,
        )

        self.labels_vocab = train_dataset.labels_vocab

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
            is_training=False,
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
            import json

            from sklearn.preprocessing import StandardScaler

            aa_data = json.load(f_in)
            self.aa_features = np.array([np.array(v) for v in aa_data["data"]])

            scaler = StandardScaler()
            self.aa_features = scaler.fit_transform(self.aa_features)
            self.aa_features = torch.from_numpy(self.aa_features)

        if load_data:
            self.data, self.labels, self.labels_s = self._load_data(
                dataset_name=self.dataset_name, mode=self.mode, config=self.config
            )

            if self.is_training and self.data != []:
                self.oversample_train(self.labels)

    def oversample_train(self, train_data_y):
        """Oversample the under represented classes."""

        class_sample_count = np.array(
            [len(np.where(train_data_y == t)[0]) for t in np.unique(train_data_y)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in train_data_y])

        samples_weight = torch.from_numpy(samples_weight)
        self.sampler = WeightedRandomSampler(
            samples_weight.type("torch.FloatTensor"), len(samples_weight)
        )

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
        from random import randrange

        if self.config["input_shift"]:

            if self.config["add_noise"]:
                is_sample = randrange(2)
                shift_num = randrange(10)

                if is_sample == 0:
                    seq = "#" + seq + "$" + "*" * self.config["max_seq_length"]
                else:
                    seq = shift_num * "*" + "#" + seq + "$" + "*" * self.config["max_seq_length"]

            else:
                seq = "#" + seq + "$" + "*" * self.config["max_seq_length"]

        else:
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
        seq = self.encode_seq(self.data[idx])
        seq_tokenized = self.tokenizer([seq], pad=True)
        encoded_seq = self.encode_chem(seq_tokenized)
        raw_seq = self.data[idx]
        return (
            seq_tokenized,
            encoded_seq,
            self.labels[idx], #self.labels_s[idx], raw_seq
        )
    # Register the existing dataset loading functions
    @register_dataset("humanness")
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


        self.labels_vocab = train_df["species"].value_counts().keys().tolist()
        self.voc2idx = {v: i for i, v in enumerate(self.labels_vocab)}
        self.idx2voc = {i: v for i, v in enumerate(self.labels_vocab)}

        # Get sequences (x) and labels (species)
        x_train, y_train_species = (
            train_df["s"].tolist(),
            train_df[self.config["y_label"]].tolist(),
        )
        x_test, y_test_species = (
            test_df["s"].tolist(),
            test_df[self.config["y_label"]].tolist(),
        )

        logging.info(
            f"Number of unique labels in {self.config['y_label']} for training set: {len(list(set(y_train_species)))}"
        )
        logging.info(
            f"Number of unique labels in {self.config['y_label']} for test set: {len(list(set(y_test_species)))}"
        )


        # Turn labels into binary
        y_train = [1 if l.lower() == "human" else 0 for l in y_train_species]
        y_test = [1 if l.lower() == "human" else 0 for l in y_test_species]


        if config["validate"]:
            x_train, x_val, y_train, y_val, y_train_s, y_val_s = train_test_split(
                x_train,
                y_train,
                y_train_species,
                test_size=1 - self.config["training_data_ratio"],
                random_state=57,
                shuffle=True,
                stratify=y_train,
            )
        else:
            x_val = []
            y_val = []

        logging.info(
            f"Number of training, validation and test samples: {len(x_train), len(x_val), len(x_test)}"
        )

        return (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            y_train_s,
            y_val_s,
            y_test_species,
        )

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
        (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            y_train_s,
            y_val_s,
            y_test_s,
        ) = DATASET_REGISTRY[dataset_name_lower](self, config)

        if mode == "train":
            return x_train, y_train, y_train_s
        elif mode == "validation":
            return x_val, y_val, y_val_s
        elif mode == "test":
            return x_test, y_test, y_test_s
        else:
            logging.info(
                "Something is wrong with the data mode. Use one of three options: train, validation, and test."
            )
            exit()