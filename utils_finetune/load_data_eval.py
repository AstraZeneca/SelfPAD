"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: A library for data loaders.
"""

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
from torch.utils.data.sampler import WeightedRandomSampler

from utils_common.tokenizer import ABtokenizer

logging.basicConfig(level=logging.INFO)

# Registry to store dataset loading functions
DATASET_REGISTRY: Dict[str, Callable] = {}


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

        dataset_name = config["evaluation_dataset"]
        batch_size = config["batch_size"]
        self.config = config

        test_dataset = TabularDataset(self.config, dataset_name=dataset_name)

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config["num_workers"],
            **kwargs,
        )

class TabularDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        dataset_name: str,
    ):
        """
        Dataset class for tabular data format.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        dataset_name : str
            Name of the dataset to load
        """

        self.config = config
        self.tokenizer = ABtokenizer(f"{config['data_path']}/vocab.json")
        self.device = config["device"]
        self.dataset_name = dataset_name
        self.target_weights = []

        with open(f"{config['data_path']}/aa_idx_pc_features.json") as f_in:
            aa_data = json.load(f_in)
            self.aa_features = np.array([np.array(v) for v in aa_data["data"]])

            scaler = StandardScaler()
            self.aa_features = scaler.fit_transform(self.aa_features)
            self.aa_features = torch.from_numpy(self.aa_features)

        self.data, self.labels = self._load_data(config=self.config)

    def _filter_char(self, c):
        return True if c in self.tokenizer.vocab_to_token else False
            
    def __len__(self) -> int:
        """
        Returns number of samples in the data.

        Returns
        -------
        int
            Number of samples in the data.
        """
        return len(self.data)

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
        seq = self.encode_seq(self.data[idx])
        seq_tokenized = self.tokenizer([seq], pad=True)
        encoded_seq = self.encode_chem(seq_tokenized)
        raw_seq = self.data[idx]
        return seq_tokenized, encoded_seq, self.labels[idx], raw_seq

    def _load_data(
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


        # 5- Remove 553 sequences obtained from OASis, which will be used for humanness scoring
        dataset = pd.read_csv(f"./data/{self.config['evaluation_dataset']}.csv")
        
        if "VH" in list(dataset):
            dataset = dataset.dropna(subset=["VH", "VL"])
        if "Heavy" in list(dataset):
            dataset = dataset.dropna(subset=["Heavy", "Light"])        

        # Dataset should have either "VH"/"VL" columns or "Heavy"/"Light" for heavy and light chains respectively
        dataset_h = dataset["VH"].tolist() if "VH" in list(dataset) else dataset["Heavy"].tolist()
        dataset_l = dataset["VL"].tolist() if "VL" in list(dataset) else dataset["Light"].tolist()
        
        # Remove any white space or characters used for alignment or any character not in the vocabulary
        dataset_h = ["".join(filter(self._filter_char,sequence.replace(' ', '').replace('-', ''))) for sequence in dataset_h]    
        dataset_l = ["".join(filter(self._filter_char,sequence.replace(' ', '').replace('-', ''))) for sequence in dataset_l]       
        
        dataset_label = dataset["Label"].tolist()

        dataset_h_l, dataset_l_l, dataset_label_l = [], [], []
        for h, l, label in zip(dataset_h, dataset_l, dataset_label):
            # If either heavy or light is NaN, skip it
            if h!=h or l!=l:
                print(f"skipping it {h} and {l}....")
            else:
                dataset_h_l.append(h)
                dataset_l_l.append(l)
                dataset_label_l.append(label)


        # Prepare dataset such that we first have the list of all heavy chains, followed by all corresponding light chains
        x_test = dataset_h_l + dataset_l_l
        y_test = dataset_label_l + dataset_label_l
        
        # Cutoff is used to split the list into heavy and light chains during evaluation
        config["cutoff"] = len(dataset_h_l)

        return x_test, y_test
