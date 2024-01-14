"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: Tokenizer used to encode antibody sequences
"""

import json
from typing import List, Union

import torch


class ABtokenizer:
    """
    Tokenizer for antibodys. Both aa to token and token to aa.

    Attributes
    ----------
    vocab_to_token : dict
        A dictionary that maps amino acids to their corresponding tokens.
    vocab_to_aa : dict
        A dictionary that maps tokens back to their corresponding amino acids.
    pad_token : int
        The token value used for padding sequences.
    """

    def __init__(self, vocab_dir: str, aa_features=None):
        """
        Initialize the ABtokenizer with a given vocabulary directory.

        Parameters
        ----------
        vocab_dir : str
            The path to the vocabulary directory.
        """
        self.set_vocabs(vocab_dir)
        self.pad_token = self.vocab_to_token["*"]
        self.aa_features = aa_features

    def __call__(
        self,
        sequenceList: List[str],
        encode: bool = True,
        pad: bool = False,
        device: str = "cpu",
    ) -> Union[List, torch.Tensor]:
        """
        Encode or decode a list of sequences.

        Parameters
        ----------
        sequenceList : List[str]
            A list of antibody sequences.
        encode : bool, optional, default: True
            If True, encode the antibody sequences. If False, decode the sequences.
        pad : bool, optional, default: False
            If True, pad the encoded sequences to the same length.
        device : str, optional, default: "cpu"
            The device to use for tensor operations, either "cpu" or "gpu".

        Returns
        -------
        Union[List, torch.Tensor]
            The encoded or decoded sequences.
        """
        if encode:
            data = [self.encode(seq, device=device) for seq in sequenceList]
            if pad:
                return torch.nn.utils.rnn.pad_sequence(
                    data, batch_first=True, padding_value=self.pad_token
                )
            else:
                return data

        else:
            return [self.decode(token) for token in sequenceList]

    def set_vocabs(self, vocab_dir: str) -> None:
        """
        Set vocab_to_token and vocab_to_aa dictionaries using the given vocabulary directory.

        Parameters
        ----------
        vocab_dir : str
            The path to the vocabulary directory.
        """
        with open(vocab_dir, encoding="utf-8") as vocab_handle:
            self.vocab_to_token = json.load(vocab_handle)

        self.vocab_to_aa = {v: k for k, v in self.vocab_to_token.items()}

    def encode(self, sequence: str, device: str = "cpu") -> torch.Tensor:
        """
        Encode a antibody sequence into a tensor.

        Parameters
        ----------
        sequence : str
            The antibody sequence.
        device : str, optional, default: "cpu"
            The device to use for tensor operations, either "cpu" or "gpu".

        Returns
        -------
        torch.Tensor
            The encoded sequence as a tensor.
        """
        encoded = [self.vocab_to_token[resn] for resn in sequence]
        return torch.tensor(encoded, dtype=torch.long, device=device)

    def decode(self, seqtokens: Union[torch.Tensor, List[int]]) -> str:
        """
        Decode a sequence of tokens into a antibody sequence.

        Parameters
        ----------
        seqtokens : Union[torch.Tensor, List[int]]
            The sequence of tokens.

        Returns
        -------
        str
            The decoded antibody sequence.
        """
        if torch.is_tensor(seqtokens):
            seqtokens = seqtokens.cpu().numpy().tolist()
        decoded_seq = "".join([self.vocab_to_aa[token] for token in seqtokens])
        return decoded_seq

    def encode_with_aa_properties(
        self, sequence: str, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Encode a antibody sequence into a tensor.

        Parameters
        ----------
        sequence : str
            The antibody sequence.
        device : str, optional, default: "cpu"
            The device to use for tensor operations, either "cpu" or "gpu".

        Returns
        -------
        torch.Tensor
            The encoded sequence as a tensor.
        """
        encoded = [self.aa_features[self.vocab_to_token[resn]] for resn in sequence]
        return torch.cat(encoded, dtype=torch.long, device=device)
