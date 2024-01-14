"""
Author: Talip Ucar
email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: Library of loss functions.
"""

import numpy as np
import torch
from torch import Tensor


class JointLoss(torch.nn.Module):
    def __init__(self, options: dict):
        """
        Initialize JointLoss.
        The reference: https://arxiv.org/pdf/2002.05709.pdf

        Parameters
        ----------
        options : dict
            Configuration dictionary.
        """
        super(JointLoss, self).__init__()
        # Assign options to self
        self.options = options
        # Batch size == number of nodes in the graph
        self.batch_size = options["batch_size"]
        # Temperature to use scale logits
        self.temperature = options["tau"]
        # Device to use: GPU or CPU
        self.device = options["device"]
        # initialize softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = (
            self._cosine_simililarity
            if options["cosine_similarity"]
            else self._dot_simililarity
        )
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_mask_for_neg_samples(self):
        # Initialize an identity matrix
        diagonal = np.eye(2 * self.batch_size)
        # Initialize a diagonal matrix - 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Initialize a second diagonal matrix - 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = torch.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1.
        mask = (1 - mask).type(torch.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x:
        x = x.unsqueeze(1)
        # Reshape y:
        y = y.T.unsqueeze(0)
        # Compute similarity
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    @staticmethod
    def _cosine_simililarity(self, x, y):
        similarity = torch.nn.CosineSimilarity(dim=-1)
        # Reshape x
        x = x.unsqueeze(1)
        # Reshape y
        y = y.unsqueeze(0)
        # Compute similarity
        return similarity(x, y)

    def XNegloss(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples
        l_pos = torch.diag(similarity, self.batch_size)
        r_pos = torch.diag(similarity, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for negatives
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = torch.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.temperature
        # Get labels
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation: Tensor) -> Tensor:
        """
        Compute loss for the given representation.

        Parameters
        ----------
        representation : torch.FloatTensor

        Returns
        -------
        loss : torch.FloatTensor
            The computed joint loss.
        """
        # Overwrite batch_size as half of the first dimension of representation
        self.batch_size = int(representation.size(0) // 2)

        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)

        # Compute the loss i.e. XNegLoss
        loss = self.XNegloss(representation)

        # Return
        return loss
