"""
Author: Talip Ucar
Email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: Utility functions for evaluations.
"""

import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Configure the logging level
logging.basicConfig(level=logging.INFO)

def append_tensors_to_lists(
    list_of_lists: List[List[np.ndarray]],
    list_of_tensors: List[Union[torch.Tensor, Tuple, List, np.ndarray]],
) -> List[List[np.ndarray]]:
    """
    Appends tensors in a list to a list after converting tensors to numpy arrays.

    Parameters
    ----------
    list_of_lists : List[List[np.ndarray]]
        List of lists, each of which holds arrays.
    list_of_tensors : List[Union[torch.Tensor, Tuple, List, np.ndarray]]
        List of Pytorch tensors or other array-like structures.

    Returns
    -------
    list_of_lists : List[List[np.ndarray]]
        List of lists, each of which holds arrays.

    """

    # Go through each tensor and corresponding list
    for i in range(len(list_of_tensors)):
        # Convert tensor to numpy and append it to the corresponding list
        list_of_lists[i] += [
            list_of_tensors[i]
            if isinstance(list_of_tensors[i], (tuple, list, np.ndarray))
            else list_of_tensors[i].cpu().numpy()
        ]
    # Return the lists
    return list_of_lists


def concatenate_lists(
    list_of_lists: List[List[np.ndarray]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Concatenates each list with the main list to a numpy array.

    Parameters
    ----------
    list_of_lists : List[List[np.ndarray]]
        List of lists, each of which holds arrays.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
        A single numpy array if there's only one list in the input, or a list containing numpy arrays.

    """

    list_of_np_arrs = []
    # Pick a list of numpy arrays ([np_arr1, np_arr2, ...]), concatenate numpy arrs to a single one (np_arr_big),
    # and append it back to the list ([np_arr_big1, np_arr_big2, ...])
    for list_ in list_of_lists:
        list_of_np_arrs.append(np.concatenate(list_))
    # Return numpy arrays
    return list_of_np_arrs[0] if len(list_of_np_arrs) == 1 else list_of_np_arrs
