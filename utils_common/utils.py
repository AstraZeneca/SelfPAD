"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Utility functions.
"""

import argparse
import logging
import os
import random as python_random
import sys

import numpy as np
import torch
import yaml
from numpy.random import seed
from sklearn import manifold
from texttable import Texttable
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)


def set_seed(options: Dict[str, Any]) -> None:
    """
    Set seed to ensure reproducibility

    Parameters
    ----------
    options : Dict[str, Any]
        Dictionary containing options and arguments
    """
    seed(options["seed"])
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])
    torch.manual_seed(options["seed"])


def create_dir(dir_path: str) -> None:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        The path to the directory to create.

    Returns
    -------
    None
        This function does not return anything.

    Example
    -------
    >>> create_dir("my_folder")
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_dirs(config: Dict[str, Any]) -> None:
    """
    Sets up the directory to load processed data and src, as well as saving results.

    Directory structure example:
        results > dataset > training    -> model
                          > evaluation   > plots
                                         > loss

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing options and arguments.
    """
    # Set main results directory using database name.
    # e.g., os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = config["root_path"]
    # results
    results_dir = make_dir(path, "results")
    # results > framework
    results_dir = make_dir(results_dir, config["experiment"])
    # Directory to save results at inference time when scoring for humanness
    inference_dir = make_dir(results_dir, config["evaluation_dataset"])
    # results > framework > training
    training_dir = make_dir(results_dir, "training")
    # results > framework > evaluation
    evaluation_dir = make_dir(results_dir, "evaluation")
    # results > framework > evaluation > clusters
    _ = make_dir(evaluation_dir, "clusters")
    # results > framework > training >  model_type >  model
    _ = make_dir(training_dir, "model")
    # results > framework > training >  model_type >  plots
    _ = make_dir(training_dir, "plots")
    # results > framework > training >  model_type > loss
    _ = make_dir(training_dir, "loss")

    # Save the results dir to be used later
    config["results_dir"] = results_dir
    config["training_dir"] = training_dir
    config["evaluation_dir"] = evaluation_dir
    config["logging_path"] = results_dir
    config["inference_dir"] = inference_dir

    # Print a message.
    logging.info("Directories are set.")


def make_dir(directory_path: str, new_folder_name: str) -> None:
    """
    Creates a new directory if it does not exist.

    Parameters
    ----------
    directory_path : str
        The directory path.
    new_folder_name : str
        The name of the new folder.

    Returns
    -------
    None

    Examples
    --------
    >>> make_dir('~/my_directory/', 'new_folder')
    '~/my_directory/new_folder/'

    """
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def get_runtime_and_model_config(args: Any) -> Dict[str, Any]:
    """
    Returns runtime and model/dataset specific config file.

    Parameters
    ----------
    args : Any
        Command-line arguments or configuration options.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing runtime and model/dataset specific configuration options.
    """

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    try:
        with open(f"{path}/config/{args.dataset}.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit(f"Error reading runtime config file: {e}")

    # Copy dataset names to config to use later
    config["dataset"] = args.dataset
    config["absolute_path"] = path

    # Update the config by adding the data specific config to runtime config
    return config


def print_config(args: argparse.Namespace) -> None:
    """
    Prints the configuration options.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace object containing the command-line arguments and their values.

    Returns
    -------
    None
    """
    # Yaml config is a dictionary while parser arguments is an object. Use vars() only on parser arguments.
    if type(args) is not dict:
        args = vars(args)
    # Sort keys
    keys = sorted(args.keys())
    # Initialize table
    table = Texttable()
    # Add rows to the table under two columns ("Parameter", "Value").
    table.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    # Print the table.
    logging.info(table.draw())
