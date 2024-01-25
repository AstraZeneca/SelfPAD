"""
Loads arguments and configuration.
Author: Talip Ucar
Email: ucabtuc@gmail.com
"""

import os
import argparse
import logging
from argparse import ArgumentParser
from typing import Optional

import torch

from .utils import get_runtime_and_model_config, print_config

logging.basicConfig(level=logging.INFO)


def get_arguments(humanness=False):
    # Initialize parser
    parser = ArgumentParser()
    # Config file needs to be named after dataset. If fine-tuning for hummaness, we need to provide humanness.yaml. Else, pad.yaml
    if humanness:
        parser.add_argument("-d", "--dataset", type=str, default="humanness")
    else:
        parser.add_argument("-d", "--dataset", type=str, default="pad")
        
    # Dataset to run an evaluation on
    parser.add_argument("-ev", "--evaluate", type=str, default="test")
    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument("-device", "--device_number", type=str, default="0")
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=str, default="pretraining")
    # Tune a pre-trained model
    parser.add_argument("-t", "--tune", type=bool, default=False)
    # Load model saved at specific epoch
    parser.add_argument("-m", "--model_at_epoch", type=int, default=None)
    # Return parser arguments
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> dict:
    """
    Load and return the configuration settings.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments passed to the program.

    Returns
    -------
    dict
        A dictionary containing the configuration settings.
    """
    # Load runtime config from config folder: ./config/
    config = get_runtime_and_model_config(args)

    # Define which device to use: GPU or CPU
    config["device"] = torch.device(
        "cuda:" + args.device_number if torch.cuda.is_available() else "cpu"
    )

    # Model at specific epoch
    config["experiment"] = args.experiment
    
    # Evaluation dataset
    config["evaluation_dataset"] = args.evaluate

    # Model at specific epoch
    config["model_at_epoch"] = args.model_at_epoch
    
    # Define number of workers
    config["num_workers"] = os.cpu_count()

    # Print the device type
    logging.info(f"Device being used is {config['device']}")

    # Return the configuration settings
    return config


def print_config_summary(config: dict, args: Optional[dict] = None) -> None:
    """Prints out summary of options and arguments used.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration options.
    args : dict, optional
        Dictionary containing arguments. Defaults to None.
    """
    # Summarize config on the screen as a sanity check
    logging.info(100 * "=")
    logging.info("Here is the configuration being used:\n")
    print_config(config)
    logging.info(100 * "=")

    if args is not None:
        logging.info("Arguments being used:\n")
        print_config(args)
        logging.info(100 * "=")
