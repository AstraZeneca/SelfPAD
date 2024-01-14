"""
Author: Talip Ucar
email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: A sample script to fine-tune the model on Patent DB model for humanness.
"""
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from src.selfpad_humanness import PADFintune
from utils_common.utils import set_dirs
from utils_common.arguments import get_arguments, get_config, print_config_summary
from utils_finetune.load_data_humanness import PADLoader


# Configure the logging level
logging.basicConfig(level=logging.INFO)


def train(data_loader, config: Dict[str, Any]) -> None:
    """
    Fine-tunes SelfPAD with a new classification head and evaluates it on the test set.

    Parameters
    ----------
    data_loader : IterableDataset
        PyTorch data loader.
    config : Dict[str, Any]
        Dictionary containing configuration options and arguments.
    """
    # Set the random seed to make the training deterministic
    pl.seed_everything(seed=config["seed"])
    # Initialize the model
    model = PADFintune(config)
    # Fit the model to the data
    f1, recall, prec, auc, acc = model.fit(data_loader)
    # Save the model for future use
    model.save_weights()

    return f1, recall, prec, auc, acc


def main():
    f1_l, recall_l, prec_l, auc_l, acc_l = [], [], [], [], []

    # Number of repeats to compute stdev --- variation due to model initialisation
    num_repeat = 1
    
    for i in range(num_repeat):

        # Get parser / command line arguments for pre-trained model
        args = get_arguments(humanness=False)
        # Get configuration file
        config = get_config(args)
        
        # Get parser / command line arguments for fine-tuning the model
        args = get_arguments(humanness=True)
        # Get configuration file
        config_finetune = get_config(args)
        
        # Update the config with options for fine-tuning
        config.update(**config_finetune)
        
        # Change the seed and update the experiment name. Keep experiment name if you want to keep overwriting the model. Else append it with str(i)
        config["seed"] = config["seed"] + i
        config["experiment"] = "humanness" # + str(i)
        
        print(print_config_summary(config))
        # Ser directories (or create if they don't exist)
        set_dirs(config)

        # Get data loader
        data_loader = PADLoader(config, drop_last=True, is_training=True)

        f1, recall, prec, auc, acc = train(data_loader, config=config)
        f1_l.append(f1)
        recall_l.append(recall)
        prec_l.append(prec)
        auc_l.append(auc)
        acc_l.append(acc)

    print(f"F1: {np.mean(f1_l)} +/- {np.std(f1_l)}")
    print(f"Recall: {np.mean(recall_l)} +/- {np.std(recall_l)}")
    print(f"Precision: {np.mean(prec_l)} +/- {np.std(prec_l)}")
    print(f"ROC AUC: {np.mean(auc_l)} +/- {np.std(auc_l)}")
    print(f"ACC.: {np.mean(acc_l)} +/- {np.std(acc_l)}")

    summary_dict = {
        "F1": f1_l,
        "Recall": recall_l,
        "Precision": prec_l,
        "ROC AUC": auc_l,
        "Accuracy": acc_l,
    }
    
    # Save the results
    summary_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in summary_dict.items()])
    )

    filename = f"{config['dataset']}_summary.csv" 
    file_path = os.path.join(config["results_dir"], filename)
    summary_df.to_csv(file_path)

    print(summary_df)


if __name__ == "__main__":
    main()