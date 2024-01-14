"""
Author: Talip Ucar
email: ucabtuc@gmail.com or talip.ucar@astrazeneca.com

Description: A sample script to score antibody sequences for humanness.
"""
import copy
import logging
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from src.selfpad_humanness import PADFintune

from utils_common.utils import set_dirs
from utils_common.arguments import get_arguments, get_config
from utils_finetune.load_data_eval import PADLoader
from torcheval.metrics.functional import binary_auprc
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

# Configure the logging level
logging.basicConfig(level=logging.INFO)


def eval(data_loader, config: Dict[str, Any]) -> None:
    """
    Fine-tunes a Large Protein Model (LPM) with a new classification head and evaluates it on the test set.

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
    model.load_models()
    transformer = model.transformer
    transformer.to(config["device"])
    transformer.eval()
    transformer.config["add_noise"] = False

    val_loss = []
    preds_l = []
    preds_raw_l = []
    labels_l = []
    embs_l = []
    species_l = []
    raw_seqs_l = []
    embs_ext_l = []
    x1_l = []

    test_loader = data_loader.test_loader
    # Attach progress bar to data_loader
    test_tqdm = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

    with torch.no_grad():
        # pass data through model
        for batch in test_tqdm:

            x1, x1aa, y1, raw_seqs = batch[1]
            labels = y1.reshape(
                -1,
            )
            # species_l.extend(y1_s)

            preds, h, h_ext = transformer(x1, x1aa, agg_dim=model.agg_dim)

            # Generate labels and predictions
            preds_raw = preds.cpu()
            preds_idx = preds[:, 1].cpu()
            
            # Save the results
            preds_l.append(preds_idx)
            preds_raw_l.append(preds_raw)
            labels_l.append(labels)
            embs_l.append(h)
            embs_ext_l.append(h_ext)
            x1_l.append(x1)
            raw_seqs_l.extend(raw_seqs)

            loss = model.ce_loss(preds, labels.to(model.device))
            batch_loss = loss.item()
            val_loss.append(batch_loss)
            del loss

        preds_l = torch.cat(preds_l)
        preds_raw_l = torch.cat(preds_raw_l)
        labels_l = torch.cat(labels_l)
        embeddings = torch.cat(embs_l)
        embs_ext = torch.cat(embs_ext_l)
        x1_cat = torch.cat(x1_l)

        preds_l_single = copy.deepcopy(preds_l)
        preds_raw_l_single = copy.deepcopy(preds_l)
        labels_l_single = copy.deepcopy(labels_l)

        preds_l_single[preds_l_single >= config["threshold"]] = 1
        preds_l_single[preds_l_single  < config["threshold"]] = 0


        # Compute metrics
        f1 = 100 * model.f1_score(preds_l_single, labels_l_single)
        acc = 100 * model.acc_score(preds_l_single, labels_l_single)
        recall = 100 * model.recall_score(preds_l_single, labels_l_single)
        auc = 100 * model.roc_auc(preds_raw_l_single, labels_l_single)
        prec = 100 * model.precision_score(preds_l_single, labels_l_single)
        pr_auc = 100 * binary_auprc(preds_raw_l_single, labels_l_single)

        summary_dict_single = {
            "F1": f1.numpy(),
            "Recall": recall.numpy(),
            "Precision": prec.numpy(),
            "ROC AUC": auc.numpy(),
            "Accuracy": acc.numpy(),
            "PR AUC": pr_auc.numpy(),
        }
        summary_df_single = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in summary_dict_single.items()])
        )
        
        print("=================================")
        print("Single chain performance")
        print(summary_df_single)

        # Results for paired sequence
        cutoff = config["cutoff"]
        preds_l1 = preds_l[:cutoff]
        preds_l2 = preds_l[cutoff:]

        thresh = config["threshold"]
        preds_l1[preds_l1 >= thresh] = 1
        preds_l1[preds_l1 < thresh] = 0

        preds_l2[preds_l2 >= thresh] = 1
        preds_l2[preds_l2 < thresh] = 0

        preds_l = (preds_l1 + preds_l2) / 2
        preds_raw_l = (preds_raw_l[:cutoff] + preds_raw_l[cutoff:]) / 2
        preds_raw_l = preds_raw_l[:, 1]
        preds_l[preds_l >= 0.5] = 1
        labels_l = labels_l[:cutoff]

        # Compute metrics
        f1 = 100 * model.f1_score(preds_l, labels_l)
        acc = 100 * model.acc_score(preds_l, labels_l)
        recall = 100 * model.recall_score(preds_l, labels_l)
        auc = 100 * model.roc_auc(preds_raw_l, labels_l)
        prec = 100 * model.precision_score(preds_l, labels_l)
        pr_auc = 100 * binary_auprc(preds_raw_l, labels_l)

        labels_l = (
            labels_l.reshape(
                -1,
            )
            .cpu()
            .numpy()
            .tolist()
        )
        
        # Save the scores as csv file
        raw_seqs_l_h = raw_seqs_l[:cutoff]
        raw_seqs_l_l = raw_seqs_l[:cutoff]
        scores_df = pd.DataFrame({
            "Heavy": raw_seqs_l_h,
            "Light": raw_seqs_l_l,
            "Scores": preds_raw_l,
            "Prediction": preds_l,
            "TrueLabel": labels_l,
        })
        
        filename = f"{config['dataset']}_scores.csv" 
        file_path = os.path.join(config["inference_dir"], filename)
        scores_df.to_csv(file_path)
        print("=================================")
        print(f"Raw scores are saved as csv file at: {file_path}")

    
        return (
            f1.numpy(),
            recall.numpy(),
            prec.numpy(),
            auc.numpy(),
            acc.numpy(),
            pr_auc.numpy(),
        )



def main():

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
    config["add_noise"] = False
    config["experiment"] = "humanness"
    
    # Ser directories (or create if they don't exist)
    set_dirs(config)

    # Get data loader.
    data_loader = PADLoader(config, drop_last=True, is_training=True)

    f1, recall, prec, auc, acc, pr_auc = eval(data_loader, config=config)
       
    summary_dict = {
        "F1": [f1],
        "Recall": [recall],
        "Precision": [prec],
        "Accuracy": [acc],
        "ROC AUC": [auc],
        "PR AUC": [pr_auc],
    }
    
    # Save the results
    summary_df_paired = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in summary_dict.items()])
    )

    filename = f"{config['dataset']}_summary.csv" 
    file_path = os.path.join(config["inference_dir"], filename)
    summary_df_paired.to_csv(file_path)
    print("Paired sequence performance")
    print(summary_df_paired)
    print("=================================")
    print(f"Summary of evaluation metrics are saved as csv file at: {file_path}")
    

if __name__ == "__main__":
    main()
