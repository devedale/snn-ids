# -*- coding: utf-8 -*-
"""
Workflows for running different SNN-IDS experiments.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import itertools
import pandas as pd
from dataclasses import dataclass
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import *
from snn_ids.preprocessing.process import preprocess_pipeline
from snn_ids.training.train import train_model, build_model, train_model_with_per_class_loss
from snn_ids.evaluation.metrics import evaluate_model_comprehensive
from snn_ids.federated.fl_simulation import FedAvgServer, FedClient, split_dataset_iid
from snn_ids.federated.he import HEContext

# ... (helper functions from before)

def run_centralized(smoke_test=False, full_benchmark=False, sample_size=None, models_to_test=None):
    # ... (implementation from before)
    pass

def run_progressive(sample_size: int = None, models_to_test: List[str] = None):
    # ... (implementation from before)
    pass

def run_best_config_tuning(sample_size: int = None):
    # ... (implementation from before)
    pass

def run_federated(use_he=False, use_dp=False, sweep=False, sample_size=None):
    # ... (implementation from before)
    pass

def run_mlp_deep_analysis(sample_size: int = None):
    """
    Runs a deep analysis on a 4-layer MLP, tracking per-class loss and metrics.
    """
    print("🚀 Avvio analisi MLP approfondita...")

    data_path = DATA_CONFIG['dataset_path']
    sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
    X, y, label_encoder = preprocess_pipeline(data_path=data_path, sample_size=sample_size)

    num_classes = len(np.unique(y))
    hyperparams = {'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001}

    model, loss_df, metrics_df = train_model_with_per_class_loss(
        X, y, 'mlp_4_layer', hyperparams, num_classes, label_encoder
    )

    output_dir = os.path.join("benchmark_results", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mlp_analysis")
    os.makedirs(output_dir, exist_ok=True)

    loss_csv_path = os.path.join(output_dir, "per_class_epoch_loss.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"✅ Log delle loss per classe salvato in: {loss_csv_path}")

    metrics_csv_path = os.path.join(output_dir, "per_class_metrics.csv")
    metrics_df.to_csv(metrics_csv_path)
    print(f"✅ Metriche per classe salvate in: {metrics_csv_path}")

    # Plot 1: Per-class loss on subplots
    num_labels = len(label_encoder.classes_)
    cols = 4
    rows = (num_labels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle('Per-Class Loss vs. Epochs', fontsize=16)

    for i, class_name in enumerate(label_encoder.classes_):
        ax = axes[i]
        loss_col = f'loss_{class_name}'
        if loss_col in loss_df.columns:
            ax.plot(loss_df['epoch'], loss_df[loss_col])
            ax.set_title(class_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

    for i in range(num_labels, len(axes)): axes[i].set_visible(False)

    plot_path_loss = os.path.join(output_dir, "per_class_loss_subplots.png")
    plt.savefig(plot_path_loss)
    plt.close(fig)
    print(f"✅ Grafico loss per classe salvato in: {plot_path_loss}")

    # Plot 2 & 3: F1 Score and Recall per class
    metrics_to_plot = {'f1-score': 'F1 Score', 'recall': 'Recall'}
    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        plot_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
        sns.barplot(x=plot_df.index, y=plot_df[metric])
        plt.title(f'{title} per Classe')
        plt.xticks(rotation=90)
        plt.ylabel(title)
        plot_path_metric = os.path.join(output_dir, f"per_class_{metric}.png")
        plt.savefig(plot_path_metric, bbox_inches='tight')
        plt.close()
        print(f"✅ Grafico {title} per classe salvato in: {plot_path_metric}")

    print("✅ Analisi MLP completata.")
