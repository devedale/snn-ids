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
from snn_ids.evaluation.metrics import evaluate_model_comprehensive, create_benchmark_summary_csv
from snn_ids.federated.fl_simulation import FedAvgServer, FedClient, split_dataset_iid
from snn_ids.federated.he import HEContext

# Define hyperparameter sets from config
SMOKE_TEST_HYPERPARAMETERS = {
    'dense': {**TRAINING_CONFIG['hyperparameters'], **BENCHMARK_CONFIG['smoke_test']['hyperparameters']}
}
FULL_BENCHMARK_HYPERPARAMETERS = {
    'dense': TRAINING_CONFIG['hyperparameters'],
    'gru': TRAINING_CONFIG['hyperparameters'],
    'lstm': TRAINING_CONFIG['hyperparameters'],
}

def run_centralized(smoke_test=False, full_benchmark=False, sample_size=None, models_to_test=None):
    """
    Runs a centralized training and evaluation benchmark.
    """
    print("🚀 Starting centralized benchmark...")

    # Determine models and hyperparameters
    if smoke_test:
        print("🔥 Running in smoke test mode.")
        models_to_test = models_to_test or ['dense']
        hyperparameters = {'dense': SMOKE_TEST_HYPERPARAMETERS['dense']}
        sample_size = sample_size or 1000
    elif full_benchmark:
        print("🚀 Running in full benchmark mode.")
        models_to_test = models_to_test or ['dense', 'gru', 'lstm']
        hyperparameters = FULL_BENCHMARK_HYPERPARAMETERS
        sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
    else:
        print("Standard run...")
        models_to_test = models_to_test or ['dense']
        hyperparameters = {'dense': FULL_BENCHMARK_HYPERPARAMETERS['dense']}
        sample_size = sample_size or 10000

    # Load and preprocess data
    data_path = DATA_CONFIG['dataset_path']
    X, y, label_encoder = preprocess_pipeline(data_path=data_path, sample_size=sample_size)

    all_results = []

    # Create a main output directory for the benchmark
    main_output_dir = os.path.join("benchmark_results", f"centralized_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(main_output_dir, exist_ok=True)

    for model_type in models_to_test:
        print(f"\n--- Training and evaluating model: {model_type.upper()} ---")
        model_hyperparams = hyperparameters[model_type]

        model, (X_train, X_test, y_train, y_test) = train_model(
            X, y, model_type, model_hyperparams, test_size=0.2, random_state=42
        )

        model_config = {
            'model_type': model_type,
            'hyperparameters': model_hyperparams,
            'sample_size': sample_size
        }

        # Create a specific output directory for this model's artifacts
        model_output_dir = os.path.join(main_output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)

        report = evaluate_model_comprehensive(
            model, X_test, y_test, label_encoder.classes_,
            output_dir=os.path.join(model_output_dir, 'visualizations'),
            model_config=model_config
        )
        all_results.append(report)

    # After all models are evaluated, create a summary CSV
    if len(all_results) > 1:
        create_benchmark_summary_csv(all_results, main_output_dir)

    print("\n✅ Centralized benchmark finished.")


def run_progressive(sample_size: int = None, models_to_test: List[str] = None):
    """
    Runs a benchmark with progressively larger data samples.
    """
    print("🚀 Starting progressive benchmark...")

    models_to_test = models_to_test or ['dense']
    sample_fractions = [0.1, 0.2, 0.5, 1.0]

    # Load full dataset once
    data_path = DATA_CONFIG['dataset_path']
    full_sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
    X_full, y_full, label_encoder = preprocess_pipeline(data_path=data_path, sample_size=full_sample_size)

    all_results = []
    main_output_dir = os.path.join("benchmark_results", f"progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(main_output_dir, exist_ok=True)

    for model_type in models_to_test:
        for fraction in sample_fractions:
            current_sample_size = int(len(X_full) * fraction)
            print(f"\n--- Running model {model_type.upper()} with {fraction*100:.0f}% of data ({current_sample_size} samples) ---")

            # Sub-sample the data
            indices = np.random.choice(len(X_full), current_sample_size, replace=False)
            X, y = X_full[indices], y_full[indices]

            model_hyperparams = FULL_BENCHMARK_HYPERPARAMETERS[model_type]

            model, (X_train, X_test, y_train, y_test) = train_model(
                X, y, model_type, model_hyperparams, test_size=0.2, random_state=42
            )

            model_config = {
                'model_type': f"{model_type}_frac{fraction*100:.0f}",
                'hyperparameters': model_hyperparams,
                'sample_size': current_sample_size,
                'data_fraction': fraction
            }

            model_output_dir = os.path.join(main_output_dir, f"{model_type}_frac{fraction*100:.0f}")
            os.makedirs(model_output_dir, exist_ok=True)

            report = evaluate_model_comprehensive(
                model, X_test, y_test, label_encoder.classes_,
                output_dir=os.path.join(model_output_dir, 'visualizations'),
                model_config=model_config
            )
            all_results.append(report)

    # Create summary CSV for the entire progressive benchmark
    create_benchmark_summary_csv(all_results, main_output_dir)

    print("\n✅ Progressive benchmark finished.")

def run_best_config_tuning(sample_size: int = None):
    # ... (implementation from before)
    pass

def run_federated(use_he=False, use_dp=False, sweep=False, sample_size=None):
    # ... (implementation from before)
    pass

def run_mlp_deep_analysis(sample_size: int = None, epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001):
    """
    Runs a deep analysis on a 4-layer MLP, tracking per-class loss and metrics.
    """
    print("🚀 Avvio analisi MLP approfondita...")

    data_path = DATA_CONFIG['dataset_path']
    sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
    X, y, label_encoder = preprocess_pipeline(data_path=data_path, sample_size=sample_size)

    num_classes = len(np.unique(y))
    hyperparams = {'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}

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
