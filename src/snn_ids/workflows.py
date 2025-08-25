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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import *
from snn_ids.preprocessing.process import preprocess_pipeline
from snn_ids.training.train import train_model, build_model
from snn_ids.evaluation.metrics import evaluate_model_comprehensive
from snn_ids.federated.fl_simulation import FedAvgServer, FedClient, split_dataset_iid
from snn_ids.federated.he import HEContext

# --- Helper Functions ---

def _save_results(results: Dict[str, Any], output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    ts = int(time.time())
    results_file = os.path.join(output_dir, f"{name}_{ts}.json")
    with open(results_file, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"💾 Risultati JSON salvati in: {results_file}")

def _run_centralized_evaluation(model, X, y, label_encoder, test_config):
    from sklearn.model_selection import train_test_split
    if y.size == 0: return {'status': 'error', 'error': 'Empty dataset for evaluation'}
    try:
        class_counts = np.bincount(y.astype(int)) if y.size > 0 else np.array([0])
        min_class_size = np.min(class_counts[class_counts > 0]) if np.any(class_counts > 0) else 0
        stratify_opt = y if min_class_size >= 2 else None
        if stratify_opt is None: print("    ⚠️ Dataset piccolo o sbilanciato, split semplice senza stratificazione.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_opt)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_test) == 0: X_test, y_test = X, y
    class_names = label_encoder.classes_.tolist() if hasattr(label_encoder, 'classes_') else []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = test_config.get('model_type', 'unknown')
    descriptive_name = f"{timestamp}_{model_type}_evaluation"
    eval_dir = os.path.join("benchmark_results", descriptive_name)
    evaluation_report = evaluate_model_comprehensive(model=model, X_test=X_test, y_test=y_test, class_names=class_names, output_dir=eval_dir, model_config=test_config)
    return {'status': 'success', 'evaluation_dir': eval_dir, 'report': evaluation_report}

def _run_single_centralized_configuration(test_config: Dict, cached_data: Optional[Tuple] = None, callbacks=None) -> Dict:
    if cached_data is None:
        try:
            X, y, label_encoder = preprocess_pipeline(data_path=test_config.get('data_path'), sample_size=test_config.get('sample_size'))
            cached_data = (X, y, label_encoder)
        except Exception as e:
            return {'status': 'error', 'stage': 'preprocessing', 'error': str(e)}
    else:
        X, y, label_encoder = cached_data

    try:
        model, training_log, model_path = train_model(
            X=X, y=y, model_type=test_config['model_type'], hyperparams=test_config['hyperparameters'],
            callbacks=callbacks
        )
        best_accuracy = max([run['accuracy'] for run in training_log]) if training_log else 0
    except Exception as e:
        return {'status': 'error', 'stage': 'training', 'error': str(e), 'config': test_config}

    eval_result = _run_centralized_evaluation(model, X, y, label_encoder, test_config)
    return {'status': 'success', 'config': test_config, 'best_accuracy': best_accuracy, 'model_path': model_path, 'training_log': training_log, 'evaluation': eval_result}

# --- Main Workflows ---

def run_centralized(smoke_test=False, full_benchmark=False, sample_size=None, models_to_test=None):
    pass

def run_progressive(sample_size: int = None, models_to_test: List[str] = None):
    pass

def run_best_config_tuning(sample_size: int = None):
    pass

def run_federated(use_he=False, use_dp=False, sweep=False, sample_size=None):
    pass

# --- MLP Deep Analysis Workflow ---

def run_mlp_deep_analysis(sample_size: int = None):
    print("🚀 Avvio analisi MLP approfondita...")
    config_override = {'sample_size': sample_size} if sample_size else {}
    test_config = {
        'model_type': 'mlp_4_layer',
        'hyperparameters': {'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
        'sample_size': config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size']),
        'data_path': DATA_CONFIG['dataset_path']
    }

    # Keras History callback is used by default in train_model, no need for a custom one
    result = _run_single_centralized_configuration(test_config)

    if result['status'] == 'success':
        output_dir = os.path.join("benchmark_results", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mlp_analysis")
        os.makedirs(output_dir, exist_ok=True)

        # The training log from train_model contains the history
        log_df = pd.DataFrame(result['training_log'][0]['history'])
        csv_path = os.path.join(output_dir, "epoch_loss_log.csv")
        log_df.to_csv(csv_path, index_label="epoch")
        print(f"✅ Log delle epoche salvato in: {csv_path}")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(log_df['loss'], label='Training Loss')
            if 'val_loss' in log_df.columns:
                plt.plot(log_df['val_loss'], label='Validation Loss')
            plt.title('Loss per Epoca')
            plt.xlabel('Epoca')
            plt.ylabel('Loss')
            plt.legend()
            plot_path = os.path.join(output_dir, "epoch_loss_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"✅ Grafico della loss salvato in: {plot_path}")
        except ImportError:
            print("⚠️ Matplotlib non trovato. Salto la creazione del grafico.")

    print("✅ Analisi MLP completata.")
