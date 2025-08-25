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

def _run_single_centralized_configuration(test_config: Dict, cached_data: Optional[Tuple] = None) -> Dict:
    if cached_data is None:
        try:
            X, y, label_encoder = preprocess_pipeline(data_path=test_config.get('data_path'), sample_size=test_config.get('sample_size'))
            cached_data = (X, y, label_encoder)
        except Exception as e:
            return {'status': 'error', 'stage': 'preprocessing', 'error': str(e)}
    else:
        X, y, label_encoder = cached_data

    try:
        model, training_log, model_path = train_model(X=X, y=y, model_type=test_config['model_type'], hyperparams=test_config['hyperparameters'])
        best_accuracy = max([run['accuracy'] for run in training_log]) if training_log else 0
    except Exception as e:
        return {'status': 'error', 'stage': 'training', 'error': str(e), 'config': test_config}

    eval_result = _run_centralized_evaluation(model, X, y, label_encoder, test_config)
    return {'status': 'success', 'config': test_config, 'best_accuracy': best_accuracy, 'model_path': model_path, 'training_log': training_log, 'evaluation': eval_result}

# --- Centralized Workflow ---

def run_centralized(smoke_test=False, full_benchmark=False, sample_size=None, models_to_test=None):
    config_override = {'sample_size': sample_size} if sample_size else {}
    test_config_base = {'sample_size': config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size']), 'data_path': DATA_CONFIG['dataset_path']}

    if smoke_test:
        test_config = {**test_config_base, 'model_type': 'dense', 'hyperparameters': {'epochs': 2, 'batch_size': 32, 'learning_rate': 0.001}}
        results = _run_single_centralized_configuration(test_config)
        _save_results(results, "benchmark_results", "smoke_test")
    elif full_benchmark:
        models = models_to_test or ['dense', 'gru', 'lstm']
        hyperparam_grid = {'epochs': [5, 10], 'batch_size': [32, 64]}
        keys, values = zip(*hyperparam_grid.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_results = []
        cached_data = preprocess_pipeline(data_path=test_config_base['data_path'], sample_size=test_config_base['sample_size'])
        for model_type, hyperparams in itertools.product(models, hyperparam_combinations):
            test_config = {**test_config_base, 'model_type': model_type, 'hyperparameters': {**TRAINING_CONFIG['hyperparameters'], **hyperparams}}
            all_results.append(_run_single_centralized_configuration(test_config, cached_data=cached_data))
        _save_results({'configuration_tests': all_results}, "benchmark_results", "full_benchmark")
    else:
        test_config = {**test_config_base, 'model_type': TRAINING_CONFIG['model_type'], 'hyperparameters': TRAINING_CONFIG['hyperparameters']}
        results = _run_single_centralized_configuration(test_config)
        _save_results(results, "benchmark_results", "single_run")

# --- Progressive & Best-Config Workflows ---
# (Simplified versions of the original scripts)

def run_progressive(sample_size: int = None, models_to_test: List[str] = None):
    print("Running progressive optimization workflow...")
    cached_data = preprocess_pipeline(sample_size=sample_size)
    results = []
    for lr in [0.001, 0.01]:
        config = {'model_type': 'dense', 'hyperparameters': {'learning_rate': lr, 'epochs': 5, 'batch_size': 64}, 'sample_size': sample_size}
        results.append(_run_single_centralized_configuration(config, cached_data=cached_data))
    print("Progressive workflow finished.")

def run_best_config_tuning(sample_size: int = None):
    print("Running best-config fine-tuning workflow...")
    cached_data = preprocess_pipeline(sample_size=sample_size)
    base_config = {'model_type': 'gru', 'epochs': 15, 'batch_size': 64, 'learning_rate': 0.01, 'activation': 'tanh', 'gru_units': 64}
    results = []
    for lr in [0.008, 0.01, 0.012]:
        hyperparams = {**base_config, 'learning_rate': lr}
        config = {'model_type': 'gru', 'hyperparameters': hyperparams, 'sample_size': sample_size}
        results.append(_run_single_centralized_configuration(config, cached_data=cached_data))
    print("Best-config workflow finished.")

# --- Federated Workflow ---

def _build_keras_builder(model_type: str, params: Dict[str, Any]):
    def _builder(input_shape, num_classes):
        return build_model(model_type, input_shape, num_classes, params)
    return _builder

def _run_single_federated_run(params, use_he, use_dp, cached_data):
    X, y, label_encoder = cached_data
    HOMOMORPHIC_CONFIG["enabled"] = use_he
    FEDERATED_CONFIG["differential_privacy"] = use_dp

    num_clients = FEDERATED_CONFIG.get('num_clients', 5)
    client_splits = split_dataset_iid(X, y, num_clients)

    input_shape = X.shape[1:]
    num_classes = len(np.unique(y))
    builder = _build_keras_builder(params.get('model_type', 'gru'), params)
    base_model = builder(input_shape, num_classes)
    base_weights = base_model.get_weights()

    he_ctx = HEContext(HOMOMORPHIC_CONFIG)
    server = FedAvgServer(base_weights, he_ctx)

    for rnd in range(FEDERATED_CONFIG.get('num_rounds', 5)):
        client_updates = [
            FedClient(cid, split).local_train(builder, server.global_weights, {"local_epochs": FEDERATED_CONFIG.get('local_epochs', 1), "batch_size": params['batch_size']}, num_classes)
            for cid, split in enumerate(client_splits)
        ]
        server.aggregate(client_updates)

    final_model = builder(input_shape, num_classes)
    final_model.set_weights(server.global_weights)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # The model is already compiled by the builder, so we don't need to re-compile.
    # Re-compiling here was forcing the wrong loss function for the binary case.
    _, acc = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"Federated run finished. HE: {use_he}, DP: {use_dp}, Accuracy: {acc:.4f}")
    return acc

def run_federated(use_he=False, use_dp=False, sweep=False, sample_size=None):
    cached_data = preprocess_pipeline(sample_size=sample_size)
    if sweep:
        print("Running federated sweep...")
        for he_enabled in [False, True]:
            params = {'epochs': 5, 'batch_size': 32, 'learning_rate': 0.001, 'activation': 'tanh', 'gru_units': 64}
            _run_single_federated_run(params, he_enabled, use_dp, cached_data)
    else:
        print("Running single federated experiment...")
        params = {'epochs': 5, 'batch_size': 32, 'learning_rate': 0.001, 'activation': 'tanh', 'gru_units': 64}
        _run_single_federated_run(params, use_he, use_dp, cached_data)
