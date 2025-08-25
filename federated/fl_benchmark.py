# -*- coding: utf-8 -*-
"""
Benchmark federato (best-config) con opzione HE per feature sensibili.

Offre un entry-point simile a benchmark-best-config ma in setting FL.
Genera anche un semplice report privacy per HE/DP.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np

from config import PREPROCESSING_CONFIG, DATA_CONFIG, TRAINING_CONFIG, HOMOMORPHIC_CONFIG, FEDERATED_CONFIG
from preprocessing.process import preprocess_pipeline
from training.train import build_model
from federated.fl_simulation import FedAvgServer, FedClient, split_dataset_iid
from evaluation.metrics import evaluate_model_comprehensive
from federated.he import HEContext


def _build_keras_builder(model_type: str, params: Dict[str, Any]):
    import tensorflow as tf

    def _builder(input_shape, num_classes):
        return build_model(model_type, input_shape, num_classes, params)

    return _builder


def _create_privacy_report(he_cfg: Dict[str, Any], dp_enabled: bool) -> Dict[str, Any]:
    # Stima qualitativa molto semplice; non sostituisce un'analisi formale
    he_active = he_cfg.get("enabled", False)
    features = he_cfg.get("features_to_encrypt", []) if he_active else []

    return {
        "homomorphic_encryption": {
            "enabled": he_active,
            "scheme": he_cfg.get("scheme"),
            "sensitive_features_protected": len(features),
            "protected_feature_names": features,
        },
        "differential_privacy": {
            "enabled": bool(dp_enabled),
            "noise_scale": FEDERATED_CONFIG.get("dp_noise_scale", 0.0) if dp_enabled else 0.0,
        },
        "confidentiality_score_simple": int(he_active) * 0.6 + (0.4 if dp_enabled else 0.0)
    }


def run_fl_best_config(sample_size: int = None, data_path: str = None) -> Dict[str, Any]:
    # Parametri base best-config ispirati al benchmark best
    params = {
        'epochs': 5,  # per modello globale finale
        'batch_size': 32,
        'learning_rate': 0.001,
        'activation': 'tanh',
        'gru_units': 64
    }

    model_type = 'gru'
    sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
    data_path = data_path or DATA_CONFIG['dataset_path']

    start = time.time()

    # Preprocess una volta
    X, y, _ = preprocess_pipeline(data_path=data_path, sample_size=sample_size)

    # Split per client
    num_clients = FEDERATED_CONFIG.get('num_clients', 5)
    client_splits = split_dataset_iid(X, y, num_clients)

    # Costruisci modello base per inizializzazione pesi
    input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
    num_classes = max(len(np.unique(y)), int(np.max(y)) + 1)

    builder = _build_keras_builder(model_type, params)
    base_model = builder(input_shape, num_classes)
    base_weights = base_model.get_weights()
    print(f"🔁 FL Setup | HE={'ON' if HOMOMORPHIC_CONFIG.get('enabled', False) else 'OFF'} | lr={params['learning_rate']} | units={params['gru_units']} | epochs_global={params['epochs']}")

    # Server + HE
    he_ctx = HEContext(HOMOMORPHIC_CONFIG)
    server = FedAvgServer(base_weights, he_ctx)

    # Round federati
    num_rounds = FEDERATED_CONFIG.get('num_rounds', 5)
    local_epochs = FEDERATED_CONFIG.get('local_epochs', 1)
    batch_size = HOMOMORPHIC_CONFIG.get('batch_size_encrypted', 32) if he_ctx.is_active() else TRAINING_CONFIG['hyperparameters'].get('batch_size', [64])[0]

    history = []
    for rnd in range(num_rounds):
        print(f"  🔄 Round {rnd+1}/{num_rounds} | local_epochs={local_epochs} | eff_batch={batch_size} | clients={num_clients}")
        client_updates = []
        for cid, split in enumerate(client_splits):
            client = FedClient(cid, split)
            print(f"    🧩 Client {cid+1}/{num_clients} local_train")
            updated = client.local_train(
                lambda inp, nc: builder(inp, nc),
                server.global_weights,
                {"local_epochs": local_epochs, "batch_size": batch_size},
                num_classes
            )
            client_updates.append(updated)

        # Aggregazione (FedAvg)
        new_weights = server.aggregate(client_updates)
        history.append({"round": rnd + 1})

    # Modello finale
    final_model = builder(input_shape, num_classes)
    final_model.set_weights(server.global_weights)

    # Valutazione semplice (ri-uso split: 20% test) con fallback se classi rare (<2 campioni)
    from sklearn.model_selection import train_test_split
    class_counts = np.bincount(y.astype(int)) if y.size > 0 else np.array([0])
    min_class_size = np.min(class_counts[class_counts > 0]) if np.any(class_counts > 0) else 0
    stratify_opt = y if min_class_size >= 2 else None
    if stratify_opt is None:
        print("    ⚠️ Dataset con classi rare (min<2), split senza stratificazione.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_opt
    )
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    final_model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
    _, acc = final_model.evaluate(X_test, y_test, verbose=0)

    # Visualizzazioni e report valutazione completi (riutilizzo pipeline esistente)
    # Stessa directory usata dal benchmark generale per consistenza
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join("benchmark_results", f"{ts}_GRU_FL_HE", "visualizations")
    model_config = {
        'model_type': model_type,
        'hyperparameters': {
            'epochs': [params['epochs']],
            'batch_size': [batch_size],
            'learning_rate': [params['learning_rate']],
            'activation': [params['activation']],
            'gru_units': [params['gru_units']]
        }
    }
    class_names = []  # opzionale; non indispensabile per salvataggio visual
    eval_report = evaluate_model_comprehensive(final_model, X_test, y_test, class_names, eval_dir, model_config)

    # Top-10 like (qui solo il finale; struttura compatibile con altri report)
    top_models = [{
        'rank': 1,
        'phase': 'Federated Final',
        'test_id': 'fl_final',
        'model_type': model_type,
        'accuracy': float(acc),
        'training_time': 0,
        'total_time': time.time() - start,
        'epochs': params['epochs'],
        'batch_size': batch_size,
        'learning_rate': params['learning_rate'],
        'activation': params['activation'],
        'gru_units': params['gru_units'],
        'dropout': None
    }]

    report = {
        "status": "success",
        "mode": "federated",
        "best_accuracy": float(acc),
        "training_time": time.time() - start,
        "config": {
            "model_type": model_type,
            "hyperparameters": {
                "epochs": [params['epochs']],
                "batch_size": [batch_size],
                "learning_rate": [params['learning_rate']],
                "activation": [params['activation']],
                "gru_units": [params['gru_units']]
            }
        },
        "evaluation_visuals_dir": eval_dir,
        "evaluation_report": eval_report,
        "top_10_models": top_models,
        "privacy_report": _create_privacy_report(HOMOMORPHIC_CONFIG, FEDERATED_CONFIG.get("differential_privacy", False))
    }

    return report


def _run_fl_once(X: np.ndarray, y: np.ndarray, params: Dict[str, Any], he_enabled: bool, model_type: str = 'gru') -> Dict[str, Any]:
    """Esegue un run FL singolo riusando X,y già preprocessati."""
    # Salva stato HE e setta
    prev_he = HOMOMORPHIC_CONFIG.get('enabled', False)
    HOMOMORPHIC_CONFIG['enabled'] = bool(he_enabled)

    start = time.time()
    input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
    num_classes = max(len(np.unique(y)), int(np.max(y)) + 1)

    builder = _build_keras_builder(model_type, params)
    base_model = builder(input_shape, num_classes)
    base_weights = base_model.get_weights()

    # Server/Client
    num_clients = FEDERATED_CONFIG.get('num_clients', 5)
    client_splits = split_dataset_iid(X, y, num_clients)
    he_ctx = HEContext(HOMOMORPHIC_CONFIG)
    server = FedAvgServer(base_weights, he_ctx)

    num_rounds = FEDERATED_CONFIG.get('num_rounds', 5)
    local_epochs = FEDERATED_CONFIG.get('local_epochs', 1)
    batch_size = HOMOMORPHIC_CONFIG.get('batch_size_encrypted', 32) if he_ctx.is_active() else TRAINING_CONFIG['hyperparameters'].get('batch_size', [64])[0]

    for _ in range(num_rounds):
        client_updates = []
        for cid, split in enumerate(client_splits):
            client = FedClient(cid, split)
            updated = client.local_train(
                lambda inp, nc: builder(inp, nc),
                server.global_weights,
                {"local_epochs": local_epochs, "batch_size": batch_size},
                num_classes
            )
            client_updates.append(updated)
        server.aggregate(client_updates)

    # Valutazione semplice
    from sklearn.model_selection import train_test_split
    class_counts = np.bincount(y.astype(int)) if y.size > 0 else np.array([0])
    min_class_size = np.min(class_counts[class_counts > 0]) if np.any(class_counts > 0) else 0
    stratify_opt = y if min_class_size >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_opt
    )
    final_model = builder(input_shape, num_classes)
    final_model.set_weights(server.global_weights)
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    final_model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
    _, acc = final_model.evaluate(X_test, y_test, verbose=0)

    # Visual dir naming include HE/noHE + hp
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_he = "HE" if he_enabled else "NOHE"
    hp_tag = f"lr{params['learning_rate']}_ep{params['epochs']}_bs{batch_size}_u{params.get('gru_units', params.get('lstm_units', ''))}"
    eval_dir = os.path.join("benchmark_results", f"{ts}_GRU_FL_{tag_he}_{hp_tag}", "visualizations")
    print(f"🧾 Eval dir: {os.path.dirname(eval_dir)}")
    model_config = {
        'model_type': model_type,
        'hyperparameters': {
            'epochs': [params['epochs']],
            'batch_size': [batch_size],
            'learning_rate': [params['learning_rate']],
            'activation': [params['activation']],
            'gru_units': [params['gru_units']]
        }
    }
    eval_report = evaluate_model_comprehensive(final_model, X_test, y_test, [], eval_dir, model_config)

    # Ripristina HE
    HOMOMORPHIC_CONFIG['enabled'] = prev_he

    return {
        'status': 'success',
        'he_enabled': he_enabled,
        'accuracy': float(acc),
        'total_time': time.time() - start,
        'eval_dir': eval_dir,
        'params': params,
        'batch_size_effective': batch_size,
        'model_type': model_type,
        'evaluation_report': eval_report
    }


def run_fl_grid_search(sample_size: int = None, data_path: str = None, he_enabled: bool = False,
                       lr_list: List[float] = None, units_list: List[int] = None,
                       epochs_list: List[int] = None, batch_list: List[int] = None) -> Dict[str, Any]:
    """Esegue una mini grid search in modalità FL, ritorna top-10 e best config."""
    sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
    data_path = data_path or DATA_CONFIG['dataset_path']

    # Default grid snella
    lr_list = lr_list or [0.008, 0.01, 0.012]
    units_list = units_list or [56, 64, 72]
    epochs_list = epochs_list or [3]
    batch_list = batch_list or [32]

    # Preprocess una volta
    X, y, _ = preprocess_pipeline(data_path=data_path, sample_size=sample_size)

    results = []
    for lr in lr_list:
        for units in units_list:
            for ep in epochs_list:
                for bs in batch_list:
                    params = {
                        'epochs': ep,
                        'batch_size': bs,  # usato solo in addestramento finale; Fed usa batch calcolato
                        'learning_rate': lr,
                        'activation': 'tanh',
                        'gru_units': units
                    }
                    res = _run_fl_once(X, y, params, he_enabled, 'gru')
                    results.append(res)

    # Ordina e top-10
    top10 = sorted(results, key=lambda r: r.get('accuracy', 0), reverse=True)[:10]
    best = top10[0] if top10 else None

    summary = {
        'he_enabled': he_enabled,
        'total_runs': len(results),
        'top_10': [
            {
                'rank': i + 1,
                'accuracy': r['accuracy'],
                'params': r['params'],
                'batch_size_effective': r['batch_size_effective'],
                'eval_dir': r['eval_dir']
            } for i, r in enumerate(top10)
        ],
        'best': {
            'accuracy': best['accuracy'],
            'params': best['params'],
            'batch_size_effective': best['batch_size_effective'],
            'eval_dir': best['eval_dir']
        } if best else None
    }

    return summary


def run_fl_comparison_sweep(sample_size: int = None, data_path: str = None) -> Dict[str, Any]:
    """Esegue due sweep: NO-HE e HE, e produce un confronto sintetico."""
    nohe = run_fl_grid_search(sample_size=sample_size, data_path=data_path, he_enabled=False)
    he = run_fl_grid_search(sample_size=sample_size, data_path=data_path, he_enabled=True)

    comparison = {
        'status': 'success',
        'no_he': nohe,
        'he': he,
        'delta_accuracy_best': (he['best']['accuracy'] - nohe['best']['accuracy']) if he.get('best') and nohe.get('best') else None
    }

    return comparison
