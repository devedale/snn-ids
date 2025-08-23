# -*- coding: utf-8 -*-
"""
Training Unificato SNN-IDS
Sistema semplice per training modelli GRU/LSTM/Dense con validazione.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
from typing import Tuple, Dict, Any, Optional

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING_CONFIG

def build_model(model_type: str, input_shape: tuple, num_classes: int, params: Dict) -> tf.keras.Model:
    """
    Costruisce un modello Keras.
    
    Args:
        model_type: Tipo di modello ('gru', 'lstm', 'dense')
        input_shape: Forma dell'input
        num_classes: Numero di classi
        params: Parametri del modello
        
    Returns:
        Modello Keras compilato
    """
    print(f"🏗️ Costruzione modello {model_type}")
    print(f"📊 Input shape: {input_shape}")
    print(f"🏷️ Classi: {num_classes}")
    
    units = params.get('gru_units', params.get('lstm_units', 64))
    activation = params.get('activation', 'relu')
    learning_rate = params.get('learning_rate', 0.001)
    is_binary = (num_classes == 2)
    output_units = 1 if is_binary else num_classes
    output_activation = 'sigmoid' if is_binary else 'softmax'
    
    if model_type == 'gru':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(units, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units // 2, activation=activation),
            tf.keras.layers.Dense(output_units, activation=output_activation)
        ])
    elif model_type == 'lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units // 2, activation=activation),
            tf.keras.layers.Dense(output_units, activation=output_activation)
        ])
    elif model_type == 'dense':
        # MLP con 4 hidden layers da config
        hidden = TRAINING_CONFIG.get('mlp_hidden_layers', [256, 128, 64, 32])
        dropout = float(TRAINING_CONFIG.get('dropout_rate', 0.2))
        layers = [tf.keras.layers.Input(shape=input_shape), tf.keras.layers.Flatten()]
        for units in hidden:
            layers.append(tf.keras.layers.Dense(int(units), activation=activation))
            layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(output_units, activation=output_activation))
        model = tf.keras.Sequential(layers)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")
    
    # Compilazione
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    print(f"✅ Modello {model_type} creato e compilato")
    return model

def train_model(
    X: np.ndarray, 
    y: np.ndarray, 
    model_type: str = None,
    validation_strategy: str = None,
    hyperparams: Dict = None
) -> Tuple[tf.keras.Model, Dict, str]:
    """
    Addestra un modello con validazione.
    
    Args:
        X: Features
        y: Target
        model_type: Tipo di modello (default da config)
        validation_strategy: Strategia validazione (default da config)
        hyperparams: Iperparametri (default da config)
        
    Returns:
        Modello migliore, log training, path modello salvato
    """
    # Usa valori di default
    model_type = model_type or TRAINING_CONFIG["model_type"]
    validation_strategy = validation_strategy or TRAINING_CONFIG["validation_strategy"]
    hyperparams = hyperparams or TRAINING_CONFIG["hyperparameters"]
    
    print("🚀 Avvio training")
    print(f"🏗️ Modello: {model_type}")
    print(f"📊 Strategia: {validation_strategy}")
    print(f"📊 Dataset: {X.shape}")
    
    # Prepara hyperparameters per grid search
    param_combinations = _create_param_combinations(hyperparams)
    print(f"🔍 Configurazioni da testare: {len(param_combinations)}")
    
    # Training e validazione
    best_accuracy = 0
    best_model = None
    training_log = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- Configurazione {i+1}/{len(param_combinations)} ---")
        print(f"Parametri: {params}")
        
        try:
            if validation_strategy == "k_fold":
                accuracy = _train_k_fold(X, y, model_type, params)
            else:  # train_test_split
                accuracy = _train_split(X, y, model_type, params)
            
            training_log.append({
                'params': params,
                'accuracy': float(accuracy),
                'config_id': i
            })
            
            print(f"✅ Accuratezza: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Addestra modello finale con tutti i dati
                num_classes = max(len(np.unique(y)), np.max(y) + 1)  # Fix per classi mancanti
                input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
                best_model = build_model(model_type, input_shape, num_classes, params)
                callbacks = []
                if TRAINING_CONFIG.get('log_training_history', True):
                    callbacks.append(tf.keras.callbacks.History())
                best_model.fit(
                    X, y,
                    epochs=min(params.get('epochs', 10), int(TRAINING_CONFIG.get('max_epochs', 30))),
                    batch_size=params['batch_size'],
                    verbose=0,
                    callbacks=callbacks
                )
                
                print(f"🏆 Nuovo miglior modello: {accuracy:.4f}")
        
        except Exception as e:
            print(f"❌ Errore nella configurazione {i+1}: {e}")
            training_log.append({
                'params': params,
                'accuracy': 0.0,
                'error': str(e),
                'config_id': i
            })
    
    # Salva modello migliore
    os.makedirs(TRAINING_CONFIG["output_path"], exist_ok=True)
    model_path = os.path.join(TRAINING_CONFIG["output_path"], "best_model.keras")
    
    if best_model:
        best_model.save(model_path)
        print(f"💾 Modello salvato: {model_path}")
    else:
        raise ValueError("Nessun modello valido addestrato!")
    
    # Salva log
    log_path = os.path.join(TRAINING_CONFIG["output_path"], "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"✅ Training completato!")
    print(f"🏆 Miglior accuratezza: {best_accuracy:.4f}")
    
    return best_model, training_log, model_path

def _create_param_combinations(hyperparams: Dict) -> list:
    """Crea tutte le combinazioni di iperparametri."""
    import itertools
    
    keys = hyperparams.keys()
    values = hyperparams.values()
    combinations = list(itertools.product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def _train_k_fold(X: np.ndarray, y: np.ndarray, model_type: str, params: Dict) -> float:
    """Training con K-Fold cross validation."""
    kf = StratifiedKFold(n_splits=TRAINING_CONFIG["k_fold_splits"], shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold + 1}/{TRAINING_CONFIG['k_fold_splits']}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scaling per-fold per evitare leakage
        is_sequence = len(X_train.shape) == 3
        if is_sequence:
            # (samples, timesteps, features) -> scala per features
            n_features = X_train.shape[2]
            scaler = StandardScaler()
            X_train_2d = X_train.reshape(-1, n_features)
            X_val_2d = X_val.reshape(-1, n_features)
            scaler.fit(X_train_2d)
            X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
            X_val = scaler.transform(X_val_2d).reshape(X_val.shape)
        else:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

        # Costruisci e addestra modello (usa tutto y per contare classi)
        num_classes = max(len(np.unique(y)), np.max(y) + 1)  # Fix per classi mancanti
        input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
        model = build_model(model_type, input_shape, num_classes, params)
        
        model.fit(
            X_train, y_train,
            epochs=min(params.get('epochs', 10), int(TRAINING_CONFIG.get('max_epochs', 30))),
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Valuta
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.4f}")
    
    return np.mean(accuracies)

def _train_split(X: np.ndarray, y: np.ndarray, model_type: str, params: Dict) -> float:
    """Training con train/test split."""
    test_size = TRAINING_CONFIG["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scaling per split per evitare leakage
    is_sequence = len(X_train.shape) == 3
    if is_sequence:
        n_features = X_train.shape[2]
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        scaler.fit(X_train_2d)
        X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test = scaler.transform(X_test_2d).reshape(X_test.shape)
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Costruisci e addestra modello (usa tutto y per contare classi)
    num_classes = max(len(np.unique(y)), np.max(y) + 1)  # Fix per classi mancanti
    input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
    model = build_model(model_type, input_shape, num_classes, params)
    
    history = model.fit(
        X_train, y_train,
        epochs=min(params.get('epochs', 10), int(TRAINING_CONFIG.get('max_epochs', 30))),
        batch_size=params['batch_size'],
        validation_data=(X_test, y_test),
        verbose=1
    )
    # Salva loss per epoca
    if TRAINING_CONFIG.get('log_training_history', True):
        os.makedirs(TRAINING_CONFIG['output_path'], exist_ok=True)
        with open(os.path.join(TRAINING_CONFIG['output_path'], 'training_history.json'), 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    
    # Valuta
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


def main():
    """CLI minimale: carica X.npy e y.npy se presenti e avvia il training."""
    pre_dir = os.path.join(TRAINING_CONFIG["output_path"], "preprocessed")
    X_path = os.path.join(pre_dir, "X.npy")
    y_path = os.path.join(pre_dir, "y.npy")
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Dataset non trovato. Esegui prima il preprocessing. Attesi: {X_path}, {y_path}")

    print(f"🔄 Caricamento dataset preprocessato: {pre_dir}")
    X = np.load(X_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)

    # Se X è 2D, usa 'dense', altrimenti usa config
    model_type = TRAINING_CONFIG.get("model_type", "gru")
    if len(X.shape) == 2:
        model_type = "dense"

    _, training_log, model_path = train_model(
        X, y,
        model_type=model_type,
        validation_strategy=TRAINING_CONFIG.get("validation_strategy"),
        hyperparams=TRAINING_CONFIG.get("hyperparameters")
    )
    print(f"🏁 Training finito. Modello: {model_path}")


if __name__ == "__main__":
    main()
