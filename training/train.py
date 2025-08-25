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
from typing import Tuple, Dict, Any, Optional, List

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING_CONFIG

class PerClassLossLogger(tf.keras.callbacks.Callback):
    """
    Callback to compute and log training loss for each class at the end of each epoch.
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, class_indices: List[int]):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.class_indices = class_indices
        # Initialize losses dictionary
        self.losses = {class_idx: [] for class_idx in self.class_indices}
        print(f"📊 PerClassLossLogger initialized for classes: {self.class_indices}")

    def on_epoch_end(self, epoch: int, logs=None):
        print(f"\nEpoch {epoch+1}: Calculating per-class training loss...")
        for class_idx in self.class_indices:
            # Filter data for the specific class
            class_mask = (self.y_train == class_idx)
            X_class = self.X_train[class_mask]
            y_class = self.y_train[class_mask]

            if X_class.shape[0] > 0:
                # Evaluate loss on the data for this class
                loss, _ = self.model.evaluate(X_class, y_class, verbose=0)
                self.losses[class_idx].append(loss)
                print(f"  - Class {class_idx}: Loss = {loss:.4f}")
            else:
                # Handle cases where a class might not be in a fold
                self.losses[class_idx].append(None)
                print(f"  - Class {class_idx}: No samples in this training set.")

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
    # Log iperparametri principali
    try:
        hp_epochs = params.get('epochs', '?')
        hp_batch = params.get('batch_size', '?')
        hp_units = params.get('gru_units', params.get('lstm_units', 64))
        learning_rate = params.get('learning_rate', 0.001)
        activation = params.get('activation', 'relu')
        print(f"⚙️ Hyperparams: lr={learning_rate}, act={activation}, units={hp_units}, epochs={hp_epochs}, batch={hp_batch}")
    except Exception:
        pass
    
    units = params.get('gru_units', params.get('lstm_units', 64))
    activation = params.get('activation', 'relu')
    learning_rate = params.get('learning_rate', 0.001)
    
    if model_type == 'gru':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(units, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units // 2, activation=activation),
            tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
    elif model_type == 'lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units // 2, activation=activation),
            tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
    elif model_type == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
    elif model_type == 'mlp_4_layer':
        hidden_layer_units = params.get('hidden_layer_units', [128, 64, 32, 16])
        dropout_rate = params.get('dropout', 0.2)

        if len(hidden_layer_units) != 4:
            raise ValueError(f"mlp_4_layer requires 'hidden_layer_units' to be a list of 4 integers, but got {hidden_layer_units}")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            # Hidden Layer 1
            tf.keras.layers.Dense(hidden_layer_units[0], activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            # Hidden Layer 2
            tf.keras.layers.Dense(hidden_layer_units[1], activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            # Hidden Layer 3
            tf.keras.layers.Dense(hidden_layer_units[2], activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            # Hidden Layer 4
            tf.keras.layers.Dense(hidden_layer_units[3], activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            # Output Layer
            tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")
    
    # Compilazione
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Per etichette intere (come quelle prodotte da LabelEncoder), usiamo sparse_categorical_crossentropy
    # Per il caso binario (0/1), binary_crossentropy è corretto con un output di Dense(1, activation='sigmoid')
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    print(f"✅ Modello {model_type} creato e compilato")
    return model

def train_model(
    X: np.ndarray, 
    y: np.ndarray, 
    model_type: str = None,
    validation_strategy: str = None,
    hyperparams: Dict = None,
    track_class_loss: bool = False
) -> Tuple[tf.keras.Model, Dict, str, Optional[Dict]]:
    """
    Addestra un modello con validazione.
    
    Args:
        X: Features
        y: Target
        model_type: Tipo di modello (default da config)
        validation_strategy: Strategia validazione (default da config)
        hyperparams: Iperparametri (default da config)
        track_class_loss: Se True, traccia la loss per classe ad ogni epoca.
        
    Returns:
        Modello migliore, log training, path modello salvato, dizionario delle loss per classe (se richiesto)
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
    best_per_class_losses = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- Configurazione {i+1}/{len(param_combinations)} ---")
        print(f"Parametri: {params}")
        
        try:
            if validation_strategy == "k_fold":
                # K-fold non è ideale per il tracciamento della loss per classe su un singolo training set,
                # ma implementiamo una versione base che restituisce la loss dell'ultimo fold.
                accuracy, per_class_losses = _train_k_fold(X, y, model_type, params, track_class_loss)
            else:  # train_test_split
                accuracy, per_class_losses = _train_split(X, y, model_type, params, track_class_loss)
            
            training_log.append({
                'params': params,
                'accuracy': float(accuracy),
                'config_id': i
            })
            
            print(f"✅ Accuratezza: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_per_class_losses = per_class_losses
                # Addestra modello finale con tutti i dati
                num_classes = max(len(np.unique(y)), np.max(y) + 1)
                input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
                best_model = build_model(model_type, input_shape, num_classes, params)
                best_model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                
                print(f"🏆 Nuovo miglior modello: {accuracy:.4f}")
        
        except Exception as e:
            print(f"❌ Errore nella configurazione {i+1}: {e}")
            import traceback
            traceback.print_exc()
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
    
    return best_model, training_log, model_path, best_per_class_losses

def _create_param_combinations(hyperparams: Dict) -> list:
    """
    Crea tutte le combinazioni di iperparametri, garantendo che i valori siano iterabili.
    """
    import itertools
    
    keys = hyperparams.keys()
    # Assicura che ogni valore sia una lista per evitare errori con `itertools.product`
    values = [
        val if isinstance(val, list) else [val]
        for val in hyperparams.values()
    ]

    combinations = list(itertools.product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def _train_k_fold(X: np.ndarray, y: np.ndarray, model_type: str, params: Dict, track_class_loss: bool) -> Tuple[float, Optional[Dict]]:
    """Training con K-Fold cross validation."""
    kf = StratifiedKFold(n_splits=TRAINING_CONFIG["k_fold_splits"], shuffle=True, random_state=42)
    accuracies = []
    last_fold_losses = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold + 1}/{TRAINING_CONFIG['k_fold_splits']}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        is_sequence = len(X_train.shape) == 3
        if is_sequence:
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

        num_classes = max(len(np.unique(y)), np.max(y) + 1)
        input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
        model = build_model(model_type, input_shape, num_classes, params)
        
        callbacks = []
        if track_class_loss:
            loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y)))
            callbacks.append(loss_logger)

        model.fit(X_train, y_train, 
                 epochs=params['epochs'], 
                 batch_size=params['batch_size'], 
                 callbacks=callbacks,
                 verbose=0)
        
        if track_class_loss:
            last_fold_losses = loss_logger.losses

        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.4f}")
    
    return np.mean(accuracies), last_fold_losses

def _train_split(X: np.ndarray, y: np.ndarray, model_type: str, params: Dict, track_class_loss: bool) -> Tuple[float, Optional[Dict]]:
    """Training con train/test split."""
    test_size = TRAINING_CONFIG["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
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

    num_classes = max(len(np.unique(y)), np.max(y) + 1)
    input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
    model = build_model(model_type, input_shape, num_classes, params)
    
    callbacks = []
    loss_logger = None
    if track_class_loss:
        loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y)))
        callbacks.append(loss_logger)

    model.fit(X_train, y_train,
             epochs=params['epochs'],
             batch_size=params['batch_size'],
             validation_data=(X_test, y_test),
             callbacks=callbacks,
             verbose=1)
    
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    per_class_losses = loss_logger.losses if loss_logger else None
    return accuracy, per_class_losses
