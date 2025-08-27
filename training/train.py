# -*- coding: utf-8 -*-
"""
Training Unificato SNN-IDS
Sistema semplice per training modelli GRU/LSTM/Dense con validazione.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import json
from typing import Tuple, Dict, Any, Optional, List, Union
import keras_tuner as kt

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

def build_model(model_type: str, input_shape: tuple, num_classes: int, hp_or_params: Union[kt.HyperParameters, Dict]) -> tf.keras.Model:
    """
    Costruisce un modello Keras, compatibile con KerasTuner e dizionari di parametri.
    
    Args:
        model_type: Tipo di modello ('gru', 'lstm', 'dense', 'mlp_4_layer')
        input_shape: Forma dell'input
        num_classes: Numero di classi
        hp_or_params: Oggetto HyperParameters di KerasTuner o un dizionario di iperparametri fissi.
        
    Returns:
        Modello Keras compilato
    """
    hp = hp_or_params
    if isinstance(hp_or_params, dict):
        # Converte il dizionario in un oggetto HyperParameters con valori fissi per compatibilità
        params = hp_or_params
        hp = kt.HyperParameters()

        # Imposta i valori di default se non presenti nel dizionario
        # Estrai il primo elemento se il valore è una lista, per compatibilità con la config
        def get_value(param_name, default_value):
            value = params.get(param_name, default_value)
            return value[0] if isinstance(value, list) else value

        hp.Fixed('learning_rate', get_value('learning_rate', 0.001))
        hp.Fixed('activation', get_value('activation', 'relu'))

        if model_type in ['gru', 'lstm']:
            units = get_value('gru_units', get_value('lstm_units', 64))
            hp.Fixed('units', units)

        if model_type == 'mlp_4_layer':
            hidden_units = get_value('hidden_layer_units', [128, 64, 32, 16])
            hp.Fixed('units_layer_1', hidden_units[0])
            hp.Fixed('units_layer_2', hidden_units[1])
            hp.Fixed('units_layer_3', hidden_units[2])
            hp.Fixed('units_layer_4', hidden_units[3])

    print(f"🏗️ Costruzione modello {model_type} (tuner-ready)")
    print(f"📊 Input shape: {input_shape}")
    print(f"🏷️ Classi: {num_classes}")

    # Definizione dello spazio di ricerca degli iperparametri comuni
    learning_rate = hp.Choice('learning_rate', values=[0.005, 0.001, 0.0005, 0.0001])
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    if model_type == 'gru':
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(tf.keras.layers.GRU(units, activation=activation))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units // 2, activation=activation))

    elif model_type == 'lstm':
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(tf.keras.layers.LSTM(units, activation=activation))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units // 2, activation=activation))

    elif model_type == 'dense':
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=activation))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation=activation))
        model.add(tf.keras.layers.Dropout(0.2))

    elif model_type == 'mlp_4_layer':
        # The data is pre-flattened in mlp-analysis.py, so no Flatten layer here.

        # Spazio di ricerca per le unità in ogni layer
        hp_units_1 = hp.Int('units_layer_1', min_value=64, max_value=256, step=32)
        hp_units_2 = hp.Int('units_layer_2', min_value=32, max_value=128, step=32)
        hp_units_3 = hp.Int('units_layer_3', min_value=16, max_value=64, step=16)
        hp_units_4 = hp.Int('units_layer_4', min_value=8, max_value=32, step=8)

        # Costruzione dei 4 layer nascosti (SENZA DROPOUT come richiesto)
        model.add(tf.keras.layers.Dense(units=hp_units_1, activation=activation))
        model.add(tf.keras.layers.Dense(units=hp_units_2, activation=activation))
        model.add(tf.keras.layers.Dense(units=hp_units_3, activation=activation))
        model.add(tf.keras.layers.Dense(units=hp_units_4, activation=activation))

    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")

    # Output Layer
    output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
    model.add(tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation=output_activation))

    # Compilazione
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    print(f"✅ Modello {model_type} (tuner-ready) creato e compilato")
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
    Addestra un modello con validazione per una SINGOLA configurazione di iperparametri.
    
    Args:
        X: Features
        y: Target
        model_type: Tipo di modello (default da config)
        validation_strategy: Strategia validazione (default da config)
        hyperparams: Iperparametri per la singola esecuzione.
        track_class_loss: Se True, traccia la loss per classe ad ogni epoca.
        
    Returns:
        Modello migliore, log training, path modello salvato, dizionario delle loss per classe (se richiesto)
    """
    # Usa valori di default
    model_type = model_type or TRAINING_CONFIG["model_type"]
    validation_strategy = validation_strategy or TRAINING_CONFIG["validation_strategy"]
    hp_or_params = hyperparams or TRAINING_CONFIG["hyperparameters"]
    
    print("🚀 Avvio training per singola configurazione")
    print(f"🏗️ Modello: {model_type}")
    print(f"📊 Strategia: {validation_strategy}")
    print(f"📊 Dataset: {X.shape}")
    print(f"⚙️ Parametri: {hp_or_params}")

    training_log = []
    
    try:
        if validation_strategy == "k_fold":
            accuracy, per_class_losses = _train_k_fold(X, y, model_type, hp_or_params, track_class_loss)
        else:  # train_test_split
            accuracy, per_class_losses = _train_split(X, y, model_type, hp_or_params, track_class_loss)
        
        training_log.append({
            'params': hp_or_params,
            'accuracy': float(accuracy),
        })

        print(f"✅ Accuratezza validazione: {accuracy:.4f}")
        
        # Addestra modello finale con tutti i dati
        num_classes = max(len(np.unique(y)), np.max(y) + 1)
        input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
        final_model = build_model(model_type, input_shape, num_classes, hp_or_params)

        # Estrai epoche e batch_size sia da hp che da dizionario
        if isinstance(hp_or_params, dict):
            def get_value(param_name, default_value):
                value = hp_or_params.get(param_name, default_value)
                return value[0] if isinstance(value, list) else value
            epochs = get_value('epochs', 10)
            batch_size = get_value('batch_size', 64)
        else: # È un oggetto kt.HyperParameters
            epochs = hp_or_params.get('epochs')
            batch_size = hp_or_params.get('batch_size')

        final_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        print(f"🏆 Modello finale addestrato su tutti i dati.")

    except Exception as e:
        print(f"❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()
        training_log.append({
            'params': hp_or_params,
            'accuracy': 0.0,
            'error': str(e),
        })
        raise e # Rilancia l'eccezione per far fallire il test singolo in mlp-analysis
    
    # Salva modello migliore
    os.makedirs(TRAINING_CONFIG["output_path"], exist_ok=True)
    model_path = os.path.join(TRAINING_CONFIG["output_path"], "best_model.keras")
    
    if final_model:
        final_model.save(model_path)
        print(f"💾 Modello salvato: {model_path}")
    else:
        raise ValueError("Nessun modello valido addestrato!")
    
    # Salva log
    log_path = os.path.join(TRAINING_CONFIG["output_path"], "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"✅ Training completato!")
    
    return final_model, training_log, model_path, per_class_losses

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

def _train_k_fold(X: np.ndarray, y: np.ndarray, model_type: str, hp_or_params: Union[kt.HyperParameters, Dict], track_class_loss: bool) -> Tuple[float, Optional[Dict]]:
    """Training con K-Fold cross validation."""
    # Controlla se la stratificazione è possibile
    class_counts = np.bincount(y)
    if np.min(class_counts) < TRAINING_CONFIG["k_fold_splits"]:
        print(f"  ⚠️  Disabling stratification for K-Fold due to classes with < {TRAINING_CONFIG['k_fold_splits']} samples.")
        kf = KFold(n_splits=TRAINING_CONFIG["k_fold_splits"], shuffle=True, random_state=42)
    else:
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
        model = build_model(model_type, input_shape, num_classes, hp_or_params)
        
        callbacks = []
        if track_class_loss:
            loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y)))
            callbacks.append(loss_logger)

        # Calculate class weights to handle imbalance
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("  ⚖️  Applying class weights to handle imbalance.")

        # Estrai epoche e batch_size sia da hp che da dizionario
        if isinstance(hp_or_params, dict):
            def get_value(param_name, default_value):
                value = hp_or_params.get(param_name, default_value)
                return value[0] if isinstance(value, list) else value
            epochs = get_value('epochs', 10)
            batch_size = get_value('batch_size', 64)
        else: # È un oggetto kt.HyperParameters
            epochs = hp_or_params.get('epochs')
            batch_size = hp_or_params.get('batch_size')

        model.fit(X_train, y_train, 
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=callbacks,
                 class_weight=class_weight_dict,
                 verbose=0)
        
        if track_class_loss:
            last_fold_losses = loss_logger.losses

        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.4f}")
    
    return np.mean(accuracies), last_fold_losses

def _train_split(X: np.ndarray, y: np.ndarray, model_type: str, hp_or_params: Union[kt.HyperParameters, Dict], track_class_loss: bool) -> Tuple[float, Optional[Dict]]:
    """Training con train/test split."""
    test_size = TRAINING_CONFIG["test_size"]

    # Controlla se la stratificazione è possibile
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("  ⚠️  Disabling stratification for train/test split due to classes with 1 sample.")
        stratify_opt = None
    else:
        stratify_opt = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_opt
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
    model = build_model(model_type, input_shape, num_classes, hp_or_params)
    
    callbacks = []
    loss_logger = None
    if track_class_loss:
        loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y)))
        callbacks.append(loss_logger)

    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("  ⚖️  Applying class weights to handle imbalance.")

    if isinstance(hp_or_params, dict):
        def get_value(param_name, default_value):
            value = hp_or_params.get(param_name, default_value)
            return value[0] if isinstance(value, list) else value
        epochs = get_value('epochs', 10)
        batch_size = get_value('batch_size', 64)
    else: # È un oggetto kt.HyperParameters
        epochs = hp_or_params.get('epochs')
        batch_size = hp_or_params.get('batch_size')

    model.fit(X_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(X_test, y_test),
             callbacks=callbacks,
             class_weight=class_weight_dict,
             verbose=1)
    
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    per_class_losses = loss_logger.losses if loss_logger else None
    return accuracy, per_class_losses
