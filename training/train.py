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
                valid = False
                loss_value = None
                try:
                    loss_value, _ = self.model.evaluate(X_class, y_class, verbose=0)
                    if np.isfinite(loss_value):
                        self.losses[class_idx].append(float(loss_value))
                        valid = True
                    else:
                        self.losses[class_idx].append(None)
                except Exception:
                    self.losses[class_idx].append(None)
                if valid:
                    print(f"  - Class {class_idx}: Loss = {loss_value:.4f}")
                else:
                    print(f"  - Class {class_idx}: Loss non disponibile (NaN/Inf/errore)")
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

        # Helper: estrae valore singolo (compat con liste standard in config)
        def get_value(param_name, default_value):
            value = params.get(param_name, default_value)
            return value[0] if isinstance(value, list) else value

        hp.Fixed('learning_rate', get_value('learning_rate', 0.001))
        hp.Fixed('activation', get_value('activation', 'relu'))

        if model_type in ['gru', 'lstm']:
            units = get_value('gru_units', get_value('lstm_units', 64))
            hp.Fixed('units', units)

        if model_type == 'mlp_4_layer':
            # Gestione robusta: accetta lista di 4, lista singola o intero
            units_param = params.get('hidden_layer_units', [256,128,64,32])
            if isinstance(units_param, (int, float)):
                units_list = [int(units_param)] * 4
            elif isinstance(units_param, list):
                if len(units_param) >= 4:
                    units_list = [int(units_param[0]), int(units_param[1]), int(units_param[2]), int(units_param[3])]
                elif len(units_param) == 1:
                    units_list = [int(units_param[0])] * 4
                else:
                    units_list = [128, 64, 32, 16]
            else:
                units_list = [128, 64, 32, 16]

            hp.Fixed('units_layer_1', units_list[0])
            hp.Fixed('units_layer_2', units_list[1])
            hp.Fixed('units_layer_3', units_list[2])
            hp.Fixed('units_layer_4', units_list[3])

    print(f"🏗️ Costruzione modello {model_type} (tuner-ready)")
    print(f"📊 Input shape: {input_shape}")
    print(f"🏷️ Classi: {num_classes}")

    # Definizione dello spazio di ricerca degli iperparametri comuni
    learning_rate = hp.Choice('learning_rate', values=[0.005, 0.001, 0.0005, 0.0001])
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    # Facoltativo: normalizzazione interna con statistiche pre-calcolate
    if isinstance(hp_or_params, dict):
        norm_mean = hp_or_params.get('normalization_mean')
        norm_var = hp_or_params.get('normalization_var')
        if norm_mean is not None and norm_var is not None:
            try:
                norm_layer = tf.keras.layers.Normalization(mean=np.array(norm_mean), variance=np.array(norm_var))
                model.add(norm_layer)
                print("🧪 Normalization layer inserito (mean/var precompute)")
            except Exception:
                pass

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
        # Inserisce normalizzazione interna per MLP 4-layer, altrimenti scaling esterno
        if model_type == 'mlp_4_layer':
            # Flatten a 2D se necessario
            if len(X.shape) == 3:
                n_samples = X.shape[0]
                X = X.reshape(n_samples, -1)
            # Statistiche per normalizzazione interna
            norm_mean = np.mean(X, axis=0).astype(np.float32)
            norm_var = np.var(X, axis=0).astype(np.float32) + 1e-8
        else:
            is_sequence = len(X.shape) == 3
            if is_sequence:
                n_features = X.shape[2]
                scaler = StandardScaler()
                X_2d = X.reshape(-1, n_features)
                scaler.fit(X_2d)
                X = scaler.transform(X_2d).reshape(X.shape)
            else:
                scaler = StandardScaler()
                scaler.fit(X)
                X = scaler.transform(X)

        num_classes = max(len(np.unique(y)), np.max(y) + 1)
        input_shape = X.shape[1:] if len(X.shape) > 2 else (X.shape[1],)
        # Per MLP passa mean/var al builder così la normalizzazione resta nel modello
        params_for_build = hp_or_params
        if model_type == 'mlp_4_layer':
            if not isinstance(hp_or_params, dict):
                # Converti HyperParameters in dict minimo
                params_for_build = {k: hp_or_params.get(k) for k in ['learning_rate', 'activation']}
            else:
                params_for_build = dict(hp_or_params)
            params_for_build['normalization_mean'] = norm_mean.tolist()
            params_for_build['normalization_var'] = norm_var.tolist()

        final_model = build_model(model_type, input_shape, num_classes, params_for_build)

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

        # Class weights anche nel fit finale per mitigare il bias verso BENIGN
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(cw))
        final_model.fit(X, y, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict, verbose=0)

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
    # Prepara aggregatori per media per-epoca su tutti i fold
    # Determina il numero di epoche una volta sola (stesso per tutti i fold)
    if isinstance(hp_or_params, dict):
        def get_value(param_name, default_value):
            value = hp_or_params.get(param_name, default_value)
            return value[0] if isinstance(value, list) else value
        epochs_total = get_value('epochs', 10)
    else:
        epochs_total = hp_or_params.get('epochs')

    sums_per_class: Optional[Dict[int, List[float]]] = {} if track_class_loss else None
    counts_per_class: Optional[Dict[int, List[int]]] = {} if track_class_loss else None
    
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

        # Se stiamo usando MLP 4-layer, appiattisci finestre temporali a vettori 2D
        if model_type == 'mlp_4_layer' and len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)

        num_classes = max(len(np.unique(y)), np.max(y) + 1)
        input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
        model = build_model(model_type, input_shape, num_classes, hp_or_params)
        
        callbacks = []
        if track_class_loss:
            # Traccia solo le classi effettivamente presenti in questo fold
            loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y_train)))
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
            last_fold_losses = loss_logger.losses  # Dict[class_idx -> List[loss_epoch]]
            # Media per-epoca: somma e conteggio separati
            if sums_per_class is not None and counts_per_class is not None and last_fold_losses is not None:
                for class_idx, series in last_fold_losses.items():
                    # Inizializza vettori per questa classe
                    if class_idx not in sums_per_class:
                        sums_per_class[class_idx] = [0.0] * int(epochs_total)
                        counts_per_class[class_idx] = [0] * int(epochs_total)
                    # Accumula per epoca, ignorando None/NaN/Inf
                    for epoch_i in range(min(len(series), int(epochs_total))):
                        v = series[epoch_i]
                        if v is None:
                            continue
                        try:
                            if np.isfinite(v):
                                sums_per_class[class_idx][epoch_i] += float(v)
                                counts_per_class[class_idx][epoch_i] += 1
                        except Exception:
                            continue

        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.4f}")
    
    # Calcola media per-epoca sui fold (se abilitato)
    if sums_per_class is not None and counts_per_class is not None and len(sums_per_class) > 0:
        averaged_losses: Dict[int, List[Optional[float]]] = {}
        for class_idx in sums_per_class.keys():
            avg_series: List[Optional[float]] = []
            for epoch_i in range(int(epochs_total)):
                cnt = counts_per_class[class_idx][epoch_i]
                if cnt > 0:
                    avg_series.append(sums_per_class[class_idx][epoch_i] / cnt)
                else:
                    avg_series.append(None)
            averaged_losses[class_idx] = avg_series
        return np.mean(accuracies), averaged_losses

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

    # Per MLP 4-layer, appiattisci a 2D dopo lo scaling
    if model_type == 'mlp_4_layer' and len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    num_classes = max(len(np.unique(y)), np.max(y) + 1)
    input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
    model = build_model(model_type, input_shape, num_classes, hp_or_params)
    
    callbacks = []
    loss_logger = None
    if track_class_loss:
        # Traccia solo le classi presenti nel training set corrente
        loss_logger = PerClassLossLogger(X_train, y_train, class_indices=list(np.unique(y_train)))
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
