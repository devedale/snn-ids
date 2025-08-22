# -*- coding: utf-8 -*-

"""
Modulo per il Training Avanzato del Modello.
Refattorizzato per accettare override della configurazione.
"""

import os
import itertools
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TRAINING_CONFIG as TC
from preprocessing.process import preprocess_data
from evaluation.stats import generate_comprehensive_report, print_summary_to_console

def build_model(model_type, input_shape, num_classes, params):
    """Costruisce un modello Keras in base al tipo specificato."""
    print(f"Costruzione del modello di tipo: {model_type}")
    print(f"Input shape ricevuto: {input_shape}")
    print(f"Tipo input_shape: {type(input_shape)}")
    print(f"Lunghezza input_shape: {len(input_shape) if hasattr(input_shape, '__len__') else 'N/A'}")

    units = params.get('lstm_units', params.get('gru_units', 64))

    if model_type == 'lstm':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units=units, activation=params['activation']),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=units // 2, activation=params['activation']),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == 'gru':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(units=units, activation=params['activation']),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=units // 2, activation=params['activation']),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == 'dense':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=params['activation']),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation=params['activation']),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    
    # Gestisci il caso di una sola classe
    if num_classes == 1:
        # Per una sola classe, usa binary crossentropy e sigmoid
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=params['activation']),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # Per multiple classi, usa sparse categorical crossentropy
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(X=None, y=None, config_override=None):
    """
    Orchestra il processo di training e valutazione.
    """
    train_config = deepcopy(TC)
    if config_override:
        train_config.update(config_override.get("TRAINING_CONFIG", {}))

    print(f"--- Avvio Training (Strategia: {train_config['validation_strategy']}) ---")

    if X is None or y is None:
        X, y = preprocess_data(config_override)
        if X is None:
            print("Training interrotto a causa di errori nel preprocessing.")
            return None, None

    hyperparams = train_config['hyperparameters']
    param_combinations = [dict(zip(hyperparams.keys(), v)) for v in itertools.product(*hyperparams.values())]

    best_accuracy = 0
    best_model_path = ""
    training_log = []

    for i, params in enumerate(param_combinations):
        print(f"\n--- Inizio Grid Search: Combinazione {i+1}/{len(param_combinations)} ---")
        print(f"Parametri: {params}")

        model_type = train_config['model_type']

        if train_config['validation_strategy'] == 'k_fold':
            kf = KFold(n_splits=train_config['k_fold_splits'], shuffle=True, random_state=42)
            fold_accuracies = []
            for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
                print(f"-- Fold {fold+1}/{train_config['k_fold_splits']} --")
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                model = build_model(model_type, X_train.shape[1:], len(np.unique(y)), params)
                model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                _, accuracy = model.evaluate(X_val, y_val, verbose=0)
                fold_accuracies.append(accuracy)
            avg_accuracy = np.mean(fold_accuracies)
            print(f"Accuratezza media K-Fold: {avg_accuracy:.4f}")
            current_run_accuracy = avg_accuracy

        elif train_config['validation_strategy'] == 'train_test_split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_config.get("test_size", 0.2), random_state=42, stratify=y)
            model = build_model(model_type, X_train.shape[1:], len(np.unique(y)), params)
            model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_test, y_test), verbose=1)
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Accuratezza Test Set: {accuracy:.4f}")
            current_run_accuracy = accuracy
        else:
            raise ValueError(f"Strategia di validazione non supportata: {train_config['validation_strategy']}")

        training_log.append({'params': params, 'accuracy': current_run_accuracy})

        if current_run_accuracy > best_accuracy:
            best_accuracy = current_run_accuracy
            output_path = train_config['output_path']
            os.makedirs(output_path, exist_ok=True)
            best_model_path = os.path.join(output_path, 'best_model.keras')
            try:
                model.save(best_model_path, save_format='keras')
            except:
                # Fallback per versioni più vecchie di Keras
                best_model_path = os.path.join(output_path, 'best_model.h5')
                model.save(best_model_path, save_format='h5')
            print(f"Nuovo modello migliore trovato con accuratezza {best_accuracy:.4f}. Salvato in: {best_model_path}")

    log_path = os.path.join(train_config['output_path'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)

    print(f"\nTraining completato. Migliore accuratezza ottenuta: {best_accuracy:.4f}")
    
    # Genera statistiche complete e report
    print("\n--- Generazione Statistiche e Report ---")
    
    # Carica il modello migliore per le predizioni
    best_model = tf.keras.models.load_model(best_model_path)
    
    # Genera predizioni sul dataset completo per le statistiche
    # Mantieni la forma 3D per i modelli sequenziali
    if len(X.shape) == 3:
        # Per modelli sequenziali (LSTM/GRU), mantieni la forma (samples, timesteps, features)
        X_reshaped = X
    else:
        # Per modelli densi, ridimensiona in 2D
        X_reshaped = X.reshape(-1, X.shape[-1])
    
    y_pred_proba = best_model.predict(X_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Ottieni i nomi delle classi dal preprocessing
    try:
        from preprocessing.process import DC
        target_map_path = os.path.join(os.path.dirname(train_config['output_path']), 'target_anonymization_map.json')
        if os.path.exists(target_map_path):
            with open(target_map_path, 'r') as f:
                target_map = json.load(f)
            class_names = list(target_map['inverse_map'].values())
        else:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    except:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    
    # Genera report completo
    stats_output_path = os.path.join(train_config['output_path'], 'statistics')
    report_data = generate_comprehensive_report(
        X=X_reshaped, y=y, y_pred=y_pred, class_names=class_names,
        training_log=training_log, best_model_path=best_model_path,
        output_path=stats_output_path, config=train_config
    )
    
    # Stampa riepilogo nella console
    print_summary_to_console(report_data)
    
    return training_log, best_model_path
