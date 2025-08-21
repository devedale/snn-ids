# -*- coding: utf-8 -*-

"""
Modulo per il Training Avanzato del Modello.

Supporta diverse architetture (Dense, LSTM) e strategie di validazione (K-Fold).
È stato refattorizzato per accettare dati pre-processati come input.
"""

import os
import itertools
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TRAINING_CONFIG
from preprocessing.process import preprocess_data

def build_model(model_type, input_shape, num_classes, params):
    """Costruisce un modello Keras in base al tipo specificato."""
    print(f"Costruzione del modello di tipo: {model_type}")

    if model_type == 'lstm':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units=params['lstm_units'], activation=params['activation']),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=params['lstm_units'] // 2, activation=params['activation']),
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
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X=None, y=None):
    """
    Orchestra il processo di training e valutazione.

    Args:
        X (np.array, optional): Feature pre-processate. Se None, verranno caricate.
        y (np.array, optional): Etichette pre-processate. Se None, verranno caricate.
    """
    print("Avvio del processo di training avanzato...")

    if X is None or y is None:
        print("Dati non forniti, avvio il preprocessing completo...")
        X, y = preprocess_data()
        if X is None:
            print("Training interrotto a causa di errori nel preprocessing.")
            return None, None

    hyperparams = TRAINING_CONFIG['hyperparameters']
    param_combinations = [dict(zip(hyperparams.keys(), v)) for v in itertools.product(*hyperparams.values())]

    best_accuracy = 0
    best_model_path = ""
    training_log = []

    for i, params in enumerate(param_combinations):
        print(f"\n--- Inizio Grid Search: Combinazione {i+1}/{len(param_combinations)} ---")
        print(f"Parametri: {params}")

        if TRAINING_CONFIG['validation_strategy'] == 'k_fold':
            kf = KFold(n_splits=TRAINING_CONFIG['k_fold_splits'], shuffle=True, random_state=42)
            fold_accuracies = []

            for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
                print(f"-- Fold {fold+1}/{TRAINING_CONFIG['k_fold_splits']} --")
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                input_shape = X_train.shape[1:]
                num_classes = len(np.unique(y))
                model = build_model(TRAINING_CONFIG['model_type'], input_shape, num_classes, params)
                model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
                fold_accuracies.append(accuracy)

            avg_accuracy = np.mean(fold_accuracies)
            print(f"Accuratezza media K-Fold: {avg_accuracy:.4f}")
            current_run_accuracy = avg_accuracy

        elif TRAINING_CONFIG['validation_strategy'] == 'train_test_split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAINING_CONFIG.get("test_size", 0.2), random_state=42, stratify=y)
            input_shape = X_train.shape[1:]
            num_classes = len(np.unique(y))

            model = build_model(TRAINING_CONFIG['model_type'], input_shape, num_classes, params)
            model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_test, y_test), verbose=1)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Accuratezza Test Set: {accuracy:.4f}")
            current_run_accuracy = accuracy

        else:
            raise ValueError(f"Strategia di validazione non supportata: {TRAINING_CONFIG['validation_strategy']}")

        training_log.append({'params': params, 'accuracy': current_run_accuracy})

        if current_run_accuracy > best_accuracy:
            best_accuracy = current_run_accuracy
            output_path = TRAINING_CONFIG['output_path']
            os.makedirs(output_path, exist_ok=True)
            best_model_path = os.path.join(output_path, 'best_model.keras')
            model.save(best_model_path)
            print(f"Nuovo modello migliore trovato con accuratezza {best_accuracy:.4f}. Salvato in: {best_model_path}")

    log_path = os.path.join(TRAINING_CONFIG['output_path'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)

    print(f"\nTraining completato. Migliore accuratezza ottenuta: {best_accuracy:.4f}")
    return training_log, best_model_path

if __name__ == '__main__':
    train_and_evaluate()
