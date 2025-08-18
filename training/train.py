# -*- coding: utf-8 -*-

"""
Modulo per il Training del Modello.

Questo modulo contiene le funzioni per:
- Eseguire un ciclo di training per diverse combinazioni di iperparametri (Grid Search).
- Costruire, compilare e addestrare un modello di rete neurale.
- Valutare il modello e salvare il migliore.
"""

import os
import itertools
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assicura che i messaggi di log di TensorFlow siano meno verbosi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importa le configurazioni e il modulo di preprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TRAINING_CONFIG
from preprocessing.process import preprocess_data

def build_model(input_shape, num_classes, activation='relu', hidden_layer_size=32):
    """Costruisce un modello di rete neurale sequenziale."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(hidden_layer_size, activation=activation),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hidden_layer_size // 2, activation=activation),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate():
    """
    Orchestra il processo di training e valutazione.

    Esegue una grid search sugli iperparametri specificati in `config.py`,
    addestra un modello per ogni combinazione e salva il migliore.
    """
    print("Avvio del processo di training...")

    # 1. Preprocessing dei dati
    X, y, _, target_map = preprocess_data()
    if X is None:
        print("Training interrotto a causa di errori nel preprocessing.")
        return

    # 2. Grid Search degli iperparametri
    hyperparams = TRAINING_CONFIG['hyperparameters']
    keys, values = zip(*hyperparams.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_accuracy = 0
    best_model_path = ""
    training_results = []

    print(f"Inizio la Grid Search: {len(param_combinations)} combinazioni da testare.")

    # Salva l'ordine delle colonne per coerenza nella predizione
    column_order_path = os.path.join(TRAINING_CONFIG['output_path'], 'column_order.json')
    os.makedirs(os.path.dirname(column_order_path), exist_ok=True)
    with open(column_order_path, 'w') as f:
        json.dump(list(X.columns), f)
    print(f"Ordine delle colonne salvato in: {column_order_path}")

    for i, params in enumerate(param_combinations):
        print(f"\n--- Training Combinazione {i+1}/{len(param_combinations)} ---")
        print(f"Parametri: {params}")

        # Divisione dei dati
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=42, stratify=y
        )

        # Costruzione del modello
        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y))
        model = build_model(
            input_shape,
            num_classes,
            activation=params['activation'],
            hidden_layer_size=params['hidden_layer_size']
        )

        # Compilazione
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Training
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.1,
            verbose=0  # 0 = silent, 1 = progress bar, 2 = one line per epoch
        )
        print("Training completato.")

        # Valutazione
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Risultati sul test set: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

        # Salva i risultati
        run_results = {
            'params': params,
            'accuracy': accuracy,
            'loss': loss
        }
        training_results.append(run_results)

        # Controlla se è il modello migliore e salvalo
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output_path = TRAINING_CONFIG['output_path']
            os.makedirs(output_path, exist_ok=True)
            best_model_path = os.path.join(output_path, 'best_model.keras')
            model.save(best_model_path)
            print(f"Nuovo modello migliore trovato! Salvato in: {best_model_path}")

    # Salva i log della grid search
    log_path = os.path.join(TRAINING_CONFIG['output_path'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_results, f, indent=4)
    print(f"\nLog della Grid Search salvato in: {log_path}")

    print(f"\nTraining completato. Migliore accuratezza: {best_accuracy:.4f}")
    return training_results, best_model_path

if __name__ == '__main__':
    train_and_evaluate()
