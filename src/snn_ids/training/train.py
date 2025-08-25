# -*- coding: utf-8 -*-
"""
Training Unificato SNN-IDS
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING_CONFIG

def build_model(model_type: str, input_shape: tuple, num_classes: int, params: Dict) -> tf.keras.Model:
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
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f"✅ Modello {model_type} creato e compilato")
    return model

def train_model(
    X: np.ndarray, y: np.ndarray, model_type: str = None,
    validation_strategy: str = None, hyperparams: Dict = None, callbacks: List = None
) -> Tuple[tf.keras.Model, Dict, str]:
    model_type = model_type or TRAINING_CONFIG["model_type"]
    validation_strategy = validation_strategy or TRAINING_CONFIG["validation_strategy"]
    hyperparams = hyperparams or TRAINING_CONFIG["hyperparameters"]

    param_combinations = _create_param_combinations(hyperparams)
    best_accuracy = 0
    best_model = None
    training_log = []

    for i, params in enumerate(param_combinations):
        try:
            if validation_strategy == "k_fold":
                accuracy, history = _train_k_fold(X, y, model_type, params, callbacks)
            else:
                accuracy, history = _train_split(X, y, model_type, params, callbacks)

            log_entry = {'params': params, 'accuracy': float(accuracy), 'config_id': i}
            if history: log_entry['history'] = history.history
            training_log.append(log_entry)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                num_classes = len(np.unique(y))
                input_shape = X.shape[1:]
                best_model = build_model(model_type, input_shape, num_classes, params)
                best_model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=callbacks)
        except Exception as e:
            training_log.append({'params': params, 'accuracy': 0.0, 'error': str(e), 'config_id': i})

    model_path = os.path.join(TRAINING_CONFIG["output_path"], "best_model.keras")
    if best_model: best_model.save(model_path)
    else: raise ValueError("Nessun modello valido addestrato!")

    return best_model, training_log, model_path

def train_model_with_per_class_loss(
    X: np.ndarray, y: np.ndarray, model_type: str, hyperparams: Dict, num_classes: int, label_encoder
) -> Tuple[tf.keras.Model, pd.DataFrame, pd.DataFrame]:
    print("🚀 Avvio training con tracking per classe...")
    input_shape = X.shape[1:]
    model = build_model(model_type, input_shape, num_classes, hyperparams)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.get('learning_rate', 0.001))
    if num_classes > 2:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    epochs = hyperparams.get('epochs', 30)
    batch_size = hyperparams.get('batch_size', 64)
    loss_history = []

    for epoch in range(epochs):
        print(f"Epoca {epoch+1}/{epochs}")
        epoch_losses_by_class = {i: [] for i in range(num_classes)}
        for i in range(0, len(X), batch_size):
            X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                per_sample_loss = loss_fn(y_batch, y_pred)
                loss = tf.reduce_mean(per_sample_loss)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            for label_id in np.unique(y_batch):
                mask = y_batch == label_id
                if np.any(mask): epoch_losses_by_class[label_id].extend(per_sample_loss[mask].numpy())

        epoch_log = {'epoch': epoch + 1}
        for i in range(num_classes):
            class_name = label_encoder.classes_[i]
            epoch_log[f'loss_{class_name}'] = np.mean(epoch_losses_by_class[i]) if epoch_losses_by_class[i] else 0
        loss_history.append(epoch_log)

    y_pred_full = model.predict(X)
    y_pred_classes = np.argmax(y_pred_full, axis=1)

    unique_labels = np.unique(y)
    target_names_present = [label_encoder.classes_[i] for i in unique_labels]

    report = classification_report(
        y,
        y_pred_classes,
        labels=unique_labels,
        target_names=target_names_present,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    loss_df = pd.DataFrame(loss_history)
    return model, loss_df, report_df

def _create_param_combinations(hyperparams: Dict) -> list:
    keys = hyperparams.keys()
    values = [val if isinstance(val, list) else [val] for val in hyperparams.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def _train_k_fold(X: np.ndarray, y: np.ndarray, model_type: str, params: Dict, callbacks: List) -> Tuple[float, Any]:
    kf = StratifiedKFold(n_splits=TRAINING_CONFIG["k_fold_splits"], shuffle=True, random_state=42)
    accuracies, histories = [], []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
        # ... (scaling logic) ...
        num_classes = len(np.unique(y))
        input_shape = X_train.shape[1:]
        model = build_model(model_type, input_shape, num_classes, params)
        history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0, callbacks=callbacks, validation_data=(X_val, y_val))
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(accuracy)
        histories.append(history)
    return np.mean(accuracies), histories[np.argmax(accuracies)]

def _train_split(X: np.ndarray, y: np.ndarray, model_type: str, params: Dict, callbacks: List) -> Tuple[float, Any]:
    # ... (implementation)
    pass
