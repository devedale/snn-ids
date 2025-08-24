# -*- coding: utf-8 -*-
"""
Factory modelli Keras (dense, gru, lstm) con head binaria/multiclasse.
"""

import tensorflow as tf
from typing import Dict, Tuple


def build_head(num_classes: int) -> Tuple[int, str, str]:
    is_binary = (num_classes == 2)
    output_units = 1 if is_binary else num_classes
    activation = 'sigmoid' if is_binary else 'softmax'
    loss = 'binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy'
    return output_units, activation, loss


def build_model(model_type: str, input_shape: tuple, num_classes: int, params: Dict) -> tf.keras.Model:
    units = params.get('gru_units', params.get('lstm_units', 64))
    activation = params.get('activation', 'relu')
    learning_rate = params.get('learning_rate', 0.001)
    output_units, output_activation, loss = build_head(num_classes)

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
        hidden = params.get('hidden', [256, 128, 64, 32])
        dropout = float(params.get('dropout', 0.2))
        layers = [tf.keras.layers.Input(shape=input_shape)]
        if len(input_shape) > 1:
            layers.append(tf.keras.layers.Flatten())
        for h in hidden:
            layers.append(tf.keras.layers.Dense(int(h), activation=activation))
            layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(output_units, activation=output_activation))
        model = tf.keras.Sequential(layers)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


