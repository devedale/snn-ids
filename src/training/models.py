# -*- coding: utf-8 -*-
"""
Model Architectures for SNN-IDS
This module defines the Keras model architectures used in the project.
Each function is responsible for building and compiling a specific type of model
(e.g., GRU, LSTM, MLP). The functions are designed to be compatible with both
fixed hyperparameters and the KerasTuner optimization framework.
"""

import tensorflow as tf
from typing import Union, List, Tuple

# A type hint for a hyperparameter that can be either a fixed value or a KerasTuner choice.
TunableInt = Union[int, 'kt.HyperParameters.Int']
TunableFloat = Union[float, 'kt.HyperParameters.Float']
TunableChoice = Union[str, 'kt.HyperParameters.Choice']

def build_gru_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    units: TunableInt = 64,
    activation: TunableChoice = 'tanh',
    learning_rate: TunableFloat = 0.0005  # Lowered default LR for RNNs
) -> tf.keras.Model:
    """
    Builds and compiles a stacked GRU-based sequential model. This improved
    architecture uses two GRU layers, which can capture more complex temporal
    patterns, and gradient clipping to stabilize training.

    Args:
        input_shape: The shape of the input data (timesteps, features).
        num_classes: The number of output classes.
        units: The number of units in the primary GRU layer.
        activation: The activation function for the dense layers.
        learning_rate: The learning rate for the Adam optimizer.

    Returns:
        A compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="input_layer"),
        # First GRU layer returns sequences to be processed by the next layer.
        tf.keras.layers.GRU(units, activation=activation, return_sequences=True, name="gru_layer_1"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        # Second GRU layer, does not return sequences.
        tf.keras.layers.GRU(units // 2, activation=activation, name="gru_layer_2"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"),
        tf.keras.layers.Dense(units // 2, activation=activation, name="dense_1"),
        tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")
    ], name="Stacked_GRU_Model")

    # Using gradient clipping (clipnorm=1.0) is a best practice for RNNs
    # to prevent the exploding gradient problem.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    units: TunableInt = 64,
    activation: TunableChoice = 'tanh',
    learning_rate: TunableFloat = 0.0005  # Lowered default LR for RNNs
) -> tf.keras.Model:
    """
    Builds and compiles a stacked LSTM-based sequential model. This improved
    architecture uses two LSTM layers and gradient clipping for better performance
    and stability.

    Args:
        input_shape: The shape of the input data (timesteps, features).
        num_classes: The number of output classes.
        units: The number of units in the primary LSTM layer.
        activation: The activation function for the dense layers.
        learning_rate: The learning rate for the Adam optimizer.

    Returns:
        A compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="input_layer"),
        tf.keras.layers.LSTM(units, activation=activation, return_sequences=True, name="lstm_layer_1"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        tf.keras.layers.LSTM(units // 2, activation=activation, name="lstm_layer_2"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"),
        tf.keras.layers.Dense(units // 2, activation=activation, name="dense_1"),
        tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")
    ], name="Stacked_LSTM_Model")

    # Using gradient clipping is crucial for training LSTMs effectively.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_dense_model(
    input_shape: Tuple[int],
    num_classes: int,
    activation: TunableChoice = 'relu',
    learning_rate: TunableFloat = 0.001
) -> tf.keras.Model:
    """
    Builds and compiles a simple Dense (fully-connected) model.
    This model includes a Flatten layer to handle sequential input if provided.

    Args:
        input_shape: The shape of the input data.
        num_classes: The number of output classes.
        activation: The activation function for the hidden layers.
        learning_rate: The learning rate for the Adam optimizer.

    Returns:
        A compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="input_layer"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation=activation, name="dense_1"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        tf.keras.layers.Dense(64, activation=activation, name="dense_2"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"),
        tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")
    ], name="Dense_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_mlp_4_layer_model(
    input_shape: Tuple[int],
    num_classes: int,
    units_layer_1: TunableInt = 256,
    units_layer_2: TunableInt = 128,
    units_layer_3: TunableInt = 64,
    units_layer_4: TunableInt = 32,
    activation: TunableChoice = 'relu',
    learning_rate: TunableFloat = 0.001,
    normalization_mean: List[float] = None,
    normalization_variance: List[float] = None
) -> tf.keras.Model:
    """
    Builds and compiles a 4-hidden-layer MLP model.
    This model can optionally include a normalization layer if the mean and
    variance of the training data are provided. This is crucial for ensuring
    that data is scaled consistently during training and inference.
    NOTE: This model expects flattened (2D) input data.

    Args:
        input_shape: The shape of the input data (features,).
        num_classes: The number of output classes.
        units_layer_1: Number of units in the first hidden layer.
        units_layer_2: Number of units in the second hidden layer.
        units_layer_3: Number of units in the third hidden layer.
        units_layer_4: Number of units in the fourth hidden layer.
        activation: The activation function for the hidden layers.
        learning_rate: The learning rate for the Adam optimizer.
        normalization_mean: The mean of the training features for normalization.
        normalization_variance: The variance of the training features for normalization.

    Returns:
        A compiled Keras model.
    """
    layers = [tf.keras.layers.Input(shape=input_shape, name="input_layer")]

    # If normalization stats are provided, add the normalization layer first.
    if normalization_mean is not None and normalization_variance is not None:
        norm_layer = tf.keras.layers.Normalization(
            mean=normalization_mean,
            variance=normalization_variance,
            name="normalization_layer"
        )
        layers.append(norm_layer)
        print("✅ Internal normalization layer added to MLP model.")

    layers.extend([
        tf.keras.layers.Dense(units=units_layer_1, activation=activation, name="dense_1"),
        tf.keras.layers.Dense(units=units_layer_2, activation=activation, name="dense_2"),
        tf.keras.layers.Dense(units=units_layer_3, activation=activation, name="dense_3"),
        tf.keras.layers.Dense(units=units_layer_4, activation=activation, name="dense_4"),
        tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")
    ])

    model = tf.keras.Sequential(layers, name="MLP_4_Layer_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# A factory to easily get the correct model-building function by name.
MODEL_BUILDERS = {
    "gru": build_gru_model,
    "lstm": build_lstm_model,
    "dense": build_dense_model,
    "mlp_4_layer": build_mlp_4_layer_model,
}

def get_model_builder(model_type: str):
    """
    Retrieves the correct model-building function based on the model type string.

    Args:
        model_type: The name of the model ('gru', 'lstm', etc.).

    Returns:
        The corresponding model-building function.

    Raises:
        ValueError: If the model_type is not supported.
    """
    builder = MODEL_BUILDERS.get(model_type)
    if not builder:
        raise ValueError(f"Unsupported model type: '{model_type}'. Supported types are: {list(MODEL_BUILDERS.keys())}")
    return builder
