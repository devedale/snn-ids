# -*- coding: utf-8 -*-
"""
Training Utilities for SNN-IDS
This module provides helper functions for the training process, including data scaling,
reshaping, and hyperparameter management. Centralizing these functions helps to
reduce code duplication and improve maintainability.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def scale_data(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies StandardScaler to training and validation data.
    This function correctly handles both 2D (tabular) and 3D (sequential) data.
    For sequential data, the scaler is fit on the reshaped 2D data and then
    applied, preserving the temporal structure.

    Args:
        X_train: Training features.
        X_val: Validation features.

    Returns:
        A tuple containing the scaled training and validation features.
    """
    scaler = StandardScaler()

    # If data is sequential (3D), we need to reshape to 2D for scaling
    if X_train.ndim == 3:
        # Original shapes
        orig_shape_train = X_train.shape
        orig_shape_val = X_val.shape

        # Reshape to (num_samples * timesteps, num_features)
        X_train_2d = X_train.reshape(-1, orig_shape_train[2])
        X_val_2d = X_val.reshape(-1, orig_shape_val[2])

        # Fit scaler ONLY on training data and transform both
        X_train_scaled_2d = scaler.fit_transform(X_train_2d)
        X_val_scaled_2d = scaler.transform(X_val_2d)

        # Reshape back to the original 3D shape
        X_train_scaled = X_train_scaled_2d.reshape(orig_shape_train)
        X_val_scaled = X_val_scaled_2d.reshape(orig_shape_val)

        return X_train_scaled, X_val_scaled

    # For 2D data, the process is straightforward
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled

def prepare_data_for_model(X: np.ndarray, model_type: str) -> np.ndarray:
    """
    Reshapes the input data based on the requirements of the specified model type.
    - For MLP/Dense models, it flattens 3D sequential data into 2D.
    - For Recurrent models (GRU/LSTM), it ensures the data remains 3D.

    Args:
        X: The input data array.
        model_type: The type of model (e.g., 'mlp_4_layer', 'gru').

    Returns:
        The appropriately shaped data array.
    """
    is_dense_model = model_type in ['dense', 'mlp_4_layer']

    # If it's a dense model and the input data is 3D, it must be flattened.
    if is_dense_model and X.ndim == 3:
        return X.reshape(X.shape[0], -1)

    return X

def get_model_hyperparameters(config: Dict, model_type: str) -> Dict[str, Any]:
    """
    Constructs the final hyperparameter dictionary for a given model type.
    It merges the 'common' hyperparameters with the 'model_specific' ones from the config.
    Model-specific parameters will override common ones if there are any conflicts.

    Args:
        config: The TRAINING_CONFIG dictionary.
        model_type: The type of model for which to get hyperparameters.

    Returns:
        A dictionary of hyperparameters for the specified model.
    """
    hyperparams_config = config.get("hyperparameters", {})
    common_params = hyperparams_config.get("common", {}).copy()
    specific_params = hyperparams_config.get("model_specific", {}).get(model_type, {}).copy()

    # The specific parameters override the common ones
    common_params.update(specific_params)

    return common_params
