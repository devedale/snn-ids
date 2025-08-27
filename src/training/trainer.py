# -*- coding: utf-8 -*-
"""
Model Training Orchestrator for SNN-IDS
This module contains the ModelTrainer class, which is responsible for orchestrating
the training and validation of models. It uses the helper functions from
`utils.py` for data preparation and the model builders from `models.py`.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.utils import class_weight
import json
from typing import Tuple, Dict, Any, Optional
import inspect

# Add project root to path to allow importing 'config' and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TRAINING_CONFIG
from .models import get_model_builder
from .utils import scale_data, prepare_data_for_model, get_model_hyperparameters

class ModelTrainer:
    """
    A class to orchestrate the model training and validation process.
    """
    def __init__(self, model_type: str, config: Dict = None):
        """
        Initializes the ModelTrainer.

        Args:
            model_type: The type of model to train (e.g., 'gru', 'lstm').
            config: The training configuration dictionary (defaults to TRAINING_CONFIG).
        """
        self.model_type = model_type
        self.config = config or TRAINING_CONFIG
        self.model_builder = get_model_builder(model_type)
        self.hyperparams = get_model_hyperparameters(self.config, model_type)
        print(f"🚀 ModelTrainer initialized for model type: {self.model_type}")
        print(f"⚙️ Hyperparameters: {self.hyperparams}")

    def _get_builder_params(self) -> Dict[str, Any]:
        """
        Inspects the model builder's signature and filters the hyperparameters
        to include only those that the builder function accepts. This prevents
        TypeError for unexpected arguments like 'epochs'.
        """
        builder_sig = inspect.signature(self.model_builder)

        # Filter the full hyperparameter list to only include keys that are
        # actual parameters of the model builder function.
        builder_params = {
            key: value[0] if isinstance(value, list) else value
            for key, value in self.hyperparams.items()
            if key in builder_sig.parameters
        }
        return builder_params

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[Dict]]:
        """
        Runs the configured validation strategy (k-fold or train-test split)
        to evaluate the model's performance with the given hyperparameters.

        Args:
            X: The feature dataset.
            y: The target labels.

        Returns:
            A tuple containing the average validation accuracy and per-class loss data.
        """
        strategy = self.config.get("validation_strategy", "k_fold")
        print(f"🏋️ Starting training with validation strategy: {strategy}")

        if strategy == "k_fold":
            return self._train_k_fold(X, y)
        elif strategy == "train_test_split":
            return self._train_split(X, y)
        else:
            raise ValueError(f"Unsupported validation strategy: {strategy}")

    def train_final_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """
        Trains a final model on the entire dataset with the configured hyperparameters.

        Args:
            X: The full feature dataset.
            y: The full target labels.

        Returns:
            The trained Keras model.
        """
        print("🏆 Training final model on the entire dataset...")

        # Prepare data (scaling and reshaping)
        X_scaled, _ = scale_data(X, X) # Use a copy of X for the 'val' set
        X_prepared = prepare_data_for_model(X_scaled, self.model_type)

        num_classes = len(np.unique(y))
        input_shape = X_prepared.shape[1:]

        # Get only the parameters relevant for the model builder
        builder_params = self._get_builder_params()
        model = self.model_builder(input_shape=input_shape, num_classes=num_classes, **builder_params)

        # Get training parameters
        epochs = self.hyperparams['epochs'][0] if isinstance(self.hyperparams.get('epochs'), list) else self.hyperparams.get('epochs', 10)
        batch_size = self.hyperparams['batch_size'][0] if isinstance(self.hyperparams.get('batch_size'), list) else self.hyperparams.get('batch_size', 64)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)

        model.fit(
            X_prepared,
            y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=dict(enumerate(class_weights)),
            verbose=1
        )

        print("✅ Final model training complete.")
        self.save_model(model)
        return model

    def _train_k_fold(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[Dict]]:
        """Private method for K-Fold cross-validation."""
        n_splits = self.config.get("k_fold_splits", 5)

        if np.min(np.bincount(y)) < n_splits:
            print(f"  ⚠️ Disabling stratification for K-Fold due to classes with < {n_splits} samples.")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        accuracies = []
        builder_params = self._get_builder_params()
        epochs = self.hyperparams['epochs'][0] if isinstance(self.hyperparams.get('epochs'), list) else self.hyperparams.get('epochs', 10)
        batch_size = self.hyperparams['batch_size'][0] if isinstance(self.hyperparams.get('batch_size'), list) else self.hyperparams.get('batch_size', 64)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
            X_train_prepared = prepare_data_for_model(X_train_scaled, self.model_type)
            X_val_prepared = prepare_data_for_model(X_val_scaled, self.model_type)

            num_classes = len(np.unique(y))
            input_shape = X_train_prepared.shape[1:]
            model = self.model_builder(input_shape=input_shape, num_classes=num_classes, **builder_params)

            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

            model.fit(
                X_train_prepared, y_train,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=dict(enumerate(class_weights)),
                validation_data=(X_val_prepared, y_val),
                verbose=1
            )

            _, accuracy = model.evaluate(X_val_prepared, y_val, verbose=0)
            accuracies.append(accuracy)
            print(f"  Fold {fold + 1} Accuracy: {accuracy:.4f}")

        avg_accuracy = np.mean(accuracies)
        print(f"\n✅ K-Fold validation complete. Average Accuracy: {avg_accuracy:.4f}")
        return avg_accuracy, None

    def _train_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[Dict]]:
        """Private method for train-test split validation."""
        test_size = self.config.get("test_size", 0.2)
        stratify_opt = y if np.min(np.bincount(y)) >= 2 else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_opt
        )

        X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
        X_train_prepared = prepare_data_for_model(X_train_scaled, self.model_type)
        X_val_prepared = prepare_data_for_model(X_val_scaled, self.model_type)

        num_classes = len(np.unique(y))
        input_shape = X_train_prepared.shape[1:]
        builder_params = self._get_builder_params()
        model = self.model_builder(input_shape=input_shape, num_classes=num_classes, **builder_params)

        epochs = self.hyperparams['epochs'][0] if isinstance(self.hyperparams.get('epochs'), list) else self.hyperparams.get('epochs', 10)
        batch_size = self.hyperparams['batch_size'][0] if isinstance(self.hyperparams.get('batch_size'), list) else self.hyperparams.get('batch_size', 64)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        model.fit(
            X_train_prepared, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_prepared, y_val),
            class_weight=dict(enumerate(class_weights)),
            verbose=1
        )

        _, accuracy = model.evaluate(X_val_prepared, y_val, verbose=0)
        print(f"✅ Train-test split validation complete. Accuracy: {accuracy:.4f}")
        return accuracy, None

    def save_model(self, model: tf.keras.Model):
        """Saves the trained model to the path specified in the config."""
        output_path = self.config.get("output_path", "models/")
        os.makedirs(output_path, exist_ok=True)
        model_path = os.path.join(output_path, f"{self.model_type}_final_model.keras")
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")

    def save_training_log(self, log_data: Dict):
        """Saves a log file with training results."""
        output_path = self.config.get("output_path", "models/")
        os.makedirs(output_path, exist_ok=True)
        log_path = os.path.join(output_path, f"{self.model_type}_training_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"📋 Training log saved to: {log_path}")
