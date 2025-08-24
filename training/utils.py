# -*- coding: utf-8 -*-
"""
Utility per training: combinazioni iperparametri, scaling 2D/3D.
"""

import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler


def param_grid(hyperparams: dict) -> list:
    keys = list(hyperparams.keys())
    values = []
    for k in keys:
        v = hyperparams[k]
        if isinstance(v, (list, tuple)):
            values.append(list(v))
        else:
            values.append([v])
    combos = list(itertools.product(*values))
    return [dict(zip(keys, c)) for c in combos]


def scale_features(X_train: np.ndarray, X_val: np.ndarray | None = None):
    is_sequence = len(X_train.shape) == 3
    if is_sequence:
        n_features = X_train.shape[2]
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, n_features)
        scaler.fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_val_scaled = None
        if X_val is not None:
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
        return X_train_scaled, X_val_scaled
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None
        return X_train_scaled, X_val_scaled


