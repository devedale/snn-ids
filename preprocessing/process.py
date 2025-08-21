# -*- coding: utf-8 -*-

# --- Data Preprocessing and Transformation Module ---

# --- Archetypal Comment (Module Level) ---
#
# Purpose & Context:
# This module is responsible for the entire data ingestion and transformation
# pipeline. Its primary role is to convert raw, multi-file CSV data into a
# clean, numerical, and optionally sequential format suitable for machine
# learning models. It handles data loading, cleaning, encoding, and normalization.
#
# Architectural Role:
# It serves as the first stage in the ML pipeline (Data Ingestion -> Preprocessing -> Training).
# It is designed to be highly configurable through the central `config.py` file,
# allowing it to adapt to different datasets and preprocessing strategies
# without altering its core source code.
#
# Key Operations:
# 1. Load & Concatenate: Reads multiple CSVs from a directory.
# 2. Clean: Handles missing or infinite values.
# 3. Encode: Converts categorical features (including IP addresses and labels)
#    into numerical representations.
# 4. Normalize: Scales numerical features to a standard range.
# 5. Sequence: Transforms the flat data into time-series windows for
#    recurrent neural networks (RNNs).
#
# ---

import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Dependency Injection via Configuration ---
# Rationale: Importing the config files directly makes this module dependent
#            on a specific file structure but centralizes control, which is a
#            valid trade-off for this project's scale.
# Alternative: A more decoupled architecture might use a dependency injection
#              framework or pass configuration objects as arguments.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PREPROCESSING_CONFIG, PREDICTION_CONFIG

def _save_json_map(data, path):
    """
    Saves a dictionary to a JSON file, creating the directory if it doesn't exist.
    This is a private utility function for this module.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Artifact saved: {path}")

def load_data_from_directory(path, sample_size=None):
    """
    --- Data Ingestion Sub-module ---

    Purpose: Loads all CSV files from a specified directory, concatenates them
             into a single pandas DataFrame, and optionally samples the result.

    Structural & Algorithmic Rationale:
    - Uses `glob` to dynamically find all files matching the *.csv pattern,
      making the process robust to the number of files.
    - Employs a list comprehension for loading and `pd.concat` for efficient
      merging of multiple DataFrames.

    Trade-offs & Assumptions:
    - Assumes all CSV files in the directory share an identical or highly
      compatible schema. Schema mismatches could lead to errors.
    - Loading all data into memory at once can be resource-intensive for
      extremely large datasets.
    - `sample_size` uses `df.head()`, which is a simple but potentially biased
      sampling method (it only takes the first N rows). For true exploratory
      analysis, random sampling (`df.sample()`) would be an alternative,
      but `head()` is deterministic and faster for quick tests.
    """
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        print(f"Warning: No CSV files found in '{path}'. Please check the path in config.py.")
        return pd.DataFrame()

    print(f"Found {len(all_files)} CSV files. Starting data loading...")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Data loading complete. Total rows: {len(df)}")

    if sample_size:
        print(f"Sampling data: The first {sample_size} rows will be used.")
        df = df.head(sample_size)

    return df

def full_preprocess(sample_size=None):
    """
    --- Main Preprocessing Pipeline ---

    Purpose: Orchestrates the end-to-end data transformation process.
    Args:
        sample_size (int, optional): Number of rows to use for a quick "smoke test" run.
                                     If None, the entire dataset is processed.

    Returns:
        A tuple (X, y) containing the processed features and labels, or (None, None) if an error occurs.
    """
    print("Starting advanced data preprocessing...")

    # --- Stage 1: Data Loading and Initial Cleaning ---
    df = load_data_from_directory(DATA_CONFIG["dataset_path"], sample_size)
    if df.empty:
        return None, None

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True) # A simple but robust strategy for this dataset.

    # --- Stage 2: Temporal Sorting ---
    # Rationale: Sorting by timestamp is crucial before creating time windows
    #            to ensure the sequential integrity of the data.
    df[DATA_CONFIG["timestamp_column"]] = pd.to_datetime(df[DATA_CONFIG["timestamp_column"]])
    df = df.sort_values(by=DATA_CONFIG["timestamp_column"]).reset_index(drop=True)

    # --- Stage 3: Categorical Feature Encoding ---
    # Rationale: Machine learning models require numerical input. LabelEncoding is used
    #            here to convert categorical identifiers (labels, IPs) into integers.
    #            The mappings are saved to JSON for use during prediction.
    target_encoder = LabelEncoder()
    df[DATA_CONFIG["target_column"]] = target_encoder.fit_transform(df[DATA_CONFIG["target_column"]])
    target_map = {
        "map": {label: int(code) for label, code in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))},
        "inverse_map": {str(code): label for code, label in enumerate(target_encoder.classes_)}
    }
    _save_json_map(target_map, PREDICTION_CONFIG["target_anonymization_map_path"])

    all_ips = pd.concat([df[col] for col in DATA_CONFIG["ip_columns_to_anonymize"]]).unique()
    ip_encoder = LabelEncoder().fit(all_ips)
    for col in DATA_CONFIG["ip_columns_to_anonymize"]:
        df[col] = ip_encoder.transform(df[col])
    ip_map = {
        "map": {ip: int(code) for ip, code in zip(ip_encoder.classes_, ip_encoder.transform(ip_encoder.classes_))},
        "inverse_map": {str(code): ip for code, ip in enumerate(ip_encoder.classes_)}
    }
    _save_json_map(ip_map, PREDICTION_CONFIG["ip_anonymization_map_path"])

    # --- Stage 4: One-Hot Encoding ---
    # Rationale: For non-ordinal categorical features, One-Hot Encoding prevents the
    #            model from assuming a false order.
    categorical_features = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns and df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_features)

    # --- Stage 5: Numerical Feature Scaling ---
    # Rationale: StandardScaler normalizes features to have zero mean and unit variance.
    #            This is critical for distance-based algorithms and helps gradient-based
    #            optimizers converge faster.
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != DATA_CONFIG["target_column"]]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # --- Stage 6: Final Feature Selection & Schema Saving ---
    # Rationale: The final order of columns is saved. This is crucial for prediction,
    #            to ensure the input for the model has the exact same schema as the training data.
    final_feature_columns = [col for col in df.columns if col not in [DATA_CONFIG["target_column"], DATA_CONFIG["timestamp_column"]]]
    _save_json_map(final_feature_columns, PREDICTION_CONFIG["column_order_path"])

    features_df = df[final_feature_columns].astype(np.float32)
    target_series = df[DATA_CONFIG["target_column"]]

    # --- Stage 7: Time Window Creation ---
    # Rationale: This block converts the data into sequences for RNNs.
    #            If disabled, it returns the data as-is for non-sequential models.
    if not PREPROCESSING_CONFIG["use_time_windows"]:
        print("Time windowing is disabled. Returning data as a flat sequence.")
        return features_df.values, target_series.values

    print(f"Creating time windows (size={PREPROCESSING_CONFIG['window_size']}, step={PREPROCESSING_CONFIG['step']})...")
    X, y = [], []
    window_size = PREPROCESSING_CONFIG['window_size']
    step = PREPROCESSING_CONFIG['step']
    for i in range(0, len(features_df) - window_size + 1, step):
        window = features_df.iloc[i : i + window_size].values
        label = target_series.iloc[i + window_size - 1]
        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Preprocessing complete. Output X shape: {X.shape}, Output y shape: {y.shape}")
    return X, y

if __name__ == '__main__':
    # --- Module Self-Test ---
    # Purpose: Provides a simple execution path to test the module's functionality
    #          independently, using a small sample size.
    print("--- SELF-TESTING PREPROCESSING MODULE WITH SAMPLING ---")
    X_processed, y_processed = full_preprocess(sample_size=10000)
    if X_processed is not None:
        print("\n--- Processed Data Sample ---")
        print("Shape of X:", X_processed.shape)
        print("Shape of y:", y_processed.shape)
        print("First sample of X:\n", X_processed[0])
        print("First label of y:", y_processed[0])
