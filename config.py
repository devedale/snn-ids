# -*- coding: utf-8 -*-

# --- Global Configuration File for the Cybersecurity ML Pipeline ---

# --- Archetypal Comment (Module Level) ---
#
# Purpose & Context:
# This file serves as the central control panel for the entire machine learning
# pipeline. It decouples the core logic (preprocessing, training, prediction)
# from the operational parameters, allowing for systematic experimentation and
# benchmarking without code modification.
#
# Architectural Role:
# This module acts as a singleton configuration provider. All other modules
# import their required parameters from here, ensuring a single source of truth
# for the pipeline's behavior. This design promotes modularity and simplifies
# scenario management (e.g., switching from a smoke test to a full run).
#
# Configurability & Testing Notes:
# Parameters are grouped by their functional area (Data, Preprocessing, etc.).
# To test different scenarios, one can create multiple versions of this file or
# dynamically override its parameters in a notebook environment. The current
# settings are tuned for the CIC-IDS-2017/2018 dataset.
#
# ---

import os

# ==============================================================================
# SECTION 1: DATA SOURCE AND SCHEMA CONFIGURATION
# ==============================================================================
#
# Purpose: Defines the data source and the semantic mapping of its columns.
# Implications: This section is critical for data loading and feature
#               engineering. Any change here directly impacts how data is
#               interpreted by the preprocessing module.
#
DATA_CONFIG = {
    # --- Data Source Path ---
    # Rationale: Points to the directory containing the dataset's CSV files.
    #            Using a directory path is a scalable choice, as the preprocessing
    #            script is designed to read and concatenate multiple files,
    #            which is common for large, time-split datasets.
    # Assumption: All .csv files within this directory are part of the dataset
    #             and share a compatible schema.
    "dataset_path": "data/CSECICIDS2018_improved/",

    # --- Column Semantic Mapping ---
    # Rationale: Explicitly defines the roles of key columns. This avoids
    #            hardcoding column names in the core logic.
    "timestamp_column": "Timestamp",
    "target_column": "Label",
    "ip_columns_to_anonymize": ["Src IP", "Dst IP"],

    # --- Feature Selection ---
    # Rationale: A curated list of numeric features chosen for their relevance
    #            in network traffic analysis. This subset balances model
    #            complexity and performance. The list can be expanded or
    #            reduced for experimentation.
    # Alternative: One could programmatically select all numeric columns, but
    #              a curated list provides more control and avoids including
    #              irrelevant or noisy features.
    "feature_columns": [
        "Src Port", "Dst Port", "Protocol", "Flow Duration", "Total Fwd Packet",
        "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
        "Flow IAT Max", "Flow IAT Min", "Fwd IAT Mean", "Bwd IAT Mean",
        "Fwd Header Length", "Bwd Header Length", "Average Packet Size",
        "Fwd Segment Size Avg", "Bwd Segment Size Avg"
    ],
}

# ==============================================================================
# SECTION 2: PREPROCESSING CONFIGURATION
# ==============================================================================
#
# Purpose: Controls the data transformation process, particularly the
#          creation of sequential windows for time-series models.
#
PREPROCESSING_CONFIG = {
    # --- Time Windowing Strategy ---
    # Rationale: If True, the flat data is converted into a sequence of
    #            "windows" (e.g., 10 consecutive events), which is required
    #            input for sequential models like LSTMs or GRUs. If False,
    #            each event is treated as an independent sample.
    "use_time_windows": True,
    "window_size": 10,
    "step": 5,  # A step < window_size creates overlapping windows, augmenting the dataset.
}

# ==============================================================================
# SECTION 3: MODEL TRAINING CONFIGURATION
# ==============================================================================
#
# Purpose: Governs the model building, training, and evaluation process.
#
TRAINING_CONFIG = {
    # --- Artifact Storage ---
    # Rationale: Centralizes the storage of all training outputs (models,
    #            logs, anonymization maps) for organization and reproducibility.
    "output_path": "models/",

    # --- Validation Strategy ---
    # Rationale: 'train_test_split' is chosen for faster iteration on large
    #            datasets. 'k_fold' is more robust but computationally expensive.
    #            This parameter allows for easy switching between the two.
    "validation_strategy": "train_test_split",
    "test_size": 0.2,  # Used only if validation_strategy is 'train_test_split'
    "k_fold_splits": 5,      # Used only if validation_strategy is 'k_fold'

    # --- Model Architecture Selection ---
    # Rationale: Allows for dynamic selection of the model architecture. This
    #            is a key parameter for experimentation (e.g., comparing
    #            LSTM vs. GRU). The 'build_model' function in 'training.py'
    #            is responsible for interpreting this value.
    "model_type": "lstm",  # Supported options: 'lstm', 'gru', 'dense'

    # --- Hyperparameter Grid for Grid Search ---
    # Rationale: Defines the search space for hyperparameter tuning. The
    #            training script will iterate through all possible combinations.
    #            For quick tests, each list should contain only one value.
    "hyperparameters": {
        "activation": ["relu"],
        "batch_size": [64],
        "epochs": [20],
        "learning_rate": [0.001],
        "lstm_units": [64],  # Used if model_type is 'lstm'
        "gru_units": [64],   # Used if model_type is 'gru'
    }
}

# ==============================================================================
# SECTION 4: PREDICTION CONFIGURATION
# ==============================================================================
#
# Purpose: Defines the paths to the artifacts required for making predictions
#          on new, unseen data.
#
PREDICTION_CONFIG = {
    # --- Model Loading ---
    # Rationale: If a specific model file is provided, it will be used.
    #            If None, the system defaults to using the best model found
    #            during the last training run.
    "model_path": None,

    # --- Artifact Paths ---
    # Rationale: Specifies the locations of the mapping files needed to
    #            transform raw prediction data into the same format used for
    #            training (e.g., applying the same anonymization and column order).
    "target_anonymization_map_path": os.path.join(TRAINING_CONFIG["output_path"], "target_anonymization_map.json"),
    "ip_anonymization_map_path": os.path.join(TRAINING_CONFIG["output_path"], "ip_anonymization_map.json"),
    "column_order_path": os.path.join(TRAINING_CONFIG["output_path"], "column_order.json")
}
