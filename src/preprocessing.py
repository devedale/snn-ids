# -*- coding: utf-8 -*-
"""
Unified Preprocessing for SNN-IDS
This module handles the complete preprocessing pipeline for CIC-IDS datasets,
including dataset caching, balanced sampling, and time-window generation.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from typing import Tuple, Optional, Dict, Any
import pickle
import json
import multiprocessing

# Add project root to path to allow importing 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, PREPROCESSING_CONFIG, TRAINING_CONFIG, RANDOM_CONFIG

def _process_file_chunk(args: Tuple[str, str, bool]) -> bool:
    """
    Helper function to process a single source CSV file in a separate process.
    It creates a dedicated cache sub-folder for the file and splits the records
    into 'benign_records.csv' and 'attack_records.csv'.

    Args:
        args: A tuple containing (file_path, cache_dir, delete_source).

    Returns:
        True if processing was successful, False otherwise.
    """
    file_path, cache_dir, delete_source = args
    file_name = os.path.basename(file_path)
    # Each source CSV gets its own cache directory to avoid file name collisions
    # and to keep the data sources separated.
    file_cache_dir = os.path.join(cache_dir, os.path.splitext(file_name)[0])

    try:
        os.makedirs(file_cache_dir, exist_ok=True)

        benign_cache_path = os.path.join(file_cache_dir, "benign_records.csv")
        attack_cache_path = os.path.join(file_cache_dir, "attack_records.csv")

        # If the cache for this specific file already exists, we can skip it.
        if os.path.exists(benign_cache_path) and os.path.exists(attack_cache_path):
            print(f"  ✅ Cache for {file_name} already exists. Skipping.")
            return True

        header_written_benign = False
        header_written_attack = False

        # Read the CSV in chunks to handle very large files without high memory usage.
        chunk_iter = pd.read_csv(file_path, chunksize=100000, low_memory=False)

        for chunk in chunk_iter:
            chunk.columns = chunk.columns.str.strip()

            if 'Label' not in chunk.columns:
                continue

            # Clean up infinite values and NaNs which can cause errors in processing.
            chunk = chunk.replace([np.inf, -np.inf], np.nan).fillna(0)

            benign_df = chunk[chunk['Label'] == 'BENIGN']
            attack_df = chunk[chunk['Label'] != 'BENIGN']

            # Append to cache files, writing the header only for the first chunk.
            if not benign_df.empty:
                benign_df.to_csv(benign_cache_path, mode='a', header=not header_written_benign, index=False)
                header_written_benign = True

            if not attack_df.empty:
                attack_df.to_csv(attack_cache_path, mode='a', header=not header_written_attack, index=False)
                header_written_attack = True

        if delete_source:
            os.remove(file_path)
            print(f"  🗑️ Source file removed: {file_name}")

        return True

    except Exception as e:
        print(f"  ⚠️ Error processing {file_name}: {e}")
        # Clean up partial files in case of an error to prevent corrupted cache.
        if os.path.exists(benign_cache_path): os.remove(benign_cache_path)
        if os.path.exists(attack_cache_path): os.remove(attack_cache_path)
        return False

def _initialize_dataset_cache(data_path: str, cache_dir: str):
    """
    Scans for source CSV files and parallelizes the creation of the cache.
    Each CSV file is processed into its own dedicated cache subdirectory.
    This step is crucial for performance, as it avoids reprocessing the raw data on every run.
    """
    print("🗂️  Initializing dataset cache (folder-based structure)...")
    os.makedirs(cache_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")

    # Check if all source files already have a corresponding cache directory.
    all_cached = all(
        os.path.exists(os.path.join(cache_dir, os.path.splitext(os.path.basename(f))[0])) for f in csv_files
    )
    if all_cached:
        print("✅ Cache is already complete for all source files. Skipping creation.")
        return

    print(f"⏳ Parallelizing caching for {len(csv_files)} files...")

    num_processes = PREPROCESSING_CONFIG.get("parallel_processes", 1)
    delete_source = PREPROCESSING_CONFIG.get("delete_processed_files", False)

    pool_args = [(file_path, cache_dir, delete_source) for file_path in csv_files]

    # Use a process pool to parallelize the work.
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_process_file_chunk, pool_args)

    successful_files = sum(1 for r in results if r)
    print(f"✅ Caching complete. {successful_files}/{len(csv_files)} files processed successfully.")

def load_and_balance_dataset(
    cache_dir: str,
    sample_size: int,
    benign_ratio: float
) -> pd.DataFrame:
    """
    Loads a balanced sample by aggregating data from all cache sub-folders.
    This function contains sophisticated sampling logic to ensure that rare attack
    labels are preserved in the final dataset, which is critical for training a robust model.
    """
    print("🔄 Aggregating and balancing dataset from the folder-based cache...")

    benign_files = glob.glob(os.path.join(cache_dir, "*", "benign_records.csv"))
    attack_files = glob.glob(os.path.join(cache_dir, "*", "attack_records.csv"))

    if not benign_files or not attack_files:
        raise FileNotFoundError("Cache files not found. Run _initialize_dataset_cache first.")

    def _sample_attack_files(attack_files: list, target_samples: int) -> pd.DataFrame:
        """
        Samples from attack files with a strategy to preserve rare labels.
        This is done by first identifying rare attacks (those with few samples globally)
        and then ensuring all instances of these rare attacks are included in the sample.
        The remainder of the quota is filled with common attacks.
        """
        if not attack_files or target_samples <= 0:
            return pd.DataFrame()

        print("    🎯 Attack sampling strategy: Preserve rare labels")

        # STAGE 1: Global analysis of attack label counts
        print("    🔍 Analyzing global label counts...")
        global_attack_counts = {}
        for file_path in attack_files:
            try:
                df = pd.read_csv(file_path, usecols=['Label'], low_memory=False)
                counts = df['Label'].value_counts().to_dict()
                for label, count in counts.items():
                    global_attack_counts[label] = global_attack_counts.get(label, 0) + count
            except Exception as e:
                print(f"    ⚠️ Error analyzing {file_path}: {e}")
                continue

        if not global_attack_counts:
            print("    ❌ No attack types found.")
            return pd.DataFrame()

        print(f"    🌐 Found {len(global_attack_counts)} unique attack types.")

        # STAGE 2: Identify rare labels based on a threshold
        samples_per_file = max(1, target_samples // len(attack_files))
        rare_threshold = samples_per_file * 0.8  # 80% of the per-file quota

        rare_attacks = {k: v for k, v in global_attack_counts.items() if v < rare_threshold}
        common_attacks = {k: v for k, v in global_attack_counts.items() if v >= rare_threshold}

        print(f"    🔴 Rare labels (< {rare_threshold:.0f} records): {len(rare_attacks)}")
        print(f"    🟢 Common labels (>= {rare_threshold:.0f} records): {len(common_attacks)}")

        # STAGE 3: Priority-based sampling
        sampled_chunks = []
        for file_path in attack_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                if 'Label' not in df.columns or df.empty:
                    continue

                file_samples = []
                # PRIORITY 1: Take all records of rare labels
                rare_df = df[df['Label'].isin(rare_attacks.keys())]
                if not rare_df.empty:
                    file_samples.append(rare_df)

                # PRIORITY 2: Fill the rest of the quota with common labels
                remaining_quota = max(0, samples_per_file - len(rare_df))
                if remaining_quota > 0:
                    common_df = df[df['Label'].isin(common_attacks.keys())]
                    if not common_df.empty:
                        file_samples.append(common_df.sample(n=min(remaining_quota, len(common_df)), random_state=RANDOM_CONFIG.get('seed', 42)))

                if file_samples:
                    sampled_chunks.append(pd.concat(file_samples, ignore_index=True))
            except Exception as e:
                print(f"    ⚠️ Error sampling from {file_path}: {e}")

        if not sampled_chunks:
            return pd.DataFrame()

        final_df = pd.concat(sampled_chunks, ignore_index=True).drop_duplicates().reset_index(drop=True)
        print(f"    🎯 Attack sampling result: {len(final_df)} total records.")
        return final_df

    def _sample_benign_files(benign_files: list, target_samples: int) -> pd.DataFrame:
        """Samples from benign files using simple uniform random sampling."""
        if not benign_files or target_samples <= 0:
            return pd.DataFrame()

        print("    🎯 Benign sampling strategy: Uniform random sampling")
        samples_per_file = max(1, target_samples // len(benign_files))
        sampled_chunks = []

        for file_path in benign_files:
            try:
                # Efficiently sample from large CSV without reading the whole file into memory
                df_sampled = _sample_from_csv(file_path, samples_per_file)
                if not df_sampled.empty:
                    sampled_chunks.append(df_sampled)
                    print(f"    ✅ Sampled {len(df_sampled)} benign records from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"    ⚠️ Error sampling from {file_path}: {e}")

        if not sampled_chunks:
            return pd.DataFrame()

        return pd.concat(sampled_chunks, ignore_index=True)

    print(f"  📂 Found {len(benign_files)} benign files and {len(attack_files)} attack files.")

    benign_needed = int(sample_size * benign_ratio)
    attack_needed = sample_size - benign_needed

    print(f"  🎯 Smart sampling: {benign_needed} benign + {attack_needed} attack records")

    benign_df = _sample_benign_files(benign_files, benign_needed)
    attack_df = _sample_attack_files(attack_files, attack_needed)

    if benign_df.empty or attack_df.empty:
        raise ValueError("One of the datasets (BENIGN or ATTACK) is empty after sampling.")

    print(f"  📊 Selected: {len(benign_df)} BENIGN + {len(attack_df)} ATTACK records")

    balanced_df = pd.concat([benign_df, attack_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_CONFIG.get('seed', 42)).reset_index(drop=True)

    # Ensure all classes have at least `k_fold_splits` samples for StratifiedKFold
    required_per_class = max(2, TRAINING_CONFIG.get("k_fold_splits", 5))
    counts = balanced_df['Label'].value_counts()
    rare_labels = counts[counts < required_per_class].index.tolist()
    if rare_labels:
        print(f"  ⚠️ Classes with fewer than {required_per_class} samples found: {len(rare_labels)}. Starting targeted oversampling...")
        oversampled_chunks = [balanced_df]
        for lbl in rare_labels:
            cur = balanced_df[balanced_df['Label'] == lbl]
            need = required_per_class - len(cur)
            if len(cur) > 0:
                dup = cur.sample(n=need, replace=True, random_state=RANDOM_CONFIG.get('seed', 42))
                oversampled_chunks.append(dup)
        balanced_df = pd.concat(oversampled_chunks, ignore_index=True).sample(frac=1.0, random_state=RANDOM_CONFIG.get('seed', 42)).reset_index(drop=True)
        print(f"  ✅ Oversampling complete. New dataset size: {len(balanced_df)}")

    print(f"✅ Balanced dataset loaded: {len(balanced_df)} rows")
    return balanced_df

def _sample_from_csv(file_path: str, n_samples: int) -> pd.DataFrame:
    """Efficiently samples n_samples rows from a CSV file."""
    if n_samples == 0:
        return pd.DataFrame()

    try:
        with open(file_path, 'r') as f:
            num_lines = sum(1 for line in f) - 1
        if num_lines <= 0:
            return pd.DataFrame()
        if n_samples >= num_lines:
            return pd.read_csv(file_path)

        skip_rows = sorted(np.random.choice(range(1, num_lines + 1), num_lines - n_samples, replace=False))
        return pd.read_csv(file_path, skiprows=skip_rows)
    except Exception as e:
        print(f"Error during sampling from {file_path}: {e}")
        return pd.DataFrame()

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Preprocesses the features of the dataset.
    This includes IP-to-octet conversion and label encoding.
    Normalization and scaling are deferred to the training phase to prevent data leakage.
    """
    print("🔄 Preprocessing features...")
    df_processed = df.copy()

    if PREPROCESSING_CONFIG["convert_ip_to_octets"]:
        df_processed = _convert_ip_to_octets(df_processed)

    label_encoder = LabelEncoder()
    if 'Label' in df_processed.columns:
        df_processed['Label_Encoded'] = label_encoder.fit_transform(df_processed['Label'])

        os.makedirs(TRAINING_CONFIG["output_path"], exist_ok=True)
        mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
        with open(os.path.join(TRAINING_CONFIG["output_path"], "label_mapping.json"), 'w') as f:
            json.dump(mapping, f, indent=2)

    feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df_processed.columns]
    for col in feature_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

    print(f"✅ Features processed: {len(feature_cols)} columns selected.")
    return df_processed, label_encoder

def _convert_ip_to_octets(df: pd.DataFrame) -> pd.DataFrame:
    """Converts IP address strings into four separate octet columns with optional Crypto-PAn anonymization."""
    print("🌐 Converting IP addresses to octets...")
    
    # Initialize Crypto-PAn if enabled
    cryptopan = None
    if PREPROCESSING_CONFIG.get("use_cryptopan", False):
        print("🔐 Crypto-PAn IP anonymization enabled")
        cryptopan = CryptoPan()
    
    for ip_col in DATA_CONFIG["ip_columns"]:
        if ip_col in df.columns:
            if cryptopan:
                # Anonymize IPs before conversion
                print(f"  🔐 Anonymizing {ip_col} with Crypto-PAn...")
                original_ips = df[ip_col].unique()
                print(f"     Processing {len(original_ips)} unique IP addresses...")
                
                # Apply anonymization
                df[ip_col] = df[ip_col].apply(cryptopan.anonymize_ip)
                
                print(f"     ✅ Anonymized {len(original_ips)} unique IPs")
            
            # Convert to octets (anonymized or original IPs)
            octets = df[ip_col].str.split('.', expand=True, n=4)
            for i in range(4):
                octet_col_name = f"{ip_col}_Octet_{i+1}"
                df[octet_col_name] = pd.to_numeric(octets[i], errors='coerce').fillna(0).astype(int)
            
            print(f"  ✅ {ip_col} -> 4 octet columns")
    
    # Save anonymization mapping if Crypto-PAn was used
    if cryptopan:
        output_dir = TRAINING_CONFIG.get("output_path", "output")
        cryptopan.save_mapping_to_file(output_dir)
        
        # Print statistics
        stats = cryptopan.get_stats()
        print(f"📊 Crypto-PAn Statistics:")
        print(f"   Total IPs processed: {stats['total_ips_processed']}")
        print(f"   Unique IPs cached: {stats['unique_ips_cached']}")
        print(f"   Key hash: {stats['key_hash']}")
    
    return df
def _ip_to_octet(ip_str: str, octet_index: int) -> int:
    """Helper to extract a specific octet from an IP string."""
    try:
        parts = str(ip_str).strip().split('.')
        if len(parts) == 4 and octet_index < 4:
            return int(parts[octet_index])
    except:
        pass
    return 0

def create_time_windows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates time windows (sequences) for recurrent models like GRU and LSTM.
    If `use_time_windows` is False in the config, it returns a 2D array.
    """
    if not PREPROCESSING_CONFIG["use_time_windows"]:
        feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns]
        X = df[feature_cols].values
        y = df['Label_Encoded'].values if 'Label_Encoded' in df.columns else np.zeros(len(df))
        return X, y

    print("⏱️  Creating time windows for sequential models...")

    if DATA_CONFIG["timestamp_column"] in df.columns:
        df = df.sort_values(DATA_CONFIG["timestamp_column"]).reset_index(drop=True)

    window_size = PREPROCESSING_CONFIG["window_size"]
    step = PREPROCESSING_CONFIG["step"]

    feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns]
    X_windows, y_windows = [], []

    for i in range(0, len(df) - window_size + 1, step):
        window_data = df.iloc[i:i + window_size]
        X_window = window_data[feature_cols].values
        # The label for a window is the label of its last element.
        y_label = window_data['Label_Encoded'].iloc[-1]
        X_windows.append(X_window)
        y_windows.append(y_label)

    X = np.array(X_windows)
    y = np.array(y_windows)

    print(f"✅ Time windows created. Shape: {X.shape}")
    return X, y

def preprocess_pipeline(
    data_path: str = None,
    sample_size: int = None
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    The main, complete preprocessing pipeline.
    It orchestrates caching, loading, balancing, feature processing, and windowing.
    It also implements a "model-ready" cache to save the final (X, y) arrays,
    allowing subsequent runs to skip all preprocessing entirely.
    """
    data_path = data_path or DATA_CONFIG["dataset_path"]
    sample_size = sample_size or PREPROCESSING_CONFIG["sample_size"]

    print("🚀 Starting complete preprocessing pipeline")
    print(f"📁 Source: {data_path}")
    print(f"📊 Sample size: {sample_size}")

    # STAGE 0: Check for a model-ready cache first for maximum speed.
    if PREPROCESSING_CONFIG.get("model_cache_enabled", False):
        cache_dir_model = PREPROCESSING_CONFIG.get("model_cache_dir", "model_cache")
        os.makedirs(cache_dir_model, exist_ok=True)
        # The cache key is deterministic based on the main processing parameters.
        feature_count = len(DATA_CONFIG.get('feature_columns', []))
        win_size = PREPROCESSING_CONFIG.get('window_size', 0) if PREPROCESSING_CONFIG.get('use_time_windows') else 0
        win_step = PREPROCESSING_CONFIG.get('step', 0) if PREPROCESSING_CONFIG.get('use_time_windows') else 0
        cache_key = f"size{sample_size}_win{win_size}_step{win_step}_features{feature_count}"

        npz_path = os.path.join(cache_dir_model, f"model_ready_{cache_key}.npz")
        le_path = os.path.join(cache_dir_model, f"label_encoder_{cache_key}.pkl")

        if os.path.exists(npz_path) and os.path.exists(le_path):
            print(f"📦 Found model-ready cache: {npz_path}. Loading final arrays...")
            data = np.load(npz_path)
            with open(le_path, 'rb') as f:
                label_encoder = pickle.load(f)
            X, y = data['X'], data['y']
            print("✅ Preprocessing complete (loaded from model-ready cache)!")
            print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
            return X, y, label_encoder

    # STAGE 1: Initialize raw data cache if model-ready cache wasn't found.
    if PREPROCESSING_CONFIG.get("cache_enabled", False):
        cache_dir = PREPROCESSING_CONFIG["cache_dir"]
        _initialize_dataset_cache(data_path, cache_dir)
        df = load_and_balance_dataset(
            cache_dir=cache_dir,
            sample_size=sample_size,
            benign_ratio=PREPROCESSING_CONFIG.get("benign_ratio", 0.5)
        )
    else:
        raise NotImplementedError("Non-cached mode is not supported. Please enable cache in config.py.")

    # STAGE 2: Process features and create labels.
    df_processed, label_encoder = preprocess_features(df)

    # STAGE 3: Create time windows for sequential models.
    X, y = create_time_windows(df_processed)

    print("✅ Preprocessing complete!")
    print(f"📊 Final X shape: {X.shape}, y shape: {y.shape}")
    if y.size > 0:
        print(f"🏷️ Unique classes found: {len(np.unique(y))}")

    # STAGE 4: Save to model-ready cache for future runs.
    if PREPROCESSING_CONFIG.get("model_cache_enabled", False):
        print(f"💾 Saving to model-ready cache for future use...")
        try:
            np.savez_compressed(npz_path, X=X, y=y)
            with open(le_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            print(f"💾 Model-ready cache saved: {npz_path}")
        except Exception as e:
            print(f"⚠️ Could not save model-ready cache: {e}")

    return X, y, label_encoder
