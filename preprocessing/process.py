# -*- coding: utf-8 -*-
"""
Preprocessing Unificato SNN-IDS
Sistema completo per preprocessare dati CIC-IDS con bilanciamento e finestre temporali.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from typing import Tuple, Optional, Dict, Any
import json

# Aggiungi path per import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, PREPROCESSING_CONFIG, TRAINING_CONFIG

def _initialize_dataset_cache(data_path: str, cache_dir: str):
    """
    Scansiona i CSV sorgente e crea file di cache separati per BENIGN e ATTACK.
    Ottimizzato per file di grandi dimensioni utilizzando la lettura a blocchi.
    """
    print("🗂️ Inizializzazione cache dataset...")
    os.makedirs(cache_dir, exist_ok=True)

    benign_cache_path = os.path.join(cache_dir, "benign_records.csv")
    attack_cache_path = os.path.join(cache_dir, "attack_records.csv")

    if os.path.exists(benign_cache_path) and os.path.exists(attack_cache_path):
        print("✅ Cache già esistente. Salto la creazione.")
        return

    print(f"⏳ Creazione cache in {cache_dir}. Potrebbe richiedere tempo...")

    # Rimuovi file parziali se esistono
    if os.path.exists(benign_cache_path): os.remove(benign_cache_path)
    if os.path.exists(attack_cache_path): os.remove(attack_cache_path)

    header_written_benign = False
    header_written_attack = False
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"Nessun file CSV trovato in {data_path}")

    for i, file_path in enumerate(csv_files):
        print(f"  📄 Processo file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
        try:
            # Leggi in blocchi per gestire file di grandi dimensioni
            chunk_iter = pd.read_csv(file_path, chunksize=100000, low_memory=False)
            
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.str.strip()
                
                if 'Label' not in chunk.columns:
                    continue

                # Pulisci e separa
                chunk = chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                benign_df = chunk[chunk['Label'] == 'BENIGN']
                attack_df = chunk[chunk['Label'] != 'BENIGN']

                # Accoda alla cache
                if not benign_df.empty:
                    benign_df.to_csv(benign_cache_path, mode='a', header=not header_written_benign, index=False)
                    header_written_benign = True
                
                if not attack_df.empty:
                    attack_df.to_csv(attack_cache_path, mode='a', header=not header_written_attack, index=False)
                    header_written_attack = True
        except Exception as e:
            print(f"  ⚠️ Errore durante la lettura di {file_path}: {e}")
            continue

    print("✅ Cache creata con successo.")

def load_and_balance_dataset(
    cache_dir: str,
    sample_size: int,
    benign_ratio: float
) -> pd.DataFrame:
    """
    Carica un campione bilanciato di dati dai file di cache.
    Ottimizzato per leggere un numero casuale di righe da file di grandi dimensioni.
    """
    print("🔄 Caricamento e bilanciamento del dataset dalla cache...")
    benign_cache_path = os.path.join(cache_dir, "benign_records.csv")
    attack_cache_path = os.path.join(cache_dir, "attack_records.csv")

    if not os.path.exists(benign_cache_path) or not os.path.exists(attack_cache_path):
        raise FileNotFoundError("File di cache non trovati. Eseguire prima _initialize_dataset_cache.")

    # Calcola il numero di campioni necessari da ogni file
    benign_needed = int(sample_size * benign_ratio)
    attack_needed = sample_size - benign_needed

    # Campiona in modo efficiente da file di grandi dimensioni
    print(f"   sampling {benign_needed} benign records...")
    benign_df = _sample_from_csv(benign_cache_path, benign_needed)
    
    print(f"  sampling {attack_needed} attack records...")
    attack_df = _sample_from_csv(attack_cache_path, attack_needed)
    
    print(f"  📊 Selezionati: {len(benign_df)} BENIGN + {len(attack_df)} ATTACCHI")

    # Combina e mescola
    balanced_df = pd.concat([benign_df, attack_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Garanzia minima per classe per StratifiedKFold
    required_per_class = max(2, TRAINING_CONFIG.get("k_fold_splits", 5))
    counts = balanced_df['Label'].value_counts()
    rare_labels = counts[counts < required_per_class].index.tolist()
    if rare_labels:
        print(f"  ⚠️ Classi con meno di {required_per_class} campioni: {len(rare_labels)}. Avvio oversampling mirato…")
        oversampled_chunks = [balanced_df]
        for lbl in rare_labels:
            cur = balanced_df[balanced_df['Label'] == lbl]
            need = required_per_class - len(cur)
            if len(cur) > 0:
                dup = cur.sample(n=need, replace=True, random_state=42)
                oversampled_chunks.append(dup)
        balanced_df = pd.concat(oversampled_chunks, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"  ✅ Oversampling completato. Nuova dimensione: {len(balanced_df)}")

    print(f"✅ Dataset bilanciato caricato: {len(balanced_df)} righe")
    return balanced_df

def _sample_from_csv(file_path: str, n_samples: int) -> pd.DataFrame:
    """Campiona n_samples righe da un file CSV in modo efficiente."""
    if n_samples == 0:
        return pd.DataFrame()
    
    try:
        # Conta le righe totali (escludendo l'intestazione)
        with open(file_path, 'r') as f:
            num_lines = sum(1 for line in f) - 1

        if num_lines <= 0:
            return pd.DataFrame()

        # Se vogliamo più campioni di quelli disponibili, prendiamoli tutti
        if n_samples >= num_lines:
            return pd.read_csv(file_path)

        # Genera indici di righe casuali da saltare
        skip_rows = sorted(np.random.choice(range(1, num_lines + 1), num_lines - n_samples, replace=False))

        return pd.read_csv(file_path, skiprows=skip_rows)

    except Exception as e:
        print(f"Errore durante il campionamento da {file_path}: {e}")
        return pd.DataFrame()

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Preprocessa le features del dataset.
    
    Args:
        df: Dataset raw
        
    Returns:
        DataFrame processato e label encoder
    """
    print("🔄 Preprocessing features...")
    
    df_processed = df.copy()
    
    # 1. Trasforma IP in ottetti
    if PREPROCESSING_CONFIG["convert_ip_to_octets"]:
        df_processed = _convert_ip_to_octets(df_processed)
    
    # 2. Encoding delle etichette
    label_encoder = LabelEncoder()
    if 'Label' in df_processed.columns:
        df_processed['Label_Encoded'] = label_encoder.fit_transform(df_processed['Label'])
        
        # Salva mapping
        os.makedirs(TRAINING_CONFIG["output_path"], exist_ok=True)
        mapping = {
            str(i): label for i, label in enumerate(label_encoder.classes_)
        }
        with open(os.path.join(TRAINING_CONFIG["output_path"], "label_mapping.json"), 'w') as f:
            json.dump(mapping, f, indent=2)
    
    # 3. Selezione features (normalizzazione spostata al training per evitare leakage)
    feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df_processed.columns]
    
    # Converti in numerico
    for col in feature_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    print(f"✅ Features processate: {len(feature_cols)} colonne")
    return df_processed, label_encoder

def _convert_ip_to_octets(df: pd.DataFrame) -> pd.DataFrame:
    """Converte indirizzi IP in ottetti separati."""
    print("🌐 Conversione IP in ottetti...")
    
    for ip_col in DATA_CONFIG["ip_columns"]:
        if ip_col in df.columns:
            # Converti IP in ottetti
            for i in range(4):
                df[f"{ip_col}_Octet_{i+1}"] = df[ip_col].apply(
                    lambda ip: _ip_to_octet(ip, i) if pd.notna(ip) else 0
                )
            print(f"  ✅ {ip_col} -> 4 ottetti")
    
    return df

def _ip_to_octet(ip_str: str, octet_index: int) -> int:
    """Estrae un ottetto specifico da un IP."""
    try:
        parts = str(ip_str).strip().split('.')
        if len(parts) == 4 and octet_index < 4:
            return int(parts[octet_index])
    except:
        pass
    return 0

def create_time_windows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea finestre temporali per modelli sequenziali.
    
    Args:
        df: DataFrame preprocessato
        
    Returns:
        Arrays X (3D) e y per training
    """
    if not PREPROCESSING_CONFIG["use_time_windows"]:
        # Modalità senza finestre temporali
        feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns]
        X = df[feature_cols].values
        y = df['Label_Encoded'].values if 'Label_Encoded' in df.columns else np.zeros(len(df))
        return X, y
    
    print("⏱️ Creazione finestre temporali...")
    
    # Ordina per timestamp
    if DATA_CONFIG["timestamp_column"] in df.columns:
        df = df.sort_values(DATA_CONFIG["timestamp_column"]).reset_index(drop=True)
    
    # Parametri finestre
    window_size = PREPROCESSING_CONFIG["window_size"]
    step = PREPROCESSING_CONFIG["step"]
    
    # Features e target
    feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns]
    X_windows, y_windows = [], []
    
    # Crea finestre scorrevoli
    for i in range(0, len(df) - window_size + 1, step):
        window_data = df.iloc[i:i + window_size]
        
        # Features della finestra
        X_window = window_data[feature_cols].values
        
        # Etichetta (dell'ultimo elemento della finestra)
        y_label = window_data['Label_Encoded'].iloc[-1] if 'Label_Encoded' in window_data.columns else 0
        
        X_windows.append(X_window)
        y_windows.append(y_label)
    
    X = np.array(X_windows)
    y = np.array(y_windows)
    
    print(f"✅ Finestre create: {X.shape}")
    return X, y

def preprocess_pipeline(
    data_path: str = None,
    sample_size: int = None,
    balance_strategy: str = None
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Pipeline completa di preprocessing.
    
    Args:
        data_path: Path ai dati (default da config)
        sample_size: Dimensione campione (default da config)
        balance_strategy: Strategia bilanciamento (default da config)
        
    Returns:
        X, y, label_encoder
    """
    # Usa valori di default dalla config
    data_path = data_path or DATA_CONFIG["dataset_path"]
    sample_size = sample_size or PREPROCESSING_CONFIG["sample_size"]
    
    print("🚀 Avvio pipeline preprocessing completa")
    print(f"📁 Dataset: {data_path}")
    print(f"📊 Sample size: {sample_size}")

    # 1. Inizializza la cache se abilitata
    if PREPROCESSING_CONFIG.get("cache_enabled", False):
        cache_dir = PREPROCESSING_CONFIG["cache_dir"]
        _initialize_dataset_cache(data_path, cache_dir)
        # Carica i dati dalla cache
        df = load_and_balance_dataset(
            cache_dir=cache_dir,
            sample_size=sample_size,
            benign_ratio=PREPROCESSING_CONFIG.get("benign_ratio", 0.5)
        )
    else:
        # TODO: implementare una logica di caricamento non basata su cache se necessario
        raise NotImplementedError("La modalità non-cache non è più supportata. Abilitare la cache in config.py.")

    # 2. Preprocessa features
    df_processed, label_encoder = preprocess_features(df)
    
    # 3. Crea finestre temporali
    X, y = create_time_windows(df_processed)
    
    print(f"✅ Preprocessing completato!")
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")
    if y.size > 0:
        print(f"🏷️ Classi: {len(np.unique(y))}")
    
    return X, y, label_encoder
