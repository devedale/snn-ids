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

def load_and_balance_dataset(
    data_path: str,
    sample_size: Optional[int] = None,
    balance_strategy: str = "security",
    benign_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Carica e bilancia il dataset CIC-IDS.
    
    Args:
        data_path: Path al dataset
        sample_size: Numero di campioni totali
        balance_strategy: Strategia di bilanciamento
        benign_ratio: Ratio di traffico benigno
        
    Returns:
        DataFrame bilanciato
    """
    print(f"🔄 Caricamento dataset da: {data_path}")
    
    # Trova tutti i file CSV
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"Nessun file CSV trovato in {data_path}")
    
    print(f"📁 Trovati {len(csv_files)} file CSV")
    
    # Carica i dati con strategia per includere attacchi
    all_data = []
    target_per_file = (sample_size // len(csv_files)) if sample_size else None
    
    for i, file_path in enumerate(csv_files):
        print(f"  📄 File {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
        
        if target_per_file:
            # ⚡ STRATEGIA SUPER-EFFICIENTE: Filtra separatamente BENIGN e ATTACCHI
            print(f"    ⚡ Campionamento intelligente per bilanciare BENIGN/ATTACCHI...")
            
            # Carica tutto il file
            df_file = pd.read_csv(file_path)
            
            if 'Label' in df_file.columns:
                # Filtra separatamente (molto più veloce dell'ordinamento)
                benign_data = df_file[df_file['Label'] == 'BENIGN']
                attack_data = df_file[df_file['Label'] != 'BENIGN']
                
                # Calcola campioni desiderati (usando benign_ratio dalla config)
                benign_needed = int(target_per_file * benign_ratio)  # Usa parametro benign_ratio
                attack_needed = target_per_file - benign_needed  # Resto per attacchi
                
                print(f"    📊 Target: {target_per_file} campioni → {benign_needed} BENIGN + {attack_needed} ATTACCHI")
                
                # Prendi campioni con shuffle per diversità degli attacchi
                benign_sample = benign_data.head(benign_needed) if len(benign_data) > 0 else pd.DataFrame()
                
                # Shuffle attacchi per massima diversità
                if len(attack_data) > 0:
                    attack_sample = attack_data.sample(
                        n=min(attack_needed, len(attack_data)), 
                        random_state=42
                    )
                else:
                    attack_sample = pd.DataFrame()
                
                # Combina
                df_file = pd.concat([benign_sample, attack_sample], ignore_index=True)
                
                print(f"    📊 Selezionati: {len(benign_sample)} BENIGN + {len(attack_sample)} ATTACCHI")
            else:
                # Fallback se non c'è colonna Label
                df_file = df_file.head(target_per_file)
        else:
            df_file = pd.read_csv(file_path)
        
        all_data.append(df_file)
        
        # Debug attacchi trovati (mostra miglioramento)
        if 'Label' in df_file.columns:
            attacks = len(df_file[df_file['Label'] != 'BENIGN'])
            total_samples = len(df_file)
            attack_percentage = (attacks / total_samples * 100) if total_samples > 0 else 0
            if attacks > 0:
                print(f"    🎯 Trovati {attacks} attacchi su {total_samples} campioni ({attack_percentage:.1f}%)")
                # Mostra diversità dei tipi di attacco
                unique_attacks = df_file[df_file['Label'] != 'BENIGN']['Label'].unique()
                print(f"    🔍 Tipi di attacco: {len(unique_attacks)} diversi ({', '.join(unique_attacks[:3])}{'...' if len(unique_attacks) > 3 else ''})")
    else:
                print(f"    📊 Solo traffico BENIGN ({total_samples} campioni)")
    
    # Combina tutti i dati
    df = pd.concat(all_data, ignore_index=True)
    print(f"📊 Dataset combinato: {len(df)} righe")
    
    # Pulizia base
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Bilanciamento
    if balance_strategy == "security" and 'Label' in df.columns:
        df = _balance_cybersecurity(df, sample_size, benign_ratio)

    return df

def _balance_cybersecurity(df: pd.DataFrame, target_size: Optional[int], benign_ratio: float) -> pd.DataFrame:
    """Bilancia il dataset per cybersecurity (50% BENIGN, 50% ATTACCHI)."""
    print("🛡️ Applicazione bilanciamento cybersecurity...")
    
    # Separa traffico benigno e maligno
    benign_df = df[df['Label'] == 'BENIGN']
    malicious_df = df[df['Label'] != 'BENIGN']
    
    print(f"  📊 BENIGN: {len(benign_df):,} campioni")
    print(f"  🔴 ATTACCHI: {len(malicious_df):,} campioni")
    
    if len(malicious_df) == 0:
        print("  ⚠️ Nessun attacco trovato! Usando solo traffico BENIGN")
        return df.sample(n=min(target_size or len(df), len(df)))
    
    # Calcola dimensioni target
    if target_size:
        benign_target = int(target_size * benign_ratio)
        malicious_target = target_size - benign_target
    else:
        benign_target = len(benign_df)
        malicious_target = len(malicious_df)
    
    # Campiona traffico benigno
    if len(benign_df) >= benign_target:
        benign_sampled = benign_df.sample(n=benign_target, random_state=42)
    else:
        benign_sampled = benign_df
    
    # Campiona attacchi mantenendo diversità
    attack_types = malicious_df['Label'].value_counts()
    print(f"  🎯 Tipi di attacco trovati: {len(attack_types)}")
    
    malicious_sampled = []
    remaining_budget = malicious_target
    
    # Distribuzione equa tra tipi di attacco
    for attack_type, count in attack_types.items():
        if remaining_budget <= 0:
            break
        
        attack_data = malicious_df[malicious_df['Label'] == attack_type]
        samples_to_take = min(count, remaining_budget // max(1, len(attack_types)))
        
        if samples_to_take > 0:
            sampled = attack_data.sample(n=samples_to_take, random_state=42)
            malicious_sampled.append(sampled)
            remaining_budget -= samples_to_take
            print(f"    {attack_type}: {samples_to_take} campioni")
    
    # Combina risultati
    malicious_combined = pd.concat(malicious_sampled, ignore_index=True) if malicious_sampled else pd.DataFrame()
    balanced_df = pd.concat([benign_sampled, malicious_combined], ignore_index=True)
    
    print(f"✅ Dataset bilanciato: {len(balanced_df)} righe")
    print(f"  📊 BENIGN: {len(benign_sampled)} ({len(benign_sampled)/len(balanced_df)*100:.1f}%)")
    print(f"  🔴 ATTACCHI: {len(malicious_combined)} ({len(malicious_combined)/len(balanced_df)*100:.1f}%)")
    
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

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
    balance_strategy = balance_strategy or PREPROCESSING_CONFIG["balance_strategy"]
    
    print("🚀 Avvio pipeline preprocessing completa")
    print(f"📁 Dataset: {data_path}")
    print(f"📊 Sample size: {sample_size}")
    print(f"⚖️ Strategia: {balance_strategy}")
    
    # 1. Carica e bilancia dataset
    benign_ratio = PREPROCESSING_CONFIG.get("benign_ratio", 0.5)
    print(f"📊 Benign ratio: {benign_ratio} ({benign_ratio*100:.0f}% BENIGN, {(1-benign_ratio)*100:.0f}% ATTACCHI)")
    df = load_and_balance_dataset(data_path, sample_size, balance_strategy, benign_ratio)
    
    # 2. Preprocessa features
    df_processed, label_encoder = preprocess_features(df)
    
    # 3. Crea finestre temporali
    X, y = create_time_windows(df_processed)
    
    print(f"✅ Preprocessing completato!")
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")
    print(f"🏷️ Classi: {len(np.unique(y))}")
    
    return X, y, label_encoder
