# -*- coding: utf-8 -*-

"""
Modulo per il Preprocessing Avanzato dei Dati.
Refattorizzato per accettare override della configurazione.
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from copy import deepcopy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG as DC, PREPROCESSING_CONFIG as PC, PREDICTION_CONFIG as PredC
from crypto import cryptopan_ip
from .ip_processor import process_ip_columns, IPProcessor

def save_json_map(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Mappa salvata in: {path}")

def generate_cryptopan_key():
    """
    Genera una chiave segreta per Crypto-PAn.
    In produzione, questa dovrebbe essere gestita in modo sicuro.
    """
    import secrets
    return secrets.token_bytes(32)  # 256 bit per SHA-256

def balance_dataset(df, target_column, strategy='balanced', max_samples_per_class=None, min_samples_per_class=100, benign_ratio=0.5):
    """
    Bilanciamento del dataset per evitare problemi di classi sbilanciate.
    
    Args:
        df: DataFrame da bilanciare
        target_column: Nome della colonna target
        strategy: Strategia di bilanciamento ('balanced', 'undersample', 'oversample', 'smart', 'security')
        max_samples_per_class: Numero massimo di campioni per classe (None = automatico)
        min_samples_per_class: Numero minimo di campioni richiesti per classe
        benign_ratio: Ratio di traffico benigno (0.5 = 50% benigno, 50% malevolo)
    
    Returns:
        DataFrame bilanciato
    """
    print(f"🔍 Analisi distribuzione classi prima del bilanciamento:")
    class_counts = df[target_column].value_counts()
    print(class_counts)
    
    if strategy == 'security':
        # Strategia cybersecurity: 50% BENIGN, 50% MALICIOUS con attacchi variegati
        print(f"🛡️ Strategia cybersecurity: {benign_ratio*100:.0f}% BENIGN, {(1-benign_ratio)*100:.0f}% MALICIOUS (variegato)")
        
        # Separa BENIGN da tutte le altre classi (malicious)
        benign_df = df[df[target_column] == 'BENIGN']
        malicious_df = df[df[target_column] != 'BENIGN']
        
        print(f"  Traffico BENIGN: {len(benign_df):,} campioni")
        print(f"  Traffico MALICIOUS: {len(malicious_df):,} campioni")
        
        # Analizza la distribuzione degli attacchi
        attack_counts = malicious_df[target_column].value_counts()
        print(f"  Distribuzione attacchi originali:")
        for attack_type, count in attack_counts.items():
            print(f"    {attack_type}: {count:,} campioni")
        
        # Calcola il numero target di campioni totali
        # Assicurati che BENIGN sia sempre il 50% del dataset finale
        if max_samples_per_class:
            # Se max_samples_per_class è specificato, calcola in base ai dati disponibili
            max_possible_benign = len(benign_df)
            max_possible_malicious = len(malicious_df)
            # Il totale deve essere tale che BENIGN sia il 50%
            total_target_samples = min(max_samples_per_class, max_possible_benign * 2, max_possible_malicious * 2)
        else:
            # Calcola in base ai dati disponibili, mantenendo sempre 50% BENIGN
            max_possible_benign = len(benign_df)
            max_possible_malicious = len(malicious_df)
            # Il totale deve essere tale che BENIGN sia il 50%
            total_target_samples = min(max_possible_benign * 2, max_possible_malicious * 2)
        
        benign_target = int(total_target_samples * benign_ratio)
        malicious_target = total_target_samples - benign_target
        
        print(f"  Target totale dataset: {total_target_samples:,} campioni")
        print(f"  Target BENIGN: {benign_target:,} campioni (50%)")
        print(f"  Target MALICIOUS: {malicious_target:,} campioni (50%)")
        
        # BILANCIAMENTO INTELLIGENTE DEGLI ATTACCHI - SOLO DATI REALI
        # Obiettivo: massimizzare la varietà senza replicare artificialmente i dati
        
        # Ordina gli attacchi per cardinalità (dal meno popoloso al più popoloso)
        attack_counts_sorted = attack_counts.sort_values()
        print(f"  Attacchi ordinati per cardinalità:")
        for attack_type, count in attack_counts_sorted.items():
            print(f"    {attack_type}: {count:,} campioni")
        
        balanced_attacks = []
        remaining_budget = malicious_target
        
        # Prima fase: prendi tutti i campioni delle classi minoritarie
        print(f"  Prima fase: includere tutte le classi minoritarie")
        minority_classes = []
        for attack_type, count in attack_counts_sorted.items():
            attack_df = malicious_df[malicious_df[target_column] == attack_type]
            if count <= remaining_budget // len(attack_counts_sorted):
                # Include tutti i campioni della classe minoritaria
                balanced_attacks.append(attack_df)
                remaining_budget -= count
                minority_classes.append(attack_type)
                print(f"    ✅ {attack_type}: inclusi tutti i {count:,} campioni")
            else:
                break
        
        # Seconda fase: distribuire il budget rimanente tra le classi dominanti
        remaining_classes = [att for att in attack_counts_sorted.index if att not in minority_classes]
        print(f"  Seconda fase: distribuire budget rimanente tra classi dominanti")
        print(f"  Budget rimanente: {remaining_budget:,}, classi rimanenti: {len(remaining_classes)}")
        
        if remaining_classes and remaining_budget > 0:
            # Calcola quanto budget dare a ciascuna classe rimanente
            budget_per_remaining = remaining_budget // len(remaining_classes)
            extra_budget = remaining_budget % len(remaining_classes)
            
            for i, attack_type in enumerate(remaining_classes):
                attack_df = malicious_df[malicious_df[target_column] == attack_type]
                current_count = len(attack_df)
                
                # Assegna budget base + eventuale budget extra
                allocated_budget = budget_per_remaining
                if i < extra_budget:  # Distribuisce il resto ai primi
                    allocated_budget += 1
                
                # Non può superare il numero di campioni disponibili
                samples_to_take = min(allocated_budget, current_count)
                
                balanced_attack = attack_df.sample(n=samples_to_take, random_state=42)
                balanced_attacks.append(balanced_attack)
                print(f"    📊 {attack_type}: campionati {samples_to_take:,} su {current_count:,} disponibili")
        
        # Combina tutti gli attacchi bilanciati
        if balanced_attacks:
            malicious_balanced = pd.concat(balanced_attacks, ignore_index=True)
            actual_malicious_count = len(malicious_balanced)
            print(f"  Totale campioni malicious ottenuti: {actual_malicious_count:,}")
            
            # Campiona BENIGN per raggiungere il target del 50%
            if len(benign_df) >= benign_target:
                benign_balanced = benign_df.sample(n=benign_target, random_state=42)
                print(f"  📉 BENIGN: campionati {benign_target:,} su {len(benign_df):,} disponibili")
            else:
                # Se non abbiamo abbastanza BENIGN, usa tutto quello che abbiamo
                benign_balanced = benign_df
                print(f"  ⚠️ BENIGN: usando tutti i {len(benign_df):,} campioni disponibili")
            
            # Combina i dataset bilanciati
            balanced_df = pd.concat([benign_balanced, malicious_balanced], ignore_index=True)
        else:
            # Caso limite: nessun attacco trovato
            print(f"  ⚠️ Nessun attacco trovato nel dataset!")
            print(f"  🔄 Passando a strategia di fallback...")
            
            # Strategia di fallback: usa solo BENIGN ma avvisa
            balanced_df = benign_df
            print(f"  📊 Dataset finale: solo BENIGN ({len(benign_df):,} campioni)")
            print(f"  ⚠️ ATTENZIONE: Dataset con una sola classe - considerare un campione più grande")
        
    elif strategy == 'smart':
        # Strategia intelligente: combina undersampling e oversampling
        print(f"🧠 Strategia intelligente: bilanciamento adattivo...")
        
        # Trova la classe con il numero medio di campioni (escludendo le estreme)
        sorted_counts = class_counts.sort_values()
        if len(sorted_counts) >= 3:
            # Usa la classe mediana come riferimento
            median_class = sorted_counts.iloc[len(sorted_counts)//2]
            target_count = median_class
        else:
            # Se poche classi, usa la media
            target_count = int(class_counts.mean())
        
        # Assicurati che ogni classe abbia almeno min_samples_per_class
        target_count = max(target_count, min_samples_per_class)
        print(f"📊 Target per classe: {target_count} campioni")
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df[target_column] == class_name]
            current_count = len(class_df)
            
            if current_count < target_count:
                # Oversampling per classi minoritarie
                if current_count < min_samples_per_class:
                    # Per classi molto piccole, usa oversampling con ripetizione
                    class_df = resample(class_df, n_samples=target_count, random_state=42)
                    print(f"  📈 {class_name}: oversampling da {current_count} a {target_count}")
                else:
                    # Per classi moderate, usa oversampling intelligente
                    class_df = resample(class_df, n_samples=target_count, random_state=42)
                    print(f"  📈 {class_name}: oversampling da {current_count} a {target_count}")
            elif current_count > target_count:
                # Undersampling per classi dominanti
                class_df = class_df.sample(n=target_count, random_state=42)
                print(f"  📉 {class_name}: undersampling da {current_count} a {target_count}")
            else:
                print(f"  ✅ {class_name}: già bilanciata ({current_count})")
            
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'balanced':
        # Strategia bilanciata: undersampling della classe dominante
        # Ma assicurati che ogni classe abbia almeno min_samples_per_class
        min_class_count = max(class_counts.min(), min_samples_per_class)
        print(f"📊 Bilanciamento: limitando ogni classe a {min_class_count} campioni")
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df[target_column] == class_name]
            if len(class_df) > min_class_count:
                # Campiona casualmente per ridurre la classe dominante
                class_df = class_df.sample(n=min_class_count, random_state=42)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'undersample':
        # Undersampling: riduci tutte le classi alla dimensione della più piccola
        # Ma assicurati che ogni classe abbia almeno min_samples_per_class
        min_class_count = max(class_counts.min(), min_samples_per_class)
        print(f"📉 Undersampling: limitando ogni classe a {min_class_count} campioni")
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df[target_column] == class_name]
            class_df = class_df.sample(n=min_class_count, random_state=42)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'oversample':
        # Oversampling: aumenta le classi minoritarie
        max_class_count = class_counts.max()
        print(f"📈 Oversampling: aumentando ogni classe a {max_class_count} campioni")
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df[target_column] == class_name]
            if len(class_df) < max_class_count:
                # Ripeti i campioni per aumentare la classe minoritaria
                class_df = resample(class_df, n_samples=max_class_count, random_state=42)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    else:
        print(f"⚠️ Strategia '{strategy}' non riconosciuta, restituendo dataset originale")
        return df
    
    # Rimescola il dataset bilanciato
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Distribuzione classi dopo il bilanciamento:")
    print(balanced_df[target_column].value_counts())
    print(f"📊 Dataset bilanciato: {len(balanced_df)} righe totali")
    
    return balanced_df

def load_data_from_directory(path, sample_size=None, balance_strategy='balanced', **balance_kwargs):
    """
    Carica i dati da una directory di file CSV con opzione di bilanciamento.
    Se sample_size è specificato, carica in modo efficiente solo le prime N righe
    dal primo file per evitare un consumo eccessivo di memoria.
    """
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        print(f"Attenzione: Nessun file CSV trovato in '{path}'.")
        return pd.DataFrame()

    if sample_size:
        print(f"Modalità sample: lettura campione rappresentativo da {len(all_files)} file...")
        # Per includere attacchi, leggiamo strategicamente da diverse parti dei file
        df_list = []
        samples_per_file = sample_size // len(all_files)
        
        # STRATEGIA MIGLIORATA: Leggere più righe dalla fine per catturare attacchi
        min_tail_samples = max(5000, samples_per_file)  # Minimo 5k righe dalla fine
        
        for i, file_path in enumerate(all_files):
            print(f"  File {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
            
            # Leggi le prime righe (solitamente BENIGN)
            benign_samples = samples_per_file // 3  # 1/3 per BENIGN
            df_head = pd.read_csv(file_path, low_memory=False, nrows=benign_samples)
            df_list.append(df_head)
            
            # Leggi le ultime righe (dove sono gli attacchi) - campione più grande
            try:
                # Conta le righe totali
                with open(file_path, 'r') as f:
                    total_lines = sum(1 for line in f)
                
                # Leggi le ultime 10k righe o 2/3 del budget per questo file
                attack_samples = max(min_tail_samples, (samples_per_file * 2) // 3)
                
                if total_lines > attack_samples:
                    skip_rows = max(0, total_lines - attack_samples - 1)
                    df_tail = pd.read_csv(file_path, skiprows=range(1, skip_rows + 1))
                    df_list.append(df_tail)
                    print(f"    ✅ Incluse {len(df_head)} (inizio) + {len(df_tail)} (fine) righe")
                    
                    # Debug: controlla se abbiamo trovato attacchi
                    if 'Label' in df_tail.columns:
                        attack_count = len(df_tail[df_tail['Label'] != 'BENIGN'])
                        if attack_count > 0:
                            print(f"    🎯 Trovati {attack_count} attacchi in questo file!")
                else:
                    print(f"    ✅ Incluse {len(df_head)} righe (file piccolo)")
            except Exception as e:
                print(f"    ⚠️ Errore lettura coda file: {e}")
                print(f"    ✅ Incluse solo {len(df_head)} righe")
        
        df = pd.concat(df_list, ignore_index=True)
        print(f"  📊 Campione totale: {len(df)} righe")
        
    else:
        # Se non stiamo campionando, carichiamo tutti i file.
        print(f"Caricamento completo del dataset da {len(all_files)} file...")
        df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)

    # Applica bilanciamento se richiesto
    if balance_strategy != 'none':
        df = balance_dataset(df, 'Label', strategy=balance_strategy, **balance_kwargs)
    
    return df

def preprocess_data(config_override=None):
    """
    Esegue il preprocessing completo. Accetta un dizionario per sovrascrivere
    le configurazioni di default al volo.
    """
    # Unisci le configurazioni di default con l'override
    data_config = deepcopy(DC)
    proc_config = deepcopy(PC)
    pred_config = deepcopy(PredC)
    if config_override:
        data_config.update(config_override.get("DATA_CONFIG", {}))
        proc_config.update(config_override.get("PREPROCESSING_CONFIG", {}))

    sample_size = proc_config.get("sample_size")
    balance_strategy = proc_config.get("balance_strategy", "balanced")
    max_samples_per_class = proc_config.get("max_samples_per_class")
    min_samples_per_class = proc_config.get("min_samples_per_class", 100)
    benign_ratio = proc_config.get("benign_ratio", 0.5)
    
    print(f"--- Inizio Preprocessing (Sample Size: {sample_size or 'Completo'}) ---")
    print(f"--- Strategia di bilanciamento: {balance_strategy} ---")
    if balance_strategy == "security":
        print(f"--- Ratio BENIGN/MALICIOUS: {benign_ratio*100:.0f}%/{(1-benign_ratio)*100:.0f}% ---")

    df = load_data_from_directory(
        data_config["dataset_path"], 
        sample_size, 
        balance_strategy=balance_strategy,
        max_samples_per_class=max_samples_per_class,
        min_samples_per_class=min_samples_per_class,
        benign_ratio=benign_ratio
    )
    if df.empty:
        return None, None

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df[data_config["timestamp_column"]] = pd.to_datetime(df[data_config["timestamp_column"]])
    df = df.sort_values(by=data_config["timestamp_column"]).reset_index(drop=True)

    target_encoder = LabelEncoder()
    df[data_config["target_column"]] = target_encoder.fit_transform(df[data_config["target_column"]])
    target_map = {
        "map": {label: int(code) for label, code in zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))},
        "inverse_map": {str(code): label for code, label in enumerate(target_encoder.classes_)}
    }
    save_json_map(target_map, pred_config["target_anonymization_map_path"])

    # Genera chiave segreta per Crypto-PAn
    cryptopan_key = generate_cryptopan_key()
    print(f"Chiave Crypto-PAn generata: {cryptopan_key.hex()}")
    
    # Anonimizza gli IP usando Crypto-PAn
    all_ips = pd.concat([df[col] for col in data_config["ip_columns_to_anonymize"]]).unique()
    
    # Crea mappa di anonimizzazione Crypto-PAn
    ip_map = {
        "cryptopan_key": cryptopan_key.hex(),  # Salva la chiave per la decrittografia
        "map": {},  # IP originale -> IP anonimizzato
        "inverse_map": {}  # IP anonimizzato -> IP originale
    }
    
    # Applica Crypto-PAn a tutti gli IP unici
    for original_ip in all_ips:
        if pd.notna(original_ip) and str(original_ip).strip():
            anonymized_ip = cryptopan_ip(str(original_ip), cryptopan_key)
            ip_map["map"][str(original_ip)] = anonymized_ip
            ip_map["inverse_map"][anonymized_ip] = str(original_ip)
    
    # Applica l'anonimizzazione alle colonne
    for col in data_config["ip_columns_to_anonymize"]:
        df[col] = df[col].apply(lambda ip: ip_map["map"].get(str(ip), ip) if pd.notna(ip) else ip)
    
    save_json_map(ip_map, pred_config["ip_anonymization_map_path"])
    
    # 🔄 TRASFORMAZIONE IP IN OTTETTI
    print("🔄 Trasformazione indirizzi IP in ottetti...")
    df = process_ip_columns(
        df, 
        ip_columns=data_config["ip_columns_to_anonymize"],
        create_new_columns=True,  # Crea nuove colonne per ogni ottetto
        drop_original=False       # Mantieni colonne originali per compatibilità
    )
    print("✅ Trasformazione IP completata!")

    categorical_features = [col for col in data_config["feature_columns"] if col in df.columns and df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_features)

    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != data_config["target_column"]]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    final_feature_columns = numeric_cols
    save_json_map(final_feature_columns, pred_config["column_order_path"])

    features_df = df[final_feature_columns].astype(np.float32)
    target_series = df[data_config["target_column"]]

    if not proc_config["use_time_windows"]:
        return features_df.values, target_series.values

    X, y = [], []
    window_size = proc_config['window_size']
    step = proc_config['step']
    for i in range(0, len(features_df) - window_size + 1, step):
        window = features_df.iloc[i : i + window_size].values
        label = target_series.iloc[i + window_size - 1]
        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Preprocessing completato. Shape di X: {X.shape}, Shape di y: {y.shape}")
    return X, y
