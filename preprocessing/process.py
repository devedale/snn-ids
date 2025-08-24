# -*- coding: utf-8 -*-
"""
Preprocessing Unificato SNN-IDS
Sistema completo per preprocessare dati CIC-IDS con bilanciamento e finestre temporali.
"""

import os
import sys
import glob
import json
import hashlib
import pickle
from datetime import timedelta
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

# Aggiungi path per import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, PREPROCESSING_CONFIG, TRAINING_CONFIG
from preprocessing.utils import (
    ensure_timestamp as utils_ensure_timestamp,
    sessionize_flows as utils_sessionize_flows,
    compute_base_cache_dir as utils_compute_base_cache_dir,
    compute_windows_cache_dir as utils_compute_windows_cache_dir,
    aggregate_features as utils_aggregate_features,
)


# ==============================================================================
# Utility di base
# ==============================================================================

def _is_malicious(label: str) -> bool:
    benign = DATA_CONFIG.get("benign_label", "BENIGN")
    return pd.notna(label) and str(label) != str(benign)


def _json_hash(data: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = str(data).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


def _compute_base_cache_dir(data_path: str, sample_size: int, balance_strategy: str, benign_ratio: float) -> str:
    return utils_compute_base_cache_dir(
        data_path=data_path,
        sample_size=sample_size,
        balance_strategy=balance_strategy,
        benign_ratio=benign_ratio,
        feature_columns=DATA_CONFIG.get("feature_columns", []),
        convert_ip_to_octets=PREPROCESSING_CONFIG.get("convert_ip_to_octets", True),
        cache_root=PREPROCESSING_CONFIG.get("cache_dir", "preprocessed_cache"),
        target_type=TRAINING_CONFIG.get("target_type")
    )


def _compute_windows_cache_dir(base_dir: str) -> str:
    params = {
        "use_time_windows": PREPROCESSING_CONFIG.get("use_time_windows", True),
        "window_size": PREPROCESSING_CONFIG.get("window_size"),
        "step": PREPROCESSING_CONFIG.get("step"),
        "flow_window_strategy": PREPROCESSING_CONFIG.get("flow_window_strategy"),
        "before": PREPROCESSING_CONFIG.get("window_before_first_malicious_s"),
        "after": PREPROCESSING_CONFIG.get("window_after_first_malicious_s"),
        "bin": PREPROCESSING_CONFIG.get("time_bin_seconds"),
        "output_mode": PREPROCESSING_CONFIG.get("output_mode"),
        "aggregation_stats": PREPROCESSING_CONFIG.get("aggregation_stats")
    }
    return utils_compute_windows_cache_dir(base_dir, params)


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    return utils_ensure_timestamp(df, DATA_CONFIG["timestamp_column"], DATA_CONFIG.get("timestamp_format"))


def _sessionize_flows(df: pd.DataFrame) -> pd.DataFrame:
    flow_col = DATA_CONFIG.get("flow_id_column", "Flow_ID")
    if flow_col not in df.columns:
        for cand in ["Flow_ID", "Flow ID", "FlowID"]:
            if cand in df.columns:
                flow_col = cand
                break
    return utils_sessionize_flows(
        _ensure_timestamp(df.copy()),
        flow_col=flow_col,
        ts_col=DATA_CONFIG["timestamp_column"],
        timeout_seconds=int(PREPROCESSING_CONFIG.get("session_timeout_seconds", 60))
    )


def _apply_noise_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica filtri di rumore sulle etichette a livello temporale per sessione.
    Supporta majority voting su finestre e temporal smoothing (media mobile).
    """
    cfg = PREPROCESSING_CONFIG.get("label_propagation", {}).get("noise_filter", {})
    if not cfg or not cfg.get("enabled", False):
        return df

    ts_col = DATA_CONFIG["timestamp_column"]
    flow_session_col = "Session_ID"
    if flow_session_col not in df.columns:
        return df
    if ts_col not in df.columns:
        return df

    df = df.copy()
    df = _ensure_timestamp(df)
    df.sort_values([flow_session_col, ts_col], inplace=True)
    benign = DATA_CONFIG.get("benign_label", "BENIGN")
    df["_is_mal"] = df[DATA_CONFIG["target_column"]].apply(_is_malicious).astype(int)

    method = cfg.get("method", "temporal_smoothing")
    if method == "majority":
        win = int(cfg.get("window", 3))
        thr = float(cfg.get("threshold", 0.5))
        df["_maj"] = (
            df.groupby(flow_session_col)["_is_mal"].rolling(window=win, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["_is_mal"] = (df["_maj"] >= thr).astype(int)
        df.drop(columns=["_maj"], inplace=True)
    else:  # temporal_smoothing (exponential)
        alpha = float(PREPROCESSING_CONFIG.get("label_propagation", {}).get("smoothing_alpha", 0.6))
        smoothed = []
        for _, g in df.groupby(flow_session_col):
            s = g["_is_mal"].astype(float)
            ema = s.ewm(alpha=alpha, adjust=False).mean()
            smoothed.append(ema)
        df["_is_mal"] = pd.concat(smoothed).sort_index()
        df["_is_mal"] = (df["_is_mal"] >= 0.5).astype(int)

    # Non sovrascrivere le label multiclass: crea una colonna binaria separata
    # Usa np.where per evitare mismatch di shape
    is_binary = (TRAINING_CONFIG.get("target_type", "multiclass") == "binary")
    df["Label_Binary"] = np.where(df["_is_mal"] == 1, "MALICIOUS", benign)
    df.drop(columns=["_is_mal"], inplace=True)
    return df


# ==============================================================================
# CLI di preprocessing (solo comandi, per notebook senza python inline)
# ==============================================================================
def main():
    """Esegue l'intera pipeline usando i parametri da config."""
    data_path = DATA_CONFIG["dataset_path"]
    X, y, _ = preprocess_pipeline(data_path=data_path)
    # Salva su disco per consumo da training
    out_dir = os.path.join(TRAINING_CONFIG["output_path"], "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    if bool(TRAINING_CONFIG.get("preprocess_per_epoch", False)) and len(X) > 0:
        flows_per_epoch = int(TRAINING_CONFIG.get("flows_per_epoch", 5000))
        total = len(X)
        n_shards = max(1, (total + flows_per_epoch - 1) // flows_per_epoch)
        print(f"🔁 Modalità per-epoca attiva: suddivido {total} campioni in {n_shards} shard da ~{flows_per_epoch}")
        for k in range(n_shards):
            s = k * flows_per_epoch
            e = min((k + 1) * flows_per_epoch, total)
            Xk = X[s:e]
            yk = y[s:e]
            np.save(os.path.join(out_dir, f"X_epoch_{k:03d}.npy"), Xk)
            np.save(os.path.join(out_dir, f"y_epoch_{k:03d}.npy"), yk)
        print(f"💾 Salvati shard per-epoca in {out_dir}")
    else:
        np.save(os.path.join(out_dir, "X.npy"), X)
        np.save(os.path.join(out_dir, "y.npy"), y)
        print(f"💾 Salvati X.npy e y.npy in {out_dir}")


 

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
            # Traccia provenienza record e indice originale
            df_file['_source_file'] = os.path.basename(file_path)
            df_file['_row_id'] = np.arange(len(df_file))
            df_file = _ensure_timestamp(df_file)
            
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
            # Traccia provenienza record e indice originale
            df_file['_source_file'] = os.path.basename(file_path)
            df_file['_row_id'] = np.arange(len(df_file))
            df_file = _ensure_timestamp(df_file)
        
        all_data.append(df_file)
        
        # Debug attacchi trovati (mostra miglioramento)
        if 'Label' in df_file.columns:
            total_samples = len(df_file)
            attacks = int((df_file['Label'] != 'BENIGN').sum())
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
    
    # Verifica colonna target
    if DATA_CONFIG["target_column"] not in df.columns:
        missing_cols = [DATA_CONFIG["target_column"]]
        raise ValueError(f"Colonna target mancante nel dataset: {missing_cols}. Verifica il path dei dati o il formato dei CSV.")

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

    # Garanzia minima per classe: necessario per StratifiedKFold
    # Richiede almeno n_splits campioni per ciascuna classe; duplichiamo con replace se serve (oversampling mirato)
    required_per_class = max(2, TRAINING_CONFIG.get("k_fold_splits", 5))
    counts = balanced_df['Label'].value_counts()
    rare_labels = counts[counts < required_per_class].index.tolist()
    if rare_labels:
        print(f"  ⚠️ Classi con meno di {required_per_class} campioni: {len(rare_labels)}. Avvio oversampling mirato…")
        oversampled_chunks = [balanced_df]
        rng = np.random.RandomState(42)
        for lbl in rare_labels:
            cur = balanced_df[balanced_df['Label'] == lbl]
            need = required_per_class - len(cur)
            if len(cur) == 0:
                continue
            dup = cur.sample(n=need, replace=True, random_state=42)
            oversampled_chunks.append(dup)
        balanced_df = pd.concat(oversampled_chunks, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"  ✅ Oversampling completato. Nuova dimensione: {len(balanced_df)}")
    
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
    
    # 0. Rimuovi Flow_ID se presente (utilizzato solo per sessionizzazione)
    flow_col = DATA_CONFIG.get("flow_id_column", "Flow_ID")
    flow_candidates = [flow_col, "Flow_ID", "Flow ID", "FlowID"]
    to_drop = [c for c in flow_candidates if c in df_processed.columns]
    if to_drop:
        print(f"🗑️ Rimozione campi flow id {to_drop} (utilizzati solo per sessionizzazione)")
        df_processed = df_processed.drop(columns=to_drop)
    
    # 1. Trasforma IP in ottetti
    if PREPROCESSING_CONFIG["convert_ip_to_octets"]:
        df_processed = _convert_ip_to_octets(df_processed)
    
    # 2. Encoding delle etichette
    label_encoder = LabelEncoder()
    # Se target binario, usa Label_Binary (creata da _apply_noise_filter); altrimenti usa Label multiclasse originale
    target_col_for_encoding = 'Label_Binary' if TRAINING_CONFIG.get('target_type', 'multiclass') == 'binary' and 'Label_Binary' in df_processed.columns else 'Label'
    if target_col_for_encoding in df_processed.columns:
        df_processed['Label_Encoded'] = label_encoder.fit_transform(df_processed[target_col_for_encoding])
        
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

def create_time_windows(df: pd.DataFrame, windows_cache_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea finestre temporali per modelli sequenziali oppure aggregati MLP.
    Supporta strategia "first_malicious_context" con N secondi prima e T dopo
    il primo evento malevolo nel flusso/sessione, e label propagation configurabile.
    """
    # Modalità senza finestre temporali: output 2D per MLP basato su record
    if not PREPROCESSING_CONFIG["use_time_windows"]:
        feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns]
        X = df[feature_cols].values
        y = df['Label_Encoded'].values if 'Label_Encoded' in df.columns else np.zeros(len(df))
        # Salva CSV semplice aggregato se richiesto
        if windows_cache_dir:
            try:
                os.makedirs(windows_cache_dir, exist_ok=True)
                out_csv = os.path.join(windows_cache_dir, "windows_summary.csv")
                meta_df = pd.DataFrame({
                    "Session_ID": df.get("Session_ID", pd.Series([None]*len(df))).values,
                    "ts_min": pd.to_datetime(df.get(DATA_CONFIG["timestamp_column"], pd.Series([None]*len(df)))).values,
                    "ts_max": pd.to_datetime(df.get(DATA_CONFIG["timestamp_column"], pd.Series([None]*len(df)))).values,
                    "label": df.get(DATA_CONFIG["target_column"], pd.Series([None]*len(df))).values
                })
                meta_df.to_csv(out_csv, index=False)
            except Exception:
                pass
        return X, y

    print("⏱️ Creazione finestre temporali / contesti N/T...")

    df = _ensure_timestamp(df)
    # Sessionizza solo se manca Session_ID
    if "Session_ID" not in df.columns:
        df = _sessionize_flows(df)
    # Se ancora assente, crea un fallback: una sessione per record
    if "Session_ID" not in df.columns:
        df = df.copy()
        df["Session_ID"] = df.index.astype(str)
        print("⚠️ Session_ID assente nei dati: creato fallback una-sessione-per-record")
    # Applica noise filter solo se abbiamo sessioni
    if "Session_ID" in df.columns:
        df = _apply_noise_filter(df)

    ts_col = DATA_CONFIG["timestamp_column"]
    tgt_col = DATA_CONFIG["target_column"]
    session_col = "Session_ID"
    if session_col not in df.columns:
        raise KeyError("Session_ID")
    benign_label = DATA_CONFIG.get("benign_label", "BENIGN")

    feature_cols = [col for col in DATA_CONFIG["feature_columns"] if col in df.columns]
    out_mode = PREPROCESSING_CONFIG.get("output_mode", "sequence")

    # Se manca il timestamp, effettua aggregazione per sessione e ritorna
    if ts_col not in df.columns:
        print("⚠️ Timestamp assente nei dati: uso aggregazione per sessione (no finestre)")
        if not feature_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = set(["Label_Encoded", tgt_col, "Session_Index"]) | set(
                [c for c in df.columns if c.endswith("_Octet_1") or c.endswith("_Octet_2") or c.endswith("_Octet_3") or c.endswith("_Octet_4")]
            )
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        X_list = []
        y_list = []
        for session_id, g in df.groupby(session_col):
            agg_row = _aggregate_features(g, feature_cols) if feature_cols else np.zeros(0)
            X_list.append(agg_row)
            y_list.append(int(np.mean([_is_malicious(l) for l in g[tgt_col]]) >= 0.5))
        X = np.vstack(X_list) if X_list else np.zeros((0, len(feature_cols) * len(PREPROCESSING_CONFIG.get("aggregation_stats", []))))
        y = np.array(y_list, dtype=int)
        print(f"✅ Aggregazioni per sessione create: X={X.shape}, y={y.shape}")
        return X, y

    strategy = PREPROCESSING_CONFIG.get("flow_window_strategy", "first_malicious_context")
    before_s = PREPROCESSING_CONFIG.get("window_before_first_malicious_s", 30)
    after_s = PREPROCESSING_CONFIG.get("window_after_first_malicious_s", 30)
    bin_s = PREPROCESSING_CONFIG.get("time_bin_seconds", 5)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows = []
    bins_rows = []

    def _label_window(labels: pd.Series) -> int:
        mode = PREPROCESSING_CONFIG.get("label_propagation", {}).get("mode", "majority")
        if mode == "any":
            return int(any(_is_malicious(l) for l in labels))
        if mode == "probabilistic":
            thr = float(PREPROCESSING_CONFIG.get("label_propagation", {}).get("prob_threshold", 0.5))
            p = np.mean([_is_malicious(l) for l in labels])
            return int(p >= thr)
        if mode == "smoothing":
            # smoothing già applicato; usa majority sul risultato filtrato
            return int(np.mean([_is_malicious(l) for l in labels]) >= 0.5)
        # default majority
        return int(np.mean([_is_malicious(l) for l in labels]) >= 0.5)

    for session_id, g in df.groupby(session_col):
        g = g.sort_values(ts_col)
        # trova primo evento malevolo
        mal_idx = np.where(g[tgt_col].apply(_is_malicious).values)[0]
        if strategy == "first_malicious_context" and len(mal_idx) > 0:
            first_mal_ts = g.iloc[mal_idx[0]][ts_col]
            last_mal_ts = g.iloc[mal_idx[-1]][ts_col]
            start_ts = first_mal_ts - pd.Timedelta(seconds=before_s)
            # Fine finestra riferita all'ULTIMO evento malevolo
            end_ts = last_mal_ts + pd.Timedelta(seconds=after_s)
            win_df = g[(g[ts_col] >= start_ts) & (g[ts_col] <= end_ts)]
        else:
            win_df = g

        if win_df.empty:
            continue

        # bin temporali per sequenze
        if out_mode == "sequence":
            win_df = win_df.copy()
            base_ts = win_df[ts_col].min()
            win_df["_bin"] = ((win_df[ts_col] - base_ts).dt.total_seconds() // bin_s).astype(int)
            bins = []
            labels = []
            for b, bg in win_df.groupby("_bin"):
                agg = _aggregate_features(bg, feature_cols)
                bins.append(agg)
                labels.append(bg[tgt_col])
                # Riga per ogni bin (solo metadati, non features)
                bins_rows.append({
                    "Session_ID": session_id,
                    "bin_index": int(b),
                    "bin_start": (base_ts + pd.Timedelta(seconds=int(b)*bin_s)),
                    "bin_end": (base_ts + pd.Timedelta(seconds=(int(b)+1)*bin_s)),
                    "records": int(len(bg)),
                    "label_bin_majority": int(np.mean([_is_malicious(l) for l in bg[tgt_col]]) >= 0.5)
                })
            X_seq = np.stack(bins, axis=0)
            y_win = _label_window(pd.concat(labels))
            X_list.append(X_seq)
            y_list.append(y_win)
            # Riga meta per finestra/sessione
            # Calcola label multiclass dominante nella finestra (se presente)
            win_labels = win_df[tgt_col]
            label_dom = (
                win_labels[win_labels != DATA_CONFIG.get("benign_label", "BENIGN")].mode().iloc[0]
                if any(win_labels != DATA_CONFIG.get("benign_label", "BENIGN")) else DATA_CONFIG.get("benign_label", "BENIGN")
            )
            meta_rows.append({
                "Session_ID": session_id,
                "ts_start": win_df[ts_col].min(),
                "ts_end": win_df[ts_col].max(),
                "n_bins": int(X_seq.shape[0]),
                "label_window": int(y_win),
                "label_dominant": str(label_dom),
                "records": int(len(win_df)),
                "flow_id": str(session_id.split('#')[0]) if isinstance(session_id, str) and '#' in session_id else str(session_id)
            })
        else:  # mlp_aggregated
            agg_row = _aggregate_features(win_df, feature_cols)
            X_list.append(agg_row)
            y_list.append(_label_window(win_df[tgt_col]))
            win_labels = win_df[tgt_col]
            label_dom = (
                win_labels[win_labels != DATA_CONFIG.get("benign_label", "BENIGN")].mode().iloc[0]
                if any(win_labels != DATA_CONFIG.get("benign_label", "BENIGN")) else DATA_CONFIG.get("benign_label", "BENIGN")
            )
            meta_rows.append({
                "Session_ID": session_id,
                "ts_start": win_df[ts_col].min(),
                "ts_end": win_df[ts_col].max(),
                "n_bins": 1,
                "label_window": int(meta_rows[-1]["label_window"]) if meta_rows else int(_label_window(win_df[tgt_col])),
                "label_dominant": str(label_dom),
                "records": int(len(win_df)),
                "flow_id": str(session_id.split('#')[0]) if isinstance(session_id, str) and '#' in session_id else str(session_id)
            })

    if out_mode == "sequence":
        # pad/truncate sequenze alla max length per batch training semplice
        max_len = max(x.shape[0] for x in X_list) if X_list else 0
        feature_dim = X_list[0].shape[1] if X_list else 0
        X = np.zeros((len(X_list), max_len, feature_dim), dtype=float)
        for i, x in enumerate(X_list):
            L = x.shape[0]
            X[i, :L, :] = x
    else:
        X = np.vstack(X_list) if X_list else np.zeros((0, len(feature_cols) * len(PREPROCESSING_CONFIG.get("aggregation_stats", []))))

    y = np.array(y_list, dtype=int)
    # Salva CSV di metadati nelle cache se richiesto
    if windows_cache_dir and meta_rows:
        try:
            os.makedirs(windows_cache_dir, exist_ok=True)
            pd.DataFrame(meta_rows).to_csv(os.path.join(windows_cache_dir, "windows_summary.csv"), index=False)
            if out_mode == "sequence" and bins_rows:
                pd.DataFrame(bins_rows).to_csv(os.path.join(windows_cache_dir, "windows_bins.csv"), index=False)
        except Exception:
            pass
    print(f"✅ Finestre/aggregazioni create: X={X.shape}, y={y.shape}")
    return X, y


def _aggregate_features(df_window: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    stats = PREPROCESSING_CONFIG.get("aggregation_stats", ["sum", "mean", "std", "min", "max"])
    return utils_aggregate_features(df_window, feature_cols, stats)

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
    
    # 0. Cache: directory base e windows
    cache_enabled = bool(PREPROCESSING_CONFIG.get("cache_enabled", False))
    benign_ratio = PREPROCESSING_CONFIG.get("benign_ratio", 0.5)
    base_cache_dir = _compute_base_cache_dir(data_path, sample_size, balance_strategy, benign_ratio) if cache_enabled else None
    windows_cache_dir = _compute_windows_cache_dir(base_cache_dir) if cache_enabled else None

    # Prova a caricare X,y dalla cache se disponibili
    if cache_enabled and windows_cache_dir and os.path.exists(os.path.join(windows_cache_dir, "X.npy")) and os.path.exists(os.path.join(windows_cache_dir, "y.npy")):
        print(f"🔁 Cache hit: caricamento X,y da {windows_cache_dir}")
        X = np.load(os.path.join(windows_cache_dir, "X.npy"), allow_pickle=True)
        y = np.load(os.path.join(windows_cache_dir, "y.npy"), allow_pickle=True)
        # Label encoder
        le_path = os.path.join(base_cache_dir, "label_mapping.json") if base_cache_dir else None
        label_encoder = LabelEncoder()
        if le_path and os.path.exists(le_path):
            with open(le_path, 'r') as f:
                mapping = json.load(f)
            classes_ = [mapping[str(i)] for i in range(len(mapping))]
            label_encoder.fit(classes_)
        else:
            # fallback
            label_encoder.fit([DATA_CONFIG.get("benign_label", "BENIGN")])
        print("✅ Preprocessing completato (from cache)!")
        print(f"📊 X shape: {X.shape}")
        print(f"📊 y shape: {y.shape}")
        if y.size > 0:
            print(f"🏷️ Classi: {len(np.unique(y))}")
        return X, y, label_encoder

    # 1. Carica e bilancia dataset (record-level rapido)
    print(f"📊 Benign ratio: {benign_ratio} ({benign_ratio*100:.0f}% BENIGN, {(1-benign_ratio)*100:.0f}% ATTACCHI)")
    # Se esiste base_df in cache, caricalo
    if cache_enabled and base_cache_dir and os.path.exists(os.path.join(base_cache_dir, "base_df.pkl")):
        print(f"🔁 Cache hit base_df: {base_cache_dir}")
        with open(os.path.join(base_cache_dir, "base_df.pkl"), 'rb') as f:
            df = pickle.load(f)
        # Carica anche label mapping
        label_encoder = LabelEncoder()
        le_path = os.path.join(base_cache_dir, "label_mapping.json")
        if os.path.exists(le_path):
            with open(le_path, 'r') as f:
                mapping = json.load(f)
            classes_ = [mapping[str(i)] for i in range(len(mapping))]
            label_encoder.fit(classes_)
        else:
            label_encoder.fit([DATA_CONFIG.get("benign_label", "BENIGN")])
    else:
        df = load_and_balance_dataset(data_path, sample_size, balance_strategy, benign_ratio)
    
    # 2. Reassembly + sessionizzazione + filtri rumore
    df = _sessionize_flows(df)
    df = _apply_noise_filter(df)

    # 3. Preprocessa features
    df_processed, label_encoder = preprocess_features(df)
    # Salva base_df in cache
    if cache_enabled and base_cache_dir:
        os.makedirs(base_cache_dir, exist_ok=True)
        with open(os.path.join(base_cache_dir, "base_df.pkl"), 'wb') as f:
            pickle.dump(df_processed, f)
        # Salva label mapping se disponibile
        lm_path = os.path.join(base_cache_dir, "label_mapping.json")
        if os.path.exists(os.path.join(TRAINING_CONFIG["output_path"], "label_mapping.json")):
            import shutil
            shutil.copy(os.path.join(TRAINING_CONFIG["output_path"], "label_mapping.json"), lm_path)
    
    # 4. Bilanciamento a livello di flusso/sessione se configurato
    df_processed = _balance_flows(df_processed)

    # 5. Crea finestre temporali / aggregazioni
    X, y = create_time_windows(df_processed, windows_cache_dir=windows_cache_dir)
    # Salva X,y in cache
    if cache_enabled and windows_cache_dir:
        os.makedirs(windows_cache_dir, exist_ok=True)
        np.save(os.path.join(windows_cache_dir, "X.npy"), X)
        np.save(os.path.join(windows_cache_dir, "y.npy"), y)
    
    print(f"✅ Preprocessing completato!")
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")
    if y.size > 0:
        print(f"🏷️ Classi: {len(np.unique(y))}")
    
    return X, y, label_encoder


def _balance_flows(df: pd.DataFrame) -> pd.DataFrame:
    cfg = PREPROCESSING_CONFIG.get("flow_balance", {"enabled": False})
    if not cfg.get("enabled", False):
        return df

    method = cfg.get("method", "undersample")
    ratio = float(cfg.get("ratio", 1.0))
    session_col = "Session_ID"
    if session_col not in df.columns:
        return df
    tgt_col = DATA_CONFIG["target_column"]
    benign_label = DATA_CONFIG.get("benign_label", "BENIGN")

    # Marca sessioni malevole se qualsiasi record malevolo presente
    session_labels = df.groupby(session_col)[tgt_col].apply(lambda s: int(any(_is_malicious(x) for x in s)))
    mal_sessions = set(session_labels[session_labels == 1].index)
    ben_sessions = set(session_labels[session_labels == 0].index)

    # undersample/oversample a livello di sessione
    if method == "smote" and SMOTE is not None:
        # Costruisci rappresentazione aggregata per sessione per SMOTE
        feature_cols = [c for c in DATA_CONFIG["feature_columns"] if c in df.columns]
        sess_agg = df.groupby(session_col).apply(lambda g: pd.Series(_aggregate_features(g, feature_cols)))
        y_sess = session_labels.loc[sess_agg.index].values
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(sess_agg.values, y_sess)
        # Seleziona sessioni risultanti (approssimazione: usa indice originale per quelle presenti)
        # In alternativa, costruire un nuovo dataset sintetico è complesso; manteniamo subset bilanciato originale
        target_pos = int(sum(y_res))
        target_neg = int(len(y_res) - target_pos)
        pos_ids = list(mal_sessions)
        neg_ids = list(ben_sessions)[:target_neg]
        keep_sessions = set(pos_ids) | set(neg_ids)
        return df[df[session_col].isin(keep_sessions)]
    else:
        # undersample benigno per pareggiare malevoli
        n_mal = len(mal_sessions)
        n_ben_keep = int(n_mal * ratio)
        ben_keep = set(list(ben_sessions)[:n_ben_keep])
        keep_sessions = set(mal_sessions) | ben_keep
        return df[df[session_col].isin(keep_sessions)]


if __name__ == "__main__":
    main()
