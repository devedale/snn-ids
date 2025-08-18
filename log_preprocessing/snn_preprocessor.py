#!/usr/bin/env python3
"""
Preprocessore per Spiking Neural Networks (SNN)
Converte log parsati in formato numerico ottimizzato per SNN
"""

import yaml
import numpy as np
import pandas as pd
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import re

# Import per Crypto-PAn (implementazione semplificata)
try:
    import ipaddress
    IP_AVAILABLE = True
except ImportError:
    IP_AVAILABLE = False

from .log_parser import ParsedLog, LogFormat

logger = logging.getLogger(__name__)


@dataclass
class SNNFeatureConfig:
    """Configurazione per una feature SNN"""
    field: str
    encoding: str
    normalize: bool = True
    max_categories: Optional[int] = None
    hash_size: Optional[int] = None
    order: Optional[List[str]] = None
    preserve_subnet: Optional[bool] = None
    subnet_bits: Optional[int] = None
    preserve_port_class: Optional[bool] = None
    description: str = ""  # Aggiunto per compatibilità con config YAML


@dataclass
class TemporalWindow:
    """Finestra temporale per raggruppamento eventi"""
    window_ms: float
    start_time: datetime
    end_time: datetime
    events: List[ParsedLog]


@dataclass
class SNNDataset:
    """Dataset preparato per SNN"""
    features: np.ndarray
    timestamps: np.ndarray
    feature_names: List[str]
    temporal_windows: List[float]
    normalization_stats: Dict[str, Any]
    encoding_mappings: Dict[str, Any]


class CryptoPAN:
    """Implementazione semplificata di Crypto-PAn per preservare struttura IP"""
    
    def __init__(self, key: str):
        self.key = bytes.fromhex(key)
        self._cache = {}

    def _hmac_digest(self, tag: str) -> bytes:
        return hmac.new(self.key, tag.encode("utf-8"), hashlib.sha256).digest()

    def anonymize_ip(self, ip_str: str, subnet_bits: int = None) -> str:
        """Anonimizza IP preservando SOLO l'appartenenza alla stessa subnet.

        - Subnet anonima: derivata tramite HMAC(key, NET|subnet)
        - Host anonimo: derivato tramite HMAC(key, HOST|ip)
        - Non conserva alcun ottetto originale
        """
        # La cache deve tener conto anche del numero di bit di subnet
        cache_key = (ip_str, subnet_bits)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            ip = ipaddress.ip_address(ip_str)

            if ip.version == 4:
                m = 24 if subnet_bits is None else max(0, min(32, int(subnet_bits)))
                network = ipaddress.IPv4Network(f"{ip}/{m}", strict=False)

                # RFC1918 networks
                net10 = ipaddress.IPv4Network("10.0.0.0/8")
                net172 = ipaddress.IPv4Network("172.16.0.0/12")
                net192 = ipaddress.IPv4Network("192.168.0.0/16")
                in_rfc1918 = (ip in net10) or (ip in net172) or (ip in net192)

                net_digest = self._hmac_digest(f"NETv4|{network.network_address}/{m}")

                # Costruisci prefisso di rete anonimo rispettando la classe (privata/pubblica)
                if in_rfc1918:
                    # Scegli blocco privato di destinazione in modo deterministico
                    choice = net_digest[0] % 3  # 0: 10/8, 1: 172.16/12, 2: 192.168/16
                    if choice == 0:
                        b0 = 10
                        b1 = net_digest[1]
                        b2 = net_digest[2]
                    elif choice == 1:
                        b0 = 172
                        b1 = 16 + (net_digest[1] % 16)  # 16..31
                        b2 = net_digest[2]
                    else:
                        b0 = 192
                        b1 = 168
                        b2 = net_digest[2]
                    b3 = 0
                    net_val_full = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
                else:
                    # Genera prefisso pubblico, evitando RFC1918
                    b0 = (net_digest[0] % 223) + 1  # 1..223
                    # Evita 10, 172, 192 (per non cadere in RFC1918) e 127
                    forbidden_first = {10, 127, 172, 192}
                    k = 1
                    while b0 in forbidden_first and k < 8:
                        b0 = ((b0 + net_digest[k]) % 223) + 1
                        k += 1
                    b1 = net_digest[1]
                    # Evita 172.16-31 e 192.168
                    if b0 == 172 and 16 <= b1 <= 31:
                        b1 = (net_digest[2] % 16)  # 0..15
                    if b0 == 192 and b1 == 168:
                        b1 = (b1 + net_digest[3] + 1) % 256
                    b2 = net_digest[2]
                    b3 = 0
                    net_val_full = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

                anon_net_prefix = (net_val_full >> (32 - m)) << (32 - m)

                # Calcola la parte host anonima
                host_bits = 32 - m
                if host_bits == 0:
                    host_val = 0
                else:
                    host_digest = self._hmac_digest(f"HOSTv4|{ip}")
                    host_space = 1 << host_bits
                    host_val = int.from_bytes(host_digest[:4], "big") % host_space
                    # Evita indirizzi speciali 0 e broadcast quando possibile
                    if host_space >= 4:
                        host_val = 1 + (host_val % (host_space - 2))

                anon_int = anon_net_prefix | host_val
                anonymized = str(ipaddress.IPv4Address(anon_int))

            else:
                m = 64 if subnet_bits is None else max(0, min(128, int(subnet_bits)))
                network = ipaddress.IPv6Network(f"{ip}/{m}", strict=False)

                net_digest = self._hmac_digest(f"NETv6|{network.network_address}/{m}")
                net_val_full = int.from_bytes(net_digest[:16], "big")
                anon_net_prefix = (net_val_full >> (128 - m)) << (128 - m)

                host_bits = 128 - m
                if host_bits == 0:
                    host_val = 0
                else:
                    host_digest = self._hmac_digest(f"HOSTv6|{ip}")
                    host_space = 1 << host_bits
                    host_val = int.from_bytes(host_digest[:16], "big") % host_space

                anon_int = anon_net_prefix | host_val
                anonymized = str(ipaddress.IPv6Address(anon_int))

            self._cache[cache_key] = anonymized
            return anonymized

        except Exception as e:
            logger.warning(f"Errore nell'anonimizzazione IP {ip_str}: {e}")
            return ip_str

    def anonymize_port(self, port_value: Union[int, str], preserve_class: bool = True) -> int:
        """Anonimizza una porta in modo deterministico con HMAC.

        - preserve_class=True mantiene la classe:
          0-1023 (well-known), 1024-49151 (registered), 49152-65535 (dynamic)
        """
        try:
            if port_value is None or port_value == "":
                return 0
            p = int(port_value)
            if p < 0:
                p = 0
        except Exception:
            # fallback per valori non numerici
            p = 0

        digest = self._hmac_digest(f"PORT|{p}")
        rnd = int.from_bytes(digest[:4], "big")

        if not preserve_class:
            # mappa in [1, 65535]
            return max(1, rnd % 65535)

        # preserva le 3 classi IANA
        if p <= 1023:
            span = 1024
            base = 0
        elif p <= 49151:
            span = 49151 - 1024 + 1
            base = 1024
        else:
            span = 65535 - 49152 + 1
            base = 49152

        mapped = base + (rnd % span)
        # evita 0
        return 1 if mapped == 0 else mapped

    def anonymize_mac(
        self,
        mac_value: Union[str, bytes],
        preserve_ig_bit: bool = True,
        force_locally_administered: bool = True,
    ) -> str:
        """Anonimizza MAC 48-bit in modo deterministico via HMAC.

        - preserve_ig_bit: preserva il bit I/G (unicast/multicast)
        - force_locally_administered: forza il bit U/L a 1 per evitare leak dell'OUI reale
        """
        if mac_value is None:
            return "00:00:00:00:00:00"

        try:
            s = str(mac_value).strip().lower()
            # estrai solo hex
            hex_only = re.sub(r"[^0-9a-f]", "", s)
            if len(hex_only) != 12:
                return "00:00:00:00:00:00"

            orig = int(hex_only, 16)
            digest = self._hmac_digest(f"MAC|{hex_only}")
            anon = int.from_bytes(digest[:6], "big")

            # aggiusta il primo byte per I/G e U/L
            orig_first = (orig >> 40) & 0xFF
            anon_first = (anon >> 40) & 0xFF

            # bit 0: I/G (0 unicast, 1 multicast)
            if preserve_ig_bit:
                ig = orig_first & 0x01
                anon_first = (anon_first & 0xFE) | ig
            else:
                # default unicast
                anon_first = (anon_first & 0xFE)

            # bit 1: U/L (0 universal, 1 local). Forza locale per evitare OUI reali
            if force_locally_administered:
                anon_first = anon_first | 0x02

            # ricostruisci i 6 byte
            rest = anon & ((1 << 40) - 1)
            anon_int = (anon_first << 40) | rest

            # formatta con ':'
            return ":".join(f"{(anon_int >> (8*(5-i))) & 0xFF:02x}" for i in range(6))
        except Exception:
            return "00:00:00:00:00:00"


class SNNPreprocessor:
    """Preprocessore principale per dati SNN"""
    
    def __init__(self, config_file: str = "snn_config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.crypto_pan = None
        self.normalization_stats = {}
        self.encoding_mappings = {}
        self._setup_crypto_pan()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carica configurazione da file YAML"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"File di configurazione {self.config_file} non trovato")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Errore nel parsing YAML: {e}")
            return {}
    
    def _setup_crypto_pan(self):
        """Inizializza Crypto-PAn se configurato"""
        if not IP_AVAILABLE:
            logger.warning("Modulo ipaddress non disponibile, Crypto-PAn disabilitato")
            return
        
        key = self.config.get('security', {}).get('crypto_pan_key')
        if key:
            try:
                self.crypto_pan = CryptoPAN(key)
                logger.info("Crypto-PAn inizializzato")
            except Exception as e:
                logger.error(f"Errore nell'inizializzazione Crypto-PAn: {e}")
    
    def get_feature_config(self, log_format: LogFormat) -> Dict[str, List[SNNFeatureConfig]]:
        """Ottiene configurazione features per un formato di log"""
        format_name = log_format.value
        if format_name in ["syslog_rfc3164", "syslog_rfc5424"]:
            format_name = "syslog"
        
        config_key = f"{format_name}_features"
        format_config = self.config.get(config_key, {})
        
        feature_configs = {
            'continuous': [],
            'categorical': [],
            'ip': [],
            'timestamp': format_config.get('timestamp_field'),
            'exclude': format_config.get('exclude_fields', [])
        }
        
        # Parse continuous features
        for feat_config in format_config.get('continuous_features', []):
            feature_configs['continuous'].append(SNNFeatureConfig(**feat_config))
        
        # Parse categorical features  
        for feat_config in format_config.get('categorical_features', []):
            feature_configs['categorical'].append(SNNFeatureConfig(**feat_config))
        
        # Parse IP features
        for feat_config in format_config.get('ip_features', []):
            feature_configs['ip'].append(SNNFeatureConfig(**feat_config))
        
        return feature_configs
    
    def create_temporal_windows(self, logs: List[ParsedLog]) -> Dict[float, List[TemporalWindow]]:
        """Crea finestre temporali per raggruppamento eventi"""
        if not logs:
            return {}
        
        # Ordina per timestamp
        sorted_logs = sorted([log for log in logs if log.timestamp], 
                           key=lambda x: x.timestamp)
        
        if not sorted_logs:
            logger.warning("Nessun log con timestamp valido")
            return {}
        
        windows_config = self.config.get('temporal_windows', {})
        window_sizes_ms = windows_config.get('windows_ms', [1000])
        window_sizes_seconds = windows_config.get('windows_seconds', [])
        
        # Converti secondi in millisecondi
        all_windows_ms = window_sizes_ms + [s * 1000 for s in window_sizes_seconds]
        
        overlap = windows_config.get('overlap', 0.1)
        
        result = {}
        
        for window_ms in all_windows_ms:
            windows = self._create_windows_for_size(sorted_logs, window_ms, overlap)
            result[window_ms] = windows
        
        return result
    
    def _create_windows_for_size(self, logs: List[ParsedLog], window_ms: float, overlap: float) -> List[TemporalWindow]:
        """Crea finestre per una specifica dimensione"""
        windows = []
        
        if not logs:
            return windows
        
        start_time = logs[0].timestamp
        end_time = logs[-1].timestamp
        
        window_delta = timedelta(milliseconds=window_ms)
        step_delta = timedelta(milliseconds=window_ms * (1 - overlap))
        
        current_start = start_time
        
        while current_start < end_time:
            current_end = current_start + window_delta
            
            # Trova eventi in questa finestra
            window_events = [
                log for log in logs 
                if current_start <= log.timestamp < current_end
            ]
            
            if window_events:
                window = TemporalWindow(
                    window_ms=window_ms,
                    start_time=current_start,
                    end_time=current_end,
                    events=window_events
                )
                windows.append(window)
            
            current_start += step_delta
        
        return windows
    
    def encode_continuous_feature(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """Encode feature numerica continua"""
        # Converti a numpy array, gestendo valori mancanti
        numeric_values = []
        # Supporto per anonimizzazione porta basata su HMAC
        if config.encoding in ("crypto_port", "hmac_port") and self.crypto_pan:
            preserve_class = True if config.preserve_port_class is None else bool(config.preserve_port_class)
            for val in values:
                anon_port = self.crypto_pan.anonymize_port(val, preserve_class=preserve_class)
                numeric_values.append(float(anon_port))
        else:
            for val in values:
                try:
                    if val is None or val == "":
                        numeric_values.append(np.nan)
                    else:
                        numeric_values.append(float(val))
                except (ValueError, TypeError):
                    numeric_values.append(np.nan)
        
        arr = np.array(numeric_values)
        
        # Gestisci encoding speciale
        if config.encoding == "log_scale":
            # Log scale per valori con grande range
            arr = np.where(arr > 0, np.log1p(arr), 0)
        
        # Normalizzazione
        if config.normalize:
            arr = self._normalize_array(arr, config.field)
        
        return arr
    
    def encode_categorical_feature(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """Encode feature categorica"""
        if config.encoding == "one_hot":
            return self._one_hot_encode(values, config)
        elif config.encoding == "hash_numeric":
            return self._hash_numeric_encode(values, config)
        elif config.encoding == "ordinal":
            return self._ordinal_encode(values, config)
        elif config.encoding in ("crypto_mac", "hmac_mac") and self.crypto_pan:
            # Anonimizza MAC e poi applica hashing numerico stabile
            anon = [self.crypto_pan.anonymize_mac(v) for v in values]
            # Mappa in [0,1] con hash stabile
            numeric_values = []
            for val in anon:
                h = hashlib.sha256(val.encode("utf-8")).digest()
                numeric_values.append(int.from_bytes(h[:4], "big") / (2**32 - 1))
            return np.array(numeric_values)
        else:
            raise ValueError(f"Encoding categorico non supportato: {config.encoding}")
    
    def encode_ip_feature(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """Encode feature IP"""
        if config.encoding == "crypto_pan" and self.crypto_pan:
            # Determina i bit di subnet dalla config; default 24 per IPv4, 64 per IPv6
            subnet_bits = config.subnet_bits
            def default_bits(v: Any) -> int:
                s = str(v)
                return 24 if "." in s else 64
            bits = subnet_bits if subnet_bits is not None else (default_bits(next((v for v in values if v is not None and v != ""), "0.0.0.0")))
            anonymized_values = [
                self.crypto_pan.anonymize_ip(str(val), subnet_bits=bits) for val in values
            ]
            return self._ip_to_numeric(anonymized_values)
        elif config.encoding == "secure_hash":
            return self._secure_hash_encode(values, config)
        else:
            return self._ip_to_numeric(values)
    
    def _normalize_array(self, arr: np.ndarray, field_name: str) -> np.ndarray:
        """Normalizza array nel range [0,1]"""
        # Rimuovi NaN per calcolo statistiche
        valid_mask = ~np.isnan(arr)
        if not np.any(valid_mask):
            return arr
        
        valid_values = arr[valid_mask]
        
        norm_config = self.config.get('normalization', {})
        method = norm_config.get('method', 'min_max')
        target_range = norm_config.get('range', [0.0, 1.0])
        clip_percentiles = norm_config.get('clip_percentiles', [0.01, 0.99])
        
        # Clip outliers
        if clip_percentiles:
            p_low, p_high = np.percentile(valid_values, clip_percentiles)
            valid_values = np.clip(valid_values, p_low, p_high)
            arr[valid_mask] = valid_values
        
        if method == "min_max":
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            if max_val > min_val:
                normalized = (arr - min_val) / (max_val - min_val)
                normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            else:
                normalized = np.full_like(arr, target_range[0])
        
        elif method == "z_score":
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            if std_val > 0:
                normalized = (arr - mean_val) / std_val
                # Remap to target range (assumendo distribuzione normale)
                normalized = np.clip(normalized, -3, 3)  # 99.7% dei dati
                normalized = (normalized + 3) / 6  # [0,1]
                normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            else:
                normalized = np.full_like(arr, target_range[0])
        
        else:
            logger.warning(f"Metodo normalizzazione {method} non supportato, uso min_max")
            return self._normalize_array(arr, field_name)
        
        # Salva statistiche per reverse mapping
        self.normalization_stats[field_name] = {
            'method': method,
            'min_val': float(np.min(valid_values)) if len(valid_values) > 0 else 0.0,
            'max_val': float(np.max(valid_values)) if len(valid_values) > 0 else 1.0,
            'mean_val': float(np.mean(valid_values)) if len(valid_values) > 0 else 0.5,
            'std_val': float(np.std(valid_values)) if len(valid_values) > 0 else 1.0,
            'target_range': target_range,
            'clip_percentiles': clip_percentiles
        }
        
        return normalized
    
    def _one_hot_encode(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """One-hot encoding per categoriali"""
        # Usa mapping esistente se disponibile, altrimenti crealo
        if config.field in self.encoding_mappings:
            mapping_info = self.encoding_mappings[config.field]
            value_to_index = mapping_info['mapping']
            unique_values = mapping_info['categories']
        else:
            unique_values = list(set(str(val) for val in values if val is not None))
            
            # Limita numero categorie
            max_categories = config.max_categories or 10
            if len(unique_values) > max_categories:
                # Prendi le più frequenti
                value_counts = {}
                for val in values:
                    str_val = str(val) if val is not None else "unknown"
                    value_counts[str_val] = value_counts.get(str_val, 0) + 1
                
                top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                unique_values = [val for val, _ in top_values[:max_categories-1]]
                unique_values.append("other")  # Categoria per il resto
            
            # Assicurati che il numero di categorie sia esattamente quello configurato
            while len(unique_values) < max_categories:
                unique_values.append(f"unused_{len(unique_values)}")
            
            unique_values = unique_values[:max_categories]  # Tronca se necessario
            
            # Crea mapping
            value_to_index = {val: i for i, val in enumerate(unique_values)}
            self.encoding_mappings[config.field] = {
                'type': 'one_hot',
                'mapping': value_to_index,
                'categories': unique_values
            }
        
        # Crea matrice one-hot con dimensione fissa
        one_hot = np.zeros((len(values), len(unique_values)))
        
        for i, val in enumerate(values):
            str_val = str(val) if val is not None else "unknown"
            if str_val in value_to_index:
                one_hot[i, value_to_index[str_val]] = 1.0
            elif "other" in value_to_index:
                one_hot[i, value_to_index["other"]] = 1.0
        
        return one_hot
    
    def _hash_numeric_encode(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """Hash numerico per categoriali ad alta cardinalità"""
        salt = self.config.get('security', {}).get('hash_salt', '')
        hash_size = config.hash_size or 32
        
        numeric_values = []
        hash_mapping = {}
        
        for val in values:
            str_val = str(val) if val is not None else "unknown"
            
            if str_val not in hash_mapping:
                hash_input = f"{salt}{str_val}".encode('utf-8')
                hash_digest = hashlib.sha256(hash_input).hexdigest()
                # Converti hash in numero [0,1]
                hash_int = int(hash_digest[:8], 16)  # Usa primi 8 caratteri hex
                hash_mapping[str_val] = hash_int / (16**8 - 1)  # Normalizza [0,1]
            
            numeric_values.append(hash_mapping[str_val])
        
        self.encoding_mappings[config.field] = {
            'type': 'hash_numeric',
            'mapping': hash_mapping,
            'hash_size': hash_size
        }
        
        return np.array(numeric_values)
    
    def _ordinal_encode(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """Encoding ordinale con ordine specificato"""
        if not config.order:
            raise ValueError(f"Ordine non specificato per feature ordinale {config.field}")
        
        # Crea mapping ordinale
        ordinal_mapping = {val: i for i, val in enumerate(config.order)}
        
        numeric_values = []
        for val in values:
            str_val = str(val) if val is not None else "unknown"
            if str_val in ordinal_mapping:
                # Normalizza in [0,1]
                numeric_values.append(ordinal_mapping[str_val] / (len(config.order) - 1))
            else:
                numeric_values.append(0.0)  # Valore di default
        
        self.encoding_mappings[config.field] = {
            'type': 'ordinal',
            'mapping': ordinal_mapping,
            'order': config.order
        }
        
        return np.array(numeric_values)
    
    def _ip_to_numeric(self, ip_values: List[Any]) -> np.ndarray:
        """Converte IP in rappresentazione numerica"""
        numeric_values = []
        
        for ip_str in ip_values:
            try:
                if IP_AVAILABLE:
                    ip = ipaddress.ip_address(str(ip_str))
                    if ip.version == 4:
                        # IPv4: converti in [0,1]
                        numeric_values.append(int(ip) / (2**32 - 1))
                    else:
                        # IPv6: usa hash
                        hash_val = hashlib.md5(str(ip).encode()).hexdigest()
                        numeric_values.append(int(hash_val[:8], 16) / (16**8 - 1))
                else:
                    # Fallback: usa hash
                    hash_val = hashlib.md5(str(ip_str).encode()).hexdigest()
                    numeric_values.append(int(hash_val[:8], 16) / (16**8 - 1))
            except Exception:
                numeric_values.append(0.0)  # IP non valido
        
        return np.array(numeric_values)
    
    def _secure_hash_encode(self, values: List[Any], config: SNNFeatureConfig) -> np.ndarray:
        """Hash sicuro per IP o altri valori sensibili"""
        salt = self.config.get('security', {}).get('hash_salt', '')
        
        numeric_values = []
        hash_mapping = {}
        
        for val in values:
            str_val = str(val) if val is not None else "unknown"
            
            if str_val not in hash_mapping:
                hash_input = f"{salt}{str_val}".encode('utf-8')
                hash_digest = hashlib.sha256(hash_input).hexdigest()
                hash_int = int(hash_digest[:8], 16)
                hash_mapping[str_val] = hash_int / (16**8 - 1)
            
            numeric_values.append(hash_mapping[str_val])
        
        # Salva mapping se richiesto (per debug/ricerca)
        if self.config.get('security', {}).get('save_feature_mapping', False):
            self.encoding_mappings[config.field] = {
                'type': 'secure_hash',
                'mapping': hash_mapping
            }
        
        return np.array(numeric_values)
    
    def process_logs_to_snn_format(self, logs: List[ParsedLog]) -> SNNDataset:
        """Converte log parsati in formato SNN"""
        if not logs:
            raise ValueError("Lista log vuota")
        
        # Determina formato predominante
        format_counts = {}
        for log in logs:
            fmt = log.format_type.value
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        main_format = max(format_counts.items(), key=lambda x: x[1])[0]
        main_format_enum = LogFormat(main_format)
        
        logger.info(f"Formato principale rilevato: {main_format}")
        
        # Ottieni configurazione features
        feature_config = self.get_feature_config(main_format_enum)
        
        # Filtra log del formato principale
        main_logs = [log for log in logs if log.format_type.value == main_format]
        
        # Crea finestre temporali
        temporal_windows = self.create_temporal_windows(main_logs)
        
        # Estrai features per ogni finestra temporale
        all_features = []
        all_timestamps = []
        feature_names = []
        
        # Processa ogni dimensione di finestra
        for window_ms, windows in temporal_windows.items():
            logger.info(f"Processando finestra {window_ms}ms con {len(windows)} finestre")
            
            for window in windows:
                if not window.events:
                    continue
                
                # Aggrega eventi nella finestra
                aggregated_features = self._aggregate_window_events(window.events, feature_config)
                
                if aggregated_features is not None:
                    all_features.append(aggregated_features)
                    all_timestamps.append(window.start_time.timestamp())
                    
                    # Genera nomi features (solo al primo giro)
                    if not feature_names:
                        feature_names = self._generate_feature_names(feature_config, window_ms)
        
        if not all_features:
            raise ValueError("Nessuna feature estratta dai log")
        
        # Converti a numpy array
        features_array = np.array(all_features)
        timestamps_array = np.array(all_timestamps)
        
        # Ordina per timestamp
        sort_indices = np.argsort(timestamps_array)
        features_array = features_array[sort_indices]
        timestamps_array = timestamps_array[sort_indices]
        
        # Validazione
        self._validate_snn_data(features_array, timestamps_array)
        
        return SNNDataset(
            features=features_array,
            timestamps=timestamps_array,
            feature_names=feature_names,
            temporal_windows=list(temporal_windows.keys()),
            normalization_stats=self.normalization_stats,
            encoding_mappings=self.encoding_mappings
        )
    
    def _aggregate_window_events(self, events: List[ParsedLog], feature_config: Dict) -> Optional[np.ndarray]:
        """Aggrega eventi in una finestra temporale"""
        if not events:
            return None
        
        aggregation_strategy = self.config.get('temporal_windows', {}).get('aggregation_strategy', 'mean')
        
        # Estrai tutte le features dagli eventi
        all_features = []
        
        for event in events:
            event_features = self._extract_features_from_event(event, feature_config)
            if event_features is not None:
                all_features.append(event_features)
        
        if not all_features:
            return None
        
        # Verifica che tutte le features abbiano la stessa dimensione
        feature_lengths = [len(f) for f in all_features]
        if len(set(feature_lengths)) > 1:
            logger.warning(f"Features con dimensioni diverse: {set(feature_lengths)}")
            # Prendi la dimensione più comune
            from collections import Counter
            most_common_length = Counter(feature_lengths).most_common(1)[0][0]
            
            # Filtra o padda per raggiungere la dimensione corretta
            normalized_features = []
            for f in all_features:
                if len(f) == most_common_length:
                    normalized_features.append(f)
                elif len(f) < most_common_length:
                    # Padda con zeri
                    padded = np.zeros(most_common_length)
                    padded[:len(f)] = f
                    normalized_features.append(padded)
                else:
                    # Tronca
                    normalized_features.append(f[:most_common_length])
            
            all_features = normalized_features
        
        try:
            features_matrix = np.array(all_features)
        except ValueError as e:
            logger.error(f"Errore nella creazione matrice features: {e}")
            logger.error(f"Dimensioni features: {[f.shape if hasattr(f, 'shape') else len(f) for f in all_features]}")
            return None
        
        # Applica strategia di aggregazione
        if aggregation_strategy == "mean":
            return np.nanmean(features_matrix, axis=0)
        elif aggregation_strategy == "sum":
            return np.nansum(features_matrix, axis=0)
        elif aggregation_strategy == "max":
            return np.nanmax(features_matrix, axis=0)
        elif aggregation_strategy == "min":
            return np.nanmin(features_matrix, axis=0)
        elif aggregation_strategy == "last":
            return features_matrix[-1]  # Ultimo evento
        else:
            logger.warning(f"Strategia aggregazione {aggregation_strategy} non supportata, uso mean")
            return np.nanmean(features_matrix, axis=0)
    
    def _extract_features_from_event(self, event: ParsedLog, feature_config: Dict) -> Optional[np.ndarray]:
        """Estrae features numeriche da un singolo evento"""
        features = []
        
        # Features continue
        for config in feature_config['continuous']:
            if config.field in event.parsed_fields:
                value = event.parsed_fields[config.field]
                encoded = self.encode_continuous_feature([value], config)
                features.extend(encoded)
            else:
                features.append(0.0)  # Valore mancante
        
        # Features categoriali - PRIMA calcola tutte le dimensioni necessarie
        for config in feature_config['categorical']:
            if config.field in event.parsed_fields:
                value = event.parsed_fields[config.field]
                # Calcola dimensione attesa per questo campo
                expected_dim = self._get_categorical_dimension(config)
                
                encoded = self.encode_categorical_feature([value], config)
                if encoded.ndim == 2:  # One-hot restituisce matrice
                    # Assicurati che sia della dimensione corretta
                    if encoded.shape[1] == expected_dim:
                        features.extend(encoded[0])
                    else:
                        # Padded o troncato alla dimensione corretta
                        padded = np.zeros(expected_dim)
                        min_len = min(encoded.shape[1], expected_dim)
                        padded[:min_len] = encoded[0][:min_len]
                        features.extend(padded)
                else:
                    features.extend(encoded)
            else:
                # Valore mancante per categoriali
                if config.encoding == "one_hot":
                    expected_dim = self._get_categorical_dimension(config)
                    features.extend([0.0] * expected_dim)
                else:
                    features.append(0.0)
        
        # Features IP
        for config in feature_config['ip']:
            if config.field in event.parsed_fields:
                value = event.parsed_fields[config.field]
                encoded = self.encode_ip_feature([value], config)
                features.extend(encoded)
            else:
                features.append(0.0)  # IP mancante
        
        return np.array(features) if features else None
    
    def _get_categorical_dimension(self, config: SNNFeatureConfig) -> int:
        """Calcola la dimensione attesa per una feature categorica"""
        if config.encoding == "one_hot":
            return config.max_categories or 10
        else:
            return 1
    
    def _generate_feature_names(self, feature_config: Dict, window_ms: float) -> List[str]:
        """Genera nomi delle features"""
        names = []
        prefix = self.config.get('output_format', {}).get('feature_prefix', 'feat_')
        window_suffix = self.config.get('output_format', {}).get('window_suffix', '_win_{window_ms}ms')
        suffix = window_suffix.format(window_ms=int(window_ms))
        
        # Features continue
        for config in feature_config['continuous']:
            names.append(f"{prefix}{config.field}{suffix}")
        
        # Features categoriali
        for config in feature_config['categorical']:
            if config.encoding == "one_hot":
                max_cat = config.max_categories or 10
                for i in range(max_cat):
                    names.append(f"{prefix}{config.field}_cat{i}{suffix}")
            else:
                names.append(f"{prefix}{config.field}{suffix}")
        
        # Features IP
        for config in feature_config['ip']:
            names.append(f"{prefix}{config.field}{suffix}")
        
        return names
    
    def _validate_snn_data(self, features: np.ndarray, timestamps: np.ndarray):
        """Valida i dati preparati per SNN"""
        validation_config = self.config.get('validation', {})
        
        # Controlla range normalizzazione
        if validation_config.get('check_normalized_range', True):
            min_val = np.nanmin(features)
            max_val = np.nanmax(features)
            target_range = self.config.get('normalization', {}).get('range', [0.0, 1.0])
            
            if min_val < target_range[0] - 0.01 or max_val > target_range[1] + 0.01:
                logger.warning(f"Valori fuori range di normalizzazione: [{min_val:.3f}, {max_val:.3f}]")
        
        # Controlla ordine temporale
        if validation_config.get('check_temporal_order', True):
            if not np.all(timestamps[:-1] <= timestamps[1:]):
                logger.warning("Sequenza temporale non ordinata")
        
        # Controlla valori NaN
        max_nan_ratio = validation_config.get('max_nan_ratio', 0.1)
        nan_ratio = np.isnan(features).sum() / features.size
        
        if nan_ratio > max_nan_ratio:
            logger.warning(f"Troppi valori NaN: {nan_ratio:.2%} > {max_nan_ratio:.2%}")
        
        logger.info(f"Validazione completata: {features.shape} features, range [{np.nanmin(features):.3f}, {np.nanmax(features):.3f}]")
    
    def save_snn_dataset(self, dataset: SNNDataset, output_path: str):
        """Salva dataset SNN su file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_format = self.config.get('output_format', {})
        file_format = output_format.get('file_format', 'csv')
        
        if file_format == 'csv':
            self._save_csv(dataset, output_path)
        elif file_format == 'json':
            self._save_json(dataset, output_path)
        elif file_format == 'parquet':
            self._save_parquet(dataset, output_path)
        else:
            raise ValueError(f"Formato output non supportato: {file_format}")
        
        # Salva metadati
        self._save_metadata(dataset, output_path)
    
    def _save_csv(self, dataset: SNNDataset, output_path: Path):
        """Salva in formato CSV"""
        # Crea DataFrame
        df_data = {}
        df_data['timestamp'] = dataset.timestamps
        
        for i, name in enumerate(dataset.feature_names):
            df_data[name] = dataset.features[:, i]
        
        df = pd.DataFrame(df_data)
        
        # Salva CSV
        separator = self.config.get('output_format', {}).get('csv_separator', ',')
        include_header = self.config.get('output_format', {}).get('include_header', True)
        
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, sep=separator, header=include_header, index=False)
        
        logger.info(f"Dataset SNN salvato in {csv_path}")
    
    def _save_json(self, dataset: SNNDataset, output_path: Path):
        """Salva in formato JSON Lines"""
        json_path = output_path.with_suffix('.jsonl')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            for i in range(len(dataset.timestamps)):
                record = {
                    'timestamp': dataset.timestamps[i],
                    'features': dataset.features[i].tolist()
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Dataset SNN salvato in {json_path}")
    
    def _save_parquet(self, dataset: SNNDataset, output_path: Path):
        """Salva in formato Parquet"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Crea tabella PyArrow
            df_data = {}
            df_data['timestamp'] = dataset.timestamps
            
            for i, name in enumerate(dataset.feature_names):
                df_data[name] = dataset.features[:, i]
            
            table = pa.table(df_data)
            
            parquet_path = output_path.with_suffix('.parquet')
            pq.write_table(table, parquet_path)
            
            logger.info(f"Dataset SNN salvato in {parquet_path}")
            
        except ImportError:
            logger.error("PyArrow non disponibile, impossibile salvare in formato Parquet")
            # Fallback a CSV
            self._save_csv(dataset, output_path)
    
    def _save_metadata(self, dataset: SNNDataset, output_path: Path):
        """Salva metadati del dataset"""
        metadata = {
            'dataset_info': {
                'num_samples': int(dataset.features.shape[0]),
                'num_features': int(dataset.features.shape[1]),
                'feature_names': dataset.feature_names,
                'temporal_windows': dataset.temporal_windows,
                'timestamp_range': [float(dataset.timestamps.min()), float(dataset.timestamps.max())]
            },
            'normalization_stats': dataset.normalization_stats,
            'encoding_mappings': dataset.encoding_mappings,
            'config_file': self.config_file,
            'preprocessing_timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadati salvati in {metadata_path}")
        
        # Salva anche file separati se richiesto
        security_config = self.config.get('security', {})
        
        if security_config.get('save_normalization_stats', False):
            norm_file = security_config.get('normalization_stats_file', 'snn_normalization_stats.json')
            norm_path = output_path.parent / norm_file
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.normalization_stats, f, indent=2)
        
        if security_config.get('save_feature_mapping', False):
            mapping_file = security_config.get('mapping_file', 'snn_feature_mapping.json')
            mapping_path = output_path.parent / mapping_file
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.encoding_mappings, f, indent=2)


def main():
    """Funzione principale per test"""
    from secure_log_parser import SecureLogParser
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Preprocessore SNN - Test ===\n")
    
    # Crea directory output se non esiste
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Carica log di esempio
    parser = SecureLogParser()
    
    test_files = [
        "/home/wls_user/tesi/input/FGT80FTK22013405.root.tlog.txt",
        "/home/wls_user/tesi/input/FGT80FTK22013405.root.elog.txt",
    ]
    
    all_logs = []
    for file_path in test_files:
        if Path(file_path).exists():
            # Usa solo il parsing, non l'anonimizzazione per test SNN
            from log_parser import UniversalLogParser
            simple_parser = UniversalLogParser()
            logs = simple_parser.parse_file(file_path)
            all_logs.extend(logs)
            print(f"Caricati {len(logs)} log da {Path(file_path).name}")
    
    if not all_logs:
        print("Nessun log caricato, uscita")
        return
    
    # Inizializza preprocessore SNN
    snn_preprocessor = SNNPreprocessor()
    
    try:
        # Processa log per SNN
        snn_dataset = snn_preprocessor.process_logs_to_snn_format(all_logs)
        
        print(f"\n=== Dataset SNN Generato ===")
        print(f"Campioni: {snn_dataset.features.shape[0]}")
        print(f"Features: {snn_dataset.features.shape[1]}")
        print(f"Finestre temporali: {snn_dataset.temporal_windows}")
        print(f"Range valori: [{np.nanmin(snn_dataset.features):.3f}, {np.nanmax(snn_dataset.features):.3f}]")
        
        # Salva dataset
        output_path = output_dir / "snn_dataset"
        snn_preprocessor.save_snn_dataset(snn_dataset, str(output_path))
        
        print(f"\n=== Dataset salvato in {output_dir}/ ===")
        
    except Exception as e:
        logger.error(f"Errore nel preprocessing SNN: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
