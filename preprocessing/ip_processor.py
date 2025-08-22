# -*- coding: utf-8 -*-
"""
Modulo per la Gestione e Trasformazione degli Indirizzi IP.
Converte indirizzi IP in feature numeriche (ottetti) per modelli ML.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import ipaddress

class IPProcessor:
    """
    Gestisce la trasformazione degli indirizzi IP in feature numeriche.
    Converte IP in ottetti separati per migliorare le performance dei modelli ML.
    """
    
    def __init__(self, ip_columns: List[str] = None):
        """
        Inizializza il processore IP.
        
        Args:
            ip_columns: Lista delle colonne contenenti indirizzi IP
        """
        if ip_columns is None:
            self.ip_columns = ["Src IP", "Dst IP"]
        else:
            self.ip_columns = ip_columns
        
        # Pattern regex per validare indirizzi IP
        self.ipv4_pattern = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
        self.ipv6_pattern = re.compile(r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$')
    
    def is_valid_ipv4(self, ip_str: str) -> bool:
        """
        Verifica se una stringa è un indirizzo IPv4 valido.
        
        Args:
            ip_str: Stringa da validare
            
        Returns:
            True se è un IPv4 valido, False altrimenti
        """
        if pd.isna(ip_str) or not isinstance(ip_str, str):
            return False
        
        try:
            ipaddress.IPv4Address(ip_str.strip())
            return True
        except:
            return False
    
    def is_valid_ipv6(self, ip_str: str) -> bool:
        """
        Verifica se una stringa è un indirizzo IPv6 valido.
        
        Args:
            ip_str: Stringa da validare
            
        Returns:
            True se è un IPv6 valido, False altrimenti
        """
        if pd.isna(ip_str) or not isinstance(ip_str, str):
            return False
        
        try:
            ipaddress.IPv6Address(ip_str.strip())
            return True
        except:
            return False
    
    def ip_to_octets(self, ip_str: str) -> List[int]:
        """
        Converte un indirizzo IP in lista di ottetti.
        
        Args:
            ip_str: Indirizzo IP come stringa
            
        Returns:
            Lista di 4 ottetti (0-255) per IPv4, 8 ottetti per IPv6
        """
        if pd.isna(ip_str) or not isinstance(ip_str, str):
            return [0, 0, 0, 0]  # Valore di default per IP mancanti
        
        ip_str = ip_str.strip()
        
        # Prova IPv4
        if self.is_valid_ipv4(ip_str):
            match = self.ipv4_pattern.match(ip_str)
            if match:
                return [int(match.group(i)) for i in range(1, 5)]
        
        # Prova IPv6 (converti in formato numerico semplificato)
        elif self.is_valid_ipv6(ip_str):
            try:
                ip_obj = ipaddress.IPv6Address(ip_str)
                # Converti in 4 ottetti per compatibilità (primi 4 byte)
                ip_int = int(ip_obj)
                return [
                    (ip_int >> 24) & 0xFF,
                    (ip_int >> 16) & 0xFF,
                    (ip_int >> 8) & 0xFF,
                    ip_int & 0xFF
                ]
            except:
                return [0, 0, 0, 0]
        
        # IP non valido, ritorna zeri
        return [0, 0, 0, 0]
    
    def process_dataframe(self, df: pd.DataFrame, 
                         create_new_columns: bool = True,
                         drop_original: bool = False) -> pd.DataFrame:
        """
        Processa un DataFrame convertendo le colonne IP in ottetti.
        
        Args:
            df: DataFrame da processare
            create_new_columns: Se True, crea nuove colonne per gli ottetti
            drop_original: Se True, rimuove le colonne IP originali
            
        Returns:
            DataFrame processato con colonne IP convertite in ottetti
        """
        df_processed = df.copy()
        
        for ip_col in self.ip_columns:
            if ip_col not in df_processed.columns:
                print(f"⚠️  Colonna IP '{ip_col}' non trovata nel DataFrame")
                continue
            
            print(f"🔄 Processando colonna IP: {ip_col}")
            
            # Converti IP in ottetti
            octet_data = df_processed[ip_col].apply(self.ip_to_octets)
            
            if create_new_columns:
                # Crea nuove colonne per ogni ottetto
                for i in range(4):
                    new_col_name = f"{ip_col}_Octet_{i+1}"
                    df_processed[new_col_name] = [octets[i] if len(octets) > i else 0 
                                                 for octets in octet_data]
                    print(f"  ✅ Creata colonna: {new_col_name}")
            
            # Sostituisci la colonna originale con la prima ottetto (per compatibilità)
            df_processed[ip_col] = [octets[0] if len(octets) > 0 else 0 
                                   for octets in octet_data]
            
            if drop_original:
                # Rimuovi le colonne IP originali se richiesto
                df_processed = df_processed.drop(columns=[ip_col])
                print(f"  🗑️  Rimossa colonna originale: {ip_col}")
        
        return df_processed
    
    def get_feature_columns(self) -> List[str]:
        """
        Restituisce la lista delle colonne features generate.
        
        Returns:
            Lista delle colonne features (inclusi ottetti IP)
        """
        feature_columns = []
        
        for ip_col in self.ip_columns:
            # Aggiungi colonne ottetti
            for i in range(4):
                feature_columns.append(f"{ip_col}_Octet_{i+1}")
        
        return feature_columns
    
    def reverse_octets_to_ip(self, octets: List[int], ip_type: str = "ipv4") -> str:
        """
        Converte ottetti di nuovo in indirizzo IP (per debugging).
        
        Args:
            octets: Lista di ottetti
            ip_type: Tipo di IP ("ipv4" o "ipv6")
            
        Returns:
            Indirizzo IP come stringa
        """
        if ip_type == "ipv4" and len(octets) >= 4:
            return f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}"
        elif ip_type == "ipv6" and len(octets) >= 8:
            # Conversione semplificata per IPv6
            return f"{octets[0]:02x}:{octets[1]:02x}:{octets[2]:02x}:{octets[3]:02x}::"
        else:
            return "0.0.0.0"

# Funzioni di utilità per uso diretto
def ip_to_octets(ip_str: str) -> List[int]:
    """
    Funzione di utilità per convertire un singolo IP in ottetti.
    
    Args:
        ip_str: Indirizzo IP come stringa
        
    Returns:
        Lista di 4 ottetti
    """
    processor = IPProcessor()
    return processor.ip_to_octets(ip_str)

def process_ip_columns(df: pd.DataFrame, 
                      ip_columns: List[str] = None,
                      create_new_columns: bool = True,
                      drop_original: bool = False) -> pd.DataFrame:
    """
    Funzione di utilità per processare direttamente un DataFrame.
    
    Args:
        df: DataFrame da processare
        ip_columns: Lista delle colonne IP (default: ["Src IP", "Dst IP"])
        create_new_columns: Se creare nuove colonne per ottetti
        drop_original: Se rimuovere colonne IP originali
        
    Returns:
        DataFrame processato
    """
    processor = IPProcessor(ip_columns)
    return processor.process_dataframe(df, create_new_columns, drop_original)
