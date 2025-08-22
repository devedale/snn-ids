#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esempio di utilizzo di Crypto-PAn per l'anonimizzazione degli IP.
Questo script mostra come:
1. Anonimizzare gli IP usando Crypto-PAn
2. Salvare la mappa di anonimizzazione
3. Decrittografare gli IP quando necessario
"""

import pandas as pd
import json
from crypto import cryptopan_ip, deanonymize_ip, deanonymize_ips_in_dataframe

def example_cryptopan_usage():
    """
    Esempio completo di utilizzo di Crypto-PAn
    """
    print("=== Esempio Crypto-PAn per Anonimizzazione IP ===\n")
    
    # Simula alcuni dati con IP
    sample_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00'],
        'source_ip': ['192.168.1.100', '10.0.0.50', '172.16.0.25'],
        'dest_ip': ['8.8.8.8', '1.1.1.1', '208.67.222.222'],
        'traffic': [1000, 500, 750]
    }
    
    df = pd.DataFrame(sample_data)
    print("Dataset originale:")
    print(df)
    print()
    
    # Genera una chiave segreta (in produzione, questa dovrebbe essere gestita in modo sicuro)
    import secrets
    cryptopan_key = secrets.token_bytes(32)
    print(f"Chiave Crypto-PAn generata: {cryptopan_key.hex()}")
    print()
    
    # Crea mappa di anonimizzazione
    ip_columns = ['source_ip', 'dest_ip']
    all_ips = pd.concat([df[col] for col in ip_columns]).unique()
    
    ip_map = {
        "cryptopan_key": cryptopan_key.hex(),
        "map": {},
        "inverse_map": {}
    }
    
    # Anonimizza tutti gli IP unici
    for original_ip in all_ips:
        anonymized_ip = cryptopan_ip(original_ip, cryptopan_key)
        ip_map["map"][original_ip] = anonymized_ip
        ip_map["inverse_map"][anonymized_ip] = original_ip
        print(f"IP originale: {original_ip} -> IP anonimizzato: {anonymized_ip}")
    
    print()
    
    # Salva la mappa (in produzione, salvala in modo sicuro)
    with open('example_ip_map.json', 'w') as f:
        json.dump(ip_map, f, indent=4)
    print("Mappa di anonimizzazione salvata in 'example_ip_map.json'")
    print()
    
    # Applica l'anonimizzazione al DataFrame
    df_anonymized = df.copy()
    for col in ip_columns:
        df_anonymized[col] = df_anonymized[col].apply(lambda ip: ip_map["map"][ip])
    
    print("Dataset anonimizzato:")
    print(df_anonymized)
    print()
    
    # Carica la mappa salvata (simula il caricamento da file)
    with open('example_ip_map.json', 'r') as f:
        loaded_ip_map = json.load(f)
    
    # Decrittografa un singolo IP
    example_anonymized_ip = df_anonymized['source_ip'].iloc[0]
    original_ip = deanonymize_ip(example_anonymized_ip, loaded_ip_map)
    print(f"Decrittografia singola: {example_anonymized_ip} -> {original_ip}")
    print()
    
    # Decrittografa tutto il DataFrame
    df_deanonymized = deanonymize_ips_in_dataframe(df_anonymized, ip_columns, loaded_ip_map)
    print("Dataset decrittografato:")
    print(df_deanonymized)
    print()
    
    # Verifica che i dati originali e decrittografati siano identici
    print("Verifica integrità:")
    print(f"Dataset originale e decrittografato sono identici: {df.equals(df_deanonymized)}")
    
    # Pulisci il file temporaneo
    import os
    if os.path.exists('example_ip_map.json'):
        os.remove('example_ip_map.json')
        print("\nFile temporaneo 'example_ip_map.json' rimosso.")

if __name__ == "__main__":
    example_cryptopan_usage()
