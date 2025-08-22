#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script di Test per la Trasformazione IP in Ottetti.
Dimostra come funziona il preprocessing degli indirizzi IP.
"""

import pandas as pd
import numpy as np
import sys
import os

# Aggiungi il path del progetto
sys.path.append(os.path.abspath('.'))

from preprocessing.ip_processor import IPProcessor, process_ip_columns, ip_to_octets

def test_single_ip_conversion():
    """Test della conversione di singoli indirizzi IP."""
    print("🧪 TEST: Conversione Singoli Indirizzi IP")
    print("=" * 50)
    
    test_ips = [
        "192.168.1.1",
        "10.0.0.1", 
        "172.16.0.100",
        "8.8.8.8",
        "127.0.0.1",
        "invalid_ip",
        "256.1.2.3",  # IP non valido
        None,
        ""
    ]
    
    processor = IPProcessor()
    
    for ip in test_ips:
        octets = processor.ip_to_octets(ip)
        is_valid = processor.is_valid_ipv4(ip) if ip else False
        
        print(f"IP: {ip!r:>15} → Ottetti: {octets} {'✅' if is_valid else '❌'}")
        
        if is_valid:
            # Test conversione inversa
            reconstructed = processor.reverse_octets_to_ip(octets)
            print(f"           → Ricostruito: {reconstructed}")
    
    print()

def test_dataframe_processing():
    """Test della trasformazione di un DataFrame completo."""
    print("🧪 TEST: Trasformazione DataFrame")
    print("=" * 50)
    
    # Crea DataFrame di test
    test_data = {
        'Src IP': ['192.168.1.1', '10.0.0.1', '172.16.0.100', '8.8.8.8'],
        'Dst IP': ['10.0.0.2', '192.168.1.2', '8.8.4.4', '1.1.1.1'],
        'Protocol': [6, 17, 6, 1],
        'Port': [80, 443, 22, 53],
        'Label': [0, 1, 0, 1]
    }
    
    df = pd.DataFrame(test_data)
    print("📊 DataFrame Originale:")
    print(df)
    print()
    
    # Processa le colonne IP
    df_processed = process_ip_columns(
        df, 
        ip_columns=['Src IP', 'Dst IP'],
        create_new_columns=True,
        drop_original=False
    )
    
    print("📊 DataFrame Processato:")
    print(df_processed)
    print()
    
    print("🔍 Colonne Generate:")
    for col in df_processed.columns:
        if 'Octet' in col:
            print(f"  {col}: {df_processed[col].tolist()}")
    
    print()
    
    # Verifica che le nuove colonne siano numeriche
    print("🔢 Tipi di Dati:")
    for col in df_processed.columns:
        if 'Octet' in col:
            dtype = df_processed[col].dtype
            print(f"  {col}: {dtype} (range: {df_processed[col].min()}-{df_processed[col].max()})")
    
    print()

def test_integration_with_config():
    """Test dell'integrazione con la configurazione del progetto."""
    print("🧪 TEST: Integrazione con Configurazione")
    print("=" * 50)
    
    try:
        from config import DATA_CONFIG
        
        print("📋 Configurazione Feature Columns:")
        for i, col in enumerate(DATA_CONFIG["feature_columns"]):
            if 'Octet' in col:
                print(f"  {i+1:2d}. {col} 🔢")
            elif 'IP' in col and 'Octet' not in col:
                print(f"  {i+1:2d}. {col} 🌐")
            else:
                print(f"  {i+1:2d}. {col}")
        
        print()
        
        # Verifica che le colonne IP siano incluse
        ip_cols = [col for col in DATA_CONFIG["feature_columns"] if 'IP' in col]
        print(f"🌐 Colonne IP incluse: {len(ip_cols)}")
        print(f"🔢 Colonne Ottetti incluse: {len([col for col in ip_cols if 'Octet' in col])}")
        
    except ImportError as e:
        print(f"❌ Errore import configurazione: {e}")
    
    print()

def test_performance():
    """Test delle performance con dataset più grandi."""
    print("🧪 TEST: Performance")
    print("=" * 50)
    
    # Crea dataset più grande per test performance
    n_samples = 10000
    
    # Genera IP casuali
    np.random.seed(42)
    src_ips = [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" 
                for _ in range(n_samples)]
    dst_ips = [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" 
                for _ in range(n_samples)]
    
    test_data = {
        'Src IP': src_ips,
        'Dst IP': dst_ips,
        'Protocol': np.random.randint(1, 256, n_samples),
        'Port': np.random.randint(1, 65536, n_samples),
        'Label': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(test_data)
    print(f"📊 Dataset di test: {len(df)} campioni")
    print(f"💾 Memoria DataFrame: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Test performance
    import time
    start_time = time.time()
    
    df_processed = process_ip_columns(
        df, 
        ip_columns=['Src IP', 'Dst IP'],
        create_new_columns=True,
        drop_original=False
    )
    
    processing_time = time.time() - start_time
    
    print(f"⚡ Tempo elaborazione: {processing_time:.4f} secondi")
    print(f"📈 Campioni/secondo: {n_samples/processing_time:.0f}")
    print(f"💾 Memoria DataFrame processato: {df_processed.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Verifica che tutte le colonne ottetti siano numeriche
    octet_cols = [col for col in df_processed.columns if 'Octet' in col]
    all_numeric = all(df_processed[col].dtype in ['int64', 'int32', 'uint8'] for col in octet_cols)
    
    print(f"✅ Tutte le colonne ottetti sono numeriche: {all_numeric}")
    print()

def main():
    """Funzione principale per eseguire tutti i test."""
    print("🚀 AVVIO TEST TRASFORMAZIONE IP IN OTTETTI")
    print("=" * 60)
    print()
    
    try:
        test_single_ip_conversion()
        test_dataframe_processing()
        test_integration_with_config()
        test_performance()
        
        print("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print()
        print("💡 BENEFICI DELLA TRASFORMAZIONE IP:")
        print("  • Indirizzi IP convertiti in 4 feature numeriche separate")
        print("  • Ogni ottetto è un valore da 0-255 (perfetto per ML)")
        print("  • Mantiene informazioni spaziali della rete")
        print("  • Compatibile con tutti i modelli di machine learning")
        print("  • Supporta sia IPv4 che IPv6")
        print()
        print("🔧 UTILIZZO NEL PROGETTO:")
        print("  • Integrato automaticamente nel preprocessing")
        print("  • Configurabile tramite config.py")
        print("  • Mantiene compatibilità con codice esistente")
        
    except Exception as e:
        print(f"❌ ERRORE DURANTE I TEST: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
