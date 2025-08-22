#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script di test per verificare il bilanciamento del dataset.
Questo script dimostra come il nuovo sistema di bilanciamento
risolve il problema delle classi sbilanciate.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from preprocessing.process import balance_dataset
import pandas as pd

def load_representative_sample(file_path, total_samples=1000000):
    """
    Carica un campione rappresentativo del dataset leggendo da diverse parti del file.
    """
    print(f"📁 Caricamento campione rappresentativo da {file_path}...")
    
    # Prima leggi le prime righe per vedere la struttura
    df_head = pd.read_csv(file_path, nrows=1000)
    print(f"  Prime 1000 righe: {df_head['Label'].value_counts().to_dict()}")
    
    # Poi leggi le ultime righe per vedere se ci sono attacchi
    # Per farlo, contiamo prima le righe totali
    print("  Contando righe totali...")
    with open(file_path, 'r') as f:
        total_lines = sum(1 for line in f)
    print(f"  File totale: {total_lines:,} righe")
    
    # Leggi un campione dalla fine del file
    skip_rows = max(0, total_lines - total_samples - 1)
    print(f"  Leggendo ultime {total_samples:,} righe...")
    df_tail = pd.read_csv(file_path, skiprows=range(1, skip_rows + 1))
    print(f"  Ultime righe: {df_tail['Label'].value_counts().to_dict()}")
    
    # Combina i due campioni
    df_combined = pd.concat([df_head, df_tail], ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    
    print(f"✅ Campione rappresentativo caricato: {len(df_combined)} righe")
    return df_combined

def test_dataset_balancing():
    """
    Testa le diverse strategie di bilanciamento su un campione del dataset.
    """
    print("🧪 TEST DEL SISTEMA DI BILANCIAMENTO DATASET")
    print("=" * 60)
    
    # Carica un campione rappresentativo del dataset
    try:
        df = load_representative_sample('data/cicids/2018/Wednesday-21-02-2018.csv')
    except Exception as e:
        print(f"❌ Errore nel caricamento: {e}")
        return
    
    # Mostra distribuzione originale
    print("\n🔍 DISTRIBUZIONE ORIGINALE DELLE CLASSI:")
    print("-" * 40)
    original_counts = df['Label'].value_counts()
    for label, count in original_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    # Se abbiamo solo una classe, prova a caricare un campione diverso
    if len(original_counts) == 1:
        print("\n⚠️ Solo una classe trovata. Provando a caricare da un file diverso...")
        try:
            df = load_representative_sample('data/cicids/2018/Tuesday-20-02-2018.csv')
            original_counts = df['Label'].value_counts()
            print("Nuova distribuzione:")
            for label, count in original_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {label}: {count:,} ({percentage:.1f}%)")
        except Exception as e:
            print(f"❌ Errore nel caricamento del secondo file: {e}")
            return
    
    # Test 1: Strategia Security (50% BENIGN, 50% MALICIOUS)
    print("\n🛡️ TEST 1: STRATEGIA SECURITY (50% BENIGN, 50% MALICIOUS)")
    print("-" * 60)
    security_df = balance_dataset(df, 'Label', strategy='security', max_samples_per_class=50000, benign_ratio=0.5)
    
    # Test 2: Strategia Smart
    print("\n🧠 TEST 2: STRATEGIA SMART")
    print("-" * 40)
    smart_df = balance_dataset(df, 'Label', strategy='smart', min_samples_per_class=1000)
    
    # Test 3: Strategia Bilanciata
    print("\n📊 TEST 3: STRATEGIA BILANCIATA")
    print("-" * 40)
    balanced_df = balance_dataset(df, 'Label', strategy='balanced', min_samples_per_class=1000)
    
    # Riepilogo finale
    print("\n🎯 RIEPILOGO FINALE:")
    print("=" * 60)
    print(f"Dataset originale: {len(df):,} righe")
    print(f"Dataset security: {len(security_df):,} righe")
    print(f"Dataset smart: {len(smart_df):,} righe")
    print(f"Dataset balanced: {len(balanced_df):,} righe")
    
    # Verifica che il bilanciamento security abbia funzionato
    print("\n✅ VERIFICA BILANCIAMENTO SECURITY:")
    print("-" * 40)
    security_counts = security_df['Label'].value_counts()
    print("Distribuzione dopo bilanciamento security:")
    
    # Calcola BENIGN vs MALICIOUS
    benign_count = security_counts.get('BENIGN', 0)
    malicious_count = len(security_df) - benign_count
    total_samples = len(security_df)
    
    benign_percentage = (benign_count / total_samples) * 100 if total_samples > 0 else 0
    malicious_percentage = (malicious_count / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"  BENIGN: {benign_count:,} ({benign_percentage:.1f}%)")
    print(f"  MALICIOUS (tutti gli attacchi): {malicious_count:,} ({malicious_percentage:.1f}%)")
    
    print("\nDettaglio per tipo di attacco:")
    for label, count in security_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    # Verifica se la proporzione 50/50 è stata rispettata
    if abs(benign_percentage - 50.0) < 5.0:  # Tolleranza del 5%
        print("\n🎉 Eccellente! Proporzione 50% BENIGN / 50% MALICIOUS rispettata!")
    else:
        print(f"\n⚠️ Proporzione non ottimale. Target: 50%/50%, Attuale: {benign_percentage:.1f}%/{malicious_percentage:.1f}%")

if __name__ == "__main__":
    test_dataset_balancing()
