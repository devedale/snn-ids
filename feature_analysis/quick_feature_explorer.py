#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Feature Explorer - Panoramica veloce di tutte le features disponibili
Cartella separata per non interferire con il progetto principale
"""

import pandas as pd
import numpy as np
import os
import sys

# Aggiungi path parent per accedere ai dati
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def explore_all_columns():
    """Esplora velocemente tutte le colonne disponibili."""
    print("🔍 QUICK FEATURE EXPLORER")
    print("=" * 50)
    
    # Path alla cache del progetto principale
    cache_dir = "../preprocessed_cache"
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache non trovata: {cache_dir}")
        print("💡 Assicurati di essere nella cartella feature_analysis/")
        return
    
    # Trova primo file disponibile
    days = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    
    if not days:
        print("❌ Nessun giorno trovato nella cache")
        return
    
    for day in days[:1]:  # Prendi solo il primo
        day_path = os.path.join(cache_dir, day)
        attack_file = os.path.join(day_path, "attack_records.csv")
        
        if os.path.exists(attack_file):
            print(f"📅 Analizzando: {day}")
            
            try:
                # Carica solo header + qualche riga
                df = pd.read_csv(attack_file, nrows=1000)
                print(f"📊 Trovate {len(df.columns)} colonne totali")
                
                # Lista tutte le colonne
                print(f"\n📋 TUTTE LE COLONNE DISPONIBILI:")
                print("-" * 60)
                
                for i, col in enumerate(df.columns):
                    # Calcola basic info per la colonna
                    sample_data = df[col].dropna()
                    unique_count = len(sample_data.unique()) if len(sample_data) > 0 else 0
                    
                    # Prova a capire il tipo
                    try:
                        numeric_data = pd.to_numeric(sample_data, errors='coerce')
                        numeric_ratio = numeric_data.notna().sum() / len(sample_data) if len(sample_data) > 0 else 0
                        col_type = "🔢 NUM" if numeric_ratio > 0.8 else "📝 CAT"
                    except:
                        col_type = "📝 CAT"
                    
                    print(f"{i+1:2d}. {col:<35} {col_type} (uniques: {unique_count:4d})")
                
                # Classifica automatica per categoria
                print(f"\n🏷️ CLASSIFICAZIONE AUTOMATICA:")
                print("-" * 60)
                
                categories = {
                    '🆔 Identificatori': [],
                    '🌐 IP/Network': [],
                    '⏰ Temporali': [],
                    '🚩 Flag TCP': [],
                    '📏 Dimensioni/Bytes': [],
                    '📈 Rate/Velocità': [],
                    '📊 Statistiche': [],
                    '🔧 Protocollo': [],
                    '🏷️ Labels': [],
                    '❓ Altri': []
                }
                
                for col in df.columns:
                    col_lower = col.lower()
                    
                    if any(x in col_lower for x in ['id', 'flow id']):
                        categories['🆔 Identificatori'].append(col)
                    elif 'ip' in col_lower:
                        categories['🌐 IP/Network'].append(col)
                    elif any(x in col_lower for x in ['timestamp', 'time', 'duration', 'iat']):
                        categories['⏰ Temporali'].append(col)
                    elif 'flag' in col_lower:
                        categories['🚩 Flag TCP'].append(col)
                    elif any(x in col_lower for x in ['length', 'size', 'bytes', 'packet']):
                        categories['📏 Dimensioni/Bytes'].append(col)
                    elif any(x in col_lower for x in ['/s', 'rate', 'ratio']):
                        categories['📈 Rate/Velocità'].append(col)
                    elif any(x in col_lower for x in ['mean', 'std', 'max', 'min', 'avg']):
                        categories['📊 Statistiche'].append(col)
                    elif any(x in col_lower for x in ['protocol', 'port']):
                        categories['🔧 Protocollo'].append(col)
                    elif any(x in col_lower for x in ['label', 'category']):
                        categories['🏷️ Labels'].append(col)
                    else:
                        categories['❓ Altri'].append(col)
                
                for category, cols in categories.items():
                    if cols:
                        print(f"\n{category} ({len(cols)}):")
                        for col in cols:
                            print(f"  • {col}")
                
                # Quick stats interessanti
                print(f"\n📈 QUICK STATISTICS (campione 1000 righe):")
                print("-" * 60)
                
                interesting_cols = [
                    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 
                    'Total Fwd Packet', 'Protocol', 'Label',
                    'Average Packet Size', 'Flow IAT Mean'
                ]
                
                for col in interesting_cols:
                    if col in df.columns:
                        if col in ['Label', 'Protocol']:
                            # Categoriche
                            try:
                                counts = df[col].value_counts().head(3)
                                values_str = ', '.join([f'{k}({v})' for k, v in counts.items()])
                                print(f"  📝 {col:<25}: {values_str}")
                            except:
                                print(f"  📝 {col:<25}: errore lettura")
                        else:
                            # Numeriche
                            try:
                                data = pd.to_numeric(df[col], errors='coerce').dropna()
                                if len(data) > 0:
                                    print(f"  🔢 {col:<25}: μ={data.mean():.1f}, σ={data.std():.1f}, range=[{data.min():.1f}, {data.max():.1f}]")
                                else:
                                    print(f"  🔢 {col:<25}: nessun dato numerico valido")
                            except:
                                print(f"  🔢 {col:<25}: errore conversione")
                
                # Confronto con config attuale
                print(f"\n⚙️ CONFRONTO CON CONFIG ATTUALE:")
                print("-" * 60)
                
                try:
                    from config import DATA_CONFIG
                    current_features = DATA_CONFIG.get('feature_columns', [])
                    print(f"📊 Features attualmente in config: {len(current_features)}")
                    print(f"📊 Features totali disponibili: {len(df.columns)}")
                    print(f"📊 Features non utilizzate: {len(df.columns) - len(current_features)}")
                    
                    missing_in_data = [f for f in current_features if f not in df.columns]
                    if missing_in_data:
                        print(f"⚠️  Features in config ma non nei dati: {missing_in_data}")
                    
                    print(f"\n💡 Features disponibili ma non in config:")
                    unused = [col for col in df.columns if col not in current_features]
                    for i, col in enumerate(unused[:15]):  # Primi 15
                        print(f"  {i+1:2d}. {col}")
                    if len(unused) > 15:
                        print(f"  ... e altre {len(unused)-15} features")
                        
                except ImportError:
                    print("⚠️  Impossibile importare config.py")
                
            except Exception as e:
                print(f"❌ Errore nell'analisi: {e}")
                return
            
            break
    
    print(f"\n" + "=" * 60)
    print(f"🎯 PROSSIMI PASSI CONSIGLIATI:")
    print(f"1. 📊 python3 feature_selector_analyzer.py  → Analisi completa con ranking")
    print(f"2. 📈 python3 statistical_analyzer.py       → Statistiche avanzate")
    print(f"3. 📝 Aggiorna config.py con features migliori")
    print(f"=" * 60)

if __name__ == "__main__":
    explore_all_columns()
