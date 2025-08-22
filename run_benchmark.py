#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Principale per il Benchmark Completo.
Esegue il benchmark con diverse finestre temporali e confronta le performance
con e senza anonimizzazione Crypto-PAn.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append(os.path.abspath('.'))

from benchmark.comparison_benchmark import (
    CryptoPanComparisonBenchmark, 
    run_quick_benchmark, 
    run_full_benchmark
)
from benchmark.visualization import visualize_benchmark_results
from preprocessing.temporal_windows import benchmark_temporal_resolutions

def load_sample_data(sample_size: int = None) -> pd.DataFrame:
    """
    Carica dati di esempio per il benchmark.
    
    Args:
        sample_size: Numero di campioni da caricare (None per tutti)
    
    Returns:
        DataFrame con i dati
    """
    print("Caricamento dati per il benchmark...")
    
    # Cerca file CSV nella directory data
    data_dir = "data/cicids"
    if not os.path.exists(data_dir):
        print(f"Directory dati non trovata: {data_dir}")
        print("Creazione dataset di esempio...")
        
        # Crea dataset di esempio per test
        return create_sample_dataset(sample_size or 10000)
    
    # Carica dati reali
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Nessun file CSV trovato in {data_dir}")
        return create_sample_dataset(sample_size or 10000)
    
    # Carica il primo file CSV
    first_file = os.path.join(data_dir, csv_files[0])
    print(f"Caricamento da: {first_file}")
    
    if sample_size:
        df = pd.read_csv(first_file, nrows=sample_size)
        print(f"Caricati {len(df)} campioni (modalità sample)")
    else:
        df = pd.read_csv(first_file)
        print(f"Caricati {len(df)} campioni (dataset completo)")
    
    # Assicura che la colonna timestamp esista
    if 'Timestamp' not in df.columns:
        print("Colonna 'Timestamp' non trovata, creazione timestamp simulato...")
        df['Timestamp'] = pd.date_range(
            start='2024-01-01 00:00:00', 
            periods=len(df), 
            freq='1S'
        )
    
    return df

def create_sample_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """
    Crea un dataset di esempio per test quando i dati reali non sono disponibili.
    
    Args:
        n_samples: Numero di campioni da generare
    
    Returns:
        DataFrame con dati di esempio
    """
    print(f"Creazione dataset di esempio con {n_samples} campioni...")
    
    import numpy as np
    
    # Genera timestamp
    timestamps = pd.date_range(
        start='2024-01-01 00:00:00', 
        periods=n_samples, 
        freq='1S'
    )
    
    # Genera IP di esempio
    src_ips = [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_samples)]
    dst_ips = [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_samples)]
    
    # Genera altre feature
    src_ports = np.random.randint(1024, 65535, n_samples)
    dst_ports = np.random.randint(1, 1024, n_samples)
    protocols = np.random.choice([6, 17, 1], n_samples)  # TCP, UDP, ICMP
    
    # Genera etichette (simula traffico normale e anomalo)
    labels = np.random.choice(['Normal', 'DoS', 'PortScan', 'BruteForce'], n_samples, p=[0.7, 0.1, 0.1, 0.1])
    
    # Crea DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Src IP': src_ips,
        'Dst IP': dst_ips,
        'Src Port': src_ports,
        'Dst Port': dst_ports,
        'Protocol': protocols,
        'Flow Duration': np.random.randint(1, 1000, n_samples),
        'Total Fwd Packet': np.random.randint(1, 100, n_samples),
        'Total Bwd packets': np.random.randint(1, 100, n_samples),
        'Total Length of Fwd Packet': np.random.randint(100, 10000, n_samples),
        'Total Length of Bwd Packet': np.random.randint(100, 10000, n_samples),
        'Flow Bytes/s': np.random.randint(1000, 100000, n_samples),
        'Flow Packets/s': np.random.randint(10, 1000, n_samples),
        'Flow IAT Mean': np.random.uniform(1, 100, n_samples),
        'Flow IAT Std': np.random.uniform(1, 50, n_samples),
        'Flow IAT Max': np.random.randint(10, 200, n_samples),
        'Flow IAT Min': np.random.randint(1, 20, n_samples),
        'Fwd IAT Mean': np.random.uniform(1, 100, n_samples),
        'Bwd IAT Mean': np.random.uniform(1, 100, n_samples),
        'Fwd Header Length': np.random.randint(20, 100, n_samples),
        'Bwd Header Length': np.random.randint(20, 100, n_samples),
        'Average Packet Size': np.random.uniform(50, 1500, n_samples),
        'Fwd Segment Size Avg': np.random.uniform(50, 1500, n_samples),
        'Bwd Segment Size Avg': np.random.uniform(50, 1500, n_samples),
        'Label': labels
    })
    
    print("Dataset di esempio creato con successo!")
    return df

def run_benchmark_analysis(df: pd.DataFrame, benchmark_type: str = 'quick'):
    """
    Esegue l'analisi completa del benchmark.
    
    Args:
        df: Dataset da analizzare
        benchmark_type: Tipo di benchmark ('quick' o 'full')
    """
    print(f"\n{'='*80}")
    print(f"AVVIO BENCHMARK: {benchmark_type.upper()}")
    print(f"{'='*80}")
    
    # Prima, analizza le finestre temporali disponibili
    print("\n--- ANALISI FINESTRE TEMPORALI ---")
    window_stats = benchmark_temporal_resolutions(df, 'Timestamp')
    
    # Esegui il benchmark
    if benchmark_type == 'quick':
        print("\n--- BENCHMARK RAPIDO ---")
        results = run_quick_benchmark(df)
    else:
        print("\n--- BENCHMARK COMPLETO ---")
        results = run_full_benchmark(df)
    
    return results

def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Benchmark Completo: Confronto Crypto-PAn vs Plain Text con Finestre Temporali'
    )
    
    parser.add_argument(
        '--type', 
        choices=['quick', 'full'], 
        default='quick',
        help='Tipo di benchmark: quick (veloce) o full (completo)'
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=50000,
        help='Numero di campioni da utilizzare (0 per tutti)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='benchmark_results',
        help='Directory di output per i risultati'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Genera grafici di visualizzazione dopo il benchmark'
    )
    
    parser.add_argument(
        '--time-resolutions', 
        nargs='+', 
        default=['1s', '5s', '10s', '1m', '5m', '10m'],
        help='Risoluzioni temporali da testare'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("BENCHMARK COMPLETO: CONFRONTO CRYPTO-PAN vs PLAIN TEXT")
    print("="*80)
    print(f"Tipo: {args.type}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'Completo'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Risoluzioni temporali: {args.time_resolutions}")
    print(f"Genera visualizzazioni: {args.visualize}")
    print("="*80)
    
    try:
        # Carica dati
        sample_size = args.sample_size if args.sample_size > 0 else None
        df = load_sample_data(sample_size)
        
        if df.empty:
            print("Errore: Impossibile caricare i dati")
            return 1
        
        print(f"\nDataset caricato: {len(df)} record")
        print(f"Colonne disponibili: {list(df.columns)}")
        print(f"Timestamp range: {df['Timestamp'].min()} - {df['Timestamp'].max()}")
        
        # Configurazione personalizzata
        config = {
            'time_resolutions': args.time_resolutions,
            'sample_size': sample_size,
            'output_dir': args.output_dir,
            'save_intermediate_results': True,
            'generate_comparison_plots': True,
            'min_window_size': 5,
            'model_type': 'gru',  # Usa modello GRU per dati sequenziali
            'test_configs': [
                {'name': 'baseline', 'use_cryptopan': False, 'description': 'Baseline senza anonimizzazione'},
                {'name': 'cryptopan', 'use_cryptopan': True, 'description': 'Con anonimizzazione Crypto-PAn'}
            ]
        }
        
        # Esegui benchmark
        if args.type == 'quick':
            benchmark = CryptoPanComparisonBenchmark(config)
            results = benchmark.run_complete_benchmark(df)
        else:
            benchmark = CryptoPanComparisonBenchmark(config)
            results = benchmark.run_complete_benchmark(df)
        
        # Genera visualizzazioni se richiesto
        if args.visualize and results:
            print("\n--- GENERAZIONE VISUALIZZAZIONI ---")
            try:
                visualizer = visualize_benchmark_results(args.output_dir)
                print("Visualizzazioni generate con successo!")
            except Exception as e:
                print(f"Errore durante la generazione delle visualizzazioni: {e}")
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETATO CON SUCCESSO!")
        print("="*80)
        print(f"Risultati salvati in: {args.output_dir}")
        
        if args.visualize:
            print(f"Grafici salvati in: {args.output_dir}/visualizations")
        
        return 0
        
    except Exception as e:
        print(f"\nErrore durante l'esecuzione del benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
