# -*- coding: utf-8 -*-
"""
Modulo per il Benchmark Completo di Confronto.
Confronta le performance del modello con diverse finestre temporali e con/senza Crypto-PAn.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import sys

# Aggiungi il path del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.temporal_windows import TemporalWindowManager, benchmark_temporal_resolutions
from crypto import cryptopan_ip, deanonymize_ip

class CryptoPanComparisonBenchmark:
    """
    Benchmark completo per confrontare le performance con e senza Crypto-PAn.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inizializza il benchmark.
        
        Args:
            config: Configurazione del benchmark
        """
        self.config = config or self._get_default_config()
        self.results = {}
        self.timestamp = datetime.now().isoformat()
        
        # Crea directory di output
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'cryptopan'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'plain_text'), exist_ok=True)
        os.makedirs(os.path.join(self.config['output_dir'], 'comparison'), exist_ok=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configurazione di default per il benchmark."""
        return {
            'time_resolutions': ['1s', '5s', '10s', '1m', '5m', '10m'],
            'min_window_size': 5,
            'sample_size': 50000,  # Per velocizzare i test
            'output_dir': 'benchmark_results',
            'save_intermediate_results': True,
            'generate_comparison_plots': True,
            'test_configs': [
                {'name': 'baseline', 'use_cryptopan': False, 'description': 'Baseline senza anonimizzazione'},
                {'name': 'cryptopan', 'use_cryptopan': True, 'description': 'Con anonimizzazione Crypto-PAn'}
            ]
        }
    
    def run_single_test(self, test_config: Dict, time_resolution: str, 
                       df: pd.DataFrame) -> Dict[str, Any]:
        """
        Esegue un singolo test per una configurazione e risoluzione temporale.
        
        Args:
            test_config: Configurazione del test
            time_resolution: Risoluzione temporale
            df: Dataset originale
        
        Returns:
            Risultati del test
        """
        test_name = test_config['name']
        use_cryptopan = test_config['use_cryptopan']
        
        print(f"\n--- Test: {test_name} - Risoluzione: {time_resolution} ---")
        
        # Crea finestre temporali
        window_manager = TemporalWindowManager([time_resolution])
        windowed_df = window_manager.create_temporal_windows(
            df, 'Timestamp', time_resolution, self.config['min_window_size']
        )
        
        if windowed_df.empty:
            print(f"  Nessuna finestra valida per {time_resolution}")
            return {'error': 'Nessuna finestra valida'}
        
        print(f"  Finestre create: {len(windowed_df.groupby(['window_start', 'window_end']))}")
        print(f"  Record totali: {len(windowed_df)}")
        
        # Preprocessing con o senza Crypto-PAn
        if use_cryptopan:
            print("  Applicazione anonimizzazione Crypto-PAn...")
            # Salva il dataset originale per confronto
            original_df = windowed_df.copy()
            
            # Applica Crypto-PAn agli IP
            ip_columns = ['Src IP', 'Dst IP']
            cryptopan_key = os.urandom(32)
            
            for col in ip_columns:
                if col in windowed_df.columns:
                    windowed_df[col] = windowed_df[col].apply(
                        lambda ip: cryptopan_ip(str(ip), cryptopan_key) if pd.notna(ip) else ip
                    )
            
            # Salva la mappa di anonimizzazione
            ip_map = {
                'cryptopan_key': cryptopan_key.hex(),
                'resolution': time_resolution,
                'test_name': test_name,
                'timestamp': self.timestamp
            }
            
            map_path = os.path.join(
                self.config['output_dir'], 'cryptopan', 
                f'ip_map_{test_name}_{time_resolution}.json'
            )
            with open(map_path, 'w') as f:
                json.dump(ip_map, f, indent=4)
        
        # Preprocessing diretto delle finestre temporali
        try:
            print("  Preprocessing finestre temporali...")
            
            # Estrai features e target dalle finestre
            feature_columns = [col for col in windowed_df.columns 
                             if col not in ['window_start', 'window_end', 'window_resolution', 'window_size', 'Label']]
            
            # Converti colonne non numeriche in numeriche
            for col in feature_columns:
                if col in windowed_df.columns:
                    # Prova a convertire in numerico, se fallisce usa 0
                    try:
                        windowed_df[col] = pd.to_numeric(windowed_df[col], errors='coerce').fillna(0)
                    except:
                        windowed_df[col] = 0
            
            # Filtra solo colonne numeriche
            feature_columns = [col for col in feature_columns 
                             if windowed_df[col].dtype in ['int64', 'float64', 'float32']]
            
            # Raggruppa per finestra e crea sequenze
            window_groups = windowed_df.groupby(['window_start', 'window_end'])
            X_processed, y_processed = [], []
            
            # Trova la dimensione massima delle finestre
            max_window_size = 0
            for (start, end), group in window_groups:
                max_window_size = max(max_window_size, len(group))
            
            # Usa una dimensione fissa per tutte le finestre
            target_window_size = min(max_window_size, 20)  # Massimo 20 timesteps
            
            for (start, end), group in window_groups:
                if len(group) >= 5:  # Minimo 5 record per finestra
                    # Estrai features
                    features = group[feature_columns].values
                    
                    # Pad o tronca alla dimensione target
                    if len(features) < target_window_size:
                        # Pad con zeri se la finestra è troppo piccola
                        padding = np.zeros((target_window_size - len(features), len(feature_columns)))
                        features = np.vstack([features, padding])
                    elif len(features) > target_window_size:
                        # Tronca se la finestra è troppo grande
                        features = features[:target_window_size]
                    
                    # Usa l'ultimo record della finestra come etichetta
                    if 'Label' in group.columns:
                        try:
                            label = int(group['Label'].iloc[-1])
                        except (ValueError, TypeError):
                            label = 0
                    else:
                        label = 0
                    
                    X_processed.append(features)
                    y_processed.append(label)
            
            if not X_processed:
                return {'error': 'Nessuna finestra valida per il training'}
            
            X_processed = np.array(X_processed)
            y_processed = np.array(y_processed)
            
            print(f"  Features processate: {X_processed.shape}")
            print(f"  Etichette: {y_processed.shape}")
            print(f"  Dimensione finestra target: {target_window_size}")
            print(f"  Numero di finestre valide: {len(X_processed)}")
            print(f"  Colonne features utilizzate: {len(feature_columns)}")
            print(f"  Esempio prima finestra: {X_processed[0].shape if len(X_processed) > 0 else 'N/A'}")
            print(f"  Etichette uniche: {np.unique(y_processed)}")
            print(f"  Distribuzione etichette: {np.bincount(y_processed)}")
            print(f"  Numero di classi: {len(np.unique(y_processed))}")
            if len(np.unique(y_processed)) == 1:
                print("  ⚠️  Attenzione: Dataset con una sola classe - il modello non può imparare pattern discriminativi")
            
            # Training e valutazione
            print("  Training modello...")
            start_time = time.time()
            
            # Import locale per evitare import circolari
            from training.train import train_and_evaluate
            
            # Usa il tipo di modello specificato nella configurazione
            training_config_override = {
                'TRAINING_CONFIG': {
                    'model_type': self.config.get('model_type', 'dense')
                }
            }
            
            training_log, best_model_path = train_and_evaluate(
                X=X_processed, y=y_processed,
                config_override=training_config_override
            )
            
            training_time = time.time() - start_time
            
            if training_log is None:
                return {'error': 'Errore nel training'}
            
            # Carica il modello per le predizioni
            import tensorflow as tf
            best_model = tf.keras.models.load_model(best_model_path)
            
            # Genera predizioni per le statistiche
            # Mantieni la forma 3D per i modelli sequenziali
            if len(X_processed.shape) == 3:
                # Per modelli sequenziali (LSTM/GRU), mantieni la forma (samples, timesteps, features)
                X_reshaped = X_processed
            else:
                # Per modelli densi, ridimensiona in 2D
                X_reshaped = X_processed.reshape(-1, X_processed.shape[-1])
            
            y_pred_proba = best_model.predict(X_reshaped, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Ottieni nomi delle classi
            try:
                target_map_path = os.path.join(
                    os.path.dirname(best_model_path), 'target_anonymization_map.json'
                )
                if os.path.exists(target_map_path):
                    with open(target_map_path, 'r') as f:
                        target_map = json.load(f)
                    class_names = list(target_map['inverse_map'].values())
                else:
                    class_names = [f"Class_{i}" for i in range(len(np.unique(y_processed)))]
            except:
                class_names = [f"Class_{i}" for i in range(len(np.unique(y_processed)))]
            
            # Genera statistiche
            print("  Generazione statistiche...")
            stats_output_path = os.path.join(
                self.config['output_dir'], test_name, 
                f'statistics_{time_resolution}'
            )
            
            # Import locale per evitare import circolari
            from evaluation.stats import generate_comprehensive_report
            report_data = generate_comprehensive_report(
                X=X_reshaped, y=y_processed, y_pred=y_pred, 
                class_names=class_names,
                training_log=training_log, 
                best_model_path=best_model_path,
                output_path=stats_output_path
            )
            
            # Raccogli risultati
            test_results = {
                'test_name': test_name,
                'time_resolution': time_resolution,
                'use_cryptopan': use_cryptopan,
                'timestamp': self.timestamp,
                'training_time': float(training_time),  # Converti in float per JSON
                'dataset_stats': {
                    'total_windows': int(len(windowed_df.groupby(['window_start', 'window_end']))),
                    'total_records': int(len(windowed_df)),
                    'features_shape': [int(x) for x in X_processed.shape],  # Converti tuple in lista di int
                    'labels_shape': [int(x) for x in y_processed.shape]
                },
                'model_performance': {
                    'best_accuracy': float(max([run['accuracy'] for run in training_log])),
                    'training_runs': int(len(training_log)),
                    'best_model_path': str(best_model_path)
                },
                'evaluation_metrics': report_data['classification_metrics']['overall_metrics'],
                'statistics_path': str(stats_output_path)
            }
            
            print(f"  Test completato in {training_time:.2f} secondi")
            return test_results
            
        except Exception as e:
            print(f"  Errore durante il test: {e}")
            return {'error': str(e)}
    
    def run_complete_benchmark(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Esegue il benchmark completo per tutte le configurazioni e risoluzioni.
        
        Args:
            df: Dataset originale
        
        Returns:
            Risultati completi del benchmark
        """
        print("="*80)
        print("BENCHMARK COMPLETO: CONFRONTO CRYPTO-PAN vs PLAIN TEXT")
        print("="*80)
        print(f"Dataset: {len(df)} record")
        print(f"Timestamp range: {df['Timestamp'].min()} - {df['Timestamp'].max()}")
        print(f"Risoluzioni temporali: {self.config['time_resolutions']}")
        print(f"Configurazioni test: {len(self.config['test_configs'])}")
        print("="*80)
        
        all_results = {}
        
        # Esegui test per ogni configurazione e risoluzione
        for test_config in self.config['test_configs']:
            test_name = test_config['name']
            all_results[test_name] = {}
            
            print(f"\n{'='*20} CONFIGURAZIONE: {test_name.upper()} {'='*20}")
            
            for resolution in self.config['time_resolutions']:
                print(f"\n--- Testando risoluzione: {resolution} ---")
                
                test_result = self.run_single_test(test_config, resolution, df)
                all_results[test_name][resolution] = test_result
                
                # Salva risultati intermedi se richiesto
                if self.config['save_intermediate_results']:
                    intermediate_path = os.path.join(
                        self.config['output_dir'], 'intermediate',
                        f'{test_name}_{resolution}_results.json'
                    )
                    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
                    
                    with open(intermediate_path, 'w') as f:
                        json.dump(test_result, f, indent=4)
        
        # Genera report di confronto
        comparison_report = self._generate_comparison_report(all_results)
        
        # Salva risultati completi
        self._save_benchmark_results(all_results, comparison_report)
        
        # Stampa riepilogo finale
        self._print_final_summary(all_results, comparison_report)
        
        return all_results
    
    def _generate_comparison_report(self, all_results: Dict) -> Dict:
        """Genera un report di confronto tra le configurazioni."""
        comparison = {
            'benchmark_timestamp': self.timestamp,
            'configurations': {},
            'resolution_comparison': {},
            'overall_summary': {}
        }
        
        # Confronta configurazioni per ogni risoluzione
        for resolution in self.config['time_resolutions']:
            comparison['resolution_comparison'][resolution] = {}
            
            baseline_result = all_results.get('baseline', {}).get(resolution, {})
            cryptopan_result = all_results.get('cryptopan', {}).get(resolution, {})
            
            if 'error' not in baseline_result and 'error' not in cryptopan_result:
                # Calcola differenze
                baseline_acc = baseline_result['evaluation_metrics']['accuracy']
                cryptopan_acc = cryptopan_result['evaluation_metrics']['accuracy']
                
                accuracy_diff = cryptopan_acc - baseline_acc
                accuracy_change_pct = (accuracy_diff / baseline_acc) * 100
                
                comparison['resolution_comparison'][resolution] = {
                    'baseline_accuracy': baseline_acc,
                    'cryptopan_accuracy': cryptopan_acc,
                    'accuracy_difference': accuracy_diff,
                    'accuracy_change_percentage': accuracy_change_pct,
                    'performance_impact': 'improvement' if accuracy_diff > 0 else 'degradation',
                    'training_time_baseline': baseline_result['training_time'],
                    'training_time_cryptopan': cryptopan_result['training_time'],
                    'time_difference': cryptopan_result['training_time'] - baseline_result['training_time']
                }
        
        return comparison
    
    def _save_benchmark_results(self, all_results: Dict, comparison_report: Dict):
        """Salva tutti i risultati del benchmark."""
        # Salva risultati completi
        results_path = os.path.join(self.config['output_dir'], 'complete_benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'benchmark_config': self.config,
                'all_results': all_results,
                'comparison_report': comparison_report,
                'timestamp': self.timestamp
            }, f, indent=4)
        
        # Salva report di confronto separato
        comparison_path = os.path.join(self.config['output_dir'], 'comparison', 'comparison_report.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_report, f, indent=4)
        
        print(f"\nRisultati completi salvati in: {results_path}")
        print(f"Report di confronto salvato in: {comparison_path}")
    
    def _print_final_summary(self, all_results: Dict, comparison_report: Dict):
        """Stampa un riepilogo finale del benchmark."""
        print("\n" + "="*80)
        print("RIEPILOGO FINALE BENCHMARK")
        print("="*80)
        
        for resolution in self.config['time_resolutions']:
            if resolution in comparison_report['resolution_comparison']:
                comp = comparison_report['resolution_comparison'][resolution]
                
                print(f"\nRisoluzione: {resolution}")
                
                if 'error' in comp:
                    print(f"  Errore: {comp['error']}")
                    if 'baseline_error' in comp:
                        print(f"  Baseline Error: {comp['baseline_error']}")
                    if 'cryptopan_error' in comp:
                        print(f"  Crypto-PAn Error: {comp['cryptopan_error']}")
                else:
                    try:
                        print(f"  Baseline Accuracy: {comp['baseline_accuracy']:.4f}")
                        print(f"  Crypto-PAn Accuracy: {comp['cryptopan_accuracy']:.4f}")
                        print(f"  Differenza: {comp['accuracy_difference']:+.4f} ({comp['accuracy_change_percentage']:+.2f}%)")
                        print(f"  Impatto Performance: {comp['performance_impact']}")
                        print(f"  Tempo Training Baseline: {comp['training_time_baseline']:.2f}s")
                        print(f"  Tempo Training Crypto-PAn: {comp['training_time_cryptopan']:.2f}s")
                        print(f"  Differenza Tempo: {comp['time_difference']:+.2f}s")
                    except KeyError as e:
                        print(f"  Errore nei dati: {e}")
                        print(f"  Dati disponibili: {list(comp.keys())}")
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETATO!")
        print("="*80)

def run_quick_benchmark(df: pd.DataFrame, config: Dict = None) -> Dict:
    """
    Esegue un benchmark rapido per test iniziali.
    
    Args:
        df: Dataset
        config: Configurazione opzionale
    
    Returns:
        Risultati del benchmark rapido
    """
    if config is None:
        config = {
            'time_resolutions': ['5s', '1m'],  # Solo 2 risoluzioni per velocità
            'sample_size': 10000,
            'output_dir': 'quick_benchmark_results'
        }
    
    benchmark = CryptoPanComparisonBenchmark(config)
    return benchmark.run_complete_benchmark(df)

def run_full_benchmark(df: pd.DataFrame, config: Dict = None) -> Dict:
    """
    Esegue il benchmark completo con tutte le risoluzioni.
    
    Args:
        df: Dataset
        config: Configurazione opzionale
    
    Returns:
        Risultati completi del benchmark
    """
    if config is None:
        config = {
            'time_resolutions': ['1s', '5s', '10s', '1m', '5m', '10m'],
            'sample_size': 50000,
            'output_dir': 'full_benchmark_results',
            'generate_comparison_plots': True
        }
    
    benchmark = CryptoPanComparisonBenchmark(config)
    return benchmark.run_complete_benchmark(df)
