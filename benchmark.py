#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark SNN-IDS
Unico entry point per tutti i test e benchmark del sistema.
"""

import os
import sys
import argparse
import time
import json
import copy
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Import moduli
sys.path.append(os.path.abspath('.'))
from config import *
from preprocessing.process import preprocess_pipeline
from training.train import train_model
from evaluation.metrics import evaluate_model_comprehensive

class SNNIDSBenchmark:
    """Benchmark unificato per SNN-IDS."""
    
    def __init__(self, config_override: Dict = None):
        """Inizializza benchmark con configurazione."""
        self.config_override = config_override or {}
        self.timestamp = datetime.now().isoformat()
        self.results = {}
    
    def run_preprocessing_test(self) -> Dict[str, Any]:
        """Test del sistema di preprocessing."""
        print("\n🧪 TEST PREPROCESSING")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Override configurazione se necessario
            sample_size = self.config_override.get('sample_size', PREPROCESSING_CONFIG['sample_size'])
            data_path = self.config_override.get('data_path', DATA_CONFIG['dataset_path'])
            
            # Esegui preprocessing
            X, y, label_encoder = preprocess_pipeline(
                data_path=data_path,
                sample_size=sample_size
            )
            
            processing_time = time.time() - start_time
            
            # Statistiche
            unique_classes = len(np.unique(y))
            class_distribution = np.bincount(y)
            
            result = {
                'status': 'success',
                'processing_time': processing_time,
                'dataset_stats': {
                    'X_shape': X.shape,
                    'y_shape': y.shape,
                    'unique_classes': unique_classes,
                    'class_distribution': class_distribution.tolist(),
                    'classes': label_encoder.classes_.tolist() if hasattr(label_encoder, 'classes_') else []
                }
            }
            
            print(f"✅ Preprocessing completato in {processing_time:.2f}s")
            print(f"📊 Dataset: {X.shape}")
            print(f"🏷️ Classi: {unique_classes}")
            
            return result, X, y, label_encoder
            
        except Exception as e:
            print(f"❌ Errore preprocessing: {e}")
            return {'status': 'error', 'error': str(e)}, None, None, None
    
    def run_training_test(self, X: np.ndarray, y: np.ndarray, label_encoder=None) -> Dict[str, Any]:
        """Test del sistema di training con valutazione completa."""
        print("\n🧪 TEST TRAINING")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Override configurazione
            model_type = self.config_override.get('model_type', TRAINING_CONFIG['model_type'])
            hyperparams = self.config_override.get('hyperparameters', TRAINING_CONFIG['hyperparameters'])
            
            print(f"🏗️ Modello: {model_type}")
            
            # Training
            model, training_log, model_path = train_model(
                X=X, y=y,
                model_type=model_type,
                hyperparams=hyperparams
            )
            
            training_time = time.time() - start_time
            best_accuracy = max([run['accuracy'] for run in training_log])
            
            # Valutazione completa con visualizzazioni
            evaluation_result = self._run_evaluation(model, X, y, label_encoder, model_type)
            
            result = {
                'status': 'success',
                'training_time': training_time,
                'model_type': model_type,
                'best_accuracy': best_accuracy,
                'model_path': model_path,
                'training_configurations': len(training_log),
                'training_log': training_log,
                'evaluation': evaluation_result
            }
            
            print(f"✅ Training completato in {training_time:.2f}s")
            print(f"🏆 Miglior accuratezza: {best_accuracy:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Errore training: {e}")
            return {'status': 'error', 'error': str(e)}

    def _run_single(self) -> Dict[str, Any]:
        """Esegue un ciclo completo preprocessing+training con le config correnti."""
        start_time = time.time()
        results: Dict[str, Any] = {}

        # Preprocessing
        prep_result, X, y, label_encoder = self.run_preprocessing_test()
        results['preprocessing'] = prep_result
        if prep_result.get('status') != 'success':
            results['overall_status'] = 'failed'
            results['total_time'] = time.time() - start_time
            results['timestamp'] = self.timestamp
            return results

        # Training
        train_result = self.run_training_test(X, y, label_encoder)
        results['training'] = train_result

        # Finalizza
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['timestamp'] = self.timestamp
        results['overall_status'] = (
            'success' if train_result.get('status') == 'success' and prep_result.get('status') == 'success' else 'failed'
        )
        return results
    
    def run_smoke_test(self) -> Dict[str, Any]:
        """Smoke test veloce per verificare funzionamento."""
        print("🔥 SMOKE TEST - Verifica funzionamento base")
        print("=" * 60)
        
        # Configurazione ridotta per test veloce
        smoke_config = {
            'sample_size': 5000,
            'time_windows': [
                {'window_size': 30, 'step': 15, 'name': '30s_15s'},
                {'window_size': 60, 'step': 30, 'name': '1m_30s'},
                {'window_size': 300, 'step': 150, 'name': '5m_2.5m'},
            ],
            'model_type': 'dense',  # Più veloce per test
            'hyperparameters': {
                'epochs': [2],
                'batch_size': [32],
                'learning_rate': [0.001],
                'activation': ['relu'],
                'gru_units': [32],
                'lstm_units': [32]
            }
        }
        
        # Merge con override utente
        config = {**smoke_config, **self.config_override}
        self.config_override = config
        
        # Esecuzione unificata
        results = self._run_single()
        print(f"\n🎉 SMOKE TEST COMPLETATO in {results.get('total_time', 0):.2f}s")
        print("✅ Tutti i test sono passati!" if results.get('overall_status') == 'success' else "❌ Alcuni test sono falliti")
        return results
    
    def run_colab_benchmark(self) -> Dict[str, Any]:
        """Benchmark ottimizzato per Google Colab con iperparametri significativi."""
        print("🚀 COLAB BENCHMARK - Iperparametri Significativi")
        print("=" * 60)
        
        # Configurazione ottimizzata per Colab
        colab_config = {
            'sample_size': 8000,
            'time_windows': [
                {'window_size': 30, 'step': 15, 'name': '30s_15s'},
                {'window_size': 60, 'step': 30, 'name': '1m_30s'},
                {'window_size': 300, 'step': 150, 'name': '5m_2.5m'},
            ],
            'models': [
                {'type': 'dense', 'name': 'Dense_Baseline', 'params': {'epochs': 3, 'batch_size': 64}},
                {'type': 'gru', 'name': 'GRU_Fast', 'params': {'epochs': 3, 'batch_size': 32, 'gru_units': 64}},
                {'type': 'lstm', 'name': 'LSTM_Optimized', 'params': {'epochs': 3, 'batch_size': 32, 'lstm_units': 64}}
            ],
            'learning_rates': [0.001, 0.0005],
            'aggregation_stats': [
                ['sum', 'mean', 'max'],
                ['sum', 'mean', 'std', 'max']
            ]
        }
        
        # Merge con override utente
        config = {**colab_config, **self.config_override}
        
        start_time = time.time()
        all_results = []
        
        total_configs = (len(config['time_windows']) * len(config['models']) * 
                        len(config['learning_rates']) * len(config['aggregation_stats']))
        
        print(f"🔢 Configurazioni totali: {total_configs}")
        print(f"⏰ Tempo stimato: ~{total_configs * 2:.0f}-{total_configs * 4:.0f} minuti")
        
        config_num = 0
        
        # Loop attraverso tutte le configurazioni
        for window_config in config['time_windows']:
            for model_config in config['models']:
                for lr in config['learning_rates']:
                    for agg_stats in config['aggregation_stats']:
                        
                        config_num += 1
                        config_start = time.time()
                        
                        print(f"\n🧪 CONFIGURAZIONE {config_num}/{total_configs}")
                        print(f"⏱️ Finestra: {window_config['name']}")
                        print(f"🤖 Modello: {model_config['name']}")
                        print(f"📈 Learning Rate: {lr}")
                        print(f"📊 Aggregazioni: {agg_stats}")
                        print("-" * 40)
                        
                        try:
                            # Configura per questo test (override temporaneo)
                            test_overrides = {
                                'sample_size': config['sample_size'],
                                'model_type': model_config['type'],
                                'hyperparameters': {
                                    **model_config['params'],
                                    'learning_rate': [lr]
                                }
                            }
                            original_config = self.config_override.copy()
                            self.config_override.update(test_overrides)

                            # Esecuzione unificata
                            single = self._run_single()
                            config_time = time.time() - config_start

                            # Accumula risultati
                            single.update({
                                'config_id': config_num,
                                'window_config': window_config,
                                'model_config': model_config,
                                'learning_rate': lr,
                                'aggregation_stats': agg_stats,
                                'config_time': config_time
                            })
                            all_results.append(single)

                            print(f"✅ Completato in {config_time:.1f}s")
                            if 'training' in single and 'best_accuracy' in single['training']:
                                print(f"📊 Accuracy: {single['training']['best_accuracy']:.3f}")

                            # Ripristina configurazione
                            self.config_override = original_config

                        except Exception as e:
                            print(f"❌ Errore configurazione {config_num}: {e}")
                            continue
        
        # Risultati finali
        total_time = time.time() - start_time
        results = {
            'timestamp': self.timestamp,
            'total_time': total_time,
            'total_configs': total_configs,
            'completed_configs': len(all_results),
            'configurations': all_results,
            'overall_status': 'success' if all_results else 'failed'
        }
        
        print(f"\n🎉 COLAB BENCHMARK COMPLETATO!")
        print(f"⏰ Tempo totale: {total_time/60:.1f} minuti")
        print(f"✅ Configurazioni completate: {len(all_results)}/{total_configs}")
        
        return results

    def run_smoke_all_profiles(self) -> Dict[str, Any]:
        """Esegue smoke-test per tutti i profili definiti in config.PROFILES."""
        print("🔥 SMOKE TEST - Tutti i profili (config.PROFILES)")
        print("=" * 60)

        runs = []
        start = time.time()

        # Salva config originali per ripristino
        orig_pre = copy.deepcopy(PREPROCESSING_CONFIG)
        orig_tr = copy.deepcopy(TRAINING_CONFIG)

        try:
            for prof_name, prof in PROFILES.items():
                print(f"\n🧪 Profilo: {prof_name} — {prof.get('description','')}")
                # Applica profilo
                if prof.get('preprocessing'):
                    PREPROCESSING_CONFIG.update(prof['preprocessing'])
                if prof.get('training'):
                    tr = prof['training']
                    if tr.get('model_type'):
                        TRAINING_CONFIG['model_type'] = tr['model_type']
                    if tr.get('validation_strategy'):
                        TRAINING_CONFIG['validation_strategy'] = tr['validation_strategy']
                    if tr.get('target_type'):
                        TRAINING_CONFIG['target_type'] = tr['target_type']
                    if tr.get('hyperparameters'):
                        TRAINING_CONFIG['hyperparameters'].update(tr['hyperparameters'])

                # Autoconfigura output-mode da modello
                auto_model = TRAINING_CONFIG.get('model_type')
                if auto_model in ['gru', 'lstm']:
                    PREPROCESSING_CONFIG['output_mode'] = 'sequence'
                elif auto_model == 'dense':
                    PREPROCESSING_CONFIG['output_mode'] = 'mlp_aggregated'

                # Imposta override minimo per velocità smoke
                original_override = self.config_override.copy()
                self.config_override.update({
                    'sample_size': min(PREPROCESSING_CONFIG.get('sample_size', 5000), 5000)
                })

                single = self._run_single()
                single['profile'] = prof_name
                runs.append(single)

                # Ripristina override
                self.config_override = original_override

                # Ripristina config per sicurezza prima del prossimo profilo
                PREPROCESSING_CONFIG.update(orig_pre)
                TRAINING_CONFIG.update(orig_tr)

        finally:
            # Ripristina config originali
            PREPROCESSING_CONFIG.update(orig_pre)
            TRAINING_CONFIG.update(orig_tr)

        overall = 'success' if all(r.get('overall_status') == 'success' for r in runs) else 'failed'
        total_time = time.time() - start
        print(f"\n🎉 SMOKE-ALL COMPLETATO in {total_time:.2f}s — stato: {overall}")
        return {
            'timestamp': self.timestamp,
            'overall_status': overall,
            'total_time': total_time,
            'runs': runs
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Benchmark completo con tutte le configurazioni."""
        print("🚀 BENCHMARK COMPLETO")
        print("=" * 60)
        
        results = {
            'timestamp': self.timestamp,
            'configuration_tests': [],
            'summary': {}
        }
        
        # Modelli da testare
        models_to_test = ['dense', 'gru', 'lstm']
        
        # Test ogni modello
        for model_type in models_to_test:
            print(f"\n🧪 TEST MODELLO: {model_type.upper()}")
            print("-" * 40)
            
            # Override per questo modello
            model_config = {
                **self.config_override,
                'model_type': model_type
            }
            
            # Crea benchmark per questo modello
            model_benchmark = SNNIDSBenchmark(model_config)
            
            # Esegui test
            start_time = time.time()
            
            # Preprocessing (una volta per tutti i modelli)
            if not hasattr(self, '_cached_data'):
                prep_result, X, y, label_encoder = model_benchmark.run_preprocessing_test()
                if prep_result['status'] == 'success':
                    self._cached_data = (X, y, label_encoder)
                else:
                    results['configuration_tests'].append({
                        'model_type': model_type,
                        'status': 'failed',
                        'error': 'Preprocessing failed'
                    })
                    continue
            else:
                X, y, label_encoder = self._cached_data
                prep_result = {'status': 'success'}
            
            # Training
            train_result = model_benchmark.run_training_test(X, y, label_encoder)
            
            test_time = time.time() - start_time
            
            # Salva risultati
            test_result = {
                'model_type': model_type,
                'test_time': test_time,
                'preprocessing': prep_result,
                'training': train_result,
                'status': 'success' if train_result['status'] == 'success' else 'failed'
            }
            
            results['configuration_tests'].append(test_result)
            
            if train_result['status'] == 'success':
                print(f"✅ {model_type.upper()}: {train_result['best_accuracy']:.4f} accuracy")
            else:
                print(f"❌ {model_type.upper()}: FAILED")
        
        # Genera summary
        successful_tests = [t for t in results['configuration_tests'] if t['status'] == 'success']
        
        if successful_tests:
            best_model = max(successful_tests, key=lambda x: x['training']['best_accuracy'])
            results['summary'] = {
                'total_models_tested': len(models_to_test),
                'successful_models': len(successful_tests),
                'best_model': best_model['model_type'],
                'best_accuracy': best_model['training']['best_accuracy'],
                'total_benchmark_time': sum(t['test_time'] for t in results['configuration_tests'])
            }
        
        print(f"\n🎉 BENCHMARK COMPLETO!")
        if successful_tests:
            print(f"🏆 Miglior modello: {results['summary']['best_model']} ({results['summary']['best_accuracy']:.4f})")
        
        return results
    
    def _run_evaluation(self, model, X, y, label_encoder, model_type):
        """Esegue valutazione completa con visualizzazioni."""
        from sklearn.model_selection import train_test_split
        
        print("\n📊 VALUTAZIONE COMPLETA")
        print("-" * 30)
        
        try:
            # Split per valutazione finale (gestione dataset piccoli)
            from collections import Counter
            class_counts = Counter(y)
            min_class_size = min(class_counts.values())
            
            if min_class_size < 2:
                # Dataset troppo piccolo per stratified split, usiamo split semplice
                print(f"    ⚠️ Dataset piccolo (min classe: {min_class_size}), split semplice")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=None
                )
            else:
                # Split stratificato normale
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # Verifica dimensione test set
            if len(X_test) < 5:
                print(f"    ⚠️ Test set troppo piccolo ({len(X_test)} campioni), usando tutto il dataset")
                X_test, y_test = X, y
            
            # Ottieni nomi delle classi
            if label_encoder and hasattr(label_encoder, 'classes_'):
                class_names = label_encoder.classes_.tolist()
            else:
                class_names = [f"Classe_{i}" for i in range(len(np.unique(y)))]
            
            # Directory organizzata per i risultati
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_dir = os.path.join(
                "benchmark_results", 
                f"{timestamp}_{model_type}_evaluation",
                "visualizations"
            )
            
            print(f"    🖼️ Generazione visualizzazioni in: {eval_dir}")
            
            # Valutazione completa
            evaluation_report = evaluate_model_comprehensive(
                model=model,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
                output_dir=eval_dir
            )
            
            return {
                'status': 'success',
                'evaluation_dir': eval_dir,
                'report': evaluation_report
            }
            
        except Exception as e:
            print(f"❌ Errore valutazione: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """Salva i risultati del benchmark."""
        os.makedirs(output_dir, exist_ok=True)
        
        # File principale
        results_file = os.path.join(output_dir, f"benchmark_results_{int(time.time())}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Risultati salvati: {results_file}")
        
        # CSV summary se è benchmark completo
        if 'configuration_tests' in results:
            self._save_csv_summary(results, output_dir)
    
    def _save_csv_summary(self, results: Dict, output_dir: str):
        """Salva summary in CSV."""
        try:
            import pandas as pd
            
            data = []
            for test in results['configuration_tests']:
                if test['status'] == 'success':
                    data.append({
                        'model_type': test['model_type'],
                        'accuracy': test['training']['best_accuracy'],
                        'training_time': test['training']['training_time'],
                        'total_time': test['test_time']
                    })
            
            if data:
                df = pd.DataFrame(data)
                csv_file = os.path.join(output_dir, "benchmark_summary.csv")
                df.to_csv(csv_file, index=False)
                print(f"📊 Summary CSV: {csv_file}")
                
        except ImportError:
            print("⚠️ pandas non disponibile per CSV export")

def main():
    """Entry point principale."""
    parser = argparse.ArgumentParser(
        description='Benchmark SNN-IDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi:
  python3 benchmark.py --smoke-test                    # Test veloce
  python3 benchmark.py --colab-benchmark               # Benchmark per Colab (36 config)
  python3 benchmark.py --full                          # Benchmark completo
  python3 benchmark.py --sample-size 10000 --model gru # Test custom

  # Profili predefiniti (vedi config.PROFILES)
  python3 benchmark.py --profile gru_sequence --data-path data/cicids/2018

  # Override puntuali di preprocessing/training
  python3 benchmark.py --smoke-test \
      --before 180 --after 60 --bin-seconds 5 --malicious-only true --min-bins 3 \
      --target-type multiclass --validation k_fold --model gru \
      --epochs 3 --batch-size 32 --learning-rate 0.001

  # Smoke test su TUTTI i profili
  python3 benchmark.py --smoke-all --data-path data/cicids/2018
        '''
    )
    
    # Argomenti principali
    parser.add_argument('--smoke-test', action='store_true',
                       help='Esegue smoke test veloce')
    parser.add_argument('--full', action='store_true',
                       help='Esegue benchmark completo con tutti i modelli')
    parser.add_argument('--colab-benchmark', action='store_true',
                       help='Benchmark ottimizzato per Google Colab con iperparametri significativi')
    parser.add_argument('--smoke-all', action='store_true',
                       help='Esegue smoke test per tutti i profili definiti in config.PROFILES')
    
    # Configurazioni custom
    parser.add_argument('--sample-size', type=int,
                       help='Numero di campioni da utilizzare')
    parser.add_argument('--model', choices=['dense', 'gru', 'lstm'],
                       help='Tipo di modello da testare')
    parser.add_argument('--data-path', type=str,
                       help='Path ai dati del dataset')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Directory output risultati')

    # Profili e override configurazioni (generali)
    parser.add_argument('--profile', type=str,
                       help='Nome profilo preset (vedi config.PROFILES)')
    # Preprocessing overrides
    parser.add_argument('--before', type=int, help='Secondi prima del primo evento (finestra)')
    parser.add_argument('--after', type=int, help='Secondi dopo l\'ultimo evento (finestra)')
    parser.add_argument('--bin-seconds', type=int, help='Granularità bin temporali per sequenze')
    parser.add_argument('--malicious-only', type=str, choices=['true','false'],
                       help='Mantieni solo intervalli/bin con eventi malevoli (true/false)')
    parser.add_argument('--min-bins', type=int, help='Minimo numero di bin per accettare una sequenza')
    # RIMOSSO: output-mode gestito automaticamente in base al modello
    # Training overrides
    parser.add_argument('--target-type', choices=['binary','multiclass'], help='Target preferito')
    parser.add_argument('--validation', choices=['k_fold','train_test_split'], help='Strategia di validazione')
    parser.add_argument('--epochs', type=int, help='Epoche (singolo valore)')
    parser.add_argument('--batch-size', type=int, help='Batch size (singolo valore)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (singolo valore)')
    
    args = parser.parse_args()
    
    # Costruisci configurazione
    config_override = {}
    if args.sample_size:
        config_override['sample_size'] = args.sample_size
    if args.model:
        config_override['model_type'] = args.model
    if args.data_path:
        config_override['data_path'] = args.data_path
    # Applica profilo, se presente
    if args.profile:
        if 'PROFILES' in globals() and args.profile in PROFILES:
            prof = PROFILES[args.profile]
            # Merge preprocessing
            if prof.get('preprocessing'):
                PREPROCESSING_CONFIG.update(prof['preprocessing'])
            # Merge training
            if prof.get('training'):
                tr = prof['training']
                # Aggiorna top-level
                if tr.get('model_type'):
                    TRAINING_CONFIG['model_type'] = tr['model_type']
                if tr.get('validation_strategy'):
                    TRAINING_CONFIG['validation_strategy'] = tr['validation_strategy']
                if tr.get('target_type'):
                    TRAINING_CONFIG['target_type'] = tr['target_type']
                # Merge hyperparameters senza perdere le chiavi esistenti
                if tr.get('hyperparameters'):
                    TRAINING_CONFIG['hyperparameters'].update(tr['hyperparameters'])
        else:
            print(f"⚠️ Profilo non trovato: {args.profile}")
    # Override puntuali PREPROCESSING
    if args.before is not None:
        PREPROCESSING_CONFIG['window_before_first_malicious_s'] = args.before
    if args.after is not None:
        PREPROCESSING_CONFIG['window_after_first_malicious_s'] = args.after
    if args.bin_seconds is not None:
        PREPROCESSING_CONFIG['time_bin_seconds'] = args.bin_seconds
    if args.malicious_only is not None:
        PREPROCESSING_CONFIG['malicious_only_windows'] = (args.malicious_only.lower() == 'true')
    if args.min_bins is not None:
        PREPROCESSING_CONFIG['min_sequence_bins'] = args.min_bins
    # Impostazione automatica output-mode in base al modello
    # RNN (gru/lstm) richiedono sequenze 3D; dense usa 2D aggregato.
    auto_model = config_override.get('model_type', TRAINING_CONFIG.get('model_type'))
    if auto_model in ['gru', 'lstm']:
        PREPROCESSING_CONFIG['output_mode'] = 'sequence'
    elif auto_model == 'dense':
        PREPROCESSING_CONFIG['output_mode'] = 'mlp_aggregated'
    # Override puntuali TRAINING
    if args.target_type is not None:
        TRAINING_CONFIG['target_type'] = args.target_type
    if args.validation is not None:
        TRAINING_CONFIG['validation_strategy'] = args.validation
    # Hyperparams – accetta scalari, il grid li normalizza a liste
    if args.epochs is not None:
        TRAINING_CONFIG['hyperparameters']['epochs'] = args.epochs
    if args.batch_size is not None:
        TRAINING_CONFIG['hyperparameters']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        TRAINING_CONFIG['hyperparameters']['learning_rate'] = args.learning_rate
    
    # Crea benchmark
    benchmark = SNNIDSBenchmark(config_override)
    
    try:
        # Esegui test appropriato
        if args.smoke_test:
            results = benchmark.run_smoke_test()
        elif args.full:
            results = benchmark.run_full_benchmark()
        elif args.colab_benchmark:
            results = benchmark.run_colab_benchmark()
        elif args.smoke_all:
            results = benchmark.run_smoke_all_profiles()
        else:
            # Test singolo: preprocessing + training
            print("🧪 TEST SINGOLO")
            print("=" * 50)
            
            prep_result, X, y, label_encoder = benchmark.run_preprocessing_test()
            if prep_result['status'] == 'success':
                train_result = benchmark.run_training_test(X, y, label_encoder)
                results = {
                    'preprocessing': prep_result,
                    'training': train_result,
                    'timestamp': benchmark.timestamp
                }
            else:
                results = {'preprocessing': prep_result}
        
        # Salva risultati
        benchmark.save_results(results, args.output_dir)
        
        return 0 if results.get('overall_status', 'success') == 'success' else 1
        
    except Exception as e:
        print(f"\n❌ Errore durante il benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
