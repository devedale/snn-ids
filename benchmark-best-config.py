#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Configurazione Migliore SNN-IDS
Test e fine-tuning della configurazione ottimale identificata:
GRU - epochs=15, batch=64, lr=0.01, tanh, units=64
Accuracy: 96.24%
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Import del benchmark esistente
sys.path.append(os.path.abspath('.'))
from benchmark import SNNIDSBenchmark
from config import TRAINING_CONFIG, PREPROCESSING_CONFIG, DATA_CONFIG, FEDERATED_CONFIG, HOMOMORPHIC_CONFIG
from federated.fl_benchmark import run_fl_best_config, run_fl_comparison_sweep

class BestConfigBenchmark:
    """Benchmark dedicato alla configurazione migliore con fine-tuning."""
    
    def __init__(self, sample_size: int = None, data_path: str = None):
        """Inizializza il benchmark per la configurazione migliore."""
        self.sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
        self.data_path = data_path or DATA_CONFIG['dataset_path']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"best_config_benchmark_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurazione base migliore identificata
        self.best_base_config = {
            'model_type': 'gru',
            'epochs': 15,
            'batch_size': 64,
            'learning_rate': 0.01,
            'activation': 'tanh',
            'gru_units': 64
        }
        
        print(f"🏆 BENCHMARK CONFIGURAZIONE MIGLIORE")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Sample size: {self.sample_size}")
        print(f"⚙️  Configurazione base: GRU, epochs=15, batch=64, lr=0.01, tanh, units=64")
    
    def test_baseline_config(self) -> Dict:
        """Testa la configurazione base per confermare le performance."""
        print(f"\n🎯 TEST BASELINE - Conferma Configurazione Migliore")
        print(f"{'─'*60}")
        
        hyperparams = {
            'epochs': [self.best_base_config['epochs']],
            'batch_size': [self.best_base_config['batch_size']],
            'learning_rate': [self.best_base_config['learning_rate']],
            'activation': [self.best_base_config['activation']],
            'gru_units': [self.best_base_config['gru_units']]
        }
        
        result = self._run_single_test('gru', hyperparams, 'baseline_confirmation')
        
        if result.get('status') == 'success':
            accuracy = result['best_accuracy']
            print(f"✅ Baseline confermato: {accuracy:.6f}")
            return result
        else:
            print(f"❌ Errore nel test baseline: {result.get('error', 'Unknown')}")
            return result
    
    def fine_tune_learning_rate(self) -> Dict:
        """Fine-tuning granulare del learning rate attorno a 0.01."""
        print(f"\n🔍 FINE-TUNING LEARNING RATE")
        print(f"📌 Range: 0.008 - 0.015 attorno al valore ottimale 0.01")
        print(f"{'─'*60}")
        
        learning_rates = [0.008, 0.009, 0.01, 0.011, 0.012, 0.015]
        results = []
        
        for lr in learning_rates:
            print(f"  🧪 Testing learning_rate = {lr}")
            
            hyperparams = {
                'epochs': [self.best_base_config['epochs']],
                'batch_size': [self.best_base_config['batch_size']],
                'learning_rate': [lr],
                'activation': [self.best_base_config['activation']],
                'gru_units': [self.best_base_config['gru_units']]
            }
            
            result = self._run_single_test('gru', hyperparams, f'lr_tuning_{lr}')
            results.append(result)
            
            if result.get('status') == 'success':
                accuracy = result['best_accuracy']
                print(f"    ✅ Accuracy: {accuracy:.6f}")
            else:
                print(f"    ❌ Errore: {result.get('error', 'Unknown')}")
        
        # Trova il migliore
        best_result = max([r for r in results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        if best_result:
            best_lr = best_result['config']['hyperparameters']['learning_rate'][0]
            best_acc = best_result['best_accuracy']
            print(f"\n🏆 Miglior learning rate: {best_lr} (accuracy: {best_acc:.6f})")
            self.best_base_config['learning_rate'] = best_lr
        
        return {'best_result': best_result, 'all_results': results}
    
    def fine_tune_gru_units(self) -> Dict:
        """Fine-tuning delle unità GRU attorno a 64."""
        print(f"\n🧠 FINE-TUNING GRU UNITS")
        print(f"📌 Range: 56-80 attorno al valore ottimale 64")
        print(f"{'─'*60}")
        
        gru_units = [56, 60, 64, 68, 72, 80]
        results = []
        
        for units in gru_units:
            print(f"  🧪 Testing gru_units = {units}")
            
            hyperparams = {
                'epochs': [self.best_base_config['epochs']],
                'batch_size': [self.best_base_config['batch_size']],
                'learning_rate': [self.best_base_config['learning_rate']],
                'activation': [self.best_base_config['activation']],
                'gru_units': [units]
            }
            
            result = self._run_single_test('gru', hyperparams, f'units_tuning_{units}')
            results.append(result)
            
            if result.get('status') == 'success':
                accuracy = result['best_accuracy']
                print(f"    ✅ Accuracy: {accuracy:.6f}")
            else:
                print(f"    ❌ Errore: {result.get('error', 'Unknown')}")
        
        # Trova il migliore
        best_result = max([r for r in results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        if best_result:
            best_units = best_result['config']['hyperparameters']['gru_units'][0]
            best_acc = best_result['best_accuracy']
            print(f"\n🏆 Miglior GRU units: {best_units} (accuracy: {best_acc:.6f})")
            self.best_base_config['gru_units'] = best_units
        
        return {'best_result': best_result, 'all_results': results}
    
    def fine_tune_epochs(self) -> Dict:
        """Fine-tuning del numero di epoche attorno a 15."""
        print(f"\n📈 FINE-TUNING EPOCHS")
        print(f"📌 Range: 12-20 attorno al valore ottimale 15")
        print(f"{'─'*60}")
        
        epochs_list = [12, 13, 15, 17, 18, 20]
        results = []
        
        for epochs in epochs_list:
            print(f"  🧪 Testing epochs = {epochs}")
            
            hyperparams = {
                'epochs': [epochs],
                'batch_size': [self.best_base_config['batch_size']],
                'learning_rate': [self.best_base_config['learning_rate']],
                'activation': [self.best_base_config['activation']],
                'gru_units': [self.best_base_config['gru_units']]
            }
            
            result = self._run_single_test('gru', hyperparams, f'epochs_tuning_{epochs}')
            results.append(result)
            
            if result.get('status') == 'success':
                accuracy = result['best_accuracy']
                training_time = result.get('training_time', 0)
                print(f"    ✅ Accuracy: {accuracy:.6f} (tempo: {training_time:.1f}s)")
            else:
                print(f"    ❌ Errore: {result.get('error', 'Unknown')}")
        
        # Trova il migliore
        best_result = max([r for r in results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        if best_result:
            best_epochs = best_result['config']['hyperparameters']['epochs'][0]
            best_acc = best_result['best_accuracy']
            print(f"\n🏆 Miglior epochs: {best_epochs} (accuracy: {best_acc:.6f})")
            self.best_base_config['epochs'] = best_epochs
        
        return {'best_result': best_result, 'all_results': results}
    
    def fine_tune_batch_size(self) -> Dict:
        """Fine-tuning della batch size attorno a 64."""
        print(f"\n📦 FINE-TUNING BATCH SIZE")
        print(f"📌 Range: 48-96 attorno al valore ottimale 64")
        print(f"{'─'*60}")
        
        batch_sizes = [48, 56, 64, 72, 80, 96]
        results = []
        
        for batch_size in batch_sizes:
            print(f"  🧪 Testing batch_size = {batch_size}")
            
            hyperparams = {
                'epochs': [self.best_base_config['epochs']],
                'batch_size': [batch_size],
                'learning_rate': [self.best_base_config['learning_rate']],
                'activation': [self.best_base_config['activation']],
                'gru_units': [self.best_base_config['gru_units']]
            }
            
            result = self._run_single_test('gru', hyperparams, f'batch_tuning_{batch_size}')
            results.append(result)
            
            if result.get('status') == 'success':
                accuracy = result['best_accuracy']
                training_time = result.get('training_time', 0)
                print(f"    ✅ Accuracy: {accuracy:.6f} (tempo: {training_time:.1f}s)")
            else:
                print(f"    ❌ Errore: {result.get('error', 'Unknown')}")
        
        # Trova il migliore
        best_result = max([r for r in results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        if best_result:
            best_batch = best_result['config']['hyperparameters']['batch_size'][0]
            best_acc = best_result['best_accuracy']
            print(f"\n🏆 Miglior batch size: {best_batch} (accuracy: {best_acc:.6f})")
            self.best_base_config['batch_size'] = best_batch
        
        return {'best_result': best_result, 'all_results': results}
    
    def test_regularization(self) -> Dict:
        """Testa l'aggiunta di dropout per prevenire overfitting."""
        print(f"\n🛡️ TEST REGULARIZATION (Dropout)")
        print(f"📌 Valori: 0.0, 0.1, 0.2, 0.3")
        print(f"{'─'*60}")
        
        dropout_values = [0.0, 0.1, 0.2, 0.3]
        results = []
        
        for dropout in dropout_values:
            print(f"  🧪 Testing dropout = {dropout}")
            
            hyperparams = {
                'epochs': [self.best_base_config['epochs']],
                'batch_size': [self.best_base_config['batch_size']],
                'learning_rate': [self.best_base_config['learning_rate']],
                'activation': [self.best_base_config['activation']],
                'gru_units': [self.best_base_config['gru_units']],
                'dropout': [dropout]
            }
            
            result = self._run_single_test('gru', hyperparams, f'dropout_tuning_{dropout}')
            results.append(result)
            
            if result.get('status') == 'success':
                accuracy = result['best_accuracy']
                print(f"    ✅ Accuracy: {accuracy:.6f}")
            else:
                print(f"    ❌ Errore: {result.get('error', 'Unknown')}")
        
        # Trova il migliore
        best_result = max([r for r in results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        if best_result:
            best_dropout = best_result['config']['hyperparameters']['dropout'][0]
            best_acc = best_result['best_accuracy']
            print(f"\n🏆 Miglior dropout: {best_dropout} (accuracy: {best_acc:.6f})")
            if best_dropout > 0:
                self.best_base_config['dropout'] = best_dropout
        
        return {'best_result': best_result, 'all_results': results}
    
    def run_complete_fine_tuning(self) -> Dict:
        """Esegue il fine-tuning completo della configurazione migliore."""
        print(f"\n{'='*70}")
        print(f"🎯 FINE-TUNING COMPLETO CONFIGURAZIONE MIGLIORE")
        print(f"{'='*70}")
        
        start_time = time.time()
        all_phases = {}
        
        # Fase 1: Test baseline
        print(f"\n📋 FASE 1/5: Conferma Baseline")
        baseline_result = self.test_baseline_config()
        all_phases['baseline'] = baseline_result
        baseline_accuracy = baseline_result.get('best_accuracy', 0) if baseline_result.get('status') == 'success' else 0
        
        # Fase 2: Fine-tuning learning rate
        print(f"\n📋 FASE 2/5: Fine-tuning Learning Rate")
        lr_results = self.fine_tune_learning_rate()
        all_phases['learning_rate'] = lr_results
        
        # Fase 3: Fine-tuning GRU units
        print(f"\n📋 FASE 3/5: Fine-tuning GRU Units")
        units_results = self.fine_tune_gru_units()
        all_phases['gru_units'] = units_results
        
        # Fase 4: Fine-tuning epochs
        print(f"\n📋 FASE 4/5: Fine-tuning Epochs")
        epochs_results = self.fine_tune_epochs()
        all_phases['epochs'] = epochs_results
        
        # Fase 5: Fine-tuning batch size
        print(f"\n📋 FASE 5/5: Fine-tuning Batch Size")
        batch_results = self.fine_tune_batch_size()
        all_phases['batch_size'] = batch_results
        
        # Fase 6: Test regularization
        print(f"\n📋 FASE BONUS: Test Regularization")
        dropout_results = self.test_regularization()
        all_phases['regularization'] = dropout_results
        
        # Test finale con configurazione ottimizzata
        print(f"\n🏁 TEST FINALE - Configurazione Ottimizzata")
        final_result = self.test_final_optimized_config()
        all_phases['final_optimized'] = final_result
        
        total_time = time.time() - start_time
        
        # Genera rapporto
        report = self._generate_fine_tuning_report(all_phases, baseline_accuracy, total_time)
        
        # Salva risultati
        self._save_results(report)
        
        # Genera e salva top 10 modelli
        self._save_top_models_summary(report)
        

        print(f"\n{'='*70}")
        print(f"🎉 FINE-TUNING COMPLETATO!")
        print(f"⏱️  Tempo totale: {total_time:.2f}s")
        print(f"📈 Miglioramento: {report.get('improvement', {}).get('absolute', 0):.6f}")
        print(f"🏆 Accuracy finale: {report.get('final_accuracy', 0):.6f}")
        print(f"📁 Risultati salvati in: {self.output_dir}")
        print(f"{'='*70}")
        
        return report
    
    def test_final_optimized_config(self) -> Dict:
        """Test finale con la configurazione completamente ottimizzata."""
        print(f"🏁 Configurazione finale ottimizzata:")
        for key, value in self.best_base_config.items():
            print(f"    {key}: {value}")
        
        hyperparams = {
            'epochs': [self.best_base_config['epochs']],
            'batch_size': [self.best_base_config['batch_size']],
            'learning_rate': [self.best_base_config['learning_rate']],
            'activation': [self.best_base_config['activation']],
            'gru_units': [self.best_base_config['gru_units']]
        }
        
        if 'dropout' in self.best_base_config:
            hyperparams['dropout'] = [self.best_base_config['dropout']]
        
        result = self._run_single_test('gru', hyperparams, 'final_optimized')
        
        if result.get('status') == 'success':
            accuracy = result['best_accuracy']
            print(f"✅ Accuracy finale: {accuracy:.6f}")
        
        return result
    
    def _run_single_test(self, model_type: str, hyperparams: Dict, test_id: str) -> Dict:
        """Esegue un singolo test usando SNNIDSBenchmark._run_single_configuration direttamente."""
        config_override = {
            'sample_size': self.sample_size,
            'data_path': self.data_path
        }
        
        benchmark = SNNIDSBenchmark(config_override)
        
        # Configurazione specifica per il singolo modello
        test_config = {
            'sample_size': self.sample_size,
            'data_path': self.data_path,
            'model_type': model_type,  # Solo il modello specificato
            'hyperparameters': hyperparams
        }
        
        # Usa direttamente _run_single_configuration per evitare il loop su tutti i modelli
        result = benchmark._run_single_configuration(test_config)
        result['test_id'] = test_id
        
        return result
    
    def _generate_fine_tuning_report(self, all_phases: Dict, baseline_accuracy: float, total_time: float) -> Dict:
        """Genera il rapporto completo del fine-tuning."""
        
        # Trova la migliore accuracy finale
        final_result = all_phases.get('final_optimized', {})
        final_accuracy = final_result.get('best_accuracy', 0) if final_result.get('status') == 'success' else 0
        
        # Calcola miglioramento
        improvement = final_accuracy - baseline_accuracy
        improvement_percent = (improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        
        # Statistiche per fase
        phase_stats = {}
        for phase_name, phase_data in all_phases.items():
            if phase_name == 'final_optimized':
                continue
                
            if isinstance(phase_data, dict) and 'all_results' in phase_data:
                successful_tests = [r for r in phase_data['all_results'] if r.get('status') == 'success']
                best_in_phase = max(successful_tests, key=lambda x: x.get('best_accuracy', 0), default=None)
                
                phase_stats[phase_name] = {
                    'tests_run': len(phase_data['all_results']),
                    'successful_tests': len(successful_tests),
                    'best_accuracy': best_in_phase.get('best_accuracy', 0) if best_in_phase else 0
                }
            elif isinstance(phase_data, dict) and 'best_accuracy' in phase_data:
                # Baseline case
                phase_stats[phase_name] = {
                    'tests_run': 1,
                    'successful_tests': 1 if phase_data.get('status') == 'success' else 0,
                    'best_accuracy': phase_data.get('best_accuracy', 0)
                }
        
        report = {
            'timestamp': self.timestamp,
            'total_time': total_time,
            'sample_size': self.sample_size,
            'baseline_accuracy': baseline_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': {
                'absolute': improvement,
                'percent': improvement_percent
            },
            'optimized_config': self.best_base_config.copy(),
            'phase_statistics': phase_stats,
            'detailed_results': all_phases
        }
        
        return report
    
    def _save_results(self, report: Dict):
        """Salva tutti i risultati del fine-tuning."""
        
        # JSON completo
        json_file = os.path.join(self.output_dir, "fine_tuning_complete.json")
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"💾 Rapporto completo: {json_file}")
        
        # Summary JSON
        summary = {
            'timestamp': report['timestamp'],
            'baseline_accuracy': report['baseline_accuracy'],
            'final_accuracy': report['final_accuracy'],
            'improvement': report['improvement'],
            'optimized_config': report['optimized_config'],
            'total_time': report['total_time']
        }
        summary_file = os.path.join(self.output_dir, "fine_tuning_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📋 Summary: {summary_file}")
        
        # CSV per analisi
        self._save_csv_analysis(report)
        
        # Report testuale
        self._save_text_report(report)
    
    def _save_csv_analysis(self, report: Dict):
        """Salva analisi CSV dei risultati."""
        try:
            all_tests = []
            
            for phase_name, phase_data in report['detailed_results'].items():
                if phase_name == 'final_optimized':
                    continue
                    
                if isinstance(phase_data, dict) and 'all_results' in phase_data:
                    for result in phase_data['all_results']:
                        if result.get('status') != 'success':
                            continue
                        
                        config = result.get('config', {})
                        hyperparams = config.get('hyperparameters', {})
                        
                        test_row = {
                            'phase': phase_name,
                            'test_id': result.get('test_id', 'unknown'),
                            'accuracy': result.get('best_accuracy'),
                            'training_time': result.get('training_time'),
                            'epochs': hyperparams.get('epochs', [None])[0],
                            'batch_size': hyperparams.get('batch_size', [None])[0],
                            'learning_rate': hyperparams.get('learning_rate', [None])[0],
                            'gru_units': hyperparams.get('gru_units', [None])[0],
                            'dropout': hyperparams.get('dropout', [None])[0] if 'dropout' in hyperparams else None
                        }
                        all_tests.append(test_row)
            
            if all_tests:
                df = pd.DataFrame(all_tests)
                csv_file = os.path.join(self.output_dir, "fine_tuning_analysis.csv")
                df.to_csv(csv_file, index=False)
                print(f"📊 Analisi CSV: {csv_file}")
                
        except ImportError:
            print("⚠️ Pandas non disponibile, CSV analysis saltata")
        except Exception as e:
            print(f"❌ Errore nel salvataggio CSV: {e}")
    
    def _save_text_report(self, report: Dict):
        """Salva rapporto testuale leggibile."""
        report_file = os.path.join(self.output_dir, "fine_tuning_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("🏆 RAPPORTO FINE-TUNING CONFIGURAZIONE MIGLIORE\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"📅 Timestamp: {report['timestamp']}\n")
            f.write(f"⏱️  Tempo totale: {report['total_time']:.2f}s\n")
            f.write(f"📊 Sample size: {report['sample_size']}\n\n")
            
            f.write("📈 RISULTATI:\n")
            f.write("-" * 30 + "\n")
            f.write(f"🎯 Accuracy baseline: {report['baseline_accuracy']:.6f}\n")
            f.write(f"🏆 Accuracy finale: {report['final_accuracy']:.6f}\n")
            f.write(f"📈 Miglioramento: +{report['improvement']['absolute']:.6f} ({report['improvement']['percent']:.2f}%)\n\n")
            
            f.write("⚙️  CONFIGURAZIONE OTTIMIZZATA:\n")
            f.write("-" * 30 + "\n")
            for param, value in report['optimized_config'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write("📊 STATISTICHE PER FASE:\n")
            f.write("-" * 30 + "\n")
            for phase_name, stats in report['phase_statistics'].items():
                f.write(f"  {phase_name}:\n")
                f.write(f"    🧪 Test: {stats['successful_tests']}/{stats['tests_run']}\n")
                f.write(f"    🎯 Miglior accuracy: {stats['best_accuracy']:.6f}\n\n")
        
        print(f"📄 Rapporto testuale: {report_file}")

    def _save_top_models_summary(self, report: Dict):
        """Salva un riepilogo dei top 10 modelli ordinati per accuratezza."""
        print(f"\n🏆 GENERAZIONE TOP 10 MODELLI")
        print(f"{'─'*50}")
        
        # Raccoglie tutti i risultati di successo da tutte le fasi
        all_successful_results = []
        
        for phase_name, phase_data in report.get('detailed_results', {}).items():
            if phase_name == 'final_optimized':
                # Gestisce il risultato finale
                if phase_data.get('status') == 'success' and phase_data.get('best_accuracy', 0) > 0:
                    config = phase_data.get('config', {})
                    hyperparams = config.get('hyperparameters', {})
                    
                    model_info = {
                        'rank': 0,
                        'phase': 'Final Optimized',
                        'test_id': phase_data.get('test_id', 'final_optimized'),
                        'model_type': config.get('model_type', 'gru'),
                        'accuracy': phase_data.get('best_accuracy', 0),
                        'training_time': phase_data.get('training_time', 0),
                        'total_time': phase_data.get('total_time', 0),
                        'epochs': hyperparams.get('epochs', [None])[0],
                        'batch_size': hyperparams.get('batch_size', [None])[0],
                        'learning_rate': hyperparams.get('learning_rate', [None])[0],
                        'activation': hyperparams.get('activation', ['tanh'])[0],
                        'gru_units': hyperparams.get('gru_units', [None])[0],
                        'dropout': hyperparams.get('dropout', [None])[0] if 'dropout' in hyperparams else None
                    }
                    all_successful_results.append(model_info)
                continue
            
            # Gestisce le fasi normali
            if isinstance(phase_data, dict) and 'all_results' in phase_data:
                for result in phase_data['all_results']:
                    if result.get('status') == 'success' and result.get('best_accuracy', 0) > 0:
                        config = result.get('config', {})
                        hyperparams = config.get('hyperparameters', {})
                        
                        model_info = {
                            'rank': 0,
                            'phase': phase_name.replace('_', ' ').title(),
                            'test_id': result.get('test_id', 'unknown'),
                            'model_type': config.get('model_type', 'gru'),
                            'accuracy': result.get('best_accuracy', 0),
                            'training_time': result.get('training_time', 0),
                            'total_time': result.get('total_time', 0),
                            'epochs': hyperparams.get('epochs', [None])[0],
                            'batch_size': hyperparams.get('batch_size', [None])[0],
                            'learning_rate': hyperparams.get('learning_rate', [None])[0],
                            'activation': hyperparams.get('activation', ['tanh'])[0],
                            'gru_units': hyperparams.get('gru_units', [None])[0],
                            'dropout': hyperparams.get('dropout', [None])[0] if 'dropout' in hyperparams else None
                        }
                        all_successful_results.append(model_info)
            elif isinstance(phase_data, dict) and 'best_accuracy' in phase_data:
                # Gestisce il baseline
                if phase_data.get('status') == 'success' and phase_data.get('best_accuracy', 0) > 0:
                    config = phase_data.get('config', {})
                    hyperparams = config.get('hyperparameters', {})
                    
                    model_info = {
                        'rank': 0,
                        'phase': 'Baseline',
                        'test_id': phase_data.get('test_id', 'baseline'),
                        'model_type': config.get('model_type', 'gru'),
                        'accuracy': phase_data.get('best_accuracy', 0),
                        'training_time': phase_data.get('training_time', 0),
                        'total_time': phase_data.get('total_time', 0),
                        'epochs': hyperparams.get('epochs', [None])[0],
                        'batch_size': hyperparams.get('batch_size', [None])[0],
                        'learning_rate': hyperparams.get('learning_rate', [None])[0],
                        'activation': hyperparams.get('activation', ['tanh'])[0],
                        'gru_units': hyperparams.get('gru_units', [None])[0],
                        'dropout': hyperparams.get('dropout', [None])[0] if 'dropout' in hyperparams else None
                    }
                    all_successful_results.append(model_info)
        
        if not all_successful_results:
            print("⚠️ Nessun risultato di successo trovato per il top 10")
            return
        
        # Ordina per accuratezza (decrescente) e prende i top 10
        top_models = sorted(all_successful_results, key=lambda x: x['accuracy'], reverse=True)[:10]
        
        # Assegna i rank
        for i, model in enumerate(top_models, 1):
            model['rank'] = i
        
        print(f"📊 Trovati {len(all_successful_results)} risultati, top 10 selezionati")
        
        # Salva JSON dei top 10
        top_models_file = os.path.join(self.output_dir, "top_10_models.json")
        with open(top_models_file, 'w') as f:
            json.dump({
                'timestamp': report.get('timestamp'),
                'baseline_accuracy': report.get('baseline_accuracy'),
                'final_accuracy': report.get('final_accuracy'),
                'total_models_tested': len(all_successful_results),
                'top_10_models': top_models
            }, f, indent=2, default=str)
        print(f"🏆 Top 10 JSON salvato: {top_models_file}")
        
        # Salva CSV dei top 10
        try:
            import pandas as pd
            df_top = pd.DataFrame(top_models)
            csv_file = os.path.join(self.output_dir, "top_10_models.csv")
            df_top.to_csv(csv_file, index=False)
            print(f"📊 Top 10 CSV salvato: {csv_file}")
        except ImportError:
            print("⚠️ Pandas non disponibile, CSV top 10 saltato")
        
        # Salva rapporto testuale dei top 10
        top_report_file = os.path.join(self.output_dir, "top_10_models_report.txt")
        with open(top_report_file, 'w') as f:
            f.write("🏆 TOP 10 MODELLI - FINE-TUNING GRU SNN-IDS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"📅 Timestamp: {report.get('timestamp')}\n")
            f.write(f"📊 Modelli testati totali: {len(all_successful_results)}\n")
            f.write(f"🎯 Sample size: {report.get('sample_size')}\n")
            f.write(f"📈 Accuracy baseline: {report.get('baseline_accuracy', 0):.6f}\n")
            f.write(f"🏆 Accuracy finale: {report.get('final_accuracy', 0):.6f}\n")
            f.write(f"📈 Miglioramento: +{report.get('improvement', {}).get('absolute', 0):.6f}\n\n")
            
            f.write("🏆 TOP 10 CONFIGURAZIONI GRU:\n")
            f.write("="*70 + "\n")
            
            for model in top_models:
                f.write(f"\n#{model['rank']:2d}. {model['model_type'].upper()} - Accuracy: {model['accuracy']:.6f}\n")
                f.write(f"     📋 Fase: {model['phase']}\n")
                f.write(f"     ⚙️  Parametri:\n")
                f.write(f"        • Epochs: {model['epochs']}\n")
                f.write(f"        • Batch size: {model['batch_size']}\n")
                f.write(f"        • Learning rate: {model['learning_rate']}\n")
                f.write(f"        • Activation: {model['activation']}\n")
                if model['gru_units']:
                    f.write(f"        • GRU units: {model['gru_units']}\n")
                if model['dropout'] is not None:
                    f.write(f"        • Dropout: {model['dropout']}\n")
                f.write(f"     ⏱️  Training time: {model['training_time']:.1f}s\n")
                f.write(f"     🆔 Test ID: {model['test_id']}\n")
        
        print(f"📄 Top 10 report salvato: {top_report_file}")
        
        # Mostra top 5 in console
        print(f"\n🏆 TOP 5 CONFIGURAZIONI GRU:")
        for i, model in enumerate(top_models[:5], 1):
            improvement = model['accuracy'] - report.get('baseline_accuracy', 0)
            print(f"  #{i}. Accuracy: {model['accuracy']:.6f} (+{improvement:.6f}) "
                  f"[lr={model['learning_rate']}, epochs={model['epochs']}, "
                  f"batch={model['batch_size']}, units={model['gru_units']}]")

def main():
    """Entry point del benchmark configurazione migliore."""
    parser = argparse.ArgumentParser(
        description='Benchmark Fine-Tuning Configurazione Migliore SNN-IDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi di utilizzo:

  # Fine-tuning completo della configurazione migliore
  python3 benchmark-best-config.py

  # Con sample size personalizzato per risultati più stabili
  python3 benchmark-best-config.py --sample-size 30000

  # Solo test baseline per conferma
  python3 benchmark-best-config.py --baseline-only

  # Solo fine-tuning learning rate
  python3 benchmark-best-config.py --lr-only
        '''
    )
    
    parser.add_argument('--sample-size', type=int, 
                       help='Numero di campioni da utilizzare')
    parser.add_argument('--data-path', type=str, 
                       help='Path ai dati del dataset')
    # Modalità Federated + HE
    parser.add_argument('--federated', action='store_true',
                       help='Esegue benchmark best-config in modalità Federated Learning')
    parser.add_argument('--he', action='store_true',
                       help='Abilita Homomorphic Encryption per feature sensibili (usa HOMOMORPHIC_CONFIG)')
    parser.add_argument('--dp', action='store_true',
                       help='Abilita Differential Privacy sui gradienti (flag in FEDERATED_CONFIG)')
    parser.add_argument('--federated-sweep', action='store_true',
                       help='Esegue una mini grid search in FL per confrontare NO-HE vs HE e produce top-10 e best config')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Esegue solo il test baseline')
    parser.add_argument('--lr-only', action='store_true',
                       help='Esegue solo il fine-tuning del learning rate')
    
    args = parser.parse_args()
    
    try:
        benchmark = BestConfigBenchmark(
            sample_size=args.sample_size,
            data_path=args.data_path
        )
        
        if args.federated_sweep:
            if args.he:
                HOMOMORPHIC_CONFIG["enabled"] = True
            if args.dp:
                FEDERATED_CONFIG["differential_privacy"] = True

            comparison = run_fl_comparison_sweep(
                sample_size=args.sample_size,
                data_path=args.data_path
            )
            out_dir = f"best_config_benchmark_{benchmark.timestamp}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "federated_sweep_comparison.json")
            with open(out_path, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"\n🏁 Federated sweep completata.")
            if comparison.get('no_he', {}).get('best'):
                print(f"NO-HE best acc: {comparison['no_he']['best']['accuracy']:.6f} params: {comparison['no_he']['best']['params']}")
            if comparison.get('he', {}).get('best'):
                print(f"HE    best acc: {comparison['he']['best']['accuracy']:.6f} params: {comparison['he']['best']['params']}")
            print(f"Δ accuracy (HE - NO-HE): {comparison.get('delta_accuracy_best')}")
            print(f"💾 Salvato in: {out_path}")
            return 0
        elif args.federated:
            # Override config runtime per flags
            if args.he:
                HOMOMORPHIC_CONFIG["enabled"] = True
            if args.dp:
                FEDERATED_CONFIG["differential_privacy"] = True

            report = run_fl_best_config(
                sample_size=args.sample_size,
                data_path=args.data_path
            )
            # Salva output
            out_dir = f"best_config_benchmark_{benchmark.timestamp}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "federated_best_config_report.json")
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n✅ Federated run completato. Accuracy: {report.get('best_accuracy', 0):.6f}")
            print(f"🔐 Privacy report: {report.get('privacy_report')}")
            print(f"💾 Salvato in: {out_path}")
            return 0
        elif args.baseline_only:
            result = benchmark.test_baseline_config()
            if result.get('status') == 'success':
                print(f"✅ Baseline confermato: {result['best_accuracy']:.6f}")
            return 0
        elif args.lr_only:
            results = benchmark.fine_tune_learning_rate()
            if results['best_result']:
                print(f"✅ Miglior LR: {results['best_result']['best_accuracy']:.6f}")
            return 0
        else:
            # Fine-tuning completo
            results = benchmark.run_complete_fine_tuning()
            
            print(f"\n🎯 Fine-tuning completato!")
            print(f"📈 Miglioramento: +{results['improvement']['absolute']:.6f} ({results['improvement']['percent']:.2f}%)")
            print(f"🏆 Accuracy finale: {results['final_accuracy']:.6f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Errore durante il fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
