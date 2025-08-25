#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Progressivo SNN-IDS
Sistema di ottimizzazione iperparametri a fasi progressive.
Utilizza il benchmark.py esistente per eseguire test mirati fase per fase.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Import del benchmark esistente
sys.path.append(os.path.abspath('.'))
from benchmark import SNNIDSBenchmark
from config import TRAINING_CONFIG, PREPROCESSING_CONFIG, DATA_CONFIG

@dataclass
class PhaseResult:
    """Risultato di una singola fase di ottimizzazione."""
    phase_name: str
    best_config: Dict
    best_accuracy: float
    best_model_type: str
    all_results: List[Dict]
    execution_time: float

class ProgressiveBenchmark:
    """Orchestratore per benchmark progressivo a fasi."""
    
    def __init__(self, sample_size: int = None, data_path: str = None, models_to_test: List[str] = None):
        """Inizializza il benchmark progressivo."""
        self.sample_size = sample_size or PREPROCESSING_CONFIG['sample_size']
        self.data_path = data_path or DATA_CONFIG['dataset_path']
        self.models_to_test = models_to_test or ['dense', 'gru', 'lstm']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"progressive_benchmark_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storia delle fasi
        self.phase_history: List[PhaseResult] = []
        self.best_params = {}
        
        print(f"🚀 BENCHMARK PROGRESSIVO INIZIATO")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Sample size: {self.sample_size}")
        print(f"🤖 Modelli da testare: {', '.join(self.models_to_test)}")
        
    def run_progressive_optimization(self) -> Dict[str, Any]:
        """Esegue l'ottimizzazione progressiva completa."""
        print(f"\n{'='*60}")
        print(f"🎯 INIZIO OTTIMIZZAZIONE PROGRESSIVA")
        print(f"{'='*60}")
        
        total_start_time = time.time()
        
        # FASE 1: Ottimizzazione Learning Rate
        phase1_result = self._run_phase_1_learning_rate()
        self.phase_history.append(phase1_result)
        self.best_params['learning_rate'] = self._extract_best_param(phase1_result, 'learning_rate')
        
        # FASE 2: Ottimizzazione Epochs
        phase2_result = self._run_phase_2_epochs()
        self.phase_history.append(phase2_result)
        self.best_params['epochs'] = self._extract_best_param(phase2_result, 'epochs')
        
        # FASE 3: Ottimizzazione Architettura
        phase3_result = self._run_phase_3_architecture()
        self.phase_history.append(phase3_result)
        self.best_params.update(self._extract_best_architecture_params(phase3_result))
        
        # FASE 4: Ottimizzazione Batch Size
        phase4_result = self._run_phase_4_batch_size()
        self.phase_history.append(phase4_result)
        self.best_params['batch_size'] = self._extract_best_param(phase4_result, 'batch_size')
        
        total_time = time.time() - total_start_time
        
        # Genera rapporto finale
        final_report = self._generate_final_report(total_time)
        
        # Salva tutti i risultati
        self._save_all_results(final_report)
        
        # Genera e salva top 10 modelli
        self._save_top_models_summary(final_report)
        

        print(f"\n{'='*60}")
        print(f"🎉 OTTIMIZZAZIONE PROGRESSIVA COMPLETATA!")
        print(f"⏱️  Tempo totale: {total_time:.2f}s")
        print(f"📁 Risultati salvati in: {self.output_dir}")
        print(f"{'='*60}")
        
        return final_report
    
    def _run_phase_1_learning_rate(self) -> PhaseResult:
        """FASE 1: Trova il learning rate ottimale."""
        print(f"\n🔍 FASE 1: Ottimizzazione Learning Rate")
        print(f"{'─'*50}")
        
        phase_start_time = time.time()
        
        # Configurazione per la fase 1
        learning_rates = [0.0001, 0.001, 0.01]
        base_config = {
            'epochs': [5],  # Fisso per velocità
            'batch_size': [64],  # Standard
            'activation': ['relu'],  # Standard
            'lstm_units': [64],
            'gru_units': [64]
        }
        
        all_results = []
        # Usa modelli configurabili
        models_to_test = self.models_to_test
        
        for model_type in models_to_test:
            for lr in learning_rates:
                print(f"  🧪 Testing {model_type} con learning_rate={lr}")
                
                # Configura iperparametri per questo test
                hyperparams = {**base_config, 'learning_rate': [lr]}
                
                # Esegui test
                result = self._run_single_test(model_type, hyperparams, f"phase1_{model_type}_lr{lr}")
                all_results.append(result)
        
        # Trova il migliore risultato
        best_result = max([r for r in all_results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        phase_time = time.time() - phase_start_time
        
        if best_result:
            best_lr = best_result['config']['hyperparameters']['learning_rate'][0]
            best_model = best_result['config']['model_type']
            best_acc = best_result['best_accuracy']
            
            print(f"  ✅ Miglior risultato: {best_model} con LR={best_lr}, accuracy={best_acc:.4f}")
        else:
            print(f"  ❌ Nessun risultato valido nella Fase 1")
            best_lr, best_model, best_acc = 0.001, 'gru', 0.0
        
        return PhaseResult(
            phase_name="Phase 1: Learning Rate",
            best_config=best_result['config'] if best_result else {},
            best_accuracy=best_acc,
            best_model_type=best_model,
            all_results=all_results,
            execution_time=phase_time
        )
    
    def _run_phase_2_epochs(self) -> PhaseResult:
        """FASE 2: Trova il numero di epoche ottimale."""
        print(f"\n📈 FASE 2: Ottimizzazione Epochs")
        print(f"📌 Usando learning_rate ottimale: {self.best_params['learning_rate']}")
        print(f"{'─'*50}")
        
        phase_start_time = time.time()
        
        # Configurazione per la fase 2
        epochs_to_test = [3, 5, 10, 15]
        base_config = {
            'batch_size': [64],
            'learning_rate': [self.best_params['learning_rate']],  # Usa il migliore dalla fase 1
            'activation': ['relu'],
            'lstm_units': [64],
            'gru_units': [64]
        }
        
        all_results = []
        models_to_test = self.models_to_test
        
        for model_type in models_to_test:
            for epochs in epochs_to_test:
                print(f"  🧪 Testing {model_type} con epochs={epochs}")
                
                hyperparams = {**base_config, 'epochs': [epochs]}
                result = self._run_single_test(model_type, hyperparams, f"phase2_{model_type}_ep{epochs}")
                all_results.append(result)
        
        best_result = max([r for r in all_results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        phase_time = time.time() - phase_start_time
        
        if best_result:
            best_epochs = best_result['config']['hyperparameters']['epochs'][0]
            best_model = best_result['config']['model_type']
            best_acc = best_result['best_accuracy']
            
            print(f"  ✅ Miglior risultato: {best_model} con epochs={best_epochs}, accuracy={best_acc:.4f}")
        else:
            print(f"  ❌ Nessun risultato valido nella Fase 2")
            best_epochs, best_model, best_acc = 5, 'gru', 0.0
        
        return PhaseResult(
            phase_name="Phase 2: Epochs",
            best_config=best_result['config'] if best_result else {},
            best_accuracy=best_acc,
            best_model_type=best_model,
            all_results=all_results,
            execution_time=phase_time
        )
    
    def _run_phase_3_architecture(self) -> PhaseResult:
        """FASE 3: Ottimizza architettura (activation, units)."""
        print(f"\n🏗️  FASE 3: Ottimizzazione Architettura")
        print(f"📌 Usando LR={self.best_params['learning_rate']}, epochs={self.best_params['epochs']}")
        print(f"{'─'*50}")
        
        phase_start_time = time.time()
        
        # Configurazione per la fase 3
        activations = ['relu', 'tanh']
        units_options = [32, 64, 128]
        
        base_config = {
            'epochs': [self.best_params['epochs']],
            'batch_size': [64],
            'learning_rate': [self.best_params['learning_rate']]
        }
        
        all_results = []
        models_to_test = self.models_to_test
        
        for model_type in models_to_test:
            for activation in activations:
                for units in units_options:
                    print(f"  🧪 Testing {model_type} con activation={activation}, units={units}")
                    
                    hyperparams = {
                        **base_config,
                        'activation': [activation],
                        'lstm_units': [units],
                        'gru_units': [units]
                    }
                    
                    result = self._run_single_test(model_type, hyperparams, 
                                                 f"phase3_{model_type}_{activation}_u{units}")
                    all_results.append(result)
        
        best_result = max([r for r in all_results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        phase_time = time.time() - phase_start_time
        
        if best_result:
            config = best_result['config']['hyperparameters']
            best_activation = config['activation'][0]
            best_units = config.get('gru_units', config.get('lstm_units', [64]))[0]
            best_model = best_result['config']['model_type']
            best_acc = best_result['best_accuracy']
            
            print(f"  ✅ Miglior risultato: {best_model} con activation={best_activation}, units={best_units}, accuracy={best_acc:.4f}")
        else:
            print(f"  ❌ Nessun risultato valido nella Fase 3")
            best_activation, best_units, best_model, best_acc = 'relu', 64, 'gru', 0.0
        
        return PhaseResult(
            phase_name="Phase 3: Architecture",
            best_config=best_result['config'] if best_result else {},
            best_accuracy=best_acc,
            best_model_type=best_model,
            all_results=all_results,
            execution_time=phase_time
        )
    
    def _run_phase_4_batch_size(self) -> PhaseResult:
        """FASE 4: Ottimizza batch size finale."""
        print(f"\n📦 FASE 4: Ottimizzazione Batch Size")
        print(f"📌 Usando parametri ottimali delle fasi precedenti")
        print(f"{'─'*50}")
        
        phase_start_time = time.time()
        
        # Configurazione per la fase 4
        batch_sizes = [32, 64, 128]
        
        base_config = {
            'epochs': [self.best_params['epochs']],
            'learning_rate': [self.best_params['learning_rate']],
            'activation': [self.best_params.get('activation', 'relu')],
            'lstm_units': [self.best_params.get('lstm_units', 64)],
            'gru_units': [self.best_params.get('gru_units', 64)]
        }
        
        all_results = []
        models_to_test = self.models_to_test
        
        for model_type in models_to_test:
            for batch_size in batch_sizes:
                print(f"  🧪 Testing {model_type} con batch_size={batch_size}")
                
                hyperparams = {**base_config, 'batch_size': [batch_size]}
                result = self._run_single_test(model_type, hyperparams, f"phase4_{model_type}_bs{batch_size}")
                all_results.append(result)
        
        best_result = max([r for r in all_results if r.get('status') == 'success'], 
                         key=lambda x: x.get('best_accuracy', 0), default=None)
        
        phase_time = time.time() - phase_start_time
        
        if best_result:
            best_batch = best_result['config']['hyperparameters']['batch_size'][0]
            best_model = best_result['config']['model_type']
            best_acc = best_result['best_accuracy']
            
            print(f"  ✅ Miglior risultato: {best_model} con batch_size={best_batch}, accuracy={best_acc:.4f}")
        else:
            print(f"  ❌ Nessun risultato valido nella Fase 4")
            best_batch, best_model, best_acc = 64, 'gru', 0.0
        
        return PhaseResult(
            phase_name="Phase 4: Batch Size",
            best_config=best_result['config'] if best_result else {},
            best_accuracy=best_acc,
            best_model_type=best_model,
            all_results=all_results,
            execution_time=phase_time
        )
    
    def _run_single_test(self, model_type: str, hyperparams: Dict, test_id: str) -> Dict:
        """Esegue un singolo test usando SNNIDSBenchmark."""
        config_override = {
            'sample_size': self.sample_size,
            'data_path': self.data_path
        }
        
        benchmark = SNNIDSBenchmark(config_override)
        
        test_config = {
            'sample_size': self.sample_size,
            'data_path': self.data_path,
            'model_type': model_type,
            'hyperparameters': hyperparams
        }
        
        result = benchmark._run_single_configuration(test_config)
        result['test_id'] = test_id
        
        return result
    
    def _extract_best_param(self, phase_result: PhaseResult, param_name: str) -> Any:
        """Estrae il valore del parametro migliore da una fase."""
        if phase_result.best_config:
            return phase_result.best_config['hyperparameters'][param_name][0]
        return None
    
    def _extract_best_architecture_params(self, phase_result: PhaseResult) -> Dict:
        """Estrae i parametri di architettura migliori dalla fase 3."""
        if not phase_result.best_config:
            return {'activation': 'relu', 'lstm_units': 64, 'gru_units': 64}
        
        hyperparams = phase_result.best_config['hyperparameters']
        return {
            'activation': hyperparams['activation'][0],
            'lstm_units': hyperparams['lstm_units'][0],
            'gru_units': hyperparams['gru_units'][0]
        }
    
    def _generate_final_report(self, total_time: float) -> Dict:
        """Genera il rapporto finale di tutte le fasi."""
        
        # Trova il miglior risultato assoluto
        all_best_results = [phase.best_accuracy for phase in self.phase_history if phase.best_accuracy > 0]
        overall_best_accuracy = max(all_best_results) if all_best_results else 0.0
        
        # Trova la fase con il miglior risultato
        best_phase = max(self.phase_history, key=lambda x: x.best_accuracy, default=None)
        
        # Statistiche per fase
        phase_stats = []
        for phase in self.phase_history:
            successful_tests = [r for r in phase.all_results if r.get('status') == 'success']
            phase_stats.append({
                'phase_name': phase.phase_name,
                'execution_time': phase.execution_time,
                'total_tests': len(phase.all_results),
                'successful_tests': len(successful_tests),
                'best_accuracy': phase.best_accuracy,
                'best_model': phase.best_model_type
            })
        
        report = {
            'timestamp': self.timestamp,
            'total_execution_time': total_time,
            'sample_size': self.sample_size,
            'overall_best_accuracy': overall_best_accuracy,
            'best_phase': best_phase.phase_name if best_phase else None,
            'optimal_parameters': self.best_params,
            'phase_statistics': phase_stats,
            'detailed_phase_results': [
                {
                    'phase_name': phase.phase_name,
                    'best_config': phase.best_config,
                    'all_results': phase.all_results
                } for phase in self.phase_history
            ]
        }
        
        return report
    
    def _save_all_results(self, final_report: Dict):
        """Salva tutti i risultati in vari formati."""
        
        # 1. JSON completo
        json_file = os.path.join(self.output_dir, "progressive_benchmark_complete.json")
        with open(json_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        print(f"💾 Rapporto completo salvato: {json_file}")
        
        # 2. Summary JSON (senza i risultati dettagliati)
        summary = {k: v for k, v in final_report.items() if k != 'detailed_phase_results'}
        summary_file = os.path.join(self.output_dir, "progressive_benchmark_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📋 Summary salvato: {summary_file}")
        
        # 3. CSV per analisi
        self._save_csv_analysis(final_report)
        
        # 4. Report testuale
        self._save_text_report(final_report)
    
    def _save_csv_analysis(self, final_report: Dict):
        """Salva un'analisi CSV dei risultati."""
        try:
            all_tests = []
            
            for phase_data in final_report['detailed_phase_results']:
                phase_name = phase_data['phase_name']
                
                for result in phase_data['all_results']:
                    if result.get('status') != 'success':
                        continue
                    
                    config = result.get('config', {})
                    hyperparams = config.get('hyperparameters', {})
                    
                    test_row = {
                        'phase': phase_name,
                        'test_id': result.get('test_id', 'unknown'),
                        'model_type': config.get('model_type'),
                        'accuracy': result.get('best_accuracy'),
                        'training_time': result.get('training_time'),
                        'epochs': hyperparams.get('epochs', [None])[0],
                        'batch_size': hyperparams.get('batch_size', [None])[0],
                        'learning_rate': hyperparams.get('learning_rate', [None])[0],
                        'activation': hyperparams.get('activation', [None])[0],
                        'lstm_units': hyperparams.get('lstm_units', [None])[0],
                        'gru_units': hyperparams.get('gru_units', [None])[0]
                    }
                    all_tests.append(test_row)
            
            if all_tests:
                df = pd.DataFrame(all_tests)
                csv_file = os.path.join(self.output_dir, "progressive_benchmark_analysis.csv")
                df.to_csv(csv_file, index=False)
                print(f"📊 Analisi CSV salvata: {csv_file}")
                
        except ImportError:
            print("⚠️ Pandas non disponibile, CSV analysis saltata")
        except Exception as e:
            print(f"❌ Errore nel salvataggio CSV: {e}")
    
    def _save_text_report(self, final_report: Dict):
        """Salva un rapporto testuale leggibile."""
        report_file = os.path.join(self.output_dir, "progressive_benchmark_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("🚀 RAPPORTO BENCHMARK PROGRESSIVO SNN-IDS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"📅 Timestamp: {final_report['timestamp']}\n")
            f.write(f"⏱️  Tempo totale: {final_report['total_execution_time']:.2f}s\n")
            f.write(f"📊 Sample size: {final_report['sample_size']}\n")
            f.write(f"🏆 Miglior accuracy: {final_report['overall_best_accuracy']:.4f}\n")
            f.write(f"🥇 Miglior fase: {final_report['best_phase']}\n\n")
            
            f.write("🎯 PARAMETRI OTTIMALI TROVATI:\n")
            f.write("-" * 30 + "\n")
            for param, value in final_report['optimal_parameters'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write("📈 STATISTICHE PER FASE:\n")
            f.write("-" * 30 + "\n")
            for stat in final_report['phase_statistics']:
                f.write(f"  {stat['phase_name']}:\n")
                f.write(f"    ⏱️  Tempo: {stat['execution_time']:.2f}s\n")
                f.write(f"    🧪 Test: {stat['successful_tests']}/{stat['total_tests']}\n")
                f.write(f"    🎯 Miglior accuracy: {stat['best_accuracy']:.4f}\n")
                f.write(f"    🤖 Miglior modello: {stat['best_model']}\n\n")
        
        print(f"📄 Rapporto testuale salvato: {report_file}")

    def _save_top_models_summary(self, final_report: Dict):
        """Salva un riepilogo dei top 10 modelli ordinati per accuratezza."""
        print(f"\n🏆 GENERAZIONE TOP 10 MODELLI")
        print(f"{'─'*50}")
        
        # Raccoglie tutti i risultati di successo da tutte le fasi
        all_successful_results = []
        
        for phase_data in final_report.get('detailed_phase_results', []):
            phase_name = phase_data['phase_name']
            
            for result in phase_data.get('all_results', []):
                if result.get('status') == 'success' and result.get('best_accuracy', 0) > 0:
                    config = result.get('config', {})
                    hyperparams = config.get('hyperparameters', {})
                    
                    model_info = {
                        'rank': 0,  # Sarà assegnato dopo l'ordinamento
                        'phase': phase_name,
                        'test_id': result.get('test_id', 'unknown'),
                        'model_type': config.get('model_type', 'unknown'),
                        'accuracy': result.get('best_accuracy', 0),
                        'training_time': result.get('training_time', 0),
                        'total_time': result.get('total_time', 0),
                        'epochs': hyperparams.get('epochs', [None])[0],
                        'batch_size': hyperparams.get('batch_size', [None])[0],
                        'learning_rate': hyperparams.get('learning_rate', [None])[0],
                        'activation': hyperparams.get('activation', [None])[0],
                        'lstm_units': hyperparams.get('lstm_units', [None])[0],
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
                'timestamp': final_report.get('timestamp'),
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
            f.write("🏆 TOP 10 MODELLI - BENCHMARK PROGRESSIVO SNN-IDS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"📅 Timestamp: {final_report.get('timestamp')}\n")
            f.write(f"📊 Modelli testati totali: {len(all_successful_results)}\n")
            f.write(f"🎯 Sample size: {final_report.get('sample_size')}\n\n")
            
            f.write("🏆 TOP 10 CONFIGURAZIONI:\n")
            f.write("="*70 + "\n")
            
            for model in top_models:
                f.write(f"\n#{model['rank']:2d}. {model['model_type'].upper()} - Accuracy: {model['accuracy']:.6f}\n")
                f.write(f"     📋 Fase: {model['phase']}\n")
                f.write(f"     ⚙️  Parametri:\n")
                f.write(f"        • Epochs: {model['epochs']}\n")
                f.write(f"        • Batch size: {model['batch_size']}\n")
                f.write(f"        • Learning rate: {model['learning_rate']}\n")
                f.write(f"        • Activation: {model['activation']}\n")
                if model['model_type'] == 'gru' and model['gru_units']:
                    f.write(f"        • GRU units: {model['gru_units']}\n")
                elif model['model_type'] == 'lstm' and model['lstm_units']:
                    f.write(f"        • LSTM units: {model['lstm_units']}\n")
                if model['dropout'] is not None:
                    f.write(f"        • Dropout: {model['dropout']}\n")
                f.write(f"     ⏱️  Training time: {model['training_time']:.1f}s\n")
                f.write(f"     🆔 Test ID: {model['test_id']}\n")
        
        print(f"📄 Top 10 report salvato: {top_report_file}")
        
        # Mostra top 5 in console
        print(f"\n🏆 TOP 5 MODELLI:")
        for i, model in enumerate(top_models[:5], 1):
            print(f"  #{i}. {model['model_type'].upper()}: {model['accuracy']:.6f} "
                  f"(lr={model['learning_rate']}, epochs={model['epochs']}, batch={model['batch_size']})")

def main():
    """Entry point del benchmark progressivo."""
    parser = argparse.ArgumentParser(
        description='Benchmark Progressivo SNN-IDS - Ottimizzazione iperparametri a fasi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi di utilizzo:

  # Eseguire benchmark progressivo con sample size di default (tutti i modelli)
  python3 benchmark-progressive.py

  # Eseguire solo su GRU (ottimizzazione mirata dopo aver identificato il vincente)
  python3 benchmark-progressive.py --models gru --sample-size 20000

  # Eseguire solo su GRU e LSTM (escludere dense)
  python3 benchmark-progressive.py --models gru lstm

  # Eseguire con dataset personalizzato
  python3 benchmark-progressive.py --data-path /path/to/data --sample-size 50000
        '''
    )
    
    parser.add_argument('--sample-size', type=int, 
                       help='Numero di campioni da utilizzare (default: configurazione)')
    parser.add_argument('--data-path', type=str, 
                       help='Path ai dati del dataset (default: configurazione)')
    parser.add_argument('--models', nargs='+', choices=['dense', 'gru', 'lstm'],
                       help='Lista di modelli da testare (default: tutti) (es. --models gru lstm)')
    
    args = parser.parse_args()
    
    try:
        # Crea ed esegue il benchmark progressivo
        progressive_benchmark = ProgressiveBenchmark(
            sample_size=args.sample_size,
            data_path=args.data_path,
            models_to_test=args.models
        )
        
        results = progressive_benchmark.run_progressive_optimization()
        
        print(f"\n✅ Benchmark progressivo completato con successo!")
        print(f"🏆 Miglior accuracy raggiunta: {results['overall_best_accuracy']:.4f}")
        print(f"📁 Risultati completi disponibili in: {progressive_benchmark.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Errore durante il benchmark progressivo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())