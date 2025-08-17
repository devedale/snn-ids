#!/usr/bin/env python3
"""
Pipeline Orchestrator - Coordinatore dei 3 moduli
Gestisce il flusso completo atomico e interconnesso
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import dai moduli
from log_preprocessing import LogProcessor, ProcessingConfig
from snn_training import SNNTrainer, TrainingConfig
from result_analysis import ResultAnalyzer, AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configurazione completa della pipeline"""
    # Modulo 1: Preprocessing
    preprocessing_config: str = "log_preprocessing/anonymization_config.yaml"
    snn_config: str = "log_preprocessing/snn_config.yaml"
    
    # Modulo 2: Training  
    training_config_file: str = ""  # Se valorizzato, carica config da YAML
    training_framework: str = "mock"  # "nengo", "mock"
    training_epochs: int = 100
    model_name: str = "snn_security_model"
    
    # Modulo 3: Analysis
    enable_reconstruction: bool = True
    anomaly_threshold: float = 0.8
    
    # Output
    output_dir: str = "output"
    secure_export: bool = True


class PipelineOrchestrator:
    """
    Orchestratore principale della pipeline SNN
    
    Coordina i 3 moduli in modo atomico:
    1. Log Preprocessing (locale)
    2. SNN Training (esterno) 
    3. Result Analysis (locale)
    
    Mantiene confidenzialità e permette uso in contesti diversi
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Inizializza moduli
        self.processor = None
        self.trainer = None
        self.analyzer = None
        
        logger.info("PipelineOrchestrator inizializzato")
    
    def run_preprocessing_only(self, input_files: List[str]) -> Dict[str, Any]:
        """
        Esegue solo preprocessing (Modulo 1)
        Uso: Locale, per preparare dati per training esterno
        """
        logger.info("=== Esecuzione Modulo 1: Preprocessing ===")
        
        # Inizializza processor
        from log_preprocessing.log_processor import ProcessingConfig
        proc_config = ProcessingConfig(
            anonymization_config=self.config.preprocessing_config,
            snn_config=self.config.snn_config,
            output_dir=str(self.output_dir),
            preserve_mappings=True  # Mantieni mapping per ricostruzione locale
        )
        
        self.processor = LogProcessor(proc_config)
        
        # Processa log
        results = self.processor.process_raw_logs(input_files, "pipeline_preprocessing")
        
        # Crea pacchetto per training esterno se richiesto
        if self.config.secure_export:
            package_path = self.processor.process_for_external_training(
                input_files, "secure_training_package"
            )
            results['secure_package'] = package_path
        
        logger.info("Preprocessing completato")
        return results
    
    def run_training_only(self, snn_dataset_path: str) -> Dict[str, Any]:
        """
        Esegue solo training (Modulo 2)
        Uso: Esterno, riceve dati anonimi e restituisce modello
        """
        logger.info("=== Esecuzione Modulo 2: Training ===")
        
        # Configurazione training
        training_config: TrainingConfig
        try:
            if self.config.training_config_file:
                cfg_path = Path(self.config.training_config_file)
                if cfg_path.exists():
                    training_config = TrainingConfig.from_yaml(str(cfg_path))
                else:
                    logger.warning(f"File training_config_file non trovato: {cfg_path}. Uso config inline.")
                    training_config = TrainingConfig(
                        framework=self.config.training_framework,
                        epochs=self.config.training_epochs,
                        model_name=self.config.model_name,
                        hidden_layers=[64, 32, 16],
                        batch_size=32,
                        learning_rate=0.001
                    )
            else:
                training_config = TrainingConfig(
                    framework=self.config.training_framework,
                    epochs=self.config.training_epochs,
                    model_name=self.config.model_name,
                    hidden_layers=[64, 32, 16],
                    batch_size=32,
                    learning_rate=0.001
                )
        except Exception as e:
            logger.error(f"Errore caricando la training config: {e}. Uso config di fallback.")
            training_config = TrainingConfig(
                framework=self.config.training_framework,
                epochs=self.config.training_epochs,
                model_name=self.config.model_name,
                hidden_layers=[64, 32, 16],
                batch_size=32,
                learning_rate=0.001
            )
        
        self.trainer = SNNTrainer(training_config)
        
        # Training
        training_results = self.trainer.train(snn_dataset_path)
        
        # Salva modello
        model_path = self.trainer.save_model(str(self.output_dir / "models"))
        
        # Export per analisi
        export_path = str(self.output_dir / "model_export.json")
        self.trainer.export_for_analysis(export_path)
        
        results = {
            'training_results': training_results,
            'model_path': model_path,
            'export_path': export_path
        }
        
        logger.info("Training completato")
        return results
    
    def run_analysis_only(self, model_export_path: str, 
                         predictions_path: str,
                         mappings_path: Optional[str] = None,
                         normalization_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Esegue solo analisi (Modulo 3)
        Uso: Locale, analizza risultati e ricostruisce log
        """
        logger.info("=== Esecuzione Modulo 3: Analysis ===")
        
        # Configurazione analisi
        analysis_config = AnalysisConfig(
            enable_reconstruction=self.config.enable_reconstruction and mappings_path is not None,
            anomaly_threshold=self.config.anomaly_threshold,
            include_temporal_analysis=True,
            export_format="json"
        )
        
        self.analyzer = ResultAnalyzer(analysis_config)
        
        # Setup ricostruzione se possibile
        if mappings_path and normalization_path:
            self.analyzer.setup_reconstruction(mappings_path, normalization_path)
        
        # Carica risultati modello
        model_data = self.analyzer.load_model_results(model_export_path)
        
        # Carica predizioni (se file, altrimenti simula)
        if Path(predictions_path).exists():
            predictions = self.analyzer.load_prediction_results(predictions_path)
        else:
            # Simula predizioni per test
            import numpy as np
            predictions = np.random.random((100, 2))
            predictions[10:15] = 0.9  # Simula anomalie
        
        # Analisi
        analysis_results = self.analyzer.analyze_predictions(predictions)
        
        # Ricostruzione log anomali
        reconstructed_logs = []
        if self.analyzer.reconstructor:
            anomaly_summary = self.analyzer.get_anomaly_summary()
            if anomaly_summary['total_anomalies'] > 0:
                # Per test, simula feature names
                feature_names = [f"feat_{i}" for i in range(10)]
                reconstructed_logs = self.analyzer.reconstruct_highlighted_logs(
                    anomaly_summary['anomaly_indices'][:5],  # Prime 5 anomalie
                    "dummy_dataset.csv",  # In pratica userebbe dataset reale
                    feature_names
                )
        
        # Export report
        report_path = str(self.output_dir / "analysis_report.json")
        self.analyzer.export_analysis_report(report_path, reconstructed_logs)
        
        results = {
            'analysis_results': analysis_results,
            'anomaly_summary': self.analyzer.get_anomaly_summary(),
            'reconstructed_logs': reconstructed_logs,
            'report_path': report_path
        }
        
        logger.info("Analisi completata")
        return results
    
    def run_full_pipeline(self, input_files: List[str]) -> Dict[str, Any]:
        """
        Esegue pipeline completa (tutti e 3 i moduli)
        Uso: Locale completo, per test e sviluppo
        """
        logger.info("=== Esecuzione Pipeline Completa ===")
        
        # 1. Preprocessing
        preprocessing_results = self.run_preprocessing_only(input_files)
        
        # 2. Training (usa dati preprocessati)
        snn_dataset = preprocessing_results['output_files']['snn_dataset']
        training_results = self.run_training_only(snn_dataset)
        
        # 3. Analysis (usa risultati training + mapping locali)
        model_export = training_results['export_path']
        
        # Trova file mapping
        mappings_file = None
        normalization_file = None
        
        for file_path in self.output_dir.glob("*security_mappings.json"):
            mappings_file = str(file_path)
            break
        
        for file_path in self.output_dir.glob("*normalization_stats.json"):
            normalization_file = str(file_path)
            break
        
        analysis_results = self.run_analysis_only(
            model_export, 
            "dummy_predictions.json",  # Simula predizioni
            mappings_file,
            normalization_file
        )
        
        # Risultati completi
        full_results = {
            'preprocessing': preprocessing_results,
            'training': training_results,
            'analysis': analysis_results,
            'pipeline_config': self.config,
            'total_anomalies': analysis_results['anomaly_summary']['total_anomalies']
        }
        
        logger.info("Pipeline completa terminata")
        return full_results
    
    def export_pipeline_summary(self, results: Dict[str, Any], output_path: str):
        """Esporta riassunto completo della pipeline"""
        summary = {
            'pipeline_execution': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'config': self.config.__dict__,
                'modules_executed': list(results.keys())
            },
            'processing_summary': {
                'total_logs': results.get('preprocessing', {}).get('total_logs', 0),
                'snn_samples': results.get('preprocessing', {}).get('snn_samples', 0),
                'snn_features': results.get('preprocessing', {}).get('snn_features', 0)
            },
            'training_summary': {
                'framework': self.config.training_framework,
                'model_path': results.get('training', {}).get('model_path', 'N/A')
            },
            'analysis_summary': results.get('analysis', {}).get('anomaly_summary', {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Riassunto pipeline salvato: {output_path}")


def main():
    """Esempio di utilizzo dell'orchestratore"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Pipeline Orchestrator ===\n")
    
    # Configurazione pipeline
    config = PipelineConfig(
        training_framework="mock",
        training_epochs=50,
        enable_reconstruction=True,
        anomaly_threshold=0.8,
        secure_export=True
    )
    
    orchestrator = PipelineOrchestrator(config)
    
    # File di test
    test_files = [
        "input/FGT80FTK22013405.root.tlog.txt",
        "input/FGT80FTK22013405.root.elog.txt",
    ]
    
    try:
        print("1. Esecuzione preprocessing standalone...")
        prep_results = orchestrator.run_preprocessing_only(test_files)
        print(f"   → {prep_results['total_logs']} log processati")
        
        print("\n2. Esecuzione training standalone...")
        dataset_path = prep_results['output_files']['snn_dataset']
        train_results = orchestrator.run_training_only(dataset_path)
        print(f"   → Modello salvato: {Path(train_results['model_path']).name}")
        
        print("\n3. Esecuzione analysis standalone...")
        analysis_results = orchestrator.run_analysis_only(
            train_results['export_path'],
            "dummy_predictions.json"
        )
        print(f"   → {analysis_results['anomaly_summary']['total_anomalies']} anomalie rilevate")
        
        print("\n4. Pipeline completa...")
        full_results = orchestrator.run_full_pipeline(test_files)
        
        # Export riassunto
        summary_path = "output/pipeline_summary.json"
        orchestrator.export_pipeline_summary(full_results, summary_path)
        
        print(f"\n=== COMPLETATO ===")
        print(f"Log processati: {full_results['preprocessing']['total_logs']}")
        print(f"Anomalie rilevate: {full_results['total_anomalies']}")
        print(f"Riassunto: {summary_path}")
        
    except Exception as e:
        logger.error(f"Errore nella pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import pandas as pd  # Import necessario per timestamp
    main()
