#!/usr/bin/env python3
"""
Log Processor - Modulo 1: Preprocessing
Processore unificato per parsing e anonimizzazione log
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import dai file esistenti nel modulo
from .log_parser import UniversalLogParser, ParsedLog
from .anonymizer import LogAnonymizer
from .snn_preprocessor import SNNPreprocessor, SNNDataset

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configurazione per il processing"""
    anonymization_config: str = "anonymization_config.yaml"
    snn_config: str = "snn_config.yaml"
    output_dir: str = "output"
    preserve_mappings: bool = True
    enable_validation: bool = True


class LogProcessor:
    """
    Processore principale per log - Modulo 1
    
    Funzionalità:
    - Parsing log multi-formato
    - Anonimizzazione sicura
    - Preparazione dati SNN
    - Export sicuro per training esterno
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.parser = UniversalLogParser()
        self.anonymizer = LogAnonymizer(self.config.anonymization_config)
        self.snn_preprocessor = SNNPreprocessor(self.config.snn_config)
        
        # Crea directory output
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        logger.info("LogProcessor inizializzato")
    
    def process_raw_logs(self, input_files: List[str], 
                        output_prefix: str = "processed") -> Dict[str, Any]:
        """
        Processa log raw completi: parsing + anonimizzazione + formato SNN
        
        Args:
            input_files: Lista file di log da processare
            output_prefix: Prefisso per file di output
            
        Returns:
            Dict con informazioni sui file generati
        """
        logger.info(f"Processando {len(input_files)} file di log")
        
        # 1. Parse di tutti i log
        all_logs = []
        for file_path in input_files:
            if Path(file_path).exists():
                logs = self.parser.parse_file(file_path)
                all_logs.extend(logs)
                logger.info(f"Parsed {len(logs)} log da {Path(file_path).name}")
        
        if not all_logs:
            raise ValueError("Nessun log processato")
        
        # 2. Anonimizzazione
        anonymized_logs = []
        for log in all_logs:
            anon_result = self.anonymizer.anonymize_log(log)
            anonymized_logs.append(anon_result)
        
        # 3. Preparazione per SNN
        snn_dataset = self.snn_preprocessor.process_logs_to_snn_format(all_logs)
        
        # 4. Export file
        output_files = self._export_processed_data(
            anonymized_logs, snn_dataset, output_prefix
        )
        
        # 5. Salva mapping se richiesto
        if self.config.preserve_mappings:
            self._save_security_mappings(output_prefix)
        
        stats = {
            'total_logs': len(all_logs),
            'anonymized_logs': len(anonymized_logs),
            'snn_samples': snn_dataset.features.shape[0],
            'snn_features': snn_dataset.features.shape[1],
            'output_files': output_files,
            'processing_config': self.config
        }
        
        logger.info(f"Processing completato: {stats}")
        return stats
    
    def process_for_external_training(self, input_files: List[str],
                                    output_package: str = "training_package") -> str:
        """
        Prepara pacchetto sicuro per training esterno
        
        Args:
            input_files: File di log da processare
            output_package: Nome del pacchetto di output
            
        Returns:
            Path del pacchetto creato
        """
        logger.info("Preparando pacchetto per training esterno")
        
        # Processa con configurazione sicura
        secure_config = ProcessingConfig(
            preserve_mappings=False,  # Non salvare mapping nel pacchetto
            enable_validation=True
        )
        
        temp_processor = LogProcessor(secure_config)
        stats = temp_processor.process_raw_logs(input_files, output_package)
        
        # Crea pacchetto sicuro (solo dati SNN anonimi)
        package_path = self._create_secure_package(output_package, stats)
        
        logger.info(f"Pacchetto sicuro creato: {package_path}")
        return package_path
    
    def _export_processed_data(self, anonymized_logs: List, 
                             snn_dataset: SNNDataset, 
                             prefix: str) -> Dict[str, str]:
        """Esporta i dati processati"""
        output_files = {}
        
        # 1. Dataset SNN (formato principale per training)
        snn_path = Path(self.config.output_dir) / f"{prefix}_snn_dataset"
        self.snn_preprocessor.save_snn_dataset(snn_dataset, str(snn_path))
        output_files['snn_dataset'] = str(snn_path.with_suffix('.csv'))
        output_files['snn_metadata'] = str(snn_path.with_suffix('.metadata.json'))
        
        # 2. Log anonimizzati (backup/debug)
        import json
        anon_path = Path(self.config.output_dir) / f"{prefix}_anonymized_logs.json"
        anon_data = []
        for result in anonymized_logs:
            entry = {
                'format_type': result.original_log.format_type.value,
                'timestamp': result.original_log.timestamp.isoformat() if result.original_log.timestamp else None,
                'anonymized_fields': result.anonymized_fields,
                'anonymization_metadata': result.anonymization_metadata
            }
            anon_data.append(entry)
        
        with open(anon_path, 'w', encoding='utf-8') as f:
            json.dump(anon_data, f, indent=2, ensure_ascii=False)
        output_files['anonymized_logs'] = str(anon_path)
        
        return output_files
    
    def _save_security_mappings(self, prefix: str):
        """Salva mapping di sicurezza (solo locale)"""
        if self.config.preserve_mappings:
            # Salva mapping anonimizzazione
            mapping_path = Path(self.config.output_dir) / f"{prefix}_security_mappings.json"
            self.anonymizer.save_mapping(str(mapping_path))
            
            # Salva statistiche normalizzazione SNN
            norm_path = Path(self.config.output_dir) / f"{prefix}_normalization_stats.json"
            import json
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(self.snn_preprocessor.normalization_stats, f, indent=2)
    
    def _create_secure_package(self, package_name: str, stats: Dict) -> str:
        """Crea pacchetto sicuro per training esterno"""
        import tarfile
        
        package_path = Path(self.config.output_dir) / f"{package_name}.tar.gz"
        
        with tarfile.open(package_path, 'w:gz') as tar:
            # Include solo file SNN (anonimi)
            snn_file = stats['output_files']['snn_dataset']
            snn_meta = stats['output_files']['snn_metadata']
            
            if Path(snn_file).exists():
                tar.add(snn_file, arcname=f"{package_name}_data.csv")
            if Path(snn_meta).exists():
                tar.add(snn_meta, arcname=f"{package_name}_metadata.json")
        
        return str(package_path)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche del processore"""
        return {
            'anonymizer_stats': self.anonymizer.get_anonymization_stats(),
            'snn_config': self.snn_preprocessor.config,
            'output_dir': self.config.output_dir
        }


def main():
    """Esempio di utilizzo del LogProcessor"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Log Processor - Modulo 1 ===\n")
    
    # Configura processore
    config = ProcessingConfig(
        output_dir="output",
        preserve_mappings=True,
        enable_validation=True
    )
    
    processor = LogProcessor(config)
    
    # File di test
    test_files = [
        "input/FGT80FTK22013405.root.tlog.txt",
        "input/FGT80FTK22013405.root.elog.txt",
    ]
    
    try:
        # Processo completo
        stats = processor.process_raw_logs(test_files, "test_processing")
        
        print("=== Risultati Processing ===")
        print(f"Log totali: {stats['total_logs']}")
        print(f"Campioni SNN: {stats['snn_samples']}")
        print(f"Features SNN: {stats['snn_features']}")
        print(f"File generati: {list(stats['output_files'].keys())}")
        
        # Crea pacchetto per training esterno
        package = processor.process_for_external_training(test_files, "external_training")
        print(f"Pacchetto esterno: {package}")
        
    except Exception as e:
        logger.error(f"Errore nel processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
