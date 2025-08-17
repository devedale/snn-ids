#!/usr/bin/env python3
"""
Parser di log sicuro con anonimizzazione integrata
Combina parsing universale con anonimizzazione configurabile
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .log_parser import UniversalLogParser, LogFormat, ParsedLog
from .anonymizer import LogAnonymizer, AnonymizedResult

logger = logging.getLogger(__name__)


class SecureLogParser:
    """Parser di log con anonimizzazione integrata"""
    
    def __init__(self, anonymization_config: str = "anonymization_config.yaml"):
        """
        Inizializza il parser sicuro
        
        Args:
            anonymization_config: Path al file di configurazione anonimizzazione
        """
        self.parser = UniversalLogParser()
        self.anonymizer = LogAnonymizer(anonymization_config)
        self.processed_logs = []
        self.stats = {
            'total_processed': 0,
            'total_anonymized': 0,
            'formats_detected': {},
            'anonymization_methods_used': {}
        }
    
    def parse_and_anonymize_line(self, line: str, format_hint: Optional[LogFormat] = None) -> AnonymizedResult:
        """
        Parsa e anonimizza una singola riga
        
        Args:
            line: Riga da processare
            format_hint: Suggerimento formato (opzionale)
            
        Returns:
            AnonymizedResult con log anonimizzato
        """
        # Parse del log
        parsed_log = self.parser.parse_line(line, format_hint)
        
        # Anonimizzazione
        anonymized_result = self.anonymizer.anonymize_log(parsed_log)
        
        # Aggiorna statistiche
        self._update_stats(anonymized_result)
        
        return anonymized_result
    
    def parse_and_anonymize_file(self, 
                                file_path: str, 
                                output_file: Optional[str] = None,
                                format_hint: Optional[LogFormat] = None) -> List[AnonymizedResult]:
        """
        Parsa e anonimizza un intero file
        
        Args:
            file_path: Path del file da processare
            output_file: Path del file di output (opzionale)
            format_hint: Suggerimento formato (opzionale)
            
        Returns:
            Lista di AnonymizedResult
        """
        results = []
        
        logger.info(f"Processando file: {file_path}")
        
        # Parse del file
        parsed_logs = self.parser.parse_file(file_path, format_hint)
        
        # Anonimizzazione di ogni log
        for parsed_log in parsed_logs:
            anonymized_result = self.anonymizer.anonymize_log(parsed_log)
            results.append(anonymized_result)
            self._update_stats(anonymized_result)
        
        # Salva risultati se richiesto
        if output_file:
            self.save_anonymized_results(results, output_file)
        
        logger.info(f"Processati {len(results)} log da {file_path}")
        return results
    
    def save_anonymized_results(self, results: List[AnonymizedResult], output_file: str):
        """
        Salva i risultati anonimizzati in formato JSON
        
        Args:
            results: Lista di AnonymizedResult da salvare
            output_file: Path del file di output
        """
        output_data = []
        
        for result in results:
            # Crea struttura dati per JSON
            entry = {
                'format_type': result.original_log.format_type.value,
                'timestamp': result.original_log.timestamp.isoformat() if result.original_log.timestamp else None,
                'anonymized_fields': result.anonymized_fields,
                'raw_message_anonymized': self._anonymize_raw_message(result),
                'metadata': {
                    **result.original_log.metadata,
                    'anonymization': result.anonymization_metadata
                }
            }
            
            # Aggiungi mapping se disponibile (solo per debug)
            if result.mapping and self.anonymizer.config.output_settings.get('save_mapping', False):
                entry['_debug_mapping'] = result.mapping
            
            output_data.append(entry)
        
        # Salva su file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Risultati salvati in {output_file}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio: {e}")
    
    def _anonymize_raw_message(self, result: AnonymizedResult) -> str:
        """
        Anonimizza il messaggio raw sostituendo i valori originali con quelli anonimizzati
        """
        raw_message = result.original_log.raw_message
        
        # Sostituisce i valori nel messaggio raw
        for field, anonymized_value in result.anonymized_fields.items():
            original_value = result.original_log.parsed_fields.get(field)
            if original_value and original_value != anonymized_value:
                # Sostituisce solo se il valore è stato effettivamente anonimizzato
                raw_message = raw_message.replace(str(original_value), str(anonymized_value))
        
        return raw_message
    
    def _update_stats(self, result: AnonymizedResult):
        """Aggiorna le statistiche di processamento"""
        self.stats['total_processed'] += 1
        
        # Conta formato
        format_type = result.original_log.format_type.value
        self.stats['formats_detected'][format_type] = self.stats['formats_detected'].get(format_type, 0) + 1
        
        # Conta anonimizzazioni
        if result.anonymization_metadata:
            self.stats['total_anonymized'] += 1
            
            # Conta metodi usati
            for field_meta in result.anonymization_metadata.values():
                method = field_meta.get('method', 'unknown')
                self.stats['anonymization_methods_used'][method] = \
                    self.stats['anonymization_methods_used'].get(method, 0) + 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche complete del processamento"""
        anonymizer_stats = self.anonymizer.get_anonymization_stats()
        
        return {
            **self.stats,
            'anonymizer': anonymizer_stats,
            'anonymization_rate': self.stats['total_anonymized'] / max(self.stats['total_processed'], 1)
        }
    
    def export_stats(self, stats_file: str = "processing_stats.json"):
        """Esporta le statistiche su file"""
        stats = self.get_processing_stats()
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Statistiche esportate in {stats_file}")
        except Exception as e:
            logger.error(f"Errore nell'esportazione statistiche: {e}")


def batch_process_directory(input_dir: str, output_dir: str, config_file: str = "anonymization_config.yaml"):
    """
    Processa in batch tutti i file di log in una directory
    
    Args:
        input_dir: Directory di input
        output_dir: Directory di output
        config_file: File di configurazione anonimizzazione
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Crea directory di output se non esiste
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inizializza parser sicuro
    secure_parser = SecureLogParser(config_file)
    
    # Processà tutti i file nella directory
    log_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.log")) + list(input_path.glob("*.csv"))
    
    logger.info(f"Trovati {len(log_files)} file da processare")
    
    for file_path in log_files:
        output_file = output_path / f"{file_path.stem}_anonymized.json"
        
        try:
            secure_parser.parse_and_anonymize_file(str(file_path), str(output_file))
        except Exception as e:
            logger.error(f"Errore nel processamento di {file_path}: {e}")
    
    # Salva statistiche
    stats_file = output_path / "batch_processing_stats.json"
    secure_parser.export_stats(str(stats_file))
    
    # Salva mapping se abilitato
    if secure_parser.anonymizer.config.output_settings.get('save_mapping', False):
        mapping_file = output_path / "anonymization_mapping.json"
        secure_parser.anonymizer.save_mapping(str(mapping_file))
    
    return secure_parser.get_processing_stats()


def main():
    """Funzione principale per test e dimostrazione"""
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Parser di Log Sicuro con Anonimizzazione ===\n")
    
    # Crea directory output se non esiste
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Inizializza parser sicuro
    secure_parser = SecureLogParser()
    
    # File di test
    test_files = [
        "/home/wls_user/tesi/input/FGT80FTK22013405.root.tlog.txt",
        "/home/wls_user/tesi/input/FGT80FTK22013405.root.elog.txt",
        "/home/wls_user/tesi/input/aaa9b30098d31c056625c603cf1a98e6e10afe77_2025-06-30_2025-07-07.csv"
    ]
    
    # Processa ogni file
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\n--- Processando {Path(file_path).name} ---")
            
            # Genera nome file output nella cartella output
            output_file = output_dir / f"{Path(file_path).stem}_secure_anonymized.json"
            
            # Processa e anonimizza
            results = secure_parser.parse_and_anonymize_file(file_path, str(output_file))
            
            # Mostra esempio del primo risultato
            if results:
                first_result = results[0]
                print(f"Formato rilevato: {first_result.original_log.format_type.value}")
                print(f"Campi anonimizzati: {len(first_result.anonymization_metadata)}")
                
                # Mostra alcuni campi anonimizzati
                if first_result.anonymization_metadata:
                    print("Esempi di anonimizzazione:")
                    for field, meta in list(first_result.anonymization_metadata.items())[:3]:
                        method = meta.get('method', 'unknown')
                        original = first_result.original_log.parsed_fields.get(field, 'N/A')
                        anonymized = first_result.anonymized_fields.get(field, 'N/A')
                        print(f"  {field}: {original} -> {anonymized} (metodo: {method})")
        else:
            print(f"File non trovato: {file_path}")
    
    # Mostra statistiche finali
    print("\n=== Statistiche Finali ===")
    stats = secure_parser.get_processing_stats()
    print(f"Log processati: {stats['total_processed']}")
    print(f"Log anonimizzati: {stats['total_anonymized']}")
    print(f"Tasso di anonimizzazione: {stats['anonymization_rate']:.2%}")
    print(f"Formati rilevati: {list(stats['formats_detected'].keys())}")
    print(f"Metodi di anonimizzazione usati: {list(stats['anonymization_methods_used'].keys())}")
    
    # Esporta statistiche nella cartella output
    secure_parser.export_stats(str(output_dir / "secure_parser_stats.json"))
    
    print("\n=== Test Completato ===")
    print("File anonimizzati e statistiche salvati nella directory 'output/'.")


if __name__ == "__main__":
    main()
