#!/usr/bin/env python3
"""
Sistema di anonimizzazione per log con supporto Microsoft Presidio
Configurabile tramite file YAML per definire campi da anonimizzare
"""

import re
import yaml
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Import Presidio (con fallback se non installato)
# Import Presidio reso lazy per evitare blocchi all'import del modulo
PRESIDIO_AVAILABLE = False
_presidio_imports = {}

def _import_presidio():
    """Import lazy di Presidio"""
    global PRESIDIO_AVAILABLE, _presidio_imports
    
    if not PRESIDIO_AVAILABLE:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import RecognizerResult
            
            _presidio_imports.update({
                'AnalyzerEngine': AnalyzerEngine,
                'AnonymizerEngine': AnonymizerEngine,
                'RecognizerResult': RecognizerResult
            })
            PRESIDIO_AVAILABLE = True
            logging.info("Presidio caricato con successo")
        except ImportError as e:
            logging.warning(f"Presidio non disponibile: {e}")
            PRESIDIO_AVAILABLE = False
    
    return PRESIDIO_AVAILABLE

from .log_parser import ParsedLog, LogFormat

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationRule:
    """Regola di anonimizzazione per un campo"""
    field: str
    method: str
    placeholder: Optional[str] = None
    salt: Optional[str] = None
    pattern: Optional[str] = None
    condition: Optional[str] = None


@dataclass
class AnonymizedResult:
    """Risultato dell'anonimizzazione"""
    original_log: ParsedLog
    anonymized_fields: Dict[str, Any]
    anonymization_metadata: Dict[str, Dict[str, Any]]
    mapping: Optional[Dict[str, str]] = None


class AnonymizationConfig:
    """Carica e gestisce la configurazione di anonimizzazione"""
    
    def __init__(self, config_file: str = "anonymization_config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.global_settings = self.config.get('global_settings', {})
        self.methods = self.config.get('anonymization_methods', {})
        self.conditions = self.config.get('conditions', {})
        self.output_settings = self.config.get('output_settings', {})
    
    def _load_config(self) -> Dict[str, Any]:
        """Carica configurazione da file YAML"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"File di configurazione {self.config_file} non trovato")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Errore nel parsing YAML: {e}")
            return {}
    
    def get_rules_for_format(self, log_format: LogFormat) -> List[AnonymizationRule]:
        """Ottiene le regole di anonimizzazione per un formato specifico"""
        format_name = log_format.value
        if format_name == "syslog_rfc3164" or format_name == "syslog_rfc5424":
            format_name = "syslog"
        
        format_config = self.config.get(format_name, {})
        rules = []
        
        # Regole sempre attive
        for rule_config in format_config.get('always_anonymize', []):
            rule = AnonymizationRule(
                field=rule_config['field'],
                method=rule_config['method'],
                placeholder=rule_config.get('placeholder'),
                salt=rule_config.get('salt'),
                pattern=rule_config.get('pattern')
            )
            rules.append(rule)
        
        # Regole condizionali
        for rule_config in format_config.get('conditional_anonymize', []):
            rule = AnonymizationRule(
                field=rule_config['field'],
                method=rule_config['method'],
                placeholder=rule_config.get('placeholder'),
                salt=rule_config.get('salt'),
                pattern=rule_config.get('pattern'),
                condition=rule_config.get('condition')
            )
            rules.append(rule)
        
        return rules
    
    def get_preserve_fields(self, log_format: LogFormat) -> List[str]:
        """Ottiene i campi da preservare per un formato"""
        format_name = log_format.value
        if format_name == "syslog_rfc3164" or format_name == "syslog_rfc5424":
            format_name = "syslog"
        
        format_config = self.config.get(format_name, {})
        return format_config.get('preserve_fields', [])
    
    def should_auto_detect_pii(self) -> bool:
        """Verifica se il rilevamento automatico PII è abilitato"""
        return self.global_settings.get('auto_detect_pii', False)
    
    def get_confidence_threshold(self) -> float:
        """Ottiene la soglia di confidenza per Presidio"""
        return self.global_settings.get('confidence_threshold', 0.7)


class LogAnonymizer:
    """Classe principale per l'anonimizzazione dei log"""
    
    def __init__(self, config_file: str = "anonymization_config.yaml"):
        self.config = AnonymizationConfig(config_file)
        self.mapping_store = {}  # Per salvare mapping originale->anonimizzato
        
        # Inizializza Presidio se disponibile (lazy loading)
        self.analyzer = None
        self.anonymizer = None
        
        if self.config.should_auto_detect_pii():
            if _import_presidio():
                try:
                    self.analyzer = _presidio_imports['AnalyzerEngine']()
                    self.anonymizer = _presidio_imports['AnonymizerEngine']()
                    logger.info("Presidio inizializzato correttamente")
                except Exception as e:
                    logger.warning(f"Errore nell'inizializzazione Presidio: {e}")
                    self.analyzer = None
                    self.anonymizer = None
    
    def anonymize_log(self, parsed_log: ParsedLog) -> AnonymizedResult:
        """
        Anonimizza un log parsato secondo la configurazione
        
        Args:
            parsed_log: Log parsato da anonimizzare
            
        Returns:
            AnonymizedResult con i campi anonimizzati
        """
        rules = self.config.get_rules_for_format(parsed_log.format_type)
        preserve_fields = self.config.get_preserve_fields(parsed_log.format_type)
        
        anonymized_fields = parsed_log.parsed_fields.copy()
        metadata = {}
        mapping = {}
        
        for rule in rules:
            if rule.field in anonymized_fields:
                original_value = anonymized_fields[rule.field]
                
                # Verifica se il campo deve essere preservato
                if rule.field in preserve_fields:
                    continue
                
                # Verifica condizione se presente
                if rule.condition and not self._check_condition(original_value, rule.condition):
                    continue
                
                # Applica anonimizzazione
                anonymized_value, field_metadata = self._apply_anonymization(
                    original_value, rule
                )
                
                if anonymized_value != original_value:
                    anonymized_fields[rule.field] = anonymized_value
                    metadata[rule.field] = field_metadata
                    mapping[str(original_value)] = str(anonymized_value)
        
        # Auto-detect PII se abilitato
        if self.config.should_auto_detect_pii() and self.analyzer:
            for field, value in anonymized_fields.items():
                if field not in preserve_fields and field not in metadata:
                    pii_results = self._detect_pii(str(value))
                    if pii_results:
                        anonymized_value, field_metadata = self._anonymize_with_presidio(
                            str(value), pii_results
                        )
                        if anonymized_value != str(value):
                            anonymized_fields[field] = anonymized_value
                            metadata[field] = field_metadata
                            mapping[str(value)] = anonymized_value
        
        # Salva mapping se richiesto
        if self.config.output_settings.get('save_mapping', False) and mapping:
            self.mapping_store.update(mapping)
        
        return AnonymizedResult(
            original_log=parsed_log,
            anonymized_fields=anonymized_fields,
            anonymization_metadata=metadata,
            mapping=mapping if self.config.output_settings.get('save_mapping', False) else None
        )
    
    def _check_condition(self, value: Any, condition_name: str) -> bool:
        """Verifica se una condizione è soddisfatta"""
        condition_config = self.config.conditions.get(condition_name, {})
        value_str = str(value)
        
        # Controllo regex
        if 'regex' in condition_config:
            return bool(re.search(condition_config['regex'], value_str))
        
        # Controllo keywords
        if 'keywords' in condition_config:
            value_lower = value_str.lower()
            return any(keyword in value_lower for keyword in condition_config['keywords'])
        
        # Controllo Presidio
        if condition_config.get('presidio_check', False) and self.analyzer:
            pii_results = self._detect_pii(value_str)
            return len(pii_results) > 0
        
        return False
    
    def _apply_anonymization(self, value: Any, rule: AnonymizationRule) -> tuple[Any, Dict[str, Any]]:
        """Applica il metodo di anonimizzazione specificato"""
        original_value = value
        value_str = str(value)
        metadata = {
            'method': rule.method,
            'original_length': len(value_str)
        }
        
        if rule.method == 'replace':
            placeholder = rule.placeholder or self.config.global_settings.get('default_placeholder', '[REDACTED]')
            return placeholder, metadata
        
        elif rule.method == 'hash':
            salt = rule.salt or self.config.methods.get('hash', {}).get('options', {}).get('salt', 'default_salt')
            hash_input = f"{value_str}{salt}".encode('utf-8')
            hashed = hashlib.sha256(hash_input).hexdigest()[:16]  # Primi 16 caratteri
            return f"[HASH_{hashed}]", metadata
        
        elif rule.method == 'mask':
            pattern = rule.pattern or '***'
            return self._apply_mask(value_str, pattern), metadata
        
        elif rule.method == 'presidio' and self.analyzer and self.anonymizer:
            pii_results = self._detect_pii(value_str)
            if pii_results:
                anonymized, presidio_metadata = self._anonymize_with_presidio(value_str, pii_results)
                metadata.update(presidio_metadata)
                return anonymized, metadata
        
        # Se nessun metodo applicabile, restituisci valore originale
        return original_value, metadata
    
    def _apply_mask(self, value: str, pattern: str) -> str:
        """Applica maschera secondo il pattern specificato"""
        if '...' in pattern:
            # Pattern tipo "****...****" - mostra inizio e fine
            parts = pattern.split('...')
            if len(parts) == 2:
                start_len = len(parts[0])
                end_len = len(parts[1])
                if len(value) > start_len + end_len:
                    return value[:start_len] + '...' + value[-end_len:] if end_len > 0 else value[:start_len] + '...'
        
        elif 'xx:xx:xx:xx:**:**' in pattern:
            # Pattern MAC address
            if ':' in value and len(value.split(':')) == 6:
                parts = value.split(':')
                return f"xx:xx:xx:xx:{parts[4]}:{parts[5]}"
        
        # Pattern generico con *
        mask_char = '*'
        if len(pattern) >= len(value):
            return mask_char * len(value)
        else:
            return pattern.replace('*', mask_char)
    
    def _detect_pii(self, text: str) -> List:
        """Rileva PII usando Presidio"""
        if not self.analyzer:
            return []
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language=self.config.global_settings.get('language', 'en'),
                score_threshold=self.config.get_confidence_threshold()
            )
            return results
        except Exception as e:
            logger.warning(f"Errore nel rilevamento PII: {e}")
            return []
    
    def _anonymize_with_presidio(self, text: str, pii_results: List) -> tuple[str, Dict[str, Any]]:
        """Anonimizza usando Presidio"""
        if not self.anonymizer:
            return text, {}
        
        try:
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=pii_results,
                operators={
                    "DEFAULT": {"type": "replace", "new_value": "[PII]"},
                    "PHONE_NUMBER": {"type": "replace", "new_value": "[PHONE]"},
                    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
                    "IP_ADDRESS": {"type": "replace", "new_value": "[IP]"},
                    "PERSON": {"type": "replace", "new_value": "[PERSON]"},
                }
            )
            
            metadata = {
                'presidio_entities': [result.entity_type for result in pii_results],
                'presidio_scores': [result.score for result in pii_results]
            }
            
            return anonymized_result.text, metadata
        
        except Exception as e:
            logger.warning(f"Errore nell'anonimizzazione Presidio: {e}")
            return text, {'error': str(e)}
    
    def save_mapping(self, filename: str = None):
        """Salva il mapping di anonimizzazione su file"""
        if not self.mapping_store:
            return
        
        # Se non specificato, usa directory output
        if filename is None:
            from pathlib import Path
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            filename = str(output_dir / self.config.output_settings.get('mapping_file', 'anonymization_mapping.json'))
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.mapping_store, f, indent=2, ensure_ascii=False)
            logger.info(f"Mapping salvato in {filename}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio mapping: {e}")
    
    def get_anonymization_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche sull'anonimizzazione"""
        return {
            'total_mappings': len(self.mapping_store),
            'presidio_available': PRESIDIO_AVAILABLE,
            'auto_detect_enabled': self.config.should_auto_detect_pii(),
            'config_file': self.config.config_file
        }


def main():
    """Funzione principale per test"""
    from log_parser import UniversalLogParser
    from pathlib import Path
    
    # Crea directory output se non esiste
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Test dell'anonimizzazione
    parser = UniversalLogParser()
    anonymizer = LogAnonymizer()
    
    # Test con una riga di esempio
    test_line = 'logver=123 srcip=192.168.1.100 dstip=10.0.0.1 user="john.doe@example.com" devname="firewall-01"'
    
    print("=== Test Anonimizzazione ===")
    print(f"Input: {test_line}")
    
    # Parsa il log
    parsed_log = parser.parse_line(test_line)
    print(f"Campi parsati: {parsed_log.parsed_fields}")
    
    # Anonimizza
    anonymized = anonymizer.anonymize_log(parsed_log)
    print(f"Campi anonimizzati: {anonymized.anonymized_fields}")
    print(f"Metadata: {anonymized.anonymization_metadata}")
    
    # Statistiche
    stats = anonymizer.get_anonymization_stats()
    print(f"Statistiche: {stats}")


if __name__ == "__main__":
    main()
