"""
Modulo 1: Log Preprocessing
Parsing, anonimizzazione e preparazione dati per SNN
"""

from .log_parser import UniversalLogParser, ParsedLog, LogFormat
from .anonymizer import LogAnonymizer, AnonymizedResult
from .snn_preprocessor import SNNPreprocessor, SNNDataset
from .log_processor import LogProcessor, ProcessingConfig

__all__ = [
    'UniversalLogParser',
    'ParsedLog', 
    'LogFormat',
    'LogAnonymizer',
    'AnonymizedResult',
    'SNNPreprocessor',
    'SNNDataset',
    'LogProcessor',
    'ProcessingConfig'
]

# Configurazione modulo
MODULE_NAME = "log_preprocessing"
MODULE_VERSION = "1.0.0"
MODULE_DESCRIPTION = "Preprocessing e anonimizzazione log per SNN"