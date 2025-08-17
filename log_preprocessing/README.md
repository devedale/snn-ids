# Modulo 1: Log Preprocessing

## Descrizione
Modulo per preprocessing e anonimizzazione log per SNN.

## Funzionalità
- ✅ Parsing log multi-formato (FortiGate, CSV, CEF, Syslog)
- ✅ Anonimizzazione sicura con Presidio
- ✅ Preparazione dati per SNN (normalizzazione [0,1])
- ✅ Export sicuro per training esterno
- ✅ Gestione mapping per ricostruzione

## File Principali
- `log_processor.py` - Processore principale
- `log_parser.py` - Parser universale log
- `anonymizer.py` - Sistema anonimizzazione
- `snn_preprocessor.py` - Preparazione dati SNN
- `anonymization_config.yaml` - Config anonimizzazione
- `snn_config.yaml` - Config preprocessing SNN

## Utilizzo

### Uso Standalone
```python
from log_preprocessing import LogProcessor, ConfigManager

# Configurazione
config = ProcessingConfig(
    output_dir="output",
    preserve_mappings=True
)

processor = LogProcessor(config)

# Processing completo
results = processor.process_raw_logs(
    ["input/log1.txt", "input/log2.csv"],
    "my_processing"
)

# Export per training esterno (sicuro)
package = processor.process_for_external_training(
    ["input/log1.txt"],
    "external_package"
)
```

### Via Orchestratore
```python
from pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
results = orchestrator.run_preprocessing_only(["input/logs.txt"])
```

## Output
- `*_snn_dataset.csv` - Dataset formattato per SNN
- `*_snn_dataset.metadata.json` - Metadati del dataset
- `*_anonymized_logs.json` - Log anonimizzati (backup)
- `*_security_mappings.json` - Mapping per ricostruzione (solo locale)
- `*_normalization_stats.json` - Statistiche normalizzazione

## Sicurezza
- ✅ Crypto-PAn per IP preservando subnet
- ✅ Hash sicuri con salt configurabile
- ✅ One-hot encoding per categoriali
- ✅ Mapping separati per ricostruzione locale
- ✅ Pacchetti sicuri senza mapping per training esterno
