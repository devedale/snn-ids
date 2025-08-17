# SNN Security Analytics - Sistema Modulare per Analisi Log di Sicurezza

Sistema modulare completo per analisi log di sicurezza tramite Spiking Neural Networks (SNN), con architettura atomica e interconnessa per garantire confidenzialità e flessibilità d'uso.

## Formati Supportati

- **FortiGate** (tlog/elog): Log FortiGate con formato `chiave=valore`
- **CSV**: File CSV strutturati con header
- **CEF**: Common Event Format
- **Syslog RFC3164**: Formato syslog tradizionale
- **Syslog RFC5424**: Formato syslog moderno

## 🎯 Architettura a 3 Moduli

### **📦 Modulo 1: Log Preprocessing** (Locale)
- 🔍 **Parsing multi-formato** (FortiGate, CSV, CEF, Syslog)
- 🛡️ **Anonimizzazione sicura** con Microsoft Presidio
- 📊 **Preparazione dati SNN** (normalizzazione [0,1])
- 🔢 **Preservazione formato** (zero iniziali, sequenzialità)
- 📤 **Export sicuro** per training esterno

### **🧠 Modulo 2: SNN Training** (Esterno)
- 🎭 **Framework multipli** (Nengo, Norse, SNNTorch)
- ⚙️ **Architetture configurabili** 
- 🔒 **Training su dati anonimi** (nessun accesso a mapping)
- 📈 **Metriche di training** complete
- 📤 **Export modelli** per analisi

### **📊 Modulo 3: Result Analysis** (Locale)
- 🔍 **Rilevamento anomalie** configurabile
- ⏰ **Analisi pattern temporali**
- 🔄 **Ricostruzione log originali** da dati anonimi
- 📋 **Report completi** con evidenza anomalie
- 🛡️ **Accesso sicuro** ai mapping locali

## Installazione

### Installazione Automatica

```bash
# Esegui lo script di installazione
./install_dependencies.sh
```

### Installazione Manuale

```bash
# Installa dipendenze Python
pip3 install -r requirements.txt

# Installa modello spaCy per Presidio
python3 -m spacy download en_core_web_sm
```

### Test Rapido

```bash
# Test del parser base
python3 log_parser.py

# Test del parser con anonimizzazione
python3 secure_log_parser.py
```

## Utilizzo Rapido

### Parsing con Anonimizzazione

```python
from secure_log_parser import SecureLogParser

# Inizializza parser sicuro con configurazione
parser = SecureLogParser("anonymization_config.yaml")

# Parsa e anonimizza una singola riga
result = parser.parse_and_anonymize_line('logver=123 srcip=192.168.1.1 devname="firewall"')
print(result.anonymized_fields)
print(result.anonymization_metadata)

# Parsa e anonimizza un intero file
results = parser.parse_and_anonymize_file('input/log_file.txt', 'output_anonymized.json')
```

### Parsing Base (Solo Parsing)

```python
from log_parser import UniversalLogParser

parser = UniversalLogParser()

# Parsa una singola riga
result = parser.parse_line('logver=123 type="traffic" srcip=1.2.3.4')
print(result.parsed_fields)
```

### Uso Modulare

```bash
# 🎯 Pipeline completa (locale)
python3 pipeline_orchestrator.py

# 📦 Solo preprocessing (locale)
python3 -m log_preprocessing.log_processor

# 🧠 Solo training (esterno, su dati anonimi)
python3 -m snn_training.snn_trainer dataset.csv

# 📊 Solo analisi (locale, con ricostruzione)
python3 -m result_analysis.result_analyzer
```

### Uso Distribuito (Produzione)

```python
# AMBIENTE LOCALE (sicuro)
from pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

# 1. Preprocessing sicuro
prep_results = orchestrator.run_preprocessing_only(["logs.txt"])
secure_package = prep_results['secure_package']  # → Invia per training

# 3. Analisi con ricostruzione (dopo training esterno)
analysis_results = orchestrator.run_analysis_only(
    model_export, predictions, mappings, normalization
)

# AMBIENTE ESTERNO (solo dati anonimi)
# 2. Training senza accesso a dati sensibili
train_results = orchestrator.run_training_only(secure_package)
```

## Esempi per Formato

### FortiGate (tlog/elog)
```
logver=0702111740 devid="FGT80FTK22013405" type="traffic" action="deny" srcip=1.2.3.4
```
**Output:**
```json
{
  "logver": 702111740,
  "devid": "FGT80FTK22013405",
  "type": "traffic",
  "action": "deny",
  "srcip": "1.2.3.4"
}
```

### CEF
```
CEF:0|Vendor|Product|1.0|100|Test Event|High|src=1.2.3.4 act=blocked
```
**Output:**
```json
{
  "cef_version": "0",
  "device_vendor": "Vendor",
  "device_product": "Product",
  "src": "1.2.3.4",
  "act": "blocked"
}
```

### Syslog RFC3164
```
<134>Dec 20 10:15:30 hostname app: Test message
```
**Output:**
```json
{
  "facility": 16,
  "severity": 6,
  "hostname": "hostname",
  "tag": "app",
  "message": "Test message"
}
```

## API Principale

### Classe `UniversalLogParser`

#### Metodi Principali

- `parse_line(line, format_hint=None)` → `ParsedLog`
- `parse_file(file_path, format_hint=None)` → `List[ParsedLog]`
- `detect_format(line)` → `LogFormat`

### Classe `ParsedLog`

```python
@dataclass
class ParsedLog:
    format_type: LogFormat        # Tipo di formato rilevato
    timestamp: Optional[datetime] # Timestamp estratto
    raw_message: str             # Messaggio originale
    parsed_fields: Dict[str, Any] # Campi parsati
    metadata: Dict[str, Any]      # Metadati (errori, info file)
```

## File di Test Inclusi

### Struttura Directory

- `input/` - File di log di esempio:
  - `FGT80FTK22013405.root.tlog.txt` - Log di traffico FortiGate
  - `FGT80FTK22013405.root.elog.txt` - Log di eventi FortiGate  
  - `*.csv` - Log in formato CSV

- `output/` - Directory per file generati automaticamente:
  - File JSON parsati e anonimizzati
  - Statistiche di processamento
  - Mapping di anonimizzazione (se abilitato)

## Funzionalità Avanzate

### Gestione Errori

Il parser gestisce gracefully errori di parsing:

```python
result = parser.parse_line("invalid log line")
if result.metadata.get('error'):
    print(f"Errore: {result.metadata['error']}")
```

### Timestamp Intelligente

Il parser estrae automaticamente timestamp da:
- Campi `date`/`time` FortiGate
- Campi `eventtime` (nanoseconds)
- Timestamp CEF (`rt`, `start`, etc.)
- Formato syslog RFC3164/RFC5424
- Campi CSV comuni (`Event Time`, etc.)

### Output JSON

```python
# I file vengono salvati automaticamente nella directory output/
results = parser.parse_and_anonymize_file('input/log_file.txt')
# Output salvato in: output/log_file_secure_anonymized.json
```

## Estensibilità

Il parser è progettato per essere facilmente estendibile:

```python
class CustomParser:
    @staticmethod
    def parse(line: str) -> Dict[str, Any]:
        # Implementa parsing personalizzato
        return {}

# Aggiungi al parser universale
parser = UniversalLogParser()
# Estendi con logica personalizzata
```

## Risoluzione Problemi

### Formato non Riconosciuto

Se il formato non viene rilevato automaticamente:

```python
# Specifica il formato manualmente
result = parser.parse_line(line, format_hint=LogFormat.FORTIGATE)
```

### Errori di Encoding

```python
# Per file con encoding specifico
with open('file.log', 'r', encoding='latin-1') as f:
    for line in f:
        result = parser.parse_line(line)
```

## Performance

- **Memory efficient**: Parsing line-by-line
- **Fast regex**: Pattern compilati e ottimizzati
- **Lazy loading**: Timestamp calcolati solo se richiesti

## Contributi

Per aggiungere nuovi formati:

1. Implementa una classe parser specifica
2. Aggiungi il formato all'enum `LogFormat`
3. Aggiorna il metodo `detect_format()`
4. Aggiungi test per il nuovo formato

## Licenza

Questo progetto è parte di una tesi universitaria.
