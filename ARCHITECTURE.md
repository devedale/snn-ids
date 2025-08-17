# Architettura Modulare SNN Security

## Overview
Sistema modulare per analisi log di sicurezza con Spiking Neural Networks, progettato per funzionamento atomico e interconnesso mantenendo la confidenzialità.

## 🏗️ Architettura a 3 Moduli

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MODULO 1      │    │   MODULO 2      │    │   MODULO 3      │
│ Log Preprocessing│    │  SNN Training   │    │ Result Analysis │
│                 │    │                 │    │                 │
│ 📍 LOCALE       │───▶│ 🌐 ESTERNO      │───▶│ 📍 LOCALE       │
│ ✅ Confidenziale│    │ 🔒 Anonimo      │    │ ✅ Confidenziale│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Modulo 1: Log Preprocessing

### Responsabilità
- Parsing log multi-formato
- Anonimizzazione sicura
- Preparazione dati SNN
- Export pacchetti sicuri

### Componenti
```
log_preprocessing/
├── __init__.py
├── log_processor.py       # 🎯 Entry point principale
├── log_parser.py          # Parser universale
├── anonymizer.py          # Anonimizzazione Presidio
├── snn_preprocessor.py    # Preparazione SNN
├── anonymization_config.yaml
└── snn_config.yaml
```

### Input/Output
- **Input**: Log raw (FortiGate, CSV, CEF, Syslog)
- **Output**: 
  - Dataset SNN anonimizzato
  - Mapping sicuri (locale)
  - Pacchetto training (esterno)

## 🧠 Modulo 2: SNN Training

### Responsabilità
- Training SNN con framework multipli
- Gestione modelli
- Export risultati

### Componenti
```
snn_training/
├── __init__.py
├── snn_trainer.py         # 🎯 Entry point principale
├── model_manager.py       # Gestione modelli (TODO)
└── training_config.py     # Configurazioni (TODO)
```

### Framework Supportati
- **Nengo** - Framework completo SNN
- **Mock** - Per test senza dipendenze
- **Futuri**: Norse, SNNTorch, BindsNET

### Input/Output
- **Input**: Dataset SNN anonimizzato
- **Output**: 
  - Modello trained
  - Metriche training
  - Export per analisi

## 📊 Modulo 3: Result Analysis

### Responsabilità
- Analisi predizioni SNN
- Rilevamento anomalie
- Ricostruzione log originali
- Report completi

### Componenti
```
result_analysis/
├── __init__.py
├── result_analyzer.py     # 🎯 Entry point principale
├── log_reconstructor.py   # Embedded
└── evaluation_metrics.py  # TODO
```

### Input/Output
- **Input**: 
  - Predizioni modello
  - Mapping sicuri (locale)
- **Output**: 
  - Anomalie rilevate
  - Log ricostruiti
  - Report analisi

## 🎭 Orchestratore Pipeline

### Coordinamento
```python
pipeline_orchestrator.py  # 🎯 Coordinatore principale
```

### Modalità di Esecuzione
1. **Moduli Singoli** - Uso atomico indipendente
2. **Pipeline Completa** - Tutti i moduli in sequenza
3. **Uso Distribuito** - Moduli in ambienti diversi

## 🔒 Gestione Sicurezza

### Separazione Ambienti
- **Locale (Modulo 1+3)**: Accesso a dati sensibili e mapping
- **Esterno (Modulo 2)**: Solo dati anonimi, nessun mapping

### Flusso Sicurezza
```
Dati Raw ──┐
           ├─► [Modulo 1] ──► Dati Anonimi ──► [Modulo 2] ──► Predizioni ──┐
Mapping ───┘                                                                ├─► [Modulo 3] ──► Log Ricostruiti
                                                                Mapping ────┘
```

### Meccanismi Sicurezza
- **Crypto-PAn** per IP preservando subnet
- **Hash sicuri** con salt configurabile
- **Mapping separati** mai esportati
- **Pacchetti anonimi** per training esterno

## 🔄 Flusso Dati Completo

### 1. Preprocessing (Locale)
```
Log Raw → Parse → Anonimizza → Formato SNN → Export Sicuro
         ↓
    Salva Mapping (locale)
```

### 2. Training (Esterno)
```
Dataset Anonimo → Build Model → Train → Export Risultati
```

### 3. Analysis (Locale)
```
Predizioni + Mapping → Analisi → Ricostruzione → Report
```

## 🔧 Configurazione Modulare

### Config Files
- `anonymization_config.yaml` - Anonimizzazione
- `snn_config.yaml` - Preprocessing SNN
- `TrainingConfig` - Parametri training
- `AnalysisConfig` - Parametri analisi

### Esempi Uso

#### Uso Completo Locale
```python
orchestrator = PipelineOrchestrator()
results = orchestrator.run_full_pipeline(["logs.txt"])
```

#### Uso Distribuito
```python
# LOCALE: Preprocessing
prep_results = orchestrator.run_preprocessing_only(["logs.txt"])
package = prep_results['secure_package']  # → Invia per training

# ESTERNO: Training  
train_results = orchestrator.run_training_only(package)
model_export = train_results['export_path']  # → Ricevi risultati

# LOCALE: Analysis
analysis_results = orchestrator.run_analysis_only(
    model_export, predictions, mappings, normalization
)
```

## 📈 Benefici Architettura

### ✅ Atomicità
- Ogni modulo funziona indipendentemente
- Interfacce standardizzate
- Testing isolato

### ✅ Interconnessione
- Flusso dati coerente
- Configurazione unificata
- Orchestrazione centrale

### ✅ Confidenzialità
- Separazione ambiente sensibile/esterno
- Mapping mai esportati
- Dati anonimi per training

### ✅ Flessibilità
- Framework SNN intercambiabili
- Configurazioni adattabili
- Uso in contesti diversi

## 🚀 Deployment

### Scenario 1: Tutto Locale
```bash
# Setup completo
./install_dependencies.sh
python pipeline_orchestrator.py
```

### Scenario 2: Distribuito
```bash
# LOCALE: Preprocessing
python -m log_preprocessing.log_processor

# ESTERNO: Training (senza mapping)
python -m snn_training.snn_trainer dataset.csv

# LOCALE: Analysis
python -m result_analysis.result_analyzer
```

Questa architettura garantisce **sicurezza**, **flessibilità** e **atomicità** per l'uso professionale in contesti di cybersecurity! 🎯
