# Guida all'Utilizzo - SNN Security Analytics

## 🎯 Panoramica Rapida

Sistema modulare per analisi log di sicurezza con **3 moduli atomici e interconnessi**:

1. **📦 Log Preprocessing** (Locale) - Parsing + Anonimizzazione + Preparazione SNN
2. **🧠 SNN Training** (Esterno) - Training su dati anonimi
3. **📊 Result Analysis** (Locale) - Analisi + Ricostruzione log originali

## 🚀 Quick Start

### Test Completo Sistema
```bash
# Verifica che tutto funzioni
python3 test_modular_structure.py

# Pipeline completa (locale)
python3 pipeline_orchestrator.py
```

## 🛠️ Guida passo‑passo (dall’input all’analisi)

1) Prerequisiti
- Python 3.9+ e dipendenze base: `pip install -r requirements.txt`
- Per usare Nengo: `pip install nengo nengo-dl`
- Per usare YAML nelle config training: `pip install pyyaml`

2) Prepara i file di input
- Copia i log in `input/`. Esempi supportati: FortiGate (`*.tlog.txt`, `*.elog.txt`), Intune XDR (.csv), CEF, Syslog.
- Puoi partire dai file di esempio in `input/` già inclusi.

3) Configura il preprocessing SNN
- File: `log_preprocessing/snn_config.yaml`
- Parametri chiave:
  - `temporal_windows`: finestre di aggregazione eventi (ms/s) e `aggregation_strategy` (mean/sum/max/min/last)
  - `normalization`: metodo (`min_max`, `z_score`, `robust`) e clipping outlier
  - `fortigate_features`/`csv_features`/`cef_features`/`syslog_features`: quali campi usare ed encoding
  - `security`: Crypto-PAn key, hash salt, salvataggio mapping/statistiche (solo in locale)
  - `output_format`: formato export (`csv`, `json`, `parquet`) e prefissi colonne

4) Configura il training SNN (consigliato via YAML)
- Crea un file `training_config.yaml` (o usa `training_config.example.yaml` come base)
- Parametri chiave:
  - `framework`: `mock` (veloce) o `nengo` (se installato)
  - Architettura: `hidden_layers`, `output_neurons`, `neuron_type` (lif/adaptive/izhikevich se supportato)
  - Iperparametri: `epochs`, `batch_size`, `learning_rate`, `validation_split`
  - SNN-specific: `dt`, `simulation_time`, `encoding_method` (rate/temporal/population)
  - Output: `model_name`, `export_format`

5) Esegui il preprocessing
```bash
python3 -c "from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig; \
orc=PipelineOrchestrator(PipelineConfig(secure_export=False)); \
print(orc.run_preprocessing_only(['input/FGT80FTK22013405.root.tlog.txt','input/FGT80FTK22013405.root.elog.txt']))"
```

6) Esegui il training (standalone)
```python
from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig

cfg = PipelineConfig(
  training_config_file="training_config.yaml",  # se assente, usa fallback inline
  training_framework="mock",                    # fallback
  training_epochs=5                              # override veloce
)
orc = PipelineOrchestrator(cfg)
results = orc.run_training_only("output/pipeline_preprocessing_snn_dataset.csv")
```

7) Esegui l’analisi (standalone)
```python
analysis = orc.run_analysis_only(
  results['export_path'],      # export del modello
  "dummy_predictions.json",   # o un file di predizioni reali (CSV/JSON)
  mappings_path=None,          # opzionale: mapping locale
  normalization_path=None      # opzionale: stats locale
)
```

8) Pipeline completa (opzionale)
```bash
python3 pipeline_orchestrator.py
```

9) Cambiare tipo di SNN (framework)
- Imposta `framework` nel `training_config.yaml`:
  - `mock`: nessuna dipendenza, training simulato veloce per test
  - `nengo`: richiede `pip install nengo nengo-dl`; esegue build e fit con `nengo-dl`
- In assenza del file o framework non supportato, si usa il fallback `mock`.

10) Verifica risultati
- Output in `output/`: dataset SNN, modelli, export analisi, mapping/statistiche (solo locale)
- Usa `quick_test.py` per un giro completo in pochi secondi: `python3 quick_test.py`

### Uso Standalone Moduli

#### 📦 Modulo 1: Preprocessing
```python
from log_preprocessing import LogProcessor, ProcessingConfig

config = ProcessingConfig(
    anonymization_config="log_preprocessing/anonymization_config.yaml",
    snn_config="log_preprocessing/snn_config.yaml",
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

#### 🧠 Modulo 2: Training
```python
from snn_training import SNNTrainer, TrainingConfig

config = TrainingConfig(
    framework="mock",  # o "nengo" se installato
    hidden_layers=[64, 32, 16],
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

trainer = SNNTrainer(config)
results = trainer.train("output/snn_dataset.csv")
model_path = trainer.save_model("output/models")
export_path = trainer.export_for_analysis("output/model_export.json")
```

### 📄 Configurazione Training via YAML (opzionale)

È possibile definire tutti i parametri di training in un file YAML e farli caricare automaticamente dall'orchestratore.

Esempio `training_config.yaml`:

```yaml
framework: mock                # nengo | mock
input_neurons: 0               # viene ricavato dal dataset
hidden_layers: [128, 64, 16]
output_neurons: 2
neuron_type: lif               # lif | adaptive | izhikevich
learning_rate: 0.001
epochs: 10
batch_size: 32
validation_split: 0.2
dt: 0.001
simulation_time: 1.0
encoding_method: rate          # rate | temporal | population
model_name: snn_security_model
save_checkpoints: true
export_format: standard        # standard | onnx | tensorflow
```

Uso nell'orchestratore:

```python
from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig

config = PipelineConfig(
    training_config_file="training_config.yaml"  # se non esiste, usa i default inline
)
orchestrator = PipelineOrchestrator(config)
results = orchestrator.run_training_only("output/pipeline_preprocessing_snn_dataset.csv")
```

#### 📊 Modulo 3: Analysis
```python
from result_analysis import ResultAnalyzer, AnalysisConfig
import numpy as np

config = AnalysisConfig(
    enable_reconstruction=True,
    anomaly_threshold=0.8,
    include_temporal_analysis=True
)

analyzer = ResultAnalyzer(config)

# Setup ricostruzione (solo locale)
analyzer.setup_reconstruction(
    "output/security_mappings.json",
    "output/normalization_stats.json"
)

# Analisi predizioni
predictions = np.random.random((100, 2))  # Simula predizioni
results = analyzer.analyze_predictions(predictions)

# Ricostruzione anomalie
anomaly_summary = analyzer.get_anomaly_summary()
reconstructed = analyzer.reconstruct_highlighted_logs(
    anomaly_summary['anomaly_indices'],
    "output/snn_dataset.csv",
    ["feat_0", "feat_1", ...]  # Feature names
)

# Export report
report_path = analyzer.export_analysis_report(
    "output/analysis_report.json",
    reconstructed
)
```

## 🔄 Flussi di Utilizzo

### Scenario 1: Uso Locale Completo
```python
from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig

config = PipelineConfig(
    training_framework="mock",
    enable_reconstruction=True,
    secure_export=False
)

orchestrator = PipelineOrchestrator(config)

# Pipeline completa
results = orchestrator.run_full_pipeline([
    "input/FGT80FTK22013405.root.tlog.txt",
    "input/FGT80FTK22013405.root.elog.txt"
])

print(f"Anomalie rilevate: {results['total_anomalies']}")
```

## ⚠️ Limiti, note e best practice

- Sicurezza e privacy
  - I file `*_security_mappings.json` e `*_normalization_stats.json` contengono informazioni sensibili per la ricostruzione: mantenerli solo in ambiente locale sicuro; non esportarli.
  - L’export per training esterno non include mapping/statistiche.

- Preprocessing
  - L’encoding delle feature è minimale per SNN: riduci cardinalità e rumore, evita campi testuali liberi.
  - Finestre temporali: finestre troppo corte aumentano il rumore, troppo lunghe appiattiscono il segnale. Parti da `windows_ms: [1,5,10,50,100]` e valuta.
  - Normalizzazione: `min_max` è semplice e funziona bene per rate coding in [0,1]. Usa `z_score` se il modello richiede valori standardizzati.

- Training
  - `mock` è per sviluppo/test rapido: non riflette qualità predittiva reale.
  - `nengo` richiede dipendenze e ha tempi maggiori; verifica GPU/CPU disponibili e dimensionamento batch.
  - `input_neurons` viene impostato automaticamente dal dataset; non forzarlo.
  - `output_neurons`: 2 nell’esempio (anomalo/ok). Adatta se fai multi‑classe.

- Analisi
  - La metrica/threshold anomalie è semplificata; tarala sul tuo dominio. Valuta soglie, finestra temporale e post‑processing (cluster/aggregazioni).

- Estendibilità
  - Framework aggiuntivi (es. `snntorch`, `norse`, `bindsnet`) possono essere integrati implementando la classe `SNNFramework` nel modulo training.
  - Aggiungi nuovi formati di log editando `log_preprocessing/snn_config.yaml` nelle sezioni dedicate.

- Troubleshooting
  - WARN Presidio in ambiente locale non bloccano l’esecuzione: il flusso continua con fallback.
  - Se `pyyaml` non è installato e usi config YAML: `pip install pyyaml`.
  - Per `nengo`: verifica versioni compatibili di `nengo` e `nengo-dl`.

### Scenario 2: Uso Distribuito (Produzione)

#### Ambiente Locale (Sicuro)
```python
orchestrator = PipelineOrchestrator()

# 1. Preprocessing sicuro
prep_results = orchestrator.run_preprocessing_only(["logs.txt"])
secure_package = prep_results['secure_package']

# → Invia 'secure_package' per training esterno

# 3. Analisi con ricostruzione (dopo training)
analysis_results = orchestrator.run_analysis_only(
    "model_export.json",     # Da training esterno
    "predictions.json",      # Da training esterno
    "mappings.json",         # Mapping locale
    "normalization.json"     # Stats locale
)
```

#### Ambiente Esterno (Solo dati anonimi)
```python
# 2. Training su dati anonimi
train_results = orchestrator.run_training_only("secure_package.csv")
# → Restituisci model_export.json e predictions.json
```

## 📂 Struttura Output

```
output/
├── *_snn_dataset.csv              # Dataset SNN formattato
├── *_snn_dataset.metadata.json    # Metadati dataset
├── *_anonymized_logs.json         # Log anonimizzati (backup)
├── *_security_mappings.json       # Mapping ricostruzione (LOCALE)
├── *_normalization_stats.json     # Stats normalizzazione (LOCALE)
├── *.tar.gz                       # Pacchetti sicuri per esterno
├── models/
│   ├── snn_model.json            # Modello trained
│   └── snn_model_config.json     # Config modello
└── analysis_report.json          # Report analisi finale
```

## 🔒 Sicurezza e Confidenzialità

### File Sensibili (Solo Locale)
- `*_security_mappings.json` - Mapping originale→anonimo
- `*_normalization_stats.json` - Statistiche denormalizzazione
- `analysis_report.json` - Log ricostruiti

### File Sicuri (Per Esterno)
- `*_snn_dataset.csv` - Solo dati anonimi normalizzati
- `*.tar.gz` - Pacchetti senza mapping

### Principi Sicurezza
✅ **Crypto-PAn** per IP preservando subnet  
✅ **Hash sicuri** con salt configurabile  
✅ **Mapping separati** mai esportati  
✅ **Ricostruzione locale** solo in ambiente sicuro  

## ⚙️ Configurazione

### Modulo 1: Preprocessing
- `log_preprocessing/anonymization_config.yaml` - Regole anonimizzazione
- `log_preprocessing/snn_config.yaml` - Config preprocessing SNN

### Modulo 2: Training
```python
TrainingConfig(
    framework="nengo",           # nengo, mock, norse, snntorch
    hidden_layers=[64, 32],      # Architettura
    epochs=100,                  # Training epochs
    batch_size=32,               # Batch size
    learning_rate=0.001,         # Learning rate
    neuron_type="lif",           # LIF, adaptive, izhikevich
    encoding_method="rate"       # rate, temporal, population
)
```

### Modulo 3: Analysis
```python
AnalysisConfig(
    enable_reconstruction=True,  # Abilita ricostruzione
    confidence_threshold=0.7,    # Soglia confidence
    anomaly_threshold=0.8,       # Soglia anomalie
    include_temporal_analysis=True
)
```

## 📋 Checklist Pre-Produzione

### Ambiente Locale
- [ ] File input presenti in `input/`
- [ ] Configurazioni in `log_preprocessing/`
- [ ] Directory `output/` scrivibile
- [ ] Dipendenze installate: `pip install -r requirements.txt`

### Ambiente Esterno (Training)
- [ ] Solo dipendenze SNN: numpy, pandas, framework SNN
- [ ] Nessun accesso a mapping o file sensibili
- [ ] Solo dataset anonimi in input

### Validazione
- [ ] Test struttura: `python3 test_modular_structure.py`
- [ ] Test pipeline: `python3 pipeline_orchestrator.py`
- [ ] Verifica output in `output/`

## 🔧 Troubleshooting

### Import Errors
```bash
# Verifica import moduli
python3 -c "from log_preprocessing import LogProcessor; print('OK')"
python3 -c "from snn_training import SNNTrainer; print('OK')"
python3 -c "from result_analysis import ResultAnalyzer; print('OK')"
```

### Config Errors
- Verifica path config files
- Controlla formato YAML
- Assicurati che file input esistano

### Presidio Slow Loading
- Import lazy implementato
- Caricamento solo quando necessario
- Fallback su processing senza Presidio

## 📚 Riferimenti

- **Architettura**: `ARCHITECTURE.md`
- **Modulo 1**: `log_preprocessing/README.md`
- **Modulo 2**: `snn_training/README.md`
- **Modulo 3**: `result_analysis/README.md`
- **Test**: `test_modular_structure.py`

---

**🎯 Sistema pronto per uso professionale in cybersecurity!**
