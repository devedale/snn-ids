# 🚀 Sistema di Benchmark Completo: Crypto-PAn vs Plain Text

Questo sistema esegue benchmark completi per confrontare le performance del modello di machine learning con diverse finestre temporali e con/senza anonimizzazione Crypto-PAn.

## 🎯 Obiettivi del Benchmark

### ✅ **Confronto Performance**
- **Baseline**: Modello con IP in chiaro
- **Crypto-PAn**: Modello con IP anonimizzati
- **Metriche**: Accuratezza, Precision, Recall, F1-Score

### ✅ **Analisi Finestre Temporali**
- **Risoluzioni**: 1s, 5s, 10s, 1m, 5m, 10m, 30m, 1h
- **Finestre Dinamiche**: Basate su approssimazioni del timestamp
- **Confronto**: Performance per ogni risoluzione temporale

### ✅ **Risultati Chiari**
- **Report Dettagliati**: Statistiche complete per ogni test
- **Visualizzazioni**: Grafici comparativi automatici
- **Confronto Diretto**: Baseline vs Crypto-PAn per ogni configurazione

## 🏗️ Architettura del Sistema

```
benchmark/
├── comparison_benchmark.py    # Core del benchmark
├── visualization.py           # Generazione grafici
└── config/
    └── benchmark_config.py    # Configurazioni predefinite

preprocessing/
└── temporal_windows.py        # Gestione finestre temporali

evaluation/
├── stats.py                   # Statistiche complete
└── quick_stats.py            # Statistiche rapide
```

## 🚀 Utilizzo Rapido

### 1. Benchmark Veloce (Test Iniziali)
```bash
# Benchmark rapido con 2 risoluzioni temporali
python run_benchmark.py --type quick --sample-size 10000 --visualize

# Benchmark rapido personalizzato
python run_benchmark.py --type quick --time-resolutions 5s 1m --sample-size 5000
```

### 2. Benchmark Completo (Analisi Dettagliata)
```bash
# Benchmark completo con tutte le risoluzioni
python run_benchmark.py --type full --sample-size 50000 --visualize

# Benchmark completo personalizzato
python run_benchmark.py --type full --time-resolutions 1s 5s 10s 1m 5m 10m
```

### 3. Benchmark con Configurazione Personalizzata
```bash
# Benchmark con risoluzioni specifiche
python run_benchmark.py \
    --type full \
    --sample-size 25000 \
    --time-resolutions 2s 7s 15s 45s 2m 7m \
    --output-dir custom_benchmark \
    --visualize
```

## ⚙️ Configurazioni Predefinite

### 🚀 **Quick** (Test Veloce)
- **Risoluzioni**: 5s, 1m
- **Sample Size**: 5,000
- **K-Fold**: 2
- **Tempo Max**: 2 minuti
- **Output**: `quick_benchmark_results/`

### 📊 **Standard** (Configurazione Default)
- **Risoluzioni**: 1s, 5s, 10s, 1m, 5m, 10m
- **Sample Size**: 25,000
- **K-Fold**: 3
- **Tempo Max**: 5 minuti
- **Output**: `standard_benchmark_results/`

### 🔬 **Extended** (Analisi Approfondita)
- **Risoluzioni**: 1s, 5s, 10s, 30s, 1m, 5m, 10m, 30m, 1h
- **Sample Size**: 100,000
- **K-Fold**: 5
- **Tempo Max**: 10 minuti
- **Output**: `extended_benchmark_results/`

### 🏭 **Production** (Ambiente Produzione)
- **Risoluzioni**: 1s, 5s, 10s, 1m, 5m, 10m
- **Sample Size**: Dataset completo
- **K-Fold**: 5
- **Tempo Max**: 30 minuti
- **Output**: `production_benchmark_results/`

## 📊 Output Generato

### 📁 **Struttura Directory**
```
benchmark_results/
├── baseline/                          # Test senza Crypto-PAn
│   ├── statistics_1s/                # Statistiche per 1 secondo
│   ├── statistics_5s/                # Statistiche per 5 secondi
│   └── ...
├── cryptopan/                        # Test con Crypto-PAn
│   ├── statistics_1s/
│   ├── statistics_5s/
│   └── ...
├── comparison/                        # Report di confronto
│   ├── comparison_report.json
│   └── ...
├── visualizations/                    # Grafici generati
│   ├── accuracy_comparison.png
│   ├── training_time_comparison.png
│   ├── performance_heatmap.png
│   └── performance_radar.png
├── intermediate/                      # Risultati intermedi
└── complete_benchmark_results.json    # Risultati completi
```

### 📈 **File di Risultati**
- **`complete_benchmark_results.json`**: Tutti i risultati del benchmark
- **`comparison_report.json`**: Confronto diretto baseline vs Crypto-PAn
- **`benchmark_summary.csv`**: Tabella riassuntiva per analisi
- **Statistiche individuali**: Per ogni configurazione e risoluzione

## 📊 Visualizzazioni Generate

### 1. **Confronto Accuratezza**
- Grafico a barre: Baseline vs Crypto-PAn
- Differenza percentuale per ogni risoluzione
- Impatto dell'anonimizzazione

### 2. **Confronto Tempi Training**
- Tempi di training per ogni configurazione
- Differenza percentuale nei tempi
- Impatto sulla performance computazionale

### 3. **Heatmap Performance**
- Matrice colorata per ogni risoluzione
- Confronto visivo immediato
- Identificazione pattern di performance

### 4. **Grafico Radar**
- Visione d'insieme normalizzata
- Confronto relativo tra configurazioni
- Identificazione punti di forza/debolezza

## 🔧 Configurazione Avanzata

### Personalizzazione Finestre Temporali
```python
from config.benchmark_config import create_custom_benchmark_config

# Configurazione personalizzata
custom_config = create_custom_benchmark_config(
    time_resolutions=['2s', '7s', '15s', '45s'],
    sample_size=30000,
    output_dir='my_custom_benchmark',
    k_fold_splits=4
)
```

### Configurazione Manuale
```python
from benchmark.comparison_benchmark import CryptoPanComparisonBenchmark

config = {
    'time_resolutions': ['1s', '5s', '10s', '1m'],
    'sample_size': 20000,
    'output_dir': 'manual_benchmark',
    'save_intermediate_results': True,
    'generate_comparison_plots': True
}

benchmark = CryptoPanComparisonBenchmark(config)
results = benchmark.run_complete_benchmark(df)
```

## 📋 Esempi di Utilizzo

### Esempio 1: Test Iniziale
```bash
# Test veloce per verificare il funzionamento
python run_benchmark.py --type quick --sample-size 5000 --visualize
```

### Esempio 2: Analisi Standard
```bash
# Benchmark completo con configurazione standard
python run_benchmark.py --type full --sample-size 25000 --visualize
```

### Esempio 3: Analisi Personalizzata
```bash
# Benchmark con risoluzioni specifiche
python run_benchmark.py \
    --type full \
    --sample-size 50000 \
    --time-resolutions 2s 7s 15s 45s 2m 7m \
    --output-dir custom_analysis \
    --visualize
```

### Esempio 4: Solo Visualizzazione
```bash
# Genera solo i grafici da risultati esistenti
python -c "
from benchmark.visualization import visualize_benchmark_results
visualize_benchmark_results('benchmark_results')
"
```

## 📊 Interpretazione dei Risultati

### 🔍 **Metriche Chiave**
- **Accuracy Difference**: Differenza di accuratezza (Crypto-PAn - Baseline)
- **Performance Impact**: "improvement", "degradation", o "no_change"
- **Time Difference**: Differenza nei tempi di training
- **Window Statistics**: Statistiche delle finestre temporali

### 📈 **Analisi delle Performance**
- **Accuratezza**: Se Crypto-PAn mantiene le performance
- **Tempi**: Overhead computazionale dell'anonimizzazione
- **Finestre**: Quale risoluzione temporale funziona meglio
- **Stabilità**: Consistenza tra diverse configurazioni

### 🎯 **Raccomandazioni**
- **Accuratezza > 0.95**: Performance eccellenti
- **Accuratezza 0.90-0.95**: Performance buone
- **Accuratezza < 0.90**: Necessita ottimizzazione
- **Time Overhead < 10%**: Accettabile
- **Time Overhead > 20%**: Richiede investigazione

## 🚨 Risoluzione Problemi

### Errore: "ModuleNotFoundError"
```bash
# Assicurati di essere nella directory root
cd /path/to/snn-ids
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Errore: "Out of Memory"
```bash
# Riduci la dimensione del sample
python run_benchmark.py --type quick --sample-size 1000
```

### Errore: "Training Timeout"
```bash
# Usa configurazione più veloce
python run_benchmark.py --type quick --sample-size 5000
```

### Errore: "No Data Found"
```bash
# Il sistema creerà automaticamente dati di esempio
# Oppure verifica la directory data/cicids/
```

## 🔄 Integrazione con Pipeline Esistente

### ✅ **Compatibilità**
- **Preprocessing**: Finestre temporali configurabili
- **Training**: Integrazione automatica con sistema esistente
- **Statistiche**: Generazione automatica di report completi
- **Configurazione**: Sistema di configurazione centralizzato

### 🔗 **Workflow**
1. **Caricamento Dati** → Dataset reale o simulato
2. **Creazione Finestre** → Finestre temporali configurabili
3. **Preprocessing** → Con o senza Crypto-PAn
4. **Training** → Modello per ogni configurazione
5. **Valutazione** → Statistiche complete
6. **Confronto** → Baseline vs Crypto-PAn
7. **Visualizzazione** → Grafici comparativi
8. **Report** → Risultati strutturati

## 📝 Note per la Produzione

### 🔒 **Sicurezza**
- **Chiavi Crypto-PAn**: Gestite in modo sicuro
- **Audit Log**: Tracciamento operazioni di anonimizzazione
- **Key Rotation**: Rotazione automatica delle chiavi

### 📊 **Performance**
- **Parallel Processing**: Per dataset grandi
- **Memory Management**: Limiti configurabili
- **Batch Optimization**: Ottimizzazione automatica

### 📈 **Scalabilità**
- **Sample Sizing**: Configurabile per diversi ambienti
- **Incremental Analysis**: Analisi per parti
- **Distributed Processing**: Supporto per cluster

## 🤝 Contributi e Estensioni

### 🆕 **Nuove Metriche**
1. Modifica `evaluation/stats.py`
2. Aggiungi configurazione in `config/benchmark_config.py`
3. Aggiorna `benchmark/visualization.py`

### 🔧 **Nuove Configurazioni**
1. Estendi `BENCHMARK_CONFIG` in `config/benchmark_config.py`
2. Aggiungi template in `get_benchmark_config()`
3. Testa con `run_benchmark.py`

### 📊 **Nuove Visualizzazioni**
1. Aggiungi metodo in `BenchmarkVisualizer`
2. Integra in `generate_comprehensive_report()`
3. Testa con dati reali

## 📚 Riferimenti

- **Crypto-PAn**: RFC 6233 - IP Address Anonymization
- **Time Series Analysis**: Finestre temporali per ML
- **Benchmark Methodology**: Best practices per confronti
- **Statistical Analysis**: Metriche per classificazione

---

**🎯 Obiettivo**: Fornire un sistema completo per valutare l'impatto dell'anonimizzazione Crypto-PAn sulle performance del modello di machine learning, con particolare attenzione alle finestre temporali configurabili.

**🚀 Risultato**: Benchmark automatizzato che produce report dettagliati e visualizzazioni comparative per supportare decisioni informate sull'uso di Crypto-PAn in produzione.
