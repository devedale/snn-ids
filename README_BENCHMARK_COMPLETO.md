# 🚀 SNN-IDS: Benchmark Completo Crypto-PAn vs Plain Text

Questo progetto implementa un sistema completo di benchmark per confrontare le performance del modello SNN-IDS con e senza anonimizzazione Crypto-PAn, utilizzando diverse finestre temporali.

## 📋 **Prerequisiti**

```bash
# Installazione dipendenze
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn requests tqdm

# Verifica versione Python
python3 --version  # Richiesto Python 3.8+
```

## 🎯 **Obiettivo del Benchmark**

Confrontare le performance del modello SNN-IDS:
- **Senza anonimizzazione** (baseline)
- **Con anonimizzazione Crypto-PAn**

Su diverse **risoluzioni temporali**: 1s, 5s, 10s, 1m, 5m, 10m

## 📥 **1. Download Dataset CICIDS2018 Completo**

### 🌐 **URL Dataset Disponibili**

```bash
# Dataset CICIDS2018 (5 giorni lavorativi)
Monday: https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Monday-WorkingHours.pcap_ISCX.csv
Tuesday: https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday: https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Wednesday-workingHours.pcap_ISCX.csv
Thursday: https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Thursday-WorkingHours.pcap_ISCX.csv
Friday: https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Friday-WorkingHours.pcap_ISCX.csv
```

### 🚀 **Download Automatico**

```bash
# Crea le directory necessarie
mkdir -p data/cicids data/raw

# Download con wget (più veloce)
cd data/cicids

# Monday
wget "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Monday-WorkingHours.pcap_ISCX.csv" -O "Monday-WorkingHours.csv"

# Tuesday
wget "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Tuesday-WorkingHours.pcap_ISCX.csv" -O "Tuesday-WorkingHours.csv"

# Wednesday
wget "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Wednesday-workingHours.pcap_ISCX.csv" -O "Wednesday-WorkingHours.csv"

# Thursday
wget "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Thursday-WorkingHours.pcap_ISCX.csv" -O "Thursday-WorkingHours.csv"

# Friday
wget "https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Friday-WorkingHours.pcap_ISCX.csv" -O "Friday-WorkingHours.csv"

cd ../..
```

### 📊 **Verifica Download**

```bash
# Conta i record per ogni file
echo "📊 Verifica dataset scaricati:"
echo "================================"

for day in Monday Tuesday Wednesday Thursday Friday; do
    if [ -f "data/cicids/${day}-WorkingHours.csv" ]; then
        lines=$(wc -l < "data/cicids/${day}-WorkingHours.csv")
        size=$(du -h "data/cicids/${day}-WorkingHours.csv" | cut -f1)
        echo "📁 ${day}: $((lines-1)) record, ${size}"
    else
        echo "❌ ${day}: File mancante"
    fi
done
```

## 🔧 **2. Preparazione Ambiente**

### 📁 **Struttura Directory**

```bash
# Crea tutte le directory necessarie
mkdir -p benchmark_results/{intermediate,comparison,visualizations}
mkdir -p models output
mkdir -p evaluation_results training_results statistics
```

### ✅ **Verifica File Richiesti**

```bash
# Verifica che tutti i file necessari siano presenti
required_files=(
    "run_benchmark.py"
    "benchmark/comparison_benchmark.py"
    "benchmark/visualization.py"
    "config/benchmark_config.py"
    "preprocessing/process.py"
    "training/train.py"
    "evaluation/stats.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MANCANTE!"
    fi
done
```

## 🚀 **3. Esecuzione Benchmark**

### 🧪 **Benchmark Rapido (Test)**

Prima di eseguire il benchmark completo, testa con un dataset ridotto:

```bash
# Benchmark rapido con 5000 record
python3 run_benchmark.py --type quick --sample-size 5000 --visualize

# Tempo stimato: 10-30 minuti
# Output: benchmark_results/
```

### 📊 **Benchmark Standard**

```bash
# Benchmark standard con 50000 record
python3 run_benchmark.py --type standard --sample-size 50000 --visualize

# Tempo stimato: 30-60 minuti
# Output: benchmark_results/
```

### 🎯 **Benchmark Completo**

```bash
# Benchmark completo con dataset intero
python3 run_benchmark.py --type full --visualize

# Tempo stimato: 2-6 ore
# Output: benchmark_results/
```

### 🔧 **Opzioni Avanzate**

```bash
# Solo configurazione baseline (senza Crypto-PAn)
python3 run_benchmark.py --type quick --sample-size 10000 --no-cryptopan

# Solo configurazione Crypto-PAn
python3 run_benchmark.py --type quick --sample-size 10000 --cryptopan-only

# Risoluzioni temporali personalizzate
python3 run_benchmark.py --type quick --sample-size 10000 --time-resolutions "1s,5s,1m"

# Output personalizzato
python3 run_benchmark.py --type quick --sample-size 10000 --output-dir "my_results"
```

## 📈 **4. Monitoraggio Progressi**

### 📁 **File Intermedi**

```bash
# Monitora i progressi in tempo reale
watch -n 10 "ls -la benchmark_results/intermediate/ | wc -l"

# Conta file completati
ls benchmark_results/intermediate/*.json | wc -l

# Ultimi file processati
ls -lt benchmark_results/intermediate/*.json | head -5
```

### 📊 **File di Output**

```bash
# Risultati principali
ls -la benchmark_results/complete_benchmark_results.json

# Report di confronto
ls -la benchmark_results/comparison/comparison_report.json

# Visualizzazioni
ls -la benchmark_results/visualizations/*.png
```

## 📊 **5. Analisi Risultati**

### 🔍 **Struttura Risultati**

```json
{
  "baseline": {
    "1s": {
      "accuracy": 0.9234,
      "training_time": 45.2,
      "memory_usage": 1024
    },
    "5s": { ... },
    "10s": { ... },
    "1m": { ... },
    "5m": { ... },
    "10m": { ... }
  },
  "cryptopan": {
    "1s": { ... },
    "5s": { ... },
    "10s": { ... },
    "1m": { ... },
    "5m": { ... },
    "10m": { ... }
  }
}
```

### 📈 **Metriche Disponibili**

- **Accuracy**: Precisione del modello
- **Training Time**: Tempo di addestramento
- **Memory Usage**: Utilizzo memoria
- **F1-Score**: Media armonica precision/recall
- **Confusion Matrix**: Matrice di confusione dettagliata

## 🎨 **6. Visualizzazioni**

### 📊 **Grafici Generati**

```bash
# Lista visualizzazioni disponibili
ls benchmark_results/visualizations/

# Grafici tipici:
# - accuracy_comparison.png: Confronto accuracy per configurazione
# - training_time_comparison.png: Confronto tempi di training
# - performance_heatmap.png: Heatmap performance per risoluzione
# - radar_chart.png: Grafico radar per metriche multiple
# - summary_table.png: Tabella riassuntiva risultati
```

### 🔧 **Generazione Manuale**

```bash
# Genera solo le visualizzazioni
python3 -c "
from benchmark.visualization import visualize_benchmark_results
visualize_benchmark_results('benchmark_results')
"
```

## 📋 **7. Configurazioni Disponibili**

### ⚙️ **Tipi di Benchmark**

| Tipo | Sample Size | Tempo | Risoluzioni | Uso |
|------|-------------|-------|-------------|-----|
| `quick` | 5K-10K | 10-30 min | 1s,5s,10s,1m,5m,10m | Test rapido |
| `standard` | 50K-100K | 30-60 min | 1s,5s,10s,1m,5m,10m | Test medio |
| `full` | Completo | 2-6 ore | 1s,5s,10s,1m,5m,10m | Benchmark completo |

### ⏰ **Risoluzioni Temporali**

- **1s**: 1 secondo - Finestre molto piccole, alta granularità
- **5s**: 5 secondi - Finestre piccole, buona granularità
- **10s**: 10 secondi - Finestre medie, equilibrio granularità/performance
- **1m**: 1 minuto - Finestre grandi, buona performance
- **5m**: 5 minuti - Finestre molto grandi, alta performance
- **10m**: 10 minuti - Finestre massime, massima performance

## 🚨 **8. Risoluzione Problemi**

### ❌ **Errori Comuni**

#### **"Input 0 of layer 'lstm' is incompatible"**
```bash
# Problema: LSTM si aspetta input 3D ma riceve 2D
# Soluzione: Il sistema usa automaticamente modello Dense
# Verifica: config['model_type'] = 'dense' in run_benchmark.py
```

#### **"Killed" (Processo terminato)**
```bash
# Problema: Memoria insufficiente
# Soluzioni:
# 1. Riduci sample_size: --sample-size 5000
# 2. Usa benchmark rapido: --type quick
# 3. Aumenta swap o RAM disponibile
```

#### **"ModuleNotFoundError"**
```bash
# Problema: Dipendenze mancanti
# Soluzione: Installa tutte le dipendenze
pip install -r requirements.txt  # se disponibile
# oppure
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

### 🔧 **Debug e Log**

```bash
# Abilita log dettagliati
export TF_CPP_MIN_LOG_LEVEL=0

# Verifica utilizzo memoria
htop
# oppure
free -h

# Verifica spazio disco
df -h
```

## 📊 **9. Interpretazione Risultati**

### 🎯 **Metriche Chiave**

1. **Accuracy**: > 0.90 = Eccellente, > 0.80 = Buono, < 0.70 = Migliorabile
2. **Training Time**: Più basso = Migliore (efficienza)
3. **Memory Usage**: Più basso = Migliore (scalabilità)

### 📈 **Confronto Crypto-PAn vs Baseline**

- **Accuracy**: Differenza < 2% = Crypto-PAn non degrada performance
- **Training Time**: Aumento < 20% = Overhead accettabile
- **Memory**: Aumento < 10% = Overhead minimo

### ⏰ **Migliore Risoluzione Temporale**

- **Alta granularità**: 1s, 5s (per dettagli temporali)
- **Equilibrio**: 10s, 1m (per uso generale)
- **Alta performance**: 5m, 10m (per produzione)

## 🎉 **10. Riepilogo e Prossimi Passi**

### ✅ **Cosa Abbiamo Ottenuto**

1. **Sistema di benchmark completo** per SNN-IDS
2. **Confronto Crypto-PAn vs Plain Text** su multiple risoluzioni
3. **Metriche dettagliate** per ogni configurazione
4. **Visualizzazioni comparative** per analisi
5. **Sistema modulare** e configurabile

### 🎯 **Prossimi Passi Consigliati**

1. **Esegui benchmark rapido** per verificare funzionamento
2. **Analizza risultati** per trovare configurazione ottimale
3. **Esegui benchmark completo** per risultati definitivi
4. **Ottimizza parametri** basandoti sui risultati
5. **Deploya configurazione migliore** in produzione

### 📚 **Risorse Aggiuntive**

- **Notebook Jupyter**: `SNN_IDS_Benchmark_Completo.ipynb`
- **Documentazione**: `README_BENCHMARK.md`
- **Configurazioni**: `config/benchmark_config.py`
- **Risultati**: `benchmark_results/`

---

**🚀 Buon Benchmark!**

Per supporto o domande, consulta la documentazione o esegui il notebook Jupyter incluso.
