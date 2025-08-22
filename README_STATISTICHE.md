# Sistema di Statistiche e Valutazione del Modello

Questo documento spiega come utilizzare il nuovo sistema di statistiche e valutazione integrato nel pipeline di machine learning per la cybersecurity.

## 🎯 Caratteristiche Principali

### ✅ Statistiche Complete
- **Matrice di Confusione**: Visualizzazione completa delle performance del modello
- **Metriche per Classe**: Precision, Recall, F1-Score per ogni classe
- **Statistiche del Dataset**: Distribuzione delle classi e statistiche delle features
- **Riepilogo del Training**: Performance di tutte le configurazioni testate

### ✅ Configurabilità Massima
- **Template Predefiniti**: Default, Minimal, Comprehensive
- **Configurazione Personalizzata**: Ogni aspetto può essere personalizzato
- **Metriche Personalizzate**: Supporto per metriche aggiuntive (Balanced Accuracy, Cohen's Kappa, etc.)

### ✅ Output Multiplo
- **File JSON**: Per analisi programmatiche
- **Plot Visivi**: Matrice di confusione e grafici
- **Console**: Riepilogo immediato delle performance
- **Report Completi**: Tutto in un unico file

## 🚀 Utilizzo Base

### 1. Statistiche Automatiche (Integrate nel Training)

Le statistiche vengono generate automaticamente dopo il training:

```python
from training.train import train_and_evaluate

# Esegui training e genera statistiche automaticamente
training_log, best_model_path = train_and_evaluate(X, y)
```

**Output generato automaticamente:**
```
models/
├── best_model.keras
├── training_log.json
└── statistics/
    ├── confusion_matrix.json
    ├── confusion_matrix.png
    ├── classification_metrics.json
    ├── dataset_statistics.json
    ├── training_summary.json
    └── comprehensive_report.json
```

### 2. Statistiche Rapide e Configurabili

```python
from evaluation.quick_stats import quick_model_evaluation, create_evaluation_config

# Usa template predefinito
config = create_evaluation_config("comprehensive")

# Genera statistiche
performance_summary = quick_model_evaluation(
    y_true=y_true, 
    y_pred=y_pred, 
    class_names=class_names,
    output_path="output/stats",
    config=config
)
```

### 3. Statistiche Complete Personalizzate

```python
from evaluation.stats import generate_comprehensive_report

# Genera report completo personalizzato
report_data = generate_comprehensive_report(
    X=X, y=y_true, y_pred=y_pred, 
    class_names=class_names,
    training_log=training_log,
    best_model_path=best_model_path,
    output_path="output/comprehensive",
    config=custom_config
)
```

## ⚙️ Configurazione

### Template Disponibili

#### Default
```python
{
    "save_confusion_matrix": True,
    "save_classification_report": True,
    "save_performance_summary": True,
    "print_to_console": True,
    "include_per_class_stats": True
}
```

#### Minimal
```python
{
    "save_confusion_matrix": False,
    "save_classification_report": False,
    "save_performance_summary": True,
    "print_to_console": True,
    "include_per_class_stats": False
}
```

#### Comprehensive
```python
{
    "save_confusion_matrix": True,
    "save_classification_report": True,
    "save_performance_summary": True,
    "print_to_console": True,
    "include_per_class_stats": True,
    "save_plots": True,
    "custom_metrics": ["balanced_accuracy", "cohen_kappa", "matthews_corrcoef"]
}
```

### Configurazione Personalizzata

```python
# Crea configurazione personalizzata
custom_config = {
    "save_confusion_matrix": True,
    "save_confusion_matrix_plot": True,
    "plot_dpi": 300,
    "plot_format": "png",
    "calculate_per_class_metrics": True,
    "include_feature_statistics": True,
    "custom_metrics": ["balanced_accuracy"]
}

# Applica configurazione
from config.stats_config import STATS_CONFIG
STATS_CONFIG.update(custom_config)
```

## 📊 Output Generati

### 1. Matrice di Confusione
- **File JSON**: Dati numerici per analisi programmatiche
- **Plot PNG**: Visualizzazione grafica per presentazioni

### 2. Metriche di Classificazione
- **Metriche Overall**: Accuratezza, Precision/Recall/F1 macro e weighted
- **Metriche per Classe**: Performance dettagliata per ogni classe
- **Support**: Numero di campioni per classe

### 3. Statistiche del Dataset
- **Distribuzione Classi**: Conteggio e percentuali
- **Statistiche Features**: Mean, std, min, max, median per ogni feature
- **Overview**: Numero totale di campioni e features

### 4. Riepilogo Training
- **Performance**: Migliore, peggiore, media e deviazione standard
- **Configurazione Ottimale**: Parametri del modello migliore
- **Log Completo**: Tutte le configurazioni testate

## 🔧 Personalizzazione Avanzata

### Metriche Personalizzate

```python
custom_metrics_config = {
    "balanced_accuracy": {
        "enabled": True,
        "description": "Accuratezza bilanciata per dataset sbilanciati"
    },
    "cohen_kappa": {
        "enabled": True,
        "description": "Coefficiente Kappa di Cohen"
    }
}

from evaluation.quick_stats import generate_custom_metrics
custom_results = generate_custom_metrics(
    y_true, y_pred, custom_metrics_config, output_path
)
```

### Stile dei Plot

```python
# Modifica configurazione plot
STATS_CONFIG["plot_style"] = "seaborn"  # "seaborn", "matplotlib", "plotly"
STATS_CONFIG["color_scheme"] = "Reds"   # Schema colori per heatmap
STATS_CONFIG["plot_dpi"] = 600          # Risoluzione plot
```

## 📁 Struttura Directory Output

```
output/
├── models/
│   ├── best_model.keras
│   └── training_log.json
├── statistics/
│   ├── confusion_matrix.json
│   ├── confusion_matrix.png
│   ├── classification_metrics.json
│   ├── dataset_statistics.json
│   ├── training_summary.json
│   └── comprehensive_report.json
└── plots/
    └── confusion_matrix.png
```

## 🎨 Esempi di Utilizzo

### Esempio 1: Training con Statistiche Complete
```python
# Nel main.py, le statistiche vengono generate automaticamente
python main.py
```

### Esempio 2: Statistiche Rapide
```python
python example_stats_usage.py
```

### Esempio 3: Configurazione Personalizzata
```python
from evaluation.quick_stats import create_evaluation_config

# Crea configurazione personalizzata
config = create_evaluation_config("comprehensive")
config["save_plots"] = True
config["custom_metrics"] = ["balanced_accuracy"]

# Applica configurazione
quick_model_evaluation(y_true, y_pred, class_names, "output", config)
```

## 🚨 Risoluzione Problemi

### Errore: "ModuleNotFoundError: No module named 'evaluation'"
```bash
# Assicurati di essere nella directory root del progetto
cd /path/to/snn-ids
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Errore: "matplotlib backend"
```python
# Aggiungi questo all'inizio del tuo script
import matplotlib
matplotlib.use('Agg')  # Per server senza display
```

### Errore: "Permission denied" per salvare file
```bash
# Crea directory con permessi appropriati
mkdir -p output/statistics
chmod 755 output/statistics
```

## 📈 Metriche Disponibili

### Metriche Standard
- **Accuracy**: Accuratezza complessiva
- **Precision**: Precisione per classe
- **Recall**: Sensibilità per classe
- **F1-Score**: Media armonica di precision e recall
- **Support**: Numero di campioni per classe

### Metriche Avanzate
- **Balanced Accuracy**: Accuratezza bilanciata per dataset sbilanciati
- **Cohen's Kappa**: Accordi tra predizioni e verità
- **Matthews Correlation**: Correlazione per classificazione binaria

## 🔄 Integrazione con Pipeline Esistente

Il sistema è completamente integrato con:
- ✅ **Preprocessing**: Anonimizzazione Crypto-PAn
- ✅ **Training**: Generazione automatica statistiche
- ✅ **Prediction**: Valutazione performance
- ✅ **Configurazione**: Sistema di configurazione centralizzato

## 📝 Note per la Produzione

1. **Sicurezza**: Le chiavi Crypto-PAn devono essere gestite in modo sicuro
2. **Performance**: Per dataset molto grandi, considera l'uso di campionamento
3. **Storage**: Le statistiche possono occupare spazio significativo
4. **Backup**: Mantieni backup delle mappe di anonimizzazione

## 🤝 Contributi

Per aggiungere nuove metriche o funzionalità:
1. Modifica `evaluation/stats.py` per metriche standard
2. Modifica `evaluation/quick_stats.py` per metriche personalizzate
3. Aggiorna `config/stats_config.py` per nuove opzioni
4. Aggiungi test in `example_stats_usage.py`
