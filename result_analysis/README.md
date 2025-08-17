# Modulo 3: Result Analysis

## Descrizione
Modulo per analisi risultati e ricostruzione log originali.

## Funzionalità
- ✅ Analisi risultati predizioni SNN
- ✅ Rilevamento anomalie con soglie configurabili
- ✅ Analisi pattern temporali
- ✅ Ricostruzione log da dati anonimizzati
- ✅ Export report completi

## File Principali
- `result_analyzer.py` - Analizzatore principale
- `log_reconstructor.py` - Embedded nel result_analyzer
- `evaluation_metrics.py` - Metriche (TODO)

## Utilizzo

### Uso Standalone
```python
from result_analysis import ResultAnalyzer, AnalysisConfig

# Configurazione analisi
config = AnalysisConfig(
    enable_reconstruction=True,
    confidence_threshold=0.7,
    anomaly_threshold=0.8,
    include_temporal_analysis=True
)

analyzer = ResultAnalyzer(config)

# Setup ricostruzione (solo locale)
analyzer.setup_reconstruction(
    "mappings.json",
    "normalization_stats.json"
)

# Analisi predizioni
predictions = np.load("predictions.npy")
results = analyzer.analyze_predictions(predictions)

# Ricostruzione log anomali
anomaly_summary = analyzer.get_anomaly_summary()
reconstructed = analyzer.reconstruct_highlighted_logs(
    anomaly_summary['anomaly_indices'],
    "original_dataset.csv",
    feature_names
)

# Export report
report_path = analyzer.export_analysis_report(
    "analysis_report.json",
    reconstructed
)
```

### Via Orchestratore
```python
from pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
results = orchestrator.run_analysis_only(
    "model_export.json",
    "predictions.json",
    "mappings.json",
    "normalization.json"
)
```

## Ricostruzione Log

### Come Funziona
1. **Denormalizzazione** - Riporta valori da [0,1] a range originale
2. **Reverse Mapping** - Usa mapping salvati per ricostruire valori
3. **Mapping Types**:
   - One-hot → Categoria originale
   - Hash → Valore più probabile
   - Ordinal → Valore da ordine
   - IP → Decifratura Crypto-PAn

### Esempio Ricostruzione
```python
# Valore anonimizzato
anonymized = {
    "feat_srcip": 0.123456,
    "feat_action_cat0": 1.0,  # one-hot
    "feat_action_cat1": 0.0,
    "feat_srcport": 0.456789
}

# Ricostruito
reconstructed = {
    "srcip": "192.168.1.100",    # Crypto-PAn reversed
    "action": "deny",            # One-hot reversed
    "srcport": "443"             # Denormalizzato
}
```

## Analisi Temporale

### Metriche Incluse
- **Trend Confidence** - Increasing/decreasing/stable
- **Anomaly Clusters** - Raggruppamenti temporali anomalie
- **Distribution Stats** - Mean, std, percentili confidence

### Pattern Detection
- Cluster anomalie consecutive
- Trend temporali
- Picchi di attività

## Output
- `analysis_report.json` - Report completo analisi
- `reconstructed_logs.csv` - Log anomali ricostruiti
- Metriche performance e confidence

## Sicurezza
- ✅ Ricostruzione solo in ambiente locale sicuro
- ✅ Mapping sensibili mai esportati
- ✅ Log ricostruiti solo per anomalie rilevate
- ✅ Controllo accesso ai file di mapping
