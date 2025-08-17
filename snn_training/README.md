# Modulo 2: SNN Training

## Descrizione
Modulo per training SNN con framework multipli.

## Funzionalità
- ✅ Training SNN con diversi framework (Nengo, mock)
- ✅ Configurazione architetture flessibili
- ✅ Algoritmi di training configurabili
- ✅ Export modelli per analisi
- ✅ Funzionamento su dati anonimi

## Framework Supportati
- **Nengo** - Framework completo per SNN
- **Mock** - Framework simulato per test
- **Futuri**: Norse, SNNTorch, BindsNET

## File Principali
- `snn_trainer.py` - Trainer principale
- `model_manager.py` - Gestione modelli (TODO)
- `training_config.py` - Configurazioni (TODO)

## Utilizzo

### Uso Standalone
```python
from snn_training import SNNTrainer, TrainingConfig

# Configurazione training
config = TrainingConfig(
    framework="nengo",  # o "mock" per test
    hidden_layers=[64, 32, 16],
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    model_name="security_snn"
)

trainer = SNNTrainer(config)

# Training da dataset CSV anonimo
results = trainer.train("data/snn_dataset.csv")

# Salva modello
model_path = trainer.save_model("models/")

# Export per analisi
export_path = trainer.export_for_analysis("model_export.json")
```

### Via Orchestratore
```python
from pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
results = orchestrator.run_training_only("snn_dataset.csv")
```

## Configurazione Training

```python
TrainingConfig(
    framework="nengo",           # Framework SNN
    input_neurons=10,            # Auto-rilevato da dataset
    hidden_layers=[64, 32],      # Architettura hidden
    output_neurons=2,            # Classi output
    neuron_type="lif",           # LIF, adaptive, izhikevich
    learning_rate=0.001,         # Learning rate
    epochs=100,                  # Numero epoche
    batch_size=32,               # Batch size
    dt=0.001,                    # Time step SNN
    simulation_time=1.0,         # Durata simulazione
    encoding_method="rate"       # rate, temporal, population
)
```

## Output
- `model.pkl` - Modello trained (framework-specific)
- `model_config.json` - Configurazione modello
- `model_history.json` - Storia training
- `model_export.json` - Export per modulo analisi

## Sicurezza
- ✅ Lavora solo su dati già anonimizzati
- ✅ Non accede a mapping originali
- ✅ Export sicuro per analisi
- ✅ Nessun dato sensibile nel modello
