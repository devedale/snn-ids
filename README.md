# Pipeline ML Avanzata e Modulare per Cybersecurity

Questo progetto implementa una pipeline di machine learning end-to-end per l'analisi di dati di rete, con un focus su **modularità, configurabilità e tecniche avanzate** per l'analisi di dati sequenziali.

È una evoluzione di una pipeline di classificazione base, ora arricchita con funzionalità per gestire dati come serie temporali e per una validazione del modello più robusta.

## Funzionalità Principali

- **Architettura Modulare**: Il codice è suddiviso in moduli specializzati (`preprocessing`, `training`, `prediction`) per la massima chiarezza e manutenibilità.
- **Configurazione Centrale (`config.py`)**: Un singolo file per controllare ogni aspetto della pipeline, dalla selezione del dataset al tuning degli iperparametri.
- **Gestione di Dati Sequenziali**:
  - **Finestre Temporali**: La pipeline può trasformare dati di log o di rete in sequenze (finestre temporali) di dimensione e sovrapposizione configurabili.
  - **Modelli Ricorrenti (LSTM)**: Supporto nativo per modelli LSTM, ideali per imparare pattern da dati sequenziali.
- **Strategie di Validazione Flessibili**:
  - **K-Fold Cross-Validation**: Utilizza la validazione incrociata per una stima più affidabile delle performance del modello.
  - **Train/Test Split**: Mantiene l'opzione di una divisione singola per test più rapidi.
- **Notebook Didattico**: `colab_notebook.ipynb` funge da guida interattiva, spiegando non solo *come* eseguire la pipeline, ma anche *perché* certe scelte tecniche sono state fatte.

## Struttura del Progetto

```
.
├── config.py                   # Pannello di controllo della pipeline
├── create_synthetic_data.py    # Script per generare dati con timestamp
├── colab_notebook.ipynb        # Notebook-guida avanzato
├── data/
│   └── cybersecurity_data.csv  # Dataset sintetico
├── preprocessing/
│   └── process.py              # Modulo di preprocessing (con logica per finestre temporali)
├── training/
│   └── train.py                # Modulo di training (con K-Fold e scelta modello)
├── prediction/
│   └── predict.py              # Modulo di predizione (per dati sequenziali)
└── README.md
```

## Come Eseguire il Progetto

Il modo più semplice e completo per usare questo progetto è attraverso il notebook per Google Colab.

1.  Carica l'intera cartella del progetto nel tuo Google Drive.
2.  Apri `colab_notebook.ipynb` in Google Colab.
3.  Esegui le celle in ordine. Il notebook ti guiderà attraverso:
    - L'installazione delle dipendenze.
    - L'analisi delle opzioni di configurazione.
    - La generazione dei dati.
    - Il preprocessing in finestre temporali.
    - Il training con K-Fold e un modello LSTM.
    - La predizione su una nuova finestra di dati.

## Personalizzazione Avanzata

La vera potenza di questa pipeline risiede in `config.py`. Per adattarla alle tue esigenze:

1.  **Usa il Tuo Dataset**: Cambia `dataset_path` e aggiorna le liste di colonne in `DATA_CONFIG`.
2.  **Modifica le Finestre Temporali**: In `PREPROCESSING_CONFIG`, regola `window_size` e `step` per cambiare come vengono create le sequenze.
3.  **Cambia Strategia di Validazione**: In `TRAINING_CONFIG`, imposta `validation_strategy` a `'k_fold'` o `'train_test_split'`.
4.  **Sperimenta con i Modelli**: Cambia `model_type` in `'lstm'` o `'dense'`. Puoi facilmente estendere `training/train.py` per aggiungere nuove architetture (es. GRU, Autoencoder).
5.  **Tuning degli Iperparametri**: Modifica le liste in `hyperparameters` per testare diverse configurazioni con la grid search.
