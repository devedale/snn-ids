# Pipeline di Machine Learning Modulare per Cybersecurity

Questo progetto implementa una pipeline di machine learning end-to-end per un task di classificazione di traffico di rete. È stato progettato per essere **modulare**, **configurabile** e **facilmente estensibile**.

L'intera logica è suddivisa in moduli specializzati, seguendo le pratiche di Single Responsibility. Un notebook per Google Colab (`colab_notebook.ipynb`) orchestra l'intero flusso di lavoro in modo chiaro e didattico.

## Struttura del Progetto

```
.
├── config.py                   # File di configurazione centrale
├── create_synthetic_data.py    # Script per generare dati fittizi
├── colab_notebook.ipynb        # Notebook per l'esecuzione guidata
├── data/
│   └── cybersecurity_data.csv  # Dataset (generato dallo script)
├── preprocessing/
│   ├── __init__.py
│   └── process.py              # Modulo di preprocessing
├── training/
│   ├── __init__.py
│   └── train.py                # Modulo di training (con Grid Search)
├── prediction/
│   ├── __init__.py
│   └── predict.py              # Modulo di predizione
└── models/
    ├── best_model.keras        # Miglior modello salvato
    ├── column_order.json       # Ordine delle feature per la predizione
    ├── ip_anonymization_map.json # Mappa di anonimizzazione per gli IP
    └── target_anonymization_map.json # Mappa di anonimizzazione per il target
```

## Come Eseguire il Progetto

### Opzione 1: Usare il Notebook per Google Colab (Consigliato)

1.  Carica l'intera cartella del progetto nel tuo Google Drive o clona il repository.
2.  Apri `colab_notebook.ipynb` in Google Colab.
3.  Esegui le celle in ordine. Il notebook si occuperà di:
    - Installare le dipendenze.
    - Generare i dati.
    - Eseguire il preprocessing.
    - Lanciare il training.
    - Fare una predizione di esempio.

### Opzione 2: Esecuzione Manuale dei Moduli

1.  **Installa le dipendenze:**
    ```bash
    pip install pandas scikit-learn tensorflow faker
    ```

2.  **Genera il dataset:**
    ```bash
    python3 create_synthetic_data.py
    ```

3.  **Esegui il training:** Questo passo eseguirà anche il preprocessing.
    ```bash
    python3 training/train.py
    ```

4.  **Esegui una predizione:** Lo script di predizione contiene un esempio hard-coded.
    ```bash
    python3 prediction/predict.py
    ```

## Configurazione e Personalizzazione

Il cuore della flessibilità del progetto è il file `config.py`. Da qui puoi controllare:
-   Il **percorso del dataset**.
-   Le **colonne** da usare come feature e come target.
-   Le colonne da sottoporre a **one-hot encoding** o a **trattamenti specifici** (es. anonimizzazione IP).
-   Gli **iperparametri** per la grid search del training (es. `learning_rate`, `batch_size`, `epochs`).

Per usare un nuovo dataset, ti basterà:
1.  Posizionare il nuovo file CSV (es. in `data/`).
2.  Aggiornare il `dataset_path` e le liste delle colonne in `DATA_CONFIG` dentro `config.py`.
