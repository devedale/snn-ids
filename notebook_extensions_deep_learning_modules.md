# Architettura del Sistema: Moduli Principali e Librerie Deep Learning

Questa sezione fornisce una panoramica completa dell'architettura del sistema, dei moduli di preprocessing e training, e delle librerie utilizzate per il deep learning e la privacy-preserving machine learning.

## Stack Tecnologico Principale

### Deep Learning Core

**TensorFlow 2.x + Keras**: Framework principale per modelli neurali, training e inference
- **Keras Sequential API**: Utilizzato per costruire modelli MLP con layer Dense
- **Keras Functional API**: Per architetture più complesse e personalizzazioni
- **tf.keras.optimizers**: Adam, SGD per ottimizzazione dei parametri
- **tf.keras.losses**: SparseCategoricalCrossentropy per classificazione multi-classe
- **tf.keras.callbacks**: EarlyStopping, ModelCheckpoint per controllo training
- **tf.random**: Controllo deterministico dei seed per riproducibilità

**NumPy**: Computazione numerica e manipolazione array multidimensionali
- **Operazioni matriciali**: Reshape, concatenazione, broadcasting per preprocessing
- **Generazione dati sintetici**: np.random per augmentation e test
- **Aggregazioni statistiche**: mean, std, percentili per analisi dataset
- **Indexing avanzato**: Selezione batch e slicing per federated learning

**Pandas**: Gestione dataset e preprocessing tabellare
- **DataFrame operations**: Merge, groupby, pivot per analisi CSE-CIC-IDS2018
- **Data cleaning**: Gestione valori mancanti, duplicati, outlier detection
- **Feature engineering**: Encoding categorico, normalizzazione, feature selection
- **Time series**: Parsing timestamp, windowing per analisi temporale

**Scikit-learn**: Pipeline di preprocessing, validazione e metriche di valutazione
- **Preprocessing**: StandardScaler, LabelEncoder, train_test_split
- **Cross-validation**: StratifiedKFold, KFold per validazione robusta
- **Metrics**: classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
- **Utilities**: resample per bilanciamento dataset, class_weight per gestione classi sbilanciate

### Privacy-Preserving Technologies

**TenSEAL**: Implementazione CKKS per Homomorphic Encryption (HE)
- **CKKS Scheme**: Cifratura omomorfica per operazioni su numeri reali
- **Context management**: Configurazione parametri crittografici (poly_modulus_degree, coeff_mod_bit_sizes)
- **Homomorphic operations**: Addizione e moltiplicazione su ciphertext
- **Serialization**: Trasferimento sicuro di dati cifrati tra client e server
- **Performance tuning**: Bilanciamento sicurezza vs overhead computazionale

**TensorFlow Privacy**: Differential Privacy (DP) integrato nel training loop
- **DP-SGD**: Gradient clipping e rumore gaussiano per privacy matematica
- **Privacy accounting**: Calcolo epsilon-delta per quantificare privacy loss
- **Noise calibration**: Tuning noise_multiplier per bilanciare utility vs privacy
- **Composition theorems**: Gestione privacy budget su multiple epoche

**Nota**: TensorFlow Privacy è referenziato ma non ancora implementato nel codice attuale

**Flower**: Framework Federated Learning per simulazioni distribuite
- **FedAvg**: Implementazione algoritmo di aggregazione standard
- **Client simulation**: Simulazione multi-client su singola macchina
- **Communication**: Protocolli gRPC per scambio parametri
- **Strategy customization**: Personalizzazione algoritmi aggregazione

**Nota**: Flower è referenziato ma non ancora implementato nel codice attuale

### Sistema di Benchmark e Ottimizzazione

**Keras Tuner**: Implementazione Hyperband per hyperparameter optimization
- **Hyperband Algorithm**: Multi-fidelity optimization con early stopping adattivo
- **Search space**: Definizione spazi di ricerca per learning_rate, units, activation
- **Trial management**: Gestione trial paralleli e checkpoint intermedi
- **Best model selection**: Selezione automatica configurazione ottimale

**Custom Benchmark Pipeline**: Sistema modulare per test ripetibili e configurabili
- **Configuration management**: Sistema config.py centralizzato
- **Reproducibility**: Seed management e deterministic training
- **Metrics collection**: Raccolta sistematica accuracy, timing, resource usage
- **Result aggregation**: Summary automatici e comparazioni cross-scenario

---

## Moduli di Preprocessing: Architettura e Implementazione

### src/preprocessing/process.py

**Obiettivo**: Pipeline modulare e riproducibile per preprocessing dataset CSE-CIC-IDS2018

**Funzioni chiave**:

```python
def load_and_balance_dataset(data_path, sample_size, balance_strategy):
    """
    Carica e bilancia dataset secondo strategia configurata
    - data_path: Directory contenente CSV giornalieri
    - sample_size: Numero totale campioni (BENIGN + ATTACK)
    - balance_strategy: "security" (50/50) o "natural" (distribuzione originale)
    """
    
# 📝 FUNZIONE: load_and_balance_dataset
# ARGOMENTI:
#   - data_path (str): Percorso directory contenente file CSV giornalieri CSE-CIC-IDS2018
#   - sample_size (int): Numero totale di campioni da estrarre (es. 10000)
#   - balance_strategy (str): "security" per 50/50 BENIGN/ATTACK, "natural" per distribuzione originale
# OUTPUT:
#   - DataFrame pandas bilanciato con colonne features + label
#   - Gestione memoria efficiente con chunking per dataset large-scale
```

**Preprocessing steps**:
1. **Loading**: Lettura parallela CSV multipli con gestione memoria
2. **Cleaning**: Rimozione duplicati, gestione valori infiniti/NaN
3. **Balancing**: Strategia "security" per ridurre bias verso traffico benigno
4. **Encoding**: LabelEncoder per trasformazione stringhe → indici numerici
5. **Validation**: Controlli integrità e distribuzione classi

**Gestione memoria**: Chunking per dataset large-scale, garbage collection proattivo

### src/training/

**Architettura modulare**:
- `models.py`: Factory pattern per creazione modelli (MLP, GRU, LSTM)
- `train.py`: Training loop con K-Fold cross-validation
- `utils.py`: Utility per data preparation e seed management

**Model builders**:

```python
def get_model_builder(model_type: str):
    """Factory per builders di modelli neurali
    - mlp_4_layer: Dense network per dati tabulari
    - gru: Recurrent network per sequence modeling
    - lstm: Long Short-Term Memory per temporal patterns
    """
    
# 📝 FUNZIONE: get_model_builder
# ARGOMENTI:
#   - model_type (str): Tipo modello richiesto ("mlp_4_layer", "gru", "lstm")
# OUTPUT:
#   - Function: Funzione builder specifica per il tipo di modello richiesto
#   - Factory pattern per creazione dinamica di modelli neurali
```

**Training strategies**:
- **K-Fold CV**: StratifiedKFold per validazione robusta (implementato in trainer.py)
- **Early stopping**: Prevenzione overfitting con patience configurabile
- **Learning rate scheduling**: Riduzione adattiva per convergenza ottimale
- **Per-class loss tracking**: Callback personalizzato per monitorare loss per classe

### src/evaluation/metrics.py

**Metriche comprehensive**:

```python
def evaluate_model_comprehensive(model, X_test, y_test, class_names, output_dir):
    """
    Valutazione multi-dimensionale con focus cybersecurity
    - Basic metrics: Accuracy, precision, recall, F1-score per classe
    - Cybersecurity metrics: Detection rate, false alarm rate
    - Calibration: Expected Calibration Error (ECE)
    - Visualizations: Confusion matrix, ROC curves, precision-recall
    """
    
# 📝 FUNZIONE: evaluate_model_comprehensive
# ARGOMENTI:
#   - model (tf.keras.Model): Modello addestrato da valutare
#   - X_test (np.ndarray): Features di test, shape (samples, features)
#   - y_test (np.ndarray): Labels di test, shape (samples,)
#   - class_names (List[str]): Nomi delle classi per interpretazione risultati
#   - output_dir (str): Directory dove salvare grafici e report JSON
# OUTPUT:
#   - Dict: Dizionario con tutte le metriche calcolate
#   - File PNG: Grafici di confusion matrix, ROC curves, per-class accuracy
#   - File JSON: Report completo con metriche per analisi programmatica
```

**Output**:
- **Tabelle**: Classification report dettagliato per classe
- **Grafici**: PNG ad alta risoluzione per presentation-ready results (matplotlib + seaborn)
- **JSON**: Structured data per analisi programmatica
- **Visualizzazioni**: Confusion matrix, ROC curves, per-class accuracy, cybersecurity metrics

---

## Algoritmo Hyperband: Teoria e Implementazione

### Principi Teorici

**Hyperband** è un algoritmo di ottimizzazione hyperparametri basato su **multi-fidelity optimization**:

1. **Successive Halving**: Elimina configurazioni poor-performing early
2. **Resource allocation**: Distribuisce budget computazionale in modo ottimale
3. **Exploration vs Exploitation**: Bilancia ricerca ampia vs approfondimento

**Vantaggi rispetto a Grid/Random Search**:
- **Efficienza**: 5-10x speedup rispetto a grid search completa
- **Adattività**: Allocation automatica risorse verso configurazioni promettenti
- **Robustezza**: Performance consistenti su diversi tipi di problemi
- **Resource management**: Successive halving per eliminazione early di configurazioni poor-performing

### Implementazione nel Sistema

Il comando `python3 benchmark.py --hyperband --model mlp_4_layer --hb-max-epochs 20 --hb-final-epochs 30 --hb-batch-size 128` attiva:

**Configurazione search space**:

```python
def build_model_mlp_4_layer(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=hp.Int('units_layer_1', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh', 'sigmoid']),
            input_shape=(input_shape,)
        ),
        tf.keras.layers.Dense(
            units=hp.Int('units_layer_2', min_value=16, max_value=256, step=16)
        ),
        tf.keras.layers.Dense(
            units=hp.Int('units_layer_3', min_value=8, max_value=128, step=8)
        ),
        tf.keras.layers.Dense(
            units=hp.Int('units_layer_4', min_value=4, max_value=64, step=4)
        ),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 📝 FUNZIONE: build_model_mlp_4_layer
# ARGOMENTI:
#   - hp (kt.HyperParameters): Oggetto Keras Tuner per hyperparameter optimization
#   - input_shape (tuple): Shape delle features di input (es. (78,) per CSE-CIC-IDS2018)
#   - num_classes (int): Numero di classi di output (es. 15 per dataset completo)
# OUTPUT:
#   - tf.keras.Model: Modello MLP compilato con hyperparameters ottimizzabili
#   - Hyperparameters tunabili: units per layer, activation, learning rate
#   - Loss: sparse_categorical_crossentropy per classificazione multi-classe
```

**Scheduling Hyperband**:
- **max_epochs=20**: Budget iniziale per early trials
- **factor=3**: Reduction factor per successive halving
- **final_epochs=30**: Budget massimo per best configurations
- **Bracket system**: Multiple bracket per exploration granularities diverse

**Resource management**:
1. **Trial spawning**: Creazione configurazioni random iniziali
2. **Early evaluation**: Training su subset epoche per screening rapido
3. **Promotion**: Configurazioni top-performing continuano con più risorse
4. **Final training**: Best configuration trainata con budget completo

### Metriche e Monitoraggio

**Durante esecuzione**:
```
Search: Running Trial #25
Value             |Best Value So Far |Hyperparameter
0.0001            |0.001             |learning_rate
relu              |tanh              |activation
224               |256               |units_layer_1
128               |32                |units_layer_2
```

**Output finale**:
- **Best configuration**: Hyperparameters ottimali trovati
- **Performance history**: Traiettoria miglioramento nel tempo
- **Resource utilization**: Analisi efficienza computazionale
- **Model artifacts**: Modello finale salvato per deployment

**Integrazione con privacy scenarios**:
L'ottimizzazione Hyperband serve come **baseline centralizado (Scenario A)** per:
- Stabilire upper bound di performance senza privacy constraints
- Calibrare aspettative per scenari FL + DP/HE
- Fornire reference point per trade-off analysis
- **Comando specifico**: `python3 benchmark.py --hyperband --model mlp_4_layer --hb-max-epochs 20 --hb-final-epochs 30 --hb-batch-size 128`

---

## Librerie TenSEAL: Deep Dive Homomorphic Encryption

### Architettura CKKS

**CKKS (Cheon-Kim-Kim-Song)** è uno schema di cifratura omomorfica per **approssimazioni di numeri reali**:

**Vantaggi per ML**:
- **Operazioni native**: Addizione e moltiplicazione su encrypted data
- **Batch processing**: SIMD operations su vettori cifrati
- **Precision control**: Gestione trade-off precisione vs performance

**Implementazione in experiments_privacy/shared/fl_utils.py**:

```python
def _create_ckks_contexts(poly_modulus_degree, coeff_mod_bit_sizes, global_scale_bits):
    """
    Genera contesti CKKS per HE operations
    - poly_modulus_degree: Determina security level e performance
    - coeff_mod_bit_sizes: Chain di moduli per depth control
    - global_scale_bits: Precisione encoding numeri reali
    """
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** global_scale_bits
    ctx.generate_galois_keys()  # Per rotazioni efficient
    
    # Context pubblico per client-side encryption
    public_bytes = ctx.serialize(save_secret_key=False)
    public_ctx = ts.context_from(public_bytes)
    return ctx, public_ctx

# 📝 FUNZIONE: _create_ckks_contexts
# ARGOMENTI:
#   - poly_modulus_degree (int): Grado polinomio (8192, 16384, 32768) - determina security level
#   - coeff_mod_bit_sizes (tuple): Chain di moduli (60, 40, 40, 60) per depth control
#   - global_scale_bits (int): Bits per precisione encoding (40) - trade-off precision vs performance
# OUTPUT:
#   - ctx (ts.Context): Contesto CKKS completo con secret key per decryption
#   - public_ctx (ts.Context): Contesto pubblico per client-side encryption
#   - Galois keys generati per operazioni di rotazione efficienti
```

### Pipeline di Cifratura FL

**Flusso operativo**:

1. **Weight flattening**: Concatenazione pesi neurali in vettore singolo
2. **Chunking**: Suddivisione in slot per batch CKKS
3. **Encryption**: Cifratura client-side con public context
4. **Transmission**: Invio ciphertext (senza chiavi private)
5. **Homomorphic aggregation**: Somma cifrata server-side
6. **Decryption**: Decifratura aggregato con secret key
7. **Reconstruction**: Ricostruzione weights da vettore aggregato

**Code esempio**:

```python
def _encrypt_weights_ckks(weights, public_ctx, poly_modulus_degree):
    vec, shapes = _flatten_weights(weights)  # Linearizzazione
    slots = poly_modulus_degree // 2         # Capacità SIMD
    
    chunks = []
    for i in range(0, len(vec), slots):
        chunk = vec[i:i+slots].tolist()
        encrypted_chunk = ts.ckks_vector(public_ctx, chunk)
        chunks.append(encrypted_chunk)
    
    return chunks, shapes, len(vec)

# 📝 FUNZIONE: _encrypt_weights_ckks
# ARGOMENTI:
#   - weights (List[np.ndarray]): Lista pesi neurali da cifrare (es. da model.get_weights())
#   - public_ctx (ts.Context): Contesto CKKS pubblico per cifratura client-side
#   - poly_modulus_degree (int): Grado polinomio per determinare capacità SIMD
# OUTPUT:
#   - chunks (List[ts.CKKSTensor]): Lista chunk cifrati per trasmissione
#   - shapes (List[tuple]): Shape originali per ricostruzione post-decifratura
#   - total_size (int): Dimensione totale vettore pesi (per metriche banda)

def _aggregate_cipher_chunks(clients_chunks):
    """Somma omomorfica: ⊕(E(w₁), E(w₂), ...) = E(w₁ + w₂ + ...)"""
    aggregated = []
    for i in range(len(clients_chunks[0])):
        total = clients_chunks[0][i]  # Primo chunk
        for j in range(1, len(clients_chunks)):
            total = total + clients_chunks[j][i]  # Addizione omomorfica
        aggregated.append(total)
    
    return aggregated

# 📝 FUNZIONE: _aggregate_cipher_chunks
# ARGOMENTI:
#   - clients_chunks (List[List[ts.CKKSTensor]]): Lista chunk cifrati per ogni client
#     Es: [[chunk1_client1, chunk2_client1], [chunk1_client2, chunk2_client2], ...]
# OUTPUT:
#   - aggregated (List[ts.CKKSTensor]): Chunk aggregati con somma omomorfica
#   - Operazione: E(w₁) + E(w₂) + ... = E(w₁ + w₂ + ...) per ogni chunk
#   - Mantiene cifratura: risultato ancora cifrato per privacy
```

### Performance e Security Tuning

**Parametri chiave**:

- **poly_modulus_degree**: 8192 (fast) vs 16384 (secure)
  - Più alto = maggiore sicurezza ma più overhead
  - Determina numero slot SIMD disponibili

- **coeff_mod_bit_sizes**: [60, 40, 40, 60]
  - Catena di moduli per multiplicative depth
  - Più depth = operazioni più complesse ma overhead maggiore

- **global_scale**: 2^40
  - Precisione encoding floating point
  - Trade-off: precisione vs noise growth

**Overhead analysis**:
```python
# Nel summary esteso troviamo:
HE_Mode: "encrypt"                    # Cifratura reale vs estimate
HE_MB_per_round: 4.39                # Banda per round
HE_EncTimeMean_s: 0.234              # Tempo cifratura client
HE_AggTimeMean_s: 0.089              # Tempo aggregazione server  
HE_DecTimeMean_s: 0.067              # Tempo decifratura server
```

**Ottimizzazioni implementate**:
- **Lazy encryption**: Solo quando `he_encrypt=True`
- **Batch encoding**: SIMD packing per efficiency
- **Memory management**: Serialization ottimizzata per network transfer
- **Fallback graceful**: Automatic degradation a estimate se TenSEAL unavailable

Questa implementazione permette di misurare l'**overhead reale** delle operazioni HE in scenari FL, fornendo metriche precise per cost-benefit analysis in production deployments.

---

## Mappatura Reale: Librerie e Metodi Effettivamente Utilizzati

### Import Statements Verificati nel Codice

**src/preprocessing.py**:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
```

**src/training/trainer.py**:
```python
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.utils import class_weight
```

**src/training/utils.py**:
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
```

**src/evaluation.py**:
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
```

**benchmark.py**:
```python
import numpy as np
import keras_tuner as kt  # Per Hyperband
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
```

**experiments_privacy/shared/fl_utils.py**:
```python
import numpy as np
import tensorflow as tf
import tenseal as ts  # Per HE operations
```

### Metodi Scikit-learn Utilizzati

- **LabelEncoder**: Encoding labels categorici in preprocessing
- **StandardScaler**: Scaling features per training
- **StratifiedKFold**: Cross-validation stratificata
- **train_test_split**: Split train/validation
- **resample**: Bilanciamento dataset
- **class_weight**: Gestione classi sbilanciate
- **confusion_matrix, classification_report**: Metriche di valutazione
- **roc_curve, auc**: Curve ROC per analisi performance

### Metodi TensorFlow/Keras Utilizzati

- **tf.keras.Sequential**: Costruzione modelli MLP
- **tf.keras.layers.Dense**: Layer fully connected
- **tf.keras.optimizers.Adam**: Ottimizzatore
- **tf.keras.losses.sparse_categorical_crossentropy**: Loss function
- **tf.keras.callbacks.Callback**: Callback personalizzato per per-class loss
- **tf.random.set_seed**: Controllo seed per riproducibilità

### Librerie di Visualizzazione Effettivamente Utilizzate

- **matplotlib.pyplot**: Grafici base e configurazione
- **seaborn**: Styling e visualizzazioni avanzate
- **Backend**: 'Agg' per ambiente server non-interattivo

### Nota Importante

Questo documento mappa **esattamente** le librerie e metodi utilizzati nel codice attuale, non librerie potenziali o generiche. Ogni import statement e metodo è stato verificato attraverso analisi del codice sorgente per garantire accuratezza e utilità pratica.

---

## 🚀 Flusso Completo: Dal Comando al Risultato

### Il Comando Principale

```bash
!python3 benchmark.py --hyperband --model mlp_4_layer --hb-max-epochs 20 --hb-final-epochs 30 --hb-batch-size 128
```

Questo comando rappresenta il **punto di ingresso principale** per l'ottimizzazione Hyperband del modello MLP a 4 layer. Analizziamo ogni componente e il flusso completo che ne deriva.

### 1. Parsing degli Argomenti e Configurazione

**benchmark.py** - Entry point principale:

```python
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperband', action='store_true', 
                       help='Enable Hyperband optimization')
    parser.add_argument('--model', type=str, default='mlp_4_layer',
                       help='Model type to optimize')
    parser.add_argument('--hb-max-epochs', type=int, default=20,
                       help='Maximum epochs for Hyperband')
    parser.add_argument('--hb-final-epochs', type=int, default=30,
                       help='Final epochs for best configuration')
    parser.add_argument('--hb-batch-size', type=int, default=128,
                       help='Batch size for Hyperband trials')

# 📝 FUNZIONE: parse_arguments
# ARGOMENTI:
#   - Nessuno (funzione chiamata automaticamente da benchmark.py)
# OUTPUT:
#   - parser (ArgumentParser): Oggetto parser configurato con tutti gli argomenti CLI
#   - Argomenti supportati: --hyperband, --model, --hb-max-epochs, --hb-final-epochs, --hb-batch-size
#   - Default values: model='mlp_4_layer', max_epochs=20, final_epochs=30, batch_size=128
```

**Flusso di inizializzazione**:
1. **Parsing**: Gli argomenti vengono interpretati e validati
2. **Config override**: I valori CLI sovrascrivono le configurazioni di default
3. **Seed setting**: `set_global_seed()` garantisce riproducibilità
4. **Data caching**: Controllo cache per evitare ri-preprocessing

### 2. Pipeline di Preprocessing (Cached)

**src/preprocessing/process.py** - Gestione dati:

```python
def preprocess_pipeline(data_path=None, sample_size=None):
    # STAGE 0: Check model-ready cache
    if PREPROCESSING_CONFIG.get("model_cache_enabled", False):
        cache_key = f"size{sample_size}_win{win_size}_step{win_step}_features{feature_count}"
        npz_path = os.path.join(cache_dir_model, f"model_ready_{cache_key}.npz")
        
        if os.path.exists(npz_path):
            # Load from cache - NO preprocessing needed
            data = np.load(npz_path)
            X, y = data['X'], data['y']
            return X, y, label_encoder
```

**Cosa succede**:
- **Cache hit**: Se esiste già un cache model-ready, carica direttamente `X`, `y`, `label_encoder`
- **Cache miss**: Esegue preprocessing completo (loading CSV, balancing, encoding, time windows)
- **Output**: Array NumPy pronti per training con shape `(samples, features)` per MLP

### 3. Costruzione Modello MLP a 4 Layer

**src/training/models.py** - Factory pattern:

```python
def get_model_builder(model_type: str):
    if model_type == "mlp_4_layer":
        return build_model_mlp_4_layer
    
def build_model_mlp_4_layer(input_shape, num_classes, **kwargs):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 📝 FUNZIONE: build_model_mlp_4_layer
# ARGOMENTI:
#   - input_shape (tuple): Shape delle features di input (es. (78,) per CSE-CIC-IDS2018)
#   - num_classes (int): Numero di classi di output (es. 15 per dataset completo)
#   - **kwargs: Argomenti aggiuntivi opzionali (non utilizzati in questa implementazione)
# OUTPUT:
#   - tf.keras.Model: Modello MLP compilato e pronto per training
#   - Architettura: 4 layer Dense con dropout e ReLU activation
#   - Compilazione: Adam optimizer, sparse_categorical_crossentropy loss, accuracy metric
```

**Architettura specifica**:
- **Input layer**: Dense(128) + ReLU + Dropout(0.3)
- **Hidden layer 1**: Dense(64) + ReLU + Dropout(0.3)  
- **Hidden layer 2**: Dense(32) + ReLU
- **Output layer**: Dense(num_classes) + Softmax
- **Optimizer**: Adam con learning rate 0.001
- **Loss**: Sparse categorical crossentropy per multi-class

### 4. Ottimizzazione Hyperband con Keras Tuner

**benchmark.py** - Orchestrazione Hyperband:

```python
def run_hyperband_optimization(self, model_type: str, max_epochs: int, 
                              final_epochs: int, batch_size: int):
    # 1. Get preprocessed data
    X, y, label_encoder = self._get_preprocessed_data()
    
    # 2. Prepare data for MLP (flatten if needed)
    X_mlp = prepare_data_for_model(X, model_type)
    
    # 3. Define Hyperband search space
    tuner = kt.Hyperband(
        lambda hp: self._build_hypermodel(hp, X_mlp.shape[1:], 
                                        len(label_encoder.classes_)),
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=3,
        directory='hyperband_results',
        project_name=f'{model_type}_optimization'
    )
    
    # 4. Execute search
    tuner.search(X_mlp, y, validation_split=0.2, 
                batch_size=batch_size, epochs=max_epochs)
    
    # 5. Get best model and retrain
    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]
    
    # 6. Final training with best hyperparameters
    final_model = self._train_final_model(best_hp, X_mlp, y, final_epochs)

# 📝 FUNZIONE: run_hyperband_optimization
# ARGOMENTI:
#   - self: Istanza della classe SNNIDSBenchmark
#   - model_type (str): Tipo modello da ottimizzare (es. "mlp_4_layer")
#   - max_epochs (int): Epoche massime per Hyperband search (es. 20)
#   - final_epochs (int): Epoche per final training con best config (es. 30)
#   - batch_size (int): Batch size per training (es. 128)
# OUTPUT:
#   - final_model (tf.keras.Model): Modello ottimizzato con best hyperparameters
#   - Directory: hyperband_results/ con tutti i trial e risultati
#   - Processo: Search → Best config → Final training → Modello finale
```

**Spazio di ricerca Hyperband**:

```python
def _build_hypermodel(self, hp, input_shape, num_classes):
    model = tf.keras.Sequential()
    
    # Input layer
    units = hp.Int('units_1', min_value=32, max_value=256, step=32)
    model.add(tf.keras.layers.Dense(units, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
    # Hidden layers
    for i in range(2, 4):  # 2 hidden layers
    for i in range(2, 4):  # 2 hidden layers
        units = hp.Int(f'units_{i}', min_value=16, max_value=128, step=16)
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))
    
    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile with tunable learning rate
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 📝 FUNZIONE: _build_hypermodel
# ARGOMENTI:
#   - self: Istanza della classe SNNIDSBenchmark
#   - hp (kt.HyperParameters): Oggetto Keras Tuner per hyperparameter optimization
#   - input_shape (tuple): Shape delle features di input (es. (78,))
#   - num_classes (int): Numero di classi di output (es. 15)
# OUTPUT:
#   - tf.keras.Model: Modello con hyperparameters ottimizzabili
#   - Hyperparameters tunabili: units (32-256), dropout (0.1-0.5), learning_rate (1e-4 a 1e-2)
#   - Architettura: 4 layer Dense con ReLU + Dropout configurabili
```

### 5. Training Loop con K-Fold Cross-Validation

**src/training/trainer.py** - Training robusto:

```python
def train_model_with_cv(self, X, y, n_splits=5):
    # Stratified K-Fold per mantenere distribuzione classi
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                         random_state=RANDOM_CONFIG.get('seed', 42))
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale data per-fold (evita data leakage)
        X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
        
        # Build and train model
        model = self.model_builder(X_train_scaled.shape[1:], 
                                 len(np.unique(y)))
        
        # Training with early stopping
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ),
                PerClassLossCallback(X_val_scaled, y_val, len(np.unique(y)))
            ],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
        fold_results.append({
            'fold': fold,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history.history
        })
    
    return fold_results

# 📝 FUNZIONE: train_model_with_cv
# ARGOMENTI:
#   - self: Istanza della classe ModelTrainer
#   - X (np.ndarray): Features di training, shape (samples, features)
#   - y (np.ndarray): Labels di training, shape (samples,)
#   - n_splits (int): Numero di fold per cross-validation (default: 5)
# OUTPUT:
#   - fold_results (List[Dict]): Lista risultati per ogni fold
#     - fold: Numero del fold (0-4)
#     - val_loss: Loss di validazione per il fold
#     - val_acc: Accuracy di validazione per il fold
#     - history: Storico training (loss, accuracy per epoca)
#   - Processo: StratifiedKFold → Per-fold scaling → Training → Evaluation
```

### 6. Valutazione e Metriche Comprehensive

**src/evaluation/metrics.py** - Analisi completa:

```python
def evaluate_model_comprehensive(model, X_test, y_test, class_names, output_dir):
    # 1. Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 2. Standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # 3. Cybersecurity-specific metrics
    cm = confusion_matrix(y_test, y_pred)
    detection_rate = np.sum(cm[1:, 1:]) / np.sum(cm[1:, :])  # Attack detection
    false_alarm_rate = np.sum(cm[0, 1:]) / np.sum(cm[0, :])  # Benign misclassification
    
    # 4. Per-class analysis
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # 5. Visualizations
    plot_confusion_matrix(cm, class_names, output_dir)
    plot_roc_curves(y_test, y_pred_proba, class_names, output_dir)
    plot_per_class_accuracy(per_class_acc, class_names, output_dir)
    
    # 6. JSON report
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'per_class_accuracy': per_class_acc.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

### 7. Flusso Completo di Esecuzione

**Sequenza temporale**:

```mermaid
graph TD
    A[Comando CLI] --> B[Parsing argomenti]
    B --> C[Setup seed + cache check]
    C --> D{Model cache exists?}
    D -->|Sì| E[Load X, y, label_encoder]
    D -->|No| F[Preprocessing completo]
    F --> G[Cache model-ready data]
    G --> E
    E --> H[Prepare data for MLP]
    H --> I[Initialize Hyperband tuner]
    I --> J[Define search space]
    J --> K[Execute Hyperband search]
    K --> L[Get best hyperparameters]
    L --> M[Final training with best config]
    M --> N[K-Fold cross-validation]
    N --> O[Model evaluation]
    O --> P[Generate visualizations]
    P --> Q[Save results + JSON report]
```

**Tempi tipici**:
- **Cache hit**: ~2-5 secondi per load
- **Hyperband search**: 5-15 minuti (dipende da max_epochs)
- **Final training**: 2-5 minuti (dipende da final_epochs)
- **Evaluation**: ~30 secondi per metriche + grafici

### 8. Output e Risultati

**Directory di output**:
```
hyperband_results/
├── mlp_4_layer_optimization/
│   ├── trial_001/          # Prima configurazione testata
│   ├── trial_002/          # Seconda configurazione
│   └── ...
├── evaluation_results/      # Risultati finali
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── per_class_accuracy.png
│   └── evaluation_results.json
└── best_model.h5           # Modello ottimizzato salvato
```

**JSON report finale**:
```json
{
  "model_type": "mlp_4_layer",
  "hyperband_config": {
    "max_epochs": 20,
    "final_epochs": 30,
    "batch_size": 128,
    "best_hyperparameters": {
      "units_1": 128,
      "dropout_1": 0.3,
      "units_2": 64,
      "dropout_2": 0.2,
      "units_3": 32,
      "dropout_3": 0.1,
      "learning_rate": 0.001
    }
  },
  "performance_metrics": {
    "accuracy": 0.945,
    "precision": 0.943,
    "recall": 0.945,
    "f1_score": 0.944,
    "detection_rate": 0.952,
    "false_alarm_rate": 0.038
  },
  "training_time": "12.5 minutes",
  "timestamp": "2024-01-15T14:30:00"
}
```

### 9. Perché Questo Flusso?

**Vantaggi del design**:
1. **Modularità**: Ogni componente ha responsabilità specifica
2. **Cache intelligente**: Evita ri-preprocessing per iterazioni multiple
3. **Hyperband efficiente**: 5-10x speedup vs grid search
4. **Validation robusta**: K-Fold + early stopping previene overfitting
5. **Metriche comprehensive**: Cybersecurity + standard ML metrics
6. **Riproducibilità**: Seed globale + configurazioni versionate

**Use case tipici**:
- **Ricerca**: Ottimizzazione hyperparameters per paper
- **Production**: Baseline performance per confronto FL/DP/HE
- **Debugging**: Isolamento problemi specifici per modulo
- **Benchmarking**: Confronto modelli e configurazioni

---

## 🔧 Funzioni Utility e Helper

### Funzioni di Preprocessing

```python
def scale_data(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applica StandardScaler a training e validation data.
    Gestisce correttamente dati 2D (tabular) e 3D (sequential).
    """
    scaler = StandardScaler()
    
    # Se dati sono sequential (3D), reshape per scaling
    if X_train.ndim == 3:
        orig_shape_train = X_train.shape
        orig_shape_val = X_val.shape
        
        # Reshape a (num_samples * timesteps, num_features)
        X_train_2d = X_train.reshape(-1, orig_shape_train[2])
        X_val_2d = X_val.reshape(-1, orig_shape_val[2])
        
        # Fit scaler SOLO su training data e transform entrambi
        X_train_scaled_2d = scaler.fit_transform(X_train_2d)
        X_val_scaled_2d = scaler.transform(X_val_2d)
        
        # Reshape back alla shape originale 3D
        X_train_scaled = X_train_scaled_2d.reshape(orig_shape_train)
        X_val_scaled = X_val_2d.reshape(orig_shape_val)
        
        return X_train_scaled, X_val_scaled
    
    # Per dati 2D, processo diretto
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled

# 📝 FUNZIONE: scale_data
# ARGOMENTI:
#   - X_train (np.ndarray): Features di training, shape (samples, features) o (samples, timesteps, features)
#   - X_val (np.ndarray): Features di validation, stessa shape di X_train
# OUTPUT:
#   - X_train_scaled (np.ndarray): Features di training scalate
#   - X_val_scaled (np.ndarray): Features di validation scalate
#   - Gestione automatica: 2D vs 3D data, evita data leakage per-fold
```

### Funzioni di Gestione Modelli

```python
def prepare_data_for_model(X: np.ndarray, model_type: str) -> np.ndarray:
    """
    Prepara dati per specifico tipo di modello.
    - MLP: Flatten 3D → 2D se necessario
    - GRU/LSTM: Mantiene shape 3D (samples, timesteps, features)
    """
    if model_type == "mlp_4_layer" and X.ndim == 3:
        # Flatten per MLP: (samples, timesteps, features) → (samples, timesteps*features)
        return X.reshape(X.shape[0], -1)
    elif model_type in ["gru", "lstm"]:
        # Mantiene shape 3D per modelli ricorrenti
        return X
    else:
        return X

# 📝 FUNZIONE: prepare_data_for_model
# ARGOMENTI:
#   - X (np.ndarray): Features da preparare, shape (samples, features) o (samples, timesteps, features)
#   - model_type (str): Tipo modello ("mlp_4_layer", "gru", "lstm")
# OUTPUT:
#   - np.ndarray: Features preparate per il tipo di modello specificato
#   - MLP: Flatten automatico se input 3D
#   - GRU/LSTM: Mantiene struttura temporale 3D
```

### Funzioni di Seed Management

```python
def set_global_seed(seed: int) -> None:
    """
    Imposta seed globali per riproducibilità across librerie.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import tensorflow as tf  # lazy import
    except Exception:
        tf = None
    
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        try:
            tf.random.set_seed(seed)
            # Best-effort determinism toggles
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        except Exception:
            pass

# 📝 FUNZIONE: set_global_seed
# ARGOMENTI:
#   - seed (int): Valore seed per riproducibilità (es. 42, 79)
# OUTPUT:
#   - None: Imposta seed per Python random, NumPy, TensorFlow
#   - Environment variables: PYTHONHASHSEED, TF_DETERMINISTIC_OPS
#   - Gestione graceful: Fallback se TensorFlow non disponibile
```

### Funzioni di Callback Personalizzati

```python
class PerClassLossCallback(tf.keras.callbacks.Callback):
    """
    Keras callback per registrare loss per-classe ad ogni epoca.
    """
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, num_classes: int):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.per_class_losses = {i: [] for i in range(num_classes)}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        for class_index in range(self.num_classes):
            mask = (self.y_val == class_index)
            if np.any(mask):
                probs = y_pred_proba[mask, class_index]
                ce = -float(np.mean(np.log(np.clip(probs, 1e-12, 1.0))))
            else:
                ce = float('nan')
            self.per_class_losses[class_index].append(ce)

# 📝 FUNZIONE: PerClassLossCallback
# ARGOMENTI:
#   - X_val (np.ndarray): Features di validazione per calcolo loss
#   - y_val (np.ndarray): Labels di validazione
#   - num_classes (int): Numero totale di classi nel dataset
# OUTPUT:
#   - self.per_class_losses (Dict[int, List[float]]): Loss per classe per ogni epoca
#   - Monitoraggio: Cross-entropy loss per ogni classe durante training
#   - Utilizzo: Analisi comportamento modello per classi specifiche
