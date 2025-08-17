#!/bin/bash
# Script per installare le dipendenze del parser di log sicuro

echo "=== Installazione Dipendenze Parser Log Sicuro ==="

# Verifica se Python è installato
if ! command -v python3 &> /dev/null; then
    echo "Errore: Python3 non trovato. Installare Python 3.8 o superiore."
    exit 1
fi

echo "Python version: $(python3 --version)"

# Installa pip se non presente
if ! command -v pip3 &> /dev/null; then
    echo "Installando pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Aggiorna pip
echo "Aggiornando pip..."
pip3 install --upgrade pip

# Installa dipendenze base
echo "Installando dipendenze da requirements.txt..."
pip3 install -r requirements.txt

# Installa il modello spaCy per l'inglese
echo "Installando modello spaCy per inglese..."
python3 -m spacy download en_core_web_sm

# Verifica installazione Presidio
echo "Verificando installazione Presidio..."
python3 -c "
import presidio_analyzer
import presidio_anonymizer
print('✓ Presidio installato correttamente')
"

if [ $? -eq 0 ]; then
    echo "✅ Tutte le dipendenze installate correttamente!"
else
    echo "❌ Errore nell'installazione di Presidio"
    echo "Provare manualmente:"
    echo "pip3 install presidio-analyzer presidio-anonymizer"
    echo "python3 -m spacy download en_core_web_sm"
fi

echo ""
echo "=== Installazione Completata ==="
echo "Per testare il sistema:"
echo "python3 secure_log_parser.py"
