#!/bin/bash

# 🚀 SNN-IDS: Download Automatico Dataset CICIDS2018
# Questo script scarica automaticamente tutti i dataset necessari per il benchmark

set -e  # Esci se qualsiasi comando fallisce

echo "================================================================================="
echo "🚀 SNN-IDS: DOWNLOAD AUTOMATICO DATASET CICIDS2018"
echo "================================================================================="

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzione per stampare messaggi colorati
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica prerequisiti
print_status "Verifica prerequisiti..."

# Verifica wget
if ! command -v wget &> /dev/null; then
    print_error "wget non trovato. Installa wget:"
    echo "  Ubuntu/Debian: sudo apt-get install wget"
    echo "  CentOS/RHEL: sudo yum install wget"
    echo "  macOS: brew install wget"
    exit 1
fi

# Verifica Python
if ! command -v python3 &> /dev/null; then
    print_warning "Python3 non trovato. Verifica l'installazione."
fi

print_success "Prerequisiti verificati!"

# Creazione directory
print_status "Creazione directory necessarie..."

mkdir -p data/cicids
mkdir -p data/raw
mkdir -p benchmark_results/{intermediate,comparison,visualizations}
mkdir -p models
mkdir -p output

print_success "Directory create!"

# Configurazione dataset
DATASET_URLS=(
    "Monday:https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Monday-WorkingHours.pcap_ISCX.csv"
    "Tuesday:https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Tuesday-WorkingHours.pcap_ISCX.csv"
    "Wednesday:https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Wednesday-workingHours.pcap_ISCX.csv"
    "Thursday:https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Thursday-WorkingHours.pcap_ISCX.csv"
    "Friday:https://raw.githubusercontent.com/ahlashkari/CICFlowMeter/master/CICFlowMeter-4.0/bin/csv/Friday-WorkingHours.pcap_ISCX.csv"
)

# Funzione per scaricare un dataset
download_dataset() {
    local day=$1
    local url=$2
    local filename="data/cicids/${day}-WorkingHours.csv"
    
    if [ -f "$filename" ]; then
        local size=$(du -h "$filename" | cut -f1)
        print_warning "${day} già presente: ${filename} (${size})"
        return 0
    fi
    
    print_status "Download ${day}..."
    echo "  URL: ${url}"
    echo "  Destinazione: ${filename}"
    
    if wget -q --show-progress -O "$filename" "$url"; then
        local size=$(du -h "$filename" | cut -f1)
        local lines=$(wc -l < "$filename")
        print_success "${day} scaricato: ${filename} (${lines} righe, ${size})"
        return 0
    else
        print_error "Download ${day} fallito!"
        return 1
    fi
}

# Download di tutti i dataset
print_status "Inizio download dataset CICIDS2018..."
echo "================================================================================="

cd data/cicids

total_downloaded=0
total_failed=0

for dataset in "${DATASET_URLS[@]}"; do
    day=$(echo "$dataset" | cut -d: -f1)
    url=$(echo "$dataset" | cut -d: -f2)
    
    if download_dataset "$day" "$url"; then
        ((total_downloaded++))
    else
        ((total_failed++))
    fi
    
    echo "---------------------------------------------------------------------------------"
done

cd ../..

# Verifica finale
print_status "Verifica finale dataset scaricati..."
echo "================================================================================="

total_records=0
total_size=0

for day in Monday Tuesday Wednesday Thursday Friday; do
    filename="data/cicids/${day}-WorkingHours.csv"
    
    if [ -f "$filename" ]; then
        lines=$(wc -l < "$filename")
        size=$(du -h "$filename" | cut -f1)
        size_bytes=$(du -b "$filename" | cut -f1)
        
        echo "📁 ${day}: $((lines-1)) record, ${size}"
        total_records=$((total_records + lines - 1))
        total_size=$((total_size + size_bytes))
    else
        echo "❌ ${day}: File mancante"
    fi
done

# Conversione size totale in formato leggibile
if [ $total_size -gt 0 ]; then
    if [ $total_size -gt 1048576 ]; then
        total_size_human=$(echo "scale=1; $total_size/1048576" | bc)
        total_size_unit="MB"
    else
        total_size_human=$(echo "scale=1; $total_size/1024" | bc)
        total_size_unit="KB"
    fi
else
    total_size_human="0"
    total_size_unit="B"
fi

echo "================================================================================="
echo "📊 RIEPILOGO DOWNLOAD:"
echo "  ✅ Dataset scaricati: ${total_downloaded}/5"
echo "  ❌ Download falliti: ${total_failed}/5"
echo "  📈 Totale record: ${total_records:,}"
echo "  💾 Dimensione totale: ${total_size_human} ${total_size_unit}"
echo "================================================================================="

# Verifica spazio disco
print_status "Verifica spazio disco disponibile..."
available_space=$(df -h . | awk 'NR==2 {print $4}')
print_success "Spazio disponibile: ${available_space}"

# Verifica integrità file
print_status "Verifica integrità file scaricati..."

for day in Monday Tuesday Wednesday Thursday Friday; do
    filename="data/cicids/${day}-WorkingHours.csv"
    
    if [ -f "$filename" ]; then
        # Verifica che il file non sia vuoto
        if [ -s "$filename" ]; then
            # Verifica che contenga dati CSV (prima riga dovrebbe contenere virgole)
            first_line=$(head -n 1 "$filename")
            if [[ "$first_line" == *","* ]]; then
                print_success "${day}: File valido"
            else
                print_warning "${day}: File potrebbe non essere CSV valido"
            fi
        else
            print_error "${day}: File vuoto!"
        fi
    fi
done

# Informazioni per il benchmark
echo ""
echo "================================================================================="
echo "🎯 PROSSIMI PASSI PER IL BENCHMARK:"
echo "================================================================================="
echo "1. Verifica che tutti i dataset siano stati scaricati correttamente"
echo "2. Esegui il benchmark rapido per test:"
echo "   python3 run_benchmark.py --type quick --sample-size 5000 --visualize"
echo "3. Se tutto funziona, esegui il benchmark completo:"
echo "   python3 run_benchmark.py --type full --visualize"
echo "4. Monitora i progressi in: benchmark_results/intermediate/"
echo "5. Visualizza i risultati in: benchmark_results/visualizations/"
echo "================================================================================="

# Verifica file di benchmark
print_status "Verifica file di benchmark necessari..."

required_files=(
    "run_benchmark.py"
    "benchmark/comparison_benchmark.py"
    "benchmark/visualization.py"
    "config/benchmark_config.py"
    "preprocessing/process.py"
    "training/train.py"
    "evaluation/stats.py"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MANCANTE!"
        ((missing_files++))
    fi
done

if [ $missing_files -eq 0 ]; then
    print_success "Tutti i file di benchmark sono presenti!"
else
    print_warning "${missing_files} file di benchmark mancanti. Verifica l'installazione."
fi

echo ""
print_success "Download completato! 🎉"
echo "Dataset disponibili in: $(pwd)/data/cicids/"
echo "Directory benchmark create in: $(pwd)/benchmark_results/"
