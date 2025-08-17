#!/usr/bin/env python3
"""
Test della struttura modulare
Verifica che tutti i moduli si importino e funzionino correttamente
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_module_imports():
    """Testa import di tutti i moduli"""
    print("=== Test Import Moduli ===")
    
    try:
        # Test Modulo 1: Log Preprocessing
        from log_preprocessing import (
            UniversalLogParser, LogAnonymizer, SNNPreprocessor, 
            LogProcessor, ProcessingConfig
        )
        print("✅ Modulo 1 (log_preprocessing) importato correttamente")
        
        # Test Modulo 2: SNN Training
        from snn_training import SNNTrainer, TrainingConfig
        print("✅ Modulo 2 (snn_training) importato correttamente")
        
        # Test Modulo 3: Result Analysis
        from result_analysis import ResultAnalyzer, AnalysisConfig
        print("✅ Modulo 3 (result_analysis) importato correttamente")
        
        # Test Orchestratore
        from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
        print("✅ Pipeline Orchestrator importato correttamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        return False

def test_module_initialization():
    """Testa inizializzazione moduli"""
    print("\n=== Test Inizializzazione Moduli ===")
    
    try:
        # Test Modulo 1
        from log_preprocessing import LogProcessor, ProcessingConfig
        config1 = ProcessingConfig(output_dir="test_output")
        processor = LogProcessor(config1)
        print("✅ LogProcessor inizializzato")
        
        # Test Modulo 2
        from snn_training import SNNTrainer, TrainingConfig
        config2 = TrainingConfig(framework="mock", epochs=5)
        trainer = SNNTrainer(config2)
        print("✅ SNNTrainer inizializzato")
        
        # Test Modulo 3
        from result_analysis import ResultAnalyzer, AnalysisConfig
        config3 = AnalysisConfig(enable_reconstruction=False)
        analyzer = ResultAnalyzer(config3)
        print("✅ ResultAnalyzer inizializzato")
        
        # Test Orchestratore
        from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
        config_main = PipelineConfig(training_framework="mock")
        orchestrator = PipelineOrchestrator(config_main)
        print("✅ PipelineOrchestrator inizializzato")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Verifica struttura file"""
    print("\n=== Test Struttura File ===")
    
    required_files = [
        # Modulo 1
        "log_preprocessing/__init__.py",
        "log_preprocessing/log_processor.py",
        "log_preprocessing/log_parser.py",
        "log_preprocessing/anonymizer.py",
        "log_preprocessing/snn_preprocessor.py",
        "log_preprocessing/anonymization_config.yaml",
        "log_preprocessing/snn_config.yaml",
        
        # Modulo 2
        "snn_training/__init__.py",
        "snn_training/snn_trainer.py",
        
        # Modulo 3
        "result_analysis/__init__.py",
        "result_analysis/result_analyzer.py",
        
        # Root
        "pipeline_orchestrator.py",
        "README.md",
        "ARCHITECTURE.md",
        ".gitignore",
        "requirements.txt",
        "requirements_full.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ File mancanti:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ Tutti i file richiesti sono presenti")
    return True

def test_basic_functionality():
    """Test funzionalità base"""
    print("\n=== Test Funzionalità Base ===")
    
    try:
        # Test parser
        from log_preprocessing import UniversalLogParser
        parser = UniversalLogParser()
        
        # Test parsing semplice
        test_line = 'logver=123 type="traffic" srcip=1.2.3.4 action=accept'
        result = parser.parse_line(test_line)
        
        if result and result.parsed_fields:
            print("✅ Parsing funziona")
        else:
            print("❌ Parsing non funziona")
            return False
        
        # Test configurazione SNN
        from log_preprocessing import SNNPreprocessor
        snn_proc = SNNPreprocessor("log_preprocessing/snn_config.yaml")
        print("✅ SNNPreprocessor configurato")
        
        # Test framework SNN mock
        from snn_training import SNNTrainer, TrainingConfig
        config = TrainingConfig(framework="mock", epochs=1)
        trainer = SNNTrainer(config)
        print("✅ SNNTrainer mock funziona")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore funzionalità: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_directory():
    """Verifica directory output"""
    print("\n=== Test Directory Output ===")
    
    output_dir = Path("output")
    if output_dir.exists() and output_dir.is_dir():
        print("✅ Directory output esiste")
        
        # Elenca file output esistenti
        output_files = list(output_dir.glob("*"))
        if output_files:
            print(f"📁 {len(output_files)} file in output/:")
            for file in output_files[:5]:  # Mostra primi 5
                print(f"   - {file.name}")
            if len(output_files) > 5:
                print(f"   ... e altri {len(output_files) - 5} file")
        else:
            print("📁 Directory output vuota")
        
        return True
    else:
        print("❌ Directory output mancante")
        return False

def main():
    """Test completo della struttura modulare"""
    print("🧪 TEST STRUTTURA MODULARE SNN SECURITY\n")
    
    tests = [
        ("Import Moduli", test_module_imports),
        ("Inizializzazione", test_module_initialization),
        ("Struttura File", test_file_structure),
        ("Funzionalità Base", test_basic_functionality),
        ("Directory Output", test_output_directory)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🔍 {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} fallito: {e}")
            results.append((test_name, False))
    
    # Riassunto
    print(f"\n{'='*50}")
    print("📊 RIASSUNTO TEST")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Risultato: {passed}/{len(tests)} test passati")
    
    if passed == len(tests):
        print("🎉 STRUTTURA MODULARE COMPLETAMENTE FUNZIONANTE!")
        return True
    else:
        print("⚠️  Alcuni test falliti - controllare la configurazione")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
