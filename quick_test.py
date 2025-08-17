#!/usr/bin/env python3
"""
Test Rapido SNN Security Analytics
Esecuzione completa in locale con dati ridotti per valutazione veloce
"""

import logging
import time
import json
from pathlib import Path

# Setup logging minimale
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def quick_test():
    """Test completo rapido del sistema"""
    print("🚀 SNN Security Analytics - Test Rapido")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # Import moduli
        print("📦 Caricamento moduli...")
        from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
        
        # Configurazione rapida
        config = PipelineConfig(
            training_config_file="training_config.example.yaml",  # YAML di esempio
            training_framework="mock",  # Fallback se YAML assente
            training_epochs=5,          # Override minimo per velocità
            enable_reconstruction=True,
            anomaly_threshold=0.7,      # Soglia permissiva per vedere anomalie
            secure_export=False         # Salta export sicuro per velocità
        )
        
        orchestrator = PipelineOrchestrator(config)
        
        # File di test esistenti
        test_files = [
            "input/FGT80FTK22013405.root.tlog.txt",
            "input/FGT80FTK22013405.root.elog.txt"
        ]
        
        # Verifica file esistenti
        existing_files = [f for f in test_files if Path(f).exists()]
        if not existing_files:
            print("❌ Nessun file di test trovato in input/")
            return False
        
        print(f"📄 Usando {len(existing_files)} file di test")
        
        # 1. PREPROCESSING RAPIDO
        print("\n🔄 Fase 1: Preprocessing...")
        prep_start = time.time()
        
        try:
            prep_results = orchestrator.run_preprocessing_only(existing_files)
            prep_time = time.time() - prep_start
            
            print(f"✅ Preprocessing completato in {prep_time:.1f}s")
            print(f"   📊 {prep_results['total_logs']} log processati")
            print(f"   🧮 {prep_results['snn_samples']} campioni SNN")
            print(f"   📈 {prep_results['snn_features']} features")
            
        except Exception as e:
            print(f"❌ Errore preprocessing: {e}")
            return False
        
        # 2. TRAINING RAPIDO
        print("\n🧠 Fase 2: Training SNN...")
        train_start = time.time()
        
        try:
            snn_dataset = prep_results['output_files']['snn_dataset']
            train_results = orchestrator.run_training_only(snn_dataset)
            train_time = time.time() - train_start
            
            print(f"✅ Training completato in {train_time:.1f}s")
            training_history = train_results['training_results']['training_history']
            print(f"   🎯 Framework: {training_history['framework']}")
            print(f"   📉 Final loss: {training_history['final_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Errore training: {e}")
            return False
        
        # 3. ANALYSIS RAPIDO
        print("\n📊 Fase 3: Analisi risultati...")
        analysis_start = time.time()
        
        try:
            # Trova file mapping automaticamente
            mapping_file = None
            norm_file = None
            
            output_dir = Path("output")
            for f in output_dir.glob("*mapping*.json"):
                mapping_file = str(f)
                break
            for f in output_dir.glob("*normalization*.json"):
                norm_file = str(f)
                break
            
            # Usa il dataset SNN reale invece di dummy per l'analisi
            snn_dataset = prep_results['output_files']['snn_dataset']
            
            analysis_results = orchestrator.run_analysis_only(
                train_results['export_path'],
                "dummy_predictions.json",  # Simula predizioni (non serve file esistente)
                mapping_file,
                norm_file
            )
            analysis_time = time.time() - analysis_start
            
            print(f"✅ Analisi completata in {analysis_time:.1f}s")
            
            # Risultati analisi
            anomaly_summary = analysis_results['anomaly_summary']
            print(f"   🚨 {anomaly_summary['total_anomalies']} anomalie rilevate")
            print(f"   📈 Tasso anomalie: {anomaly_summary['anomaly_rate']:.1%}")
            print(f"   🎯 Confidence media: {anomaly_summary.get('avg_anomaly_confidence', 0):.3f}")
            
            # Log ricostruiti
            reconstructed = analysis_results['reconstructed_logs']
            if reconstructed:
                print(f"   🔄 {len(reconstructed)} log ricostruiti")
            
        except Exception as e:
            print(f"❌ Errore analisi: {e}")
            return False
        
        # 4. RISULTATI FINALI
        total_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("🎉 TEST COMPLETATO CON SUCCESSO!")
        print("="*50)
        print(f"⏱️  Tempo totale: {total_time:.1f} secondi")
        print(f"📊 Log processati: {prep_results['total_logs']}")
        print(f"🧮 Campioni SNN: {prep_results['snn_samples']}")
        print(f"🚨 Anomalie: {anomaly_summary['total_anomalies']}")
        
        # Mostra file generati
        print(f"\n📂 File generati in output/:")
        output_files = list(Path("output").glob("*"))
        for f in sorted(output_files)[-5:]:  # Ultimi 5 file
            size = f.stat().st_size / 1024  # KB
            print(f"   📄 {f.name} ({size:.1f} KB)")
        
        # Suggerimenti
        print("\n💡 Cosa puoi fare ora:")
        print("   📋 Visualizza report: cat output/analysis_report.json")
        print("   📊 Visualizza dataset: head output/*snn_dataset.csv")
        print("   🔍 Visualizza anomalie: head output/analysis_report.json")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        print("💡 Prova: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_sample_results():
    """Mostra campioni dei risultati generati"""
    print("\n🔍 CAMPIONI RISULTATI:")
    print("="*30)
    
    # Mostra sample dataset SNN
    try:
        import pandas as pd
        
        snn_files = list(Path("output").glob("*snn_dataset.csv"))
        if snn_files:
            df = pd.read_csv(snn_files[0])
            print(f"📊 Dataset SNN ({len(df)} righe, {len(df.columns)} colonne):")
            print(df.head(3).to_string(index=False))
        
        # Mostra sample report
        report_files = list(Path("output").glob("*analysis_report.json"))
        if report_files:
            with open(report_files[0]) as f:
                report = json.load(f)
            
            print(f"\n📋 Report Analisi:")
            summary = report.get('analysis_summary', {})
            print(f"   • Campioni totali: {summary.get('total_samples', 0)}")
            print(f"   • Anomalie rilevate: {summary.get('anomalies_detected', 0)}")
            print(f"   • Tasso anomalie: {summary.get('anomaly_rate', 0):.1%}")
            
    except Exception as e:
        print(f"Errore visualizzazione: {e}")

if __name__ == "__main__":
    print("🧪 Avvio test rapido del sistema SNN...")
    
    success = quick_test()
    
    if success:
        show_sample_results()
        print("\n🎯 Sistema funzionante e pronto per uso avanzato!")
    else:
        print("\n❌ Test fallito - controlla la configurazione")
    
    print(f"\n📚 Per informazioni dettagliate: cat USAGE_GUIDE.md")
