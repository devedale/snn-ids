#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per testare il bilanciamento automatico nel pipeline normale.
Questo script simula come viene chiamato il preprocessing nel programma principale.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('.'))

from preprocessing.process import preprocess_data

def test_automatic_balancing():
    """
    Testa il bilanciamento automatico senza configurazioni specifiche.
    """
    print("🧪 TEST BILANCIAMENTO AUTOMATICO NEL PIPELINE")
    print("=" * 60)
    
    print("📋 Configurazione di default:")
    print("  - balance_strategy: security")
    print("  - benign_ratio: 0.5 (50%)")
    print("  - max_samples_per_class: 100000")
    
    print("\n🔄 Chiamata preprocess_data() senza configurazioni...")
    
    try:
        # Chiamata senza configurazioni (usa le default)
        X, y = preprocess_data()
        
        if X is not None and y is not None:
            print(f"✅ Preprocessing completato con successo!")
            print(f"📊 Shape X: {X.shape}")
            print(f"📊 Shape y: {y.shape}")
            
            # Verifica il bilanciamento
            if len(X.shape) == 3:
                # Per modelli sequenziali, prendi l'ultimo timestep
                y_flat = y
            else:
                y_flat = y
            
            unique_labels, counts = np.unique(y_flat, return_counts=True)
            total_samples = len(y_flat)
            
            print(f"\n🔍 Distribuzione classi nel dataset bilanciato:")
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                print(f"  Classe {label}: {count:,} campioni ({percentage:.1f}%)")
            
            # Verifica se abbiamo almeno 2 classi
            if len(unique_labels) >= 2:
                print(f"\n🎉 SUCCESSO! Dataset bilanciato con {len(unique_labels)} classi diverse!")
                print("   Il warning 'Dataset con una sola classe' non dovrebbe più apparire.")
            else:
                print(f"\n⚠️ ATTENZIONE: Ancora solo {len(unique_labels)} classe/i!")
                
        else:
            print("❌ Preprocessing fallito!")
            
    except Exception as e:
        print(f"❌ Errore durante il preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_automatic_balancing()
