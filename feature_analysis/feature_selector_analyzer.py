#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selector Analyzer - Analisi completa per ranking automatico delle features
Identifica le features più discriminanti per attacchi vs traffico normale
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Path per accedere al progetto principale
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FeatureSelectorAnalyzer:
    """Analizzatore avanzato per selezione features basata su metriche scientifiche."""
    
    def __init__(self, cache_dir="../preprocessed_cache"):
        self.cache_dir = cache_dir
        self.feature_metrics = {}
        
    def load_sample_data(self, max_samples=15000):
        """Carica campione bilanciato per analisi."""
        print("🔍 Caricamento dati per analisi features...")
        
        all_data = []
        days = [d for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))][:2]
        
        for day in days:
            day_path = os.path.join(self.cache_dir, day)
            print(f"  📅 {day}...")
            
            # Carica attack e benign
            for data_type, label in [("attack_records.csv", 1), ("benign_records.csv", 0)]:
                file_path = os.path.join(day_path, data_type)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, nrows=max_samples//4)
                        df['attack_label'] = label
                        all_data.append(df)
                        print(f"    ✅ {data_type}: {len(df)} records")
                    except Exception as e:
                        print(f"    ❌ Errore {data_type}: {e}")
        
        if not all_data:
            raise ValueError("Nessun dato caricato!")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Dataset: {len(combined_df)} records, {len(combined_df.columns)} colonne")
        
        return combined_df
    
    def analyze_feature_discriminative_power(self, df):
        """Analizza il potere discriminativo di ogni feature."""
        print(f"\n🎯 ANALISI POTERE DISCRIMINATIVO")
        print("=" * 70)
        
        skip_cols = ['id', 'Flow ID', 'attack_label', 'Timestamp']
        
        for col in df.columns:
            if col in skip_cols:
                continue
                
            print(f"\n🔍 {col}")
            
            # Prova conversione numerica
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                valid_ratio = numeric_data.notna().sum() / len(df)
                
                if valid_ratio > 0.7:  # Se >70% convertibili → tratta come numerico
                    metrics = self._analyze_numeric_feature(df, col, numeric_data)
                else:
                    metrics = self._analyze_categorical_feature(df, col)
                    
                self.feature_metrics[col] = metrics
                
                # Print summary
                if metrics['type'] == 'numeric':
                    disc_status = "🔴 DISCRIMINANTE" if metrics.get('is_discriminative', False) else "🟢 normale"
                    print(f"  📊 Variabilità: CV={metrics['cv']:.2f}, Range=[{metrics['min']:.1f}, {metrics['max']:.1f}]")
                    print(f"  🎯 {disc_status} (p={metrics.get('ks_pvalue', 1):.4f})")
                else:
                    disc_status = "🔴 DISCRIMINANTE" if metrics.get('is_discriminative', False) else "🟢 normale"
                    print(f"  📝 Categorie: {metrics['unique_count']}, Entropia: {metrics['entropy']:.2f}")
                    print(f"  🎯 {disc_status}")
                
                print(f"  ⭐ Score: {metrics['importance_score']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Errore: {e}")
                continue
    
    def _analyze_numeric_feature(self, df, col, numeric_data):
        """Analisi dettagliata feature numerica."""
        # Riempi NaN con 0 per analisi
        data = numeric_data.fillna(0)
        
        # Statistiche base
        metrics = {
            'type': 'numeric',
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'unique_count': len(data.unique()),
            'unique_ratio': len(data.unique()) / len(data),
            'zero_ratio': (data == 0).sum() / len(data),
            'cv': float(data.std() / data.mean()) if data.mean() != 0 else float('inf')
        }
        
        # Analisi discriminativa
        attack_data = data[df['attack_label'] == 1]
        benign_data = data[df['attack_label'] == 0]
        
        if len(attack_data) > 10 and len(benign_data) > 10:
            # Test Kolmogorov-Smirnov
            ks_stat, ks_pvalue = stats.ks_2samp(attack_data, benign_data)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(attack_data)-1)*attack_data.var() + 
                                (len(benign_data)-1)*benign_data.var()) / 
                               (len(attack_data)+len(benign_data)-2))
            cohens_d = abs(attack_data.mean() - benign_data.mean()) / pooled_std if pooled_std > 0 else 0
            
            metrics.update({
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'cohens_d': float(cohens_d),
                'attack_mean': float(attack_data.mean()),
                'benign_mean': float(benign_data.mean()),
                'mean_ratio': float(attack_data.mean() / benign_data.mean()) if benign_data.mean() != 0 else float('inf'),
                'is_discriminative': ks_pvalue < 0.05 and cohens_d > 0.2
            })
        
        # Score di importanza composito
        variance_score = min(metrics['cv'], 3.0)  # Cap a 3
        uniqueness_score = min(metrics['unique_ratio'] * 2, 2.0)  # Cap a 2
        discrimination_score = (1 - metrics.get('ks_pvalue', 1)) * metrics.get('cohens_d', 0)
        
        importance_score = (variance_score + uniqueness_score + discrimination_score * 4) / 7
        metrics['importance_score'] = float(importance_score)
        
        return metrics
    
    def _analyze_categorical_feature(self, df, col):
        """Analisi feature categorica."""
        data = df[col].fillna('missing').astype(str)
        
        # Statistiche base
        value_counts = data.value_counts()
        metrics = {
            'type': 'categorical',
            'unique_count': len(value_counts),
            'unique_ratio': len(value_counts) / len(data),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else 'none',
            'most_frequent_ratio': value_counts.iloc[0] / len(data) if len(value_counts) > 0 else 0,
            'entropy': float(stats.entropy(value_counts)) if len(value_counts) > 1 else 0
        }
        
        # Test discriminativo (Chi-quadro)
        if len(value_counts) > 1 and len(value_counts) < 50:  # Non troppe categorie
            try:
                contingency_table = pd.crosstab(data, df['attack_label'])
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    # Cramér's V (effect size)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    
                    metrics.update({
                        'chi2_statistic': float(chi2),
                        'chi2_pvalue': float(chi2_pvalue),
                        'cramers_v': float(cramers_v),
                        'is_discriminative': chi2_pvalue < 0.05 and cramers_v > 0.1
                    })
            except Exception:
                pass
        
        # Score di importanza
        entropy_score = min(metrics['entropy'], 3.0)
        uniqueness_score = min(metrics['unique_ratio'] * 2, 2.0)
        discrimination_score = (1 - metrics.get('chi2_pvalue', 1)) * metrics.get('cramers_v', 0)
        
        importance_score = (entropy_score + uniqueness_score + discrimination_score * 4) / 7
        metrics['importance_score'] = float(importance_score)
        
        return metrics
    
    def generate_ranking(self):
        """Genera ranking delle features più importanti."""
        print(f"\n🏆 RANKING FEATURES PIÙ PROMETTENTI")
        print("=" * 80)
        
        # Ordina per importance score
        sorted_features = sorted(
            self.feature_metrics.items(), 
            key=lambda x: x[1]['importance_score'], 
            reverse=True
        )
        
        print(f"{'Rank':<4} {'Feature':<35} {'Type':<6} {'Score':<8} {'Discriminative':<12} {'Details'}")
        print("-" * 80)
        
        top_features = []
        for i, (feature, metrics) in enumerate(sorted_features[:25]):
            rank = i + 1
            ftype = "NUM" if metrics['type'] == 'numeric' else "CAT"
            score = metrics['importance_score']
            discriminative = "🔴 YES" if metrics.get('is_discriminative', False) else "🟢 no"
            
            if metrics['type'] == 'numeric':
                details = f"CV={metrics['cv']:.1f}, Range=[{metrics['min']:.0f}-{metrics['max']:.0f}]"
            else:
                details = f"Categories={metrics['unique_count']}, Entropy={metrics['entropy']:.1f}"
            
            print(f"{rank:<4} {feature:<35} {ftype:<6} {score:<8.3f} {discriminative:<12} {details}")
            
            top_features.append({
                'rank': rank,
                'feature': feature,
                'score': score,
                'type': metrics['type'],
                'is_discriminative': metrics.get('is_discriminative', False),
                'metrics': metrics
            })
        
        return top_features
    
    def suggest_new_config(self, top_features, target_count=30):
        """Suggerisce nuova configurazione features."""
        print(f"\n💡 SUGGERIMENTO NUOVA CONFIGURAZIONE")
        print("=" * 60)
        
        # Filtra le migliori features
        suggested = []
        discriminative_features = []
        
        for feat in top_features:
            if feat['score'] > 0.1:  # Soglia minima
                suggested.append(feat['feature'])
                if feat['is_discriminative']:
                    discriminative_features.append(feat['feature'])
        
        # Limita al target
        suggested = suggested[:target_count]
        
        print(f"🎯 Features consigliate: {len(suggested)}")
        print(f"🔴 Features discriminanti: {len(discriminative_features)}")
        
        print(f"\n📋 NUOVA CONFIGURAZIONE per config.py:")
        print('feature_columns = [')
        for feat in suggested:
            marker = "  # 🔴 DISCRIMINANTE" if feat in discriminative_features else ""
            print(f'    "{feat}",{marker}')
        print(']')
        
        return suggested
    
    def save_results(self, top_features):
        """Salva risultati dell'analisi."""
        results = {
            'feature_metrics': self.feature_metrics,
            'top_features': top_features,
            'analysis_summary': {
                'total_features_analyzed': len(self.feature_metrics),
                'discriminative_features': len([f for f in self.feature_metrics.values() if f.get('is_discriminative', False)]),
                'top_score': max([f['score'] for f in top_features]) if top_features else 0
            }
        }
        
        output_file = 'feature_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Risultati salvati: {output_file}")
        return output_file
    
    def run_complete_analysis(self):
        """Esegue analisi completa."""
        print("🚀 FEATURE SELECTOR ANALYZER")
        print("Analisi scientifica per identificare features più discriminanti")
        print("=" * 70)
        
        # Carica dati
        df = self.load_sample_data()
        
        # Analizza features
        self.analyze_feature_discriminative_power(df)
        
        # Genera ranking
        top_features = self.generate_ranking()
        
        # Suggerimenti configurazione
        suggested = self.suggest_new_config(top_features)
        
        # Salva risultati
        self.save_results(top_features)
        
        print(f"\n🎉 ANALISI COMPLETATA!")
        print(f"📊 Features totali analizzate: {len(self.feature_metrics)}")
        print(f"🔴 Features discriminanti trovate: {len([f for f in self.feature_metrics.values() if f.get('is_discriminative', False)])}")
        print(f"💡 Features consigliate per config: {len(suggested)}")
        
        return self.feature_metrics, top_features, suggested

def main():
    analyzer = FeatureSelectorAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
