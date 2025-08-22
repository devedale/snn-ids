# -*- coding: utf-8 -*-
"""
Modulo per la Visualizzazione dei Risultati del Benchmark.
Genera grafici comparativi per analizzare le performance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import matplotlib.patches as mpatches

class BenchmarkVisualizer:
    """
    Genera visualizzazioni per i risultati del benchmark.
    """
    
    def __init__(self, results_dir: str, output_dir: str = None):
        """
        Inizializza il visualizzatore.
        
        Args:
            results_dir: Directory con i risultati del benchmark
            output_dir: Directory per salvare i grafici
        """
        self.results_dir = results_dir
        self.output_dir = output_dir or os.path.join(results_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Carica i risultati
        self.results = self._load_benchmark_results()
        self.comparison_data = self._extract_comparison_data()
    
    def _load_benchmark_results(self) -> Dict:
        """Carica i risultati del benchmark."""
        results_path = os.path.join(self.results_dir, 'complete_benchmark_results.json')
        
        if not os.path.exists(results_path):
            print(f"File risultati non trovato: {results_path}")
            return {}
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def _extract_comparison_data(self) -> pd.DataFrame:
        """Estrae i dati per il confronto in formato DataFrame."""
        if not self.results or 'comparison_report' not in self.results:
            return pd.DataFrame()
        
        comparison = self.results['comparison_report']['resolution_comparison']
        data = []
        
        for resolution, comp_data in comparison.items():
            if 'error' not in comp_data:
                data.append({
                    'resolution': resolution,
                    'baseline_accuracy': comp_data['baseline_accuracy'],
                    'cryptopan_accuracy': comp_data['cryptopan_accuracy'],
                    'accuracy_difference': comp_data['accuracy_difference'],
                    'accuracy_change_percentage': comp_data['accuracy_change_percentage'],
                    'performance_impact': comp_data['performance_impact'],
                    'training_time_baseline': comp_data['training_time_baseline'],
                    'training_time_cryptopan': comp_data['training_time_cryptopan'],
                    'time_difference': comp_data['time_difference']
                })
        
        return pd.DataFrame(data)
    
    def plot_accuracy_comparison(self, save_plot: bool = True) -> plt.Figure:
        """
        Grafico a barre per confrontare l'accuratezza tra baseline e Crypto-PAn.
        
        Args:
            save_plot: Se salvare il grafico
        
        Returns:
            Figura matplotlib
        """
        if self.comparison_data.empty:
            print("Nessun dato di confronto disponibile")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Grafico 1: Accuratezza assoluta
        x = np.arange(len(self.comparison_data))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, self.comparison_data['baseline_accuracy'], 
                        width, label='Baseline (Plain Text)', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, self.comparison_data['cryptopan_accuracy'], 
                        width, label='Crypto-PAn', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Risoluzione Temporale')
        ax1.set_ylabel('Accuratezza')
        ax1.set_title('Confronto Accuratezza: Baseline vs Crypto-PAn')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.comparison_data['resolution'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Grafico 2: Differenza percentuale
        colors = ['green' if x > 0 else 'red' for x in self.comparison_data['accuracy_change_percentage']]
        bars3 = ax2.bar(x, self.comparison_data['accuracy_change_percentage'], 
                        color=colors, alpha=0.7)
        
        ax2.set_xlabel('Risoluzione Temporale')
        ax2.set_ylabel('Cambio Accuratezza (%)')
        ax2.set_title('Impatto Crypto-PAn sull\'Accuratezza (% di Cambio)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.comparison_data['resolution'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'accuracy_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Grafico accuratezza salvato in: {plot_path}")
        
        return fig
    
    def plot_training_time_comparison(self, save_plot: bool = True) -> plt.Figure:
        """
        Grafico per confrontare i tempi di training.
        
        Args:
            save_plot: Se salvare il grafico
        
        Returns:
            Figura matplotlib
        """
        if self.comparison_data.empty:
            print("Nessun dato di confronto disponibile")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Grafico 1: Tempi assoluti
        x = np.arange(len(self.comparison_data))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, self.comparison_data['training_time_baseline'], 
                        width, label='Baseline', color='lightgreen', alpha=0.8)
        bars2 = ax1.bar(x + width/2, self.comparison_data['training_time_cryptopan'], 
                        width, label='Crypto-PAn', color='orange', alpha=0.8)
        
        ax1.set_xlabel('Risoluzione Temporale')
        ax1.set_ylabel('Tempo Training (secondi)')
        ax1.set_title('Confronto Tempi di Training: Baseline vs Crypto-PAn')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.comparison_data['resolution'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Grafico 2: Differenza percentuale
        time_diff_pct = (self.comparison_data['time_difference'] / 
                        self.comparison_data['training_time_baseline']) * 100
        
        colors = ['red' if x > 0 else 'green' for x in time_diff_pct]
        bars3 = ax2.bar(x, time_diff_pct, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Risoluzione Temporale')
        ax2.set_ylabel('Cambio Tempo Training (%)')
        ax2.set_title('Impatto Crypto-PAn sui Tempi di Training (% di Cambio)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.comparison_data['resolution'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'training_time_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Grafico tempi training salvato in: {plot_path}")
        
        return fig
    
    def plot_resolution_heatmap(self, save_plot: bool = True) -> plt.Figure:
        """
        Heatmap per visualizzare le performance per ogni risoluzione.
        
        Args:
            save_plot: Se salvare il grafico
        
        Returns:
            Figura matplotlib
        """
        if self.comparison_data.empty:
            print("Nessun dato di confronto disponibile")
            return None
        
        # Prepara i dati per la heatmap
        heatmap_data = self.comparison_data[['resolution', 'baseline_accuracy', 'cryptopan_accuracy']].copy()
        heatmap_data.set_index('resolution', inplace=True)
        
        # Crea la heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, cbar_kws={'label': 'Accuratezza'}, ax=ax)
        
        ax.set_title('Heatmap Performance per Risoluzione Temporale')
        ax.set_xlabel('Risoluzione Temporale')
        ax.set_ylabel('Configurazione')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'performance_heatmap.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap performance salvata in: {plot_path}")
        
        return fig
    
    def plot_summary_radar(self, save_plot: bool = True) -> plt.Figure:
        """
        Grafico radar per una visione d'insieme delle performance.
        
        Args:
            save_plot: Se salvare il grafico
        
        Returns:
            Figura matplotlib
        """
        if self.comparison_data.empty:
            print("Nessun dato di confronto disponibile")
            return None
        
        # Prepara i dati per il radar
        categories = self.comparison_data['resolution'].tolist()
        
        # Normalizza i valori per il radar (0-1)
        baseline_norm = (self.comparison_data['baseline_accuracy'] - 
                        self.comparison_data['baseline_accuracy'].min()) / \
                       (self.comparison_data['baseline_accuracy'].max() - 
                        self.comparison_data['baseline_accuracy'].min())
        
        cryptopan_norm = (self.comparison_data['cryptopan_accuracy'] - 
                         self.comparison_data['cryptopan_accuracy'].min()) / \
                        (self.comparison_data['cryptopan_accuracy'].max() - 
                         self.comparison_data['cryptopan_accuracy'].min())
        
        # Chiudi il grafico radar
        baseline_norm = np.concatenate([baseline_norm, [baseline_norm[0]]])
        cryptopan_norm = np.concatenate([cryptopan_norm, [cryptopan_norm[0]]])
        categories = categories + [categories[0]]
        
        # Crea il grafico radar
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Chiudi il grafico
        
        ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color='blue')
        ax.fill(angles, baseline_norm, alpha=0.25, color='blue')
        
        ax.plot(angles, cryptopan_norm, 'o-', linewidth=2, label='Crypto-PAn', color='red')
        ax.fill(angles, cryptopan_norm, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Confronto Performance: Baseline vs Crypto-PAn\n(Valori Normalizzati)', 
                    pad=20, size=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, 'performance_radar.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Grafico radar salvato in: {plot_path}")
        
        return fig
    
    def generate_comprehensive_report(self, save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Genera tutti i grafici del benchmark.
        
        Args:
            save_plots: Se salvare tutti i grafici
        
        Returns:
            Dizionario con tutte le figure generate
        """
        print("=== Generazione Grafici Benchmark ===")
        
        figures = {}
        
        try:
            # Grafico confronto accuratezza
            print("Generazione grafico confronto accuratezza...")
            figures['accuracy_comparison'] = self.plot_accuracy_comparison(save_plots)
            
            # Grafico confronto tempi training
            print("Generazione grafico confronto tempi training...")
            figures['training_time_comparison'] = self.plot_training_time_comparison(save_plots)
            
            # Heatmap performance
            print("Generazione heatmap performance...")
            figures['performance_heatmap'] = self.plot_resolution_heatmap(save_plots)
            
            # Grafico radar
            print("Generazione grafico radar...")
            figures['performance_radar'] = self.plot_summary_radar(save_plots)
            
            print("=== Generazione Grafici Completata ===")
            
        except Exception as e:
            print(f"Errore durante la generazione dei grafici: {e}")
        
        return figures
    
    def create_summary_table(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Crea una tabella riassuntiva dei risultati.
        
        Args:
            save_csv: Se salvare la tabella in CSV
        
        Returns:
            DataFrame con il riepilogo
        """
        if self.comparison_data.empty:
            print("Nessun dato di confronto disponibile")
            return pd.DataFrame()
        
        # Crea tabella riassuntiva
        summary = self.comparison_data.copy()
        
        # Aggiungi colonne calcolate
        summary['baseline_vs_cryptopan'] = summary['accuracy_difference'].apply(
            lambda x: 'Miglioramento' if x > 0 else 'Peggioramento' if x < 0 else 'Uguale'
        )
        
        summary['time_impact'] = summary['time_difference'].apply(
            lambda x: 'Più lento' if x > 0 else 'Più veloce' if x < 0 else 'Uguale'
        )
        
        # Ordina per risoluzione temporale
        summary = summary.sort_values('resolution')
        
        if save_csv:
            csv_path = os.path.join(self.output_dir, 'benchmark_summary.csv')
            summary.to_csv(csv_path, index=False)
            print(f"Tabella riassuntiva salvata in: {csv_path}")
        
        return summary

def visualize_benchmark_results(results_dir: str, output_dir: str = None) -> BenchmarkVisualizer:
    """
    Funzione di utilità per visualizzare i risultati del benchmark.
    
    Args:
        results_dir: Directory con i risultati
        output_dir: Directory per i grafici
    
    Returns:
        Visualizzatore configurato
    """
    visualizer = BenchmarkVisualizer(results_dir, output_dir)
    visualizer.generate_comprehensive_report()
    visualizer.create_summary_table()
    
    return visualizer
