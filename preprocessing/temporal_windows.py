# -*- coding: utf-8 -*-
"""
Modulo per la Gestione delle Finestre Temporali Configurabili.
Gestisce finestre temporali basate su approssimazioni del timestamp.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json

class TemporalWindowManager:
    """
    Gestisce la creazione di finestre temporali configurabili per analisi time-series.
    """
    
    def __init__(self, time_resolutions: List[str] = None):
        """
        Inizializza il gestore delle finestre temporali.
        
        Args:
            time_resolutions: Lista delle risoluzioni temporali (es: ['1s', '5s', '10s', '1m', '5m', '10m'])
        """
        if time_resolutions is None:
            self.time_resolutions = ['1s', '5s', '10s', '1m', '5m', '10m']
        else:
            self.time_resolutions = time_resolutions
        
        self.resolution_seconds = self._parse_time_resolutions()
    
    def _parse_time_resolutions(self) -> Dict[str, int]:
        """
        Converte le risoluzioni temporali in secondi.
        
        Returns:
            Dizionario {risoluzione: secondi}
        """
        resolution_map = {}
        
        for resolution in self.time_resolutions:
            if resolution.endswith('s'):
                seconds = int(resolution[:-1])
            elif resolution.endswith('m'):
                seconds = int(resolution[:-1]) * 60
            elif resolution.endswith('h'):
                seconds = int(resolution[:-1]) * 3600
            else:
                raise ValueError(f"Formato risoluzione non riconosciuto: {resolution}")
            
            resolution_map[resolution] = seconds
        
        return resolution_map
    
    def create_temporal_windows(self, df: pd.DataFrame, timestamp_col: str, 
                               resolution: str, min_window_size: int = 5) -> pd.DataFrame:
        """
        Crea finestre temporali basate su una risoluzione specifica.
        
        Args:
            df: DataFrame con i dati
            timestamp_col: Nome della colonna timestamp
            resolution: Risoluzione temporale (es: '5s', '1m')
            min_window_size: Dimensione minima della finestra per essere considerata valida
        
        Returns:
            DataFrame con finestre temporali
        """
        if resolution not in self.resolution_seconds:
            raise ValueError(f"Risoluzione {resolution} non supportata. Disponibili: {list(self.resolution_seconds.keys())}")
        
        # Assicura che il timestamp sia in formato datetime
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Ordina per timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Crea finestre temporali
        window_seconds = self.resolution_seconds[resolution]
        windows = []
        
        current_time = df[timestamp_col].min()
        end_time = df[timestamp_col].max()
        
        while current_time <= end_time:
            window_end = current_time + timedelta(seconds=window_seconds)
            
            # Trova tutti i record nella finestra temporale
            window_mask = (df[timestamp_col] >= current_time) & (df[timestamp_col] < window_end)
            window_data = df[window_mask]
            
            if len(window_data) >= min_window_size:
                # Aggiungi informazioni sulla finestra
                window_data = window_data.copy()
                window_data['window_start'] = current_time
                window_data['window_end'] = window_end
                window_data['window_resolution'] = resolution
                window_data['window_size'] = len(window_data)
                
                windows.append(window_data)
            
            current_time = window_end
        
        if windows:
            result_df = pd.concat(windows, ignore_index=True)
            return result_df
        else:
            return pd.DataFrame()
    
    def create_multiple_resolution_windows(self, df: pd.DataFrame, timestamp_col: str,
                                         min_window_size: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Crea finestre temporali per tutte le risoluzioni configurate.
        
        Args:
            df: DataFrame con i dati
            timestamp_col: Nome della colonna timestamp
            min_window_size: Dimensione minima della finestra
        
        Returns:
            Dizionario {risoluzione: DataFrame con finestre}
        """
        results = {}
        
        for resolution in self.time_resolutions:
            print(f"Creazione finestre temporali per risoluzione: {resolution}")
            try:
                windowed_df = self.create_temporal_windows(df, timestamp_col, resolution, min_window_size)
                results[resolution] = windowed_df
                print(f"  - Finestre create: {len(windowed_df)} record")
            except Exception as e:
                print(f"  - Errore per {resolution}: {e}")
                results[resolution] = pd.DataFrame()
        
        return results
    
    def get_window_statistics(self, windowed_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Calcola statistiche per ogni risoluzione temporale.
        
        Args:
            windowed_dfs: Dizionario delle finestre per risoluzione
        
        Returns:
            Statistiche per ogni risoluzione
        """
        stats = {}
        
        for resolution, df in windowed_dfs.items():
            if df.empty:
                stats[resolution] = {"error": "Nessuna finestra valida"}
                continue
            
            # Calcola statistiche delle finestre
            window_sizes = df.groupby(['window_start', 'window_end'])['window_size'].first()
            
            stats[resolution] = {
                "total_windows": len(window_sizes),
                "total_records": len(df),
                "avg_window_size": float(window_sizes.mean()),
                "min_window_size": int(window_sizes.min()),
                "max_window_size": int(window_sizes.max()),
                "std_window_size": float(window_sizes.std()),
                "coverage_start": df['window_start'].min().isoformat(),
                "coverage_end": df['window_end'].max().isoformat()
            }
        
        return stats
    
    def save_window_configuration(self, output_path: str):
        """
        Salva la configurazione delle finestre temporali.
        
        Args:
            output_path: Percorso per salvare la configurazione
        """
        config = {
            "time_resolutions": self.time_resolutions,
            "resolution_seconds": self.resolution_seconds,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configurazione finestre temporali salvata in: {output_path}")

def create_temporal_windows_simple(df: pd.DataFrame, timestamp_col: str, 
                                  resolution: str, min_window_size: int = 5) -> pd.DataFrame:
    """
    Funzione semplificata per creare finestre temporali.
    
    Args:
        df: DataFrame con i dati
        timestamp_col: Nome della colonna timestamp
        resolution: Risoluzione temporale (es: '5s', '1m')
        min_window_size: Dimensione minima della finestra
    
    Returns:
        DataFrame con finestre temporali
    """
    manager = TemporalWindowManager([resolution])
    return manager.create_temporal_windows(df, timestamp_col, resolution, min_window_size)

def benchmark_temporal_resolutions(df: pd.DataFrame, timestamp_col: str, 
                                 resolutions: List[str] = None) -> Dict[str, Dict]:
    """
    Benchmark delle diverse risoluzioni temporali.
    
    Args:
        df: DataFrame con i dati
        timestamp_col: Nome della colonna timestamp
        resolutions: Lista delle risoluzioni da testare
    
    Returns:
        Risultati del benchmark per ogni risoluzione
    """
    if resolutions is None:
        resolutions = ['1s', '5s', '10s', '1m', '5m', '10m']
    
    manager = TemporalWindowManager(resolutions)
    
    print("=== Benchmark Risoluzioni Temporali ===")
    print(f"Dataset originale: {len(df)} record")
    print(f"Timestamp range: {df[timestamp_col].min()} - {df[timestamp_col].max()}")
    print()
    
    # Crea finestre per tutte le risoluzioni
    windowed_dfs = manager.create_multiple_resolution_windows(df, timestamp_col)
    
    # Calcola statistiche
    stats = manager.get_window_statistics(windowed_dfs)
    
    # Stampa risultati
    print("\n=== Risultati Benchmark ===")
    for resolution, stat in stats.items():
        if "error" not in stat:
            print(f"{resolution}:")
            print(f"  Finestre: {stat['total_windows']}")
            print(f"  Record totali: {stat['total_records']}")
            print(f"  Dimensione media finestra: {stat['avg_window_size']:.1f}")
            print(f"  Range finestra: {stat['min_window_size']} - {stat['max_window_size']}")
            print()
        else:
            print(f"{resolution}: {stat['error']}")
    
    return stats
