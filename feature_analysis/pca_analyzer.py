#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA ANALYZER - Analizzatore Features Distintive per Dati di Rete
================================================================

GUIDA RAPIDA

Scopo:
- Identificare le features più distintive per etichetta usando PCA e correlazioni
- Creare grafici PCA comprensivi e uno scatter Features vs Labels
- Campionare i dati in modo uniforme tra i file per ogni label

Parametri principali (tutti configurabili in run_complete_pca_analysis):
- max_samples_per_label: max record per label totali (default: 10_000)
- attack_only: True = solo attacchi, False = include anche benign (se presenti)
- max_components: numero massimo di componenti PCA da analizzare (default: 10)
- features_per_component: top features per componente (default: 10)
- min_pca_score: soglia minima di importanza PCA per includere una feature (default: 0.05)
- correlation_threshold: soglia oltre cui considerare due feature troppo correlate (default: 0.95)
- target_variance: configurazione PCA principale (default: 0.95)

Esempi:
- Standard (10 features distintive):
  analyzer = PCAAnalyzer()
  analyzer.run_complete_pca_analysis()

- Più features distintive:
  analyzer.run_complete_pca_analysis(
      max_samples_per_label=15000,
      max_components=15,
      features_per_component=15,
      min_pca_score=0.03,
      correlation_threshold=0.98
  )

- Solo attacchi:
  analyzer.run_complete_pca_analysis(attack_only=True)

Output:
- pca_comprehensive_analysis.png
- features_vs_labels_importance.png
- label_specific_features_analysis.png
- pca_configurations.csv
- pca_components_analysis.csv
- distinctive_features_ranking.csv
- feature_correlations.csv
- feature_importance_per_label.csv
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# Path per accedere al progetto principale
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PCAAnalyzer:
    """Analizzatore PCA indipendente per features di rete."""

    # Parametri di default
    DEFAULT_MAX_COMPONENTS = 10
    DEFAULT_FEATURES_PER_COMPONENT = 10
    DEFAULT_MIN_PCA_SCORE = 0.05
    DEFAULT_CORRELATION_THRESHOLD = 0.95
    DEFAULT_TARGET_VARIANCE = 0.95
    DEFAULT_MAX_SAMPLES_PER_LABEL = 10000

    def __init__(self, cache_dir="../preprocessed_cache", **kwargs):
        self.cache_dir = cache_dir
        self.pca_results = {}

        # Parametri configurabili
        self.max_components = kwargs.get("max_components", self.DEFAULT_MAX_COMPONENTS)
        self.features_per_component = kwargs.get("features_per_component", self.DEFAULT_FEATURES_PER_COMPONENT)
        self.min_pca_score = kwargs.get("min_pca_score", self.DEFAULT_MIN_PCA_SCORE)
        self.correlation_threshold = kwargs.get("correlation_threshold", self.DEFAULT_CORRELATION_THRESHOLD)
        self.target_variance = kwargs.get("target_variance", self.DEFAULT_TARGET_VARIANCE)
        self.max_samples_per_label = kwargs.get("max_samples_per_label", self.DEFAULT_MAX_SAMPLES_PER_LABEL)

    # ===========================
    # Caricamento e Campionamento
    # ===========================

    def analyze_all_files_distribution(self):
        """Analizza la distribuzione di tutte le label in attack_records.csv di ogni giorno."""
        print("Analisi distribuzione label in tutti i file...")

        label_distribution = {}
        total_files = 0

        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory non trovata: {self.cache_dir}")

        days = [d for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))]

        for day in days:
            day_path = os.path.join(self.cache_dir, day)
            attack_file = os.path.join(day_path, "attack_records.csv")

            if os.path.exists(attack_file):
                total_files += 1
                try:
                    df = pd.read_csv(attack_file)
                    if 'Label' in df.columns:
                        counts = df['Label'].value_counts()
                        for label, count in counts.items():
                            info = label_distribution.setdefault(label, {'total': 0, 'files': 0})
                            info['total'] += int(count)
                            info['files'] += 1
                except Exception as e:
                    print(f"    Errore analisi {day}: {e}")

        print("Distribuzione label trovata:")
        for label, info in label_distribution.items():
            print(f"   • {label}: {info['total']:,} records in {info['files']} files")

        return label_distribution, total_files

    def calculate_samples_per_file_per_label(self, max_samples_per_label, label_distribution, total_files):
        """Calcola quanti record campionare per ogni label in ogni file (uniforme tra i file che la contengono)."""
        samples_per_file_per_label = {}

        for label, info in label_distribution.items():
            files_with_label = info['files']
            if files_with_label > 0:
                per_file_by_cap = max_samples_per_label // files_with_label
                per_file_by_avail = info['total'] // files_with_label
                samples_per_file = max(1, min(per_file_by_cap, per_file_by_avail))
                samples_per_file_per_label[label] = samples_per_file

        print("Campionamento per file:")
        for label, samples in samples_per_file_per_label.items():
            print(f"   • {label}: {samples:,} records per file")

        return samples_per_file_per_label

    def sample_from_file(self, file_path, label, samples_needed):
        """Campiona random records per una label da un file."""
        try:
            df = pd.read_csv(file_path)
            if 'Label' not in df.columns:
                return pd.DataFrame()
            label_data = df[df['Label'] == label]
            if len(label_data) == 0:
                return pd.DataFrame()
            if len(label_data) <= samples_needed:
                return label_data
            return label_data.sample(n=samples_needed, random_state=42)
        except Exception as e:
            print(f"    Errore campionamento {label} da {file_path}: {e}")
            return pd.DataFrame()

    def load_data_for_pca(self, max_samples_per_label=10000, attack_only=False):
        """Carica dati con campionamento uniforme per file/label."""
        print("Caricamento dati per analisi PCA...")

        if attack_only:
            print("Modalità ATTACCHI SOLO attivata")
        else:
            print(f"Modalità UNIFORME: max {max_samples_per_label:,} records per label totali")

        # 1) Distribuzione
        label_distribution, total_files = self.analyze_all_files_distribution()

        # 2) Quanti record per file per ogni label
        samples_per_file_per_label = self.calculate_samples_per_file_per_label(
            max_samples_per_label, label_distribution, total_files
        )

        # 3) Campionamento
        all_data = []
        days = [d for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))]
        for day in days:
            day_path = os.path.join(self.cache_dir, day)
            attack_file = os.path.join(day_path, "attack_records.csv")

            if os.path.exists(attack_file):
                try:
                    print(f"  Giorno {day}...")
                    file_data = []
                    for label, samples_per_file in samples_per_file_per_label.items():
                        sampled = self.sample_from_file(attack_file, label, samples_per_file)
                        if not sampled.empty:
                            sampled['attack_label'] = 1  # nel nostro scenario attack_records.csv
                            file_data.append(sampled)
                            print(f"    {label}: {len(sampled):,} records")
                    if file_data:
                        combined = pd.concat(file_data, ignore_index=True)
                        all_data.append(combined)
                        print(f"    Totale {day}: {len(combined):,} records")
                except Exception as e:
                    print(f"    Errore {day}: {e}")

        if not all_data:
            raise ValueError("Nessun dato caricato!")

        df = pd.concat(all_data, ignore_index=True)

        print("\nDataset finale per label:")
        for label in samples_per_file_per_label.keys():
            n = (df['Label'] == label).sum()
            print(f"   • {label}: {n:,} records")

        print(f"Dataset totale: {len(df):,} records, {len(df.columns)} colonne")
        return df

    # ===========================
    # Preparazione & PCA
    # ===========================

    def prepare_features_for_pca(self, df):
        """Seleziona colonne numeriche valide e pulisce i NaN/varianza zero."""
        print("\nPREPARAZIONE FEATURES PER PCA")
        print("=" * 50)

        exclude_cols = ['id', 'Flow ID', 'attack_label', 'Timestamp', 'Label', 'Src IP', 'Dst IP']

        numeric_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            valid_ratio = numeric_data.notna().sum() / len(df)
            if valid_ratio > 0.8:
                numeric_cols.append(col)

        print(f"Colonne numeriche trovate: {len(numeric_cols)}")

        pca_data = df[numeric_cols].copy()
        pca_data = pca_data.dropna(axis=1, thresh=int(len(pca_data) * 0.7))
        print(f"Colonne dopo rimozione NaN: {len(pca_data.columns)}")

        pca_data = pca_data.fillna(0)
        pca_data = pca_data.loc[:, pca_data.var() > 0]
        print(f"Colonne dopo rimozione varianza zero: {len(pca_data.columns)}")

        return pca_data, numeric_cols

    def perform_comprehensive_pca(self, pca_data, labels, n_components_range=None):
        """Esegue PCA per più soglie di varianza e raccoglie analisi componenti."""
        print("\nANALISI PCA COMPLETA")
        print("=" * 50)

        if n_components_range is None:
            n_components_range = [0.8, 0.9, 0.95, 0.99]

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler = StandardScaler()
        pca_data_scaled = scaler.fit_transform(pca_data)

        print(f"Dati standardizzati: {pca_data_scaled.shape}")

        pca_configs = {}
        for var_ratio in n_components_range:
            print(f"\nTestando varianza spiegata: {var_ratio:.0%}")
            pca = PCA(n_components=var_ratio)
            pca_result = pca.fit_transform(pca_data_scaled)

            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            print(f"  Componenti necessarie: {len(explained_variance_ratio)}")
            print(f"  Varianza spiegata: {cumulative_variance[-1]:.3f}")

            feature_importance = np.abs(pca.components_)
            component_analysis = []
            for i, component in enumerate(feature_importance):
                feature_scores = list(zip(pca_data.columns, component))
                feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                component_analysis.append({
                    'component': i + 1,
                    'explained_variance': explained_variance_ratio[i],
                    'cumulative_variance': cumulative_variance[i],
                    'top_features': feature_scores[:10],
                    'total_features': len(feature_scores)
                })

            pca_configs[var_ratio] = {
                'n_components': len(explained_variance_ratio),
                'explained_variance': cumulative_variance[-1],
                'pca_result': pca_result,
                'pca_model': pca,
                'component_analysis': component_analysis,
                'feature_importance': feature_importance
            }

        return pca_configs, pca_data_scaled

    def analyze_feature_correlations(self, pca_data):
        """Rileva coppie di features con alta correlazione (> soglia)."""
        print("\nANALISI CORRELAZIONI FEATURES")
        print("=" * 50)

        correlation_matrix = pca_data.corr()
        high_corr_pairs = []
        threshold = self.correlation_threshold

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        print(f"Coppie altamente correlate (|r| > {threshold}): {len(high_corr_pairs)}")

        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        for i, pair in enumerate(high_corr_pairs[:10]):
            print(f"  {i + 1:2d}. {pair['feature1']} ↔ {pair['feature2']}: r={pair['correlation']:.3f}")

        return correlation_matrix, high_corr_pairs

    # ===========================
    # Report attacchi-only
    # ===========================

    def create_attack_only_analysis(self, pca_data, distinctive_features, pca_configs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print("\nCreazione analisi features per SOLI ATTACCHI...")

            target_config = self.target_variance if self.target_variance in pca_configs else list(pca_configs.keys())[0]
            pca_config = pca_configs[target_config]

            fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=True)
            fig.suptitle('ANALISI FEATURES PER SOLI ATTACCHI', fontsize=18, fontweight='bold')

            # 1) Istogrammi
            ax = axes[0, 0]
            top_features = list(distinctive_features)[:6]
            for feature in top_features:
                if feature in pca_data.columns:
                    vals = pd.to_numeric(pca_data[feature], errors='coerce').dropna()
                    if len(vals) > 0:
                        ax.hist(vals, bins=30, alpha=0.6, label=feature, density=True)
            ax.set_xlabel('Valore Feature')
            ax.set_ylabel('Densità')
            ax.set_title('Distribuzione Top Features')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 2) Boxplot
            ax = axes[0, 1]
            plot_data, plot_labels = [], []
            for feature in top_features:
                if feature in pca_data.columns:
                    vals = pd.to_numeric(pca_data[feature], errors='coerce').dropna()
                    if len(vals) > 0:
                        plot_data.append(vals)
                        plot_labels.append(feature)
            if plot_data:
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.7)
            ax.set_xticklabels(plot_labels, rotation=45, ha='right')
            ax.set_ylabel('Valore Feature')
            ax.set_title('Boxplot Top Features')
            ax.grid(True, alpha=0.3)

            # 3) Heatmap correlazioni distintive (triangolo inferiore, senza annot)
            ax = axes[0, 2]
            if len(distinctive_features) > 1:
                feats = list(distinctive_features)[:20]  # limita a 20 per leggibilità
                corr_subset = pca_data[feats].corr()
                mask = np.triu(np.ones_like(corr_subset, dtype=bool))
                sns.heatmap(corr_subset, mask=mask, cmap='coolwarm', center=0,
                            square=True, cbar_kws={'label': 'Correlazione'}, ax=ax, annot=False)
                ax.set_title('Correlazioni Features Distintive')
            else:
                ax.text(0.5, 0.5, 'Troppo poche features', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Correlazioni Features Distintive')

            # 4) Varianza
            ax = axes[1, 0]
            variance_scores = []
            for feature in distinctive_features:
                if feature in pca_data.columns:
                    vals = pd.to_numeric(pca_data[feature], errors='coerce').dropna()
                    if len(vals) > 0:
                        variance_scores.append((feature, vals.std() ** 2))
            variance_scores.sort(key=lambda x: x[1], reverse=True)
            top_variance = variance_scores[:10]
            if top_variance:
                features, variances = zip(*top_variance)
                ax.barh(range(len(features)), variances, color='orange', alpha=0.7)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Varianza')
                ax.set_title('Top 10 Features per Varianza')
                ax.grid(True, alpha=0.3)

            # 5) Skewness/Kurtosis
            ax = axes[1, 1]
            skewness_values, kurtosis_values, feature_names = [], [], []
            for feature in distinctive_features:
                if feature in pca_data.columns:
                    vals = pd.to_numeric(pca_data[feature], errors='coerce').dropna()
                    if len(vals) > 0:
                        skewness_values.append(vals.skew())
                        kurtosis_values.append(vals.kurtosis())
                        feature_names.append(feature)
            if feature_names:
                x = np.arange(len(feature_names))
                width = 0.35
                ax.bar(x - width/2, skewness_values, width, label='Skewness', color='blue', alpha=0.7)
                ax.bar(x + width/2, kurtosis_values, width, label='Kurtosis', color='green', alpha=0.7)
                ax.set_xlabel('Features')
                ax.set_ylabel('Valore')
                ax.set_title('Skewness e Kurtosis')
                ax.set_xticks(x)
                ax.set_xticklabels(feature_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 6) Scatter 2D top 2 features
            ax = axes[1, 2]
            if len(distinctive_features) >= 2:
                f1, f2 = distinctive_features[0], distinctive_features[1]
                if f1 in pca_data.columns and f2 in pca_data.columns:
                    x_vals = pd.to_numeric(pca_data[f1], errors='coerce')
                    y_vals = pd.to_numeric(pca_data[f2], errors='coerce')
                    mask = ~(x_vals.isna() | y_vals.isna())
                    ax.scatter(x_vals[mask], y_vals[mask], alpha=0.6, color='red', s=20)
                    ax.set_xlabel(f1)
                    ax.set_ylabel(f2)
                    ax.set_title(f'{f1} vs {f2}')
                    ax.grid(True, alpha=0.3)

            plt.savefig('attack_only_features_analysis.png', dpi=300, bbox_inches='tight')
            print("Analisi features per soli attacchi salvata: attack_only_features_analysis.png")
            plt.close(fig)

        except ImportError:
            print("matplotlib/seaborn non disponibili")
        except Exception as e:
            print(f"Errore nella creazione del grafico per attacchi soli: {e}")

    # ===========================
    # Scatter Features vs Labels
    # ===========================

    def create_features_vs_labels_scatter(self, df, distinctive_features, pca_configs):
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl

            print("\nCreazione scatter plot Features vs Labels...")

            target_config = self.target_variance if self.target_variance in pca_configs else list(pca_configs.keys())[0]
            _ = pca_configs[target_config]

            if 'Label' in df.columns:
                unique_labels = sorted(df['Label'].unique())
            else:
                unique_labels = sorted(df['attack_label'].unique())

            feature_importance = {}
            for label in unique_labels:
                label_mask = (df['Label'] == label) if 'Label' in df.columns else (df['attack_label'] == label)
                label_data = df[label_mask]
                if len(label_data) == 0:
                    continue
                for feature in distinctive_features:
                    if feature in label_data.columns:
                        vals = pd.to_numeric(label_data[feature], errors='coerce').dropna()
                        if len(vals) > 0:
                            feature_importance.setdefault(feature, {})[label] = float(vals.var())

            features_list, labels_list, sizes_list, colors_list = [], [], [], []
            label_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            label_color_map = {label: label_colors[i] for i, label in enumerate(unique_labels)}

            for feature in distinctive_features:
                if feature not in feature_importance:
                    continue
                for label in unique_labels:
                    if label not in feature_importance[feature]:
                        continue
                    features_list.append(feature)
                    labels_list.append(str(label))
                    sizes_list.append(feature_importance[feature][label])
                    colors_list.append(label_color_map[label])

            if sizes_list:
                min_size = max(min(sizes_list), 1e-12)
                max_size = max(sizes_list)
                normalized_sizes = []
                for size in sizes_list:
                    if size > 0:
                        normalized = np.log(1 + size / min_size) / np.log(1 + max_size / min_size)
                        scaled = 120 + normalized * 780  # 120..900
                        normalized_sizes.append(scaled)
                    else:
                        normalized_sizes.append(120)
                sizes_list = normalized_sizes

            # Spazio extra a destra per colorbar; legenda sotto
            fig, ax = plt.subplots(figsize=(22, 12), constrained_layout=False)
            fig.subplots_adjust(right=0.85, bottom=0.2)

            sc = ax.scatter(labels_list, features_list, s=sizes_list, c=colors_list, alpha=0.75)
            ax.set_xlabel('Labels', fontsize=12, fontweight='bold')
            ax.set_ylabel('Features', fontsize=12, fontweight='bold')
            ax.set_title('Features vs Labels - Importanza per Varianza', fontsize=16, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Legenda sotto, su più colonne
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=label_color_map[label],
                                         markersize=10, label=str(label)) for label in unique_labels]
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      ncol=min(6, len(unique_labels)), frameon=False)

            # Colorbar a destra (log scale sugli originali)
            if features_list:
                originals = [feature_importance[f][l] for f, l in zip(features_list, labels_list)]
                norm = mpl.colors.LogNorm(vmin=max(min(originals), 1e-12), vmax=max(originals))
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label('Varianza Originale (Importanza)', rotation=270, labelpad=15)

            plt.savefig('features_vs_labels_importance.png', dpi=300, bbox_inches='tight')
            print("Scatter plot Features vs Labels salvato: features_vs_labels_importance.png")
            plt.close(fig)

            pd.DataFrame(feature_importance).T.to_csv('feature_importance_per_label.csv')
            print("Tabella importanza features salvata: feature_importance_per_label.csv")

        except ImportError:
            print("matplotlib/seaborn non disponibili")
        except Exception as e:
            print(f"Errore nella creazione del scatter plot Features vs Labels: {e}")

    # ===========================
    # Selezione Features Distintive
    # ===========================

    def identify_distinctive_features(self, pca_configs, correlation_matrix, high_corr_pairs,
                                      max_components=10, features_per_component=10,
                                      min_pca_score=0.05, correlation_threshold=0.95,
                                      target_variance=0.95):
        """Identifica features distintive usando PCA e rimozione per alta correlazione."""
        print("\nIDENTIFICAZIONE FEATURES DISTINTIVE")
        print("=" * 60)
        print("Parametri:")
        print(f"   • max_components: {max_components}")
        print(f"   • features_per_component: {features_per_component}")
        print(f"   • min_pca_score: {min_pca_score}")
        print(f"   • correlation_threshold: {correlation_threshold}")
        print(f"   • target_variance: {target_variance}")

        if target_variance in pca_configs:
            pca_config = pca_configs[target_variance]
        else:
            target_variance = list(pca_configs.keys())[0]
            pca_config = pca_configs[target_variance]

        distinctive_features = set()
        feature_scores = {}

        num_components = min(max_components, len(pca_config['component_analysis']))
        for comp in pca_config['component_analysis'][:num_components]:
            print(f"\nComponente {comp['component']} (var: {comp['explained_variance']:.1%})")
            top_features = comp['top_features'][:features_per_component]
            for feature, score in top_features:
                if abs(score) > min_pca_score:
                    distinctive_features.add(feature)
                    feature_scores[feature] = feature_scores.get(feature, 0.0) + abs(score) * comp['explained_variance']
                    print(f"  • {feature}: score={score:.3f}")

        print(f"\nFeatures candidate prima rimozione correlazioni: {len(distinctive_features)}")

        # Rimozione per alta correlazione
        to_remove = set()
        for pair in high_corr_pairs:
            if pair['correlation'] > correlation_threshold:
                f1, f2 = pair['feature1'], pair['feature2']
                if f1 in distinctive_features and f2 in distinctive_features:
                    s1 = feature_scores.get(f1, 0)
                    s2 = feature_scores.get(f2, 0)
                    if s1 < s2:
                        to_remove.add(f1)
                        print(f"  Rimossa {f1} (corr. con {f2}, r={pair['correlation']:.3f})")
                    else:
                        to_remove.add(f2)
                        print(f"  Rimossa {f2} (corr. con {f1}, r={pair['correlation']:.3f})")

        distinctive_features = list(distinctive_features - to_remove)
        print(f"Features distintive finali: {len(distinctive_features)}")
        print(f"{sorted(distinctive_features)}")
        return distinctive_features, feature_scores

    # ===========================
    # Salvataggi CSV
    # ===========================

    def save_pca_results_csv(self, pca_configs, distinctive_features, feature_scores, correlation_matrix, high_corr_pairs):
        print("\nSalvataggio risultati PCA in CSV...")

        # Configurazioni PCA
        config_rows = []
        for var_ratio, cfg in pca_configs.items():
            config_rows.append({
                'variance_threshold': var_ratio,
                'n_components': cfg['n_components'],
                'explained_variance': cfg['explained_variance'],
                'data_shape': cfg['pca_result'].shape[1]
            })
        pd.DataFrame(config_rows).to_csv('pca_configurations.csv', index=False)
        print("Config. PCA salvate: pca_configurations.csv")

        # Componenti (target variance)
        target_config = self.target_variance if self.target_variance in pca_configs else list(pca_configs.keys())[0]
        pca_config = pca_configs[target_config]
        comp_rows = []
        for comp in pca_config['component_analysis']:
            for feature, score in comp['top_features']:
                comp_rows.append({
                    'component': comp['component'],
                    'explained_variance': comp['explained_variance'],
                    'cumulative_variance': comp['cumulative_variance'],
                    'feature': feature,
                    'pca_score': score,
                    'abs_pca_score': abs(score),
                    'is_distinctive': feature in distinctive_features
                })
        pd.DataFrame(comp_rows).to_csv('pca_components_analysis.csv', index=False)
        print("Componenti PCA salvate: pca_components_analysis.csv")

        # Distinctive ranking
        ranked = sorted([(f, feature_scores.get(f, 0.0)) for f in distinctive_features], key=lambda x: x[1], reverse=True)
        pd.DataFrame([{'feature': f, 'pca_importance_score': s, 'rank': i + 1} for i, (f, s) in enumerate(ranked)]) \
            .to_csv('distinctive_features_ranking.csv', index=False)
        print("Features distintive salvate: distinctive_features_ranking.csv")

        # Correlazioni
        pd.DataFrame([{
            'feature1': p['feature1'],
            'feature2': p['feature2'],
            'correlation': p['correlation'],
            'correlation_abs': abs(p['correlation'])
        } for p in high_corr_pairs]).to_csv('feature_correlations.csv', index=False)
        print("Correlazioni salvate: feature_correlations.csv")

        return 'pca_configurations.csv', 'pca_components_analysis.csv', 'distinctive_features_ranking.csv', 'feature_correlations.csv'

    # ===========================
    # Grafici PCA
    # ===========================

    def create_comprehensive_plots(self, pca_configs, pca_data_scaled, labels, distinctive_features, correlation_matrix):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print("\nCreazione grafici PCA...")

            target_config = self.target_variance if self.target_variance in pca_configs else list(pca_configs.keys())[0]
            pca_config = pca_configs[target_config]
            pca_result = pca_config['pca_result']

            labels_values = np.asarray(labels)
            unique_labels = sorted(pd.Series(labels_values).unique())
            print(f"Usando {len(unique_labels)} label per i colori: {unique_labels}")

            plt.style.use('seaborn-v0_8')
            fig = plt.figure(figsize=(20, 15))

            # 1) PC1 vs PC2
            plt.subplot(3, 3, 1)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = (labels_values == label)
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1], c=[colors[i]], alpha=0.7, s=30, label=str(label))
            plt.xlabel(f'PC1 ({pca_config["component_analysis"][0]["explained_variance"]:.1%})')
            plt.ylabel(f'PC2 ({pca_config["component_analysis"][1]["explained_variance"]:.1%})')
            plt.title('PCA - Componenti 1 vs 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            # 2) PC2 vs PC3
            if len(pca_config['component_analysis']) >= 3:
                plt.subplot(3, 3, 2)
                for i, label in enumerate(unique_labels):
                    mask = (labels_values == label)
                    plt.scatter(pca_result[mask, 1], pca_result[mask, 2], c=[colors[i]], alpha=0.7, s=30, label=str(label))
                plt.xlabel(f'PC2 ({pca_config["component_analysis"][1]["explained_variance"]:.1%})')
                plt.ylabel(f'PC3 ({pca_config["component_analysis"][2]["explained_variance"]:.1%})')
                plt.title('PCA - Componenti 2 vs 3')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)

            # 3) Componenti vs explained variance
            plt.subplot(3, 3, 3)
            configs = list(pca_configs.keys())
            n_components = [pca_configs[c]['n_components'] for c in configs]
            explained_var = [pca_configs[c]['explained_variance'] for c in configs]
            bars = plt.bar(range(len(configs)), n_components, alpha=0.7)
            plt.xlabel('Configurazione')
            plt.ylabel('Numero Componenti')
            plt.title('Componenti per Configurazione')
            plt.xticks(range(len(configs)), [f'{c:.0%}' for c in configs])
            for bar, var in zip(bars, explained_var):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{var:.1%}', ha='center', va='bottom')

            # 4) Varianza spiegata per componente
            plt.subplot(3, 3, 4)
            comp = pca_config['component_analysis']
            cumulative_var = [c['cumulative_variance'] for c in comp]
            individual_var = [c['explained_variance'] for c in comp]
            x = range(1, len(cumulative_var) + 1)
            plt.bar(x, individual_var, alpha=0.7, label='Varianza per componente')
            plt.plot(x, cumulative_var, 'ro-', linewidth=2, markersize=6, label='Varianza cumulativa')
            plt.xlabel('Componente')
            plt.ylabel('Varianza Spiegata')
            plt.title('Varianza Spiegata per Componente')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 5) Top features comp1
            plt.subplot(3, 3, 5)
            top_features_comp1 = dict(pca_config['component_analysis'][0]['top_features'][:8])
            features = list(top_features_comp1.keys())
            scores = list(top_features_comp1.values())
            bar_colors = ['red' if abs(s) > 0.5 else 'blue' for s in scores]
            plt.barh(features, scores, color=bar_colors, alpha=0.7)
            plt.xlabel('Score PCA')
            plt.title('Top Features - Componente 1')
            plt.grid(True, alpha=0.3)

            # 6) Heatmap correlazioni (features distintive) - triangolo, senza annot e limitata
            plt.subplot(3, 3, 6)
            if len(distinctive_features) > 1:
                feats = list(distinctive_features)[:20]  # limita a 20
                subset = correlation_matrix.loc[feats, feats]
                mask = np.triu(np.ones_like(subset, dtype=bool))
                sns.heatmap(subset, mask=mask, cmap='coolwarm', center=0, square=True,
                            cbar_kws={'label': 'Correlazione'}, annot=False)
                plt.title('Correlazioni Features Distintive')
            else:
                plt.text(0.5, 0.5, 'Troppo poche features', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Correlazioni Features Distintive')

            # 7) 3D scatter (PC1, PC2, PC3)
            if len(pca_config['component_analysis']) >= 3:
                ax = fig.add_subplot(3, 3, 7, projection='3d')
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = (labels_values == label)
                    ax.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2], c=[colors[i]], alpha=0.7, s=30, label=str(label))
                ax.set_xlabel(f'PC1 ({pca_config["component_analysis"][0]["explained_variance"]:.1%})')
                ax.set_ylabel(f'PC2 ({pca_config["component_analysis"][1]["explained_variance"]:.1%})')
                ax.set_zlabel(f'PC3 ({pca_config["component_analysis"][2]["explained_variance"]:.1%})')
                ax.set_title('PCA 3D - Primi 3 Componenti')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 8) Conteggio features distintive per componente (prime 5)
            plt.subplot(3, 3, 8)
            comp_counts = {}
            for comp_i in pca_config['component_analysis'][:5]:
                comp_counts[f'PC{comp_i["component"]}'] = sum(1 for f, _ in comp_i['top_features'][:5] if f in distinctive_features)
            bars = plt.bar(comp_counts.keys(), comp_counts.values(), alpha=0.7, color='green')
            plt.xlabel('Componente')
            plt.ylabel('Features Distintive')
            plt.title('Features Distintive per Componente')
            plt.grid(True, alpha=0.3)
            for bar, count in zip(bars, comp_counts.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), ha='center', va='bottom')

            # 9) Scree plot (varianza per componente)
            plt.subplot(3, 3, 9)
            individual_var = [c['explained_variance'] for c in pca_config['component_analysis']]
            plt.plot(range(1, len(individual_var) + 1), individual_var, 'bo-', linewidth=2, markersize=6)
            plt.xlabel('Componente')
            plt.ylabel('Varianza Spiegata')
            plt.title('Scree Plot')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('pca_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            print("Grafici PCA completi salvati: pca_comprehensive_analysis.png")
            plt.close(fig)

        except ImportError:
            print("matplotlib/seaborn non disponibili")
        except Exception as e:
            print(f"Errore nella creazione dei grafici: {e}")

    # ===========================
    # Analisi per Label (Attack vs Benign)
    # ===========================

    def create_label_specific_analysis(self, pca_data, labels, distinctive_features, pca_configs):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print("\nCreazione analisi features per label...")

            target_config = self.target_variance if self.target_variance in pca_configs else list(pca_configs.keys())[0]
            _ = pca_configs[target_config]

            labels_series = pd.Series(labels)
            attack_only = len(labels_series.unique()) == 1 and 'BENIGN' not in labels_series.unique()

            if attack_only:
                print("Analisi SOLO ATTACCHI - Creo report dedicato")
                self.create_attack_only_analysis(pca_data, distinctive_features, pca_configs)
                return

            attack_data = pca_data[labels_series != 'BENIGN']
            benign_data = pca_data[labels_series == 'BENIGN']

            fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
            fig.suptitle('ANALISI FEATURES PER LABEL - Attack vs Benign', fontsize=16, fontweight='bold')

            # 1) Boxplot comparativi
            ax = axes[0, 0]
            top_features = list(distinctive_features)[:8]
            plot_data, plot_labels = [], []
            for feature in top_features:
                if feature in attack_data.columns and feature in benign_data.columns:
                    a = pd.to_numeric(attack_data[feature], errors='coerce').dropna()
                    b = pd.to_numeric(benign_data[feature], errors='coerce').dropna()
                    if len(a) > 0:
                        plot_data.append(a)
                        plot_labels.append(f'{feature}\n(Attack)')
                    if len(b) > 0:
                        plot_data.append(b)
                        plot_labels.append(f'{feature}\n(Benign)')
            if plot_data:
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                colors = ['red' if 'Attack' in lab else 'blue' for lab in plot_labels]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            ax.set_xticklabels(plot_labels, rotation=45, ha='right')
            ax.set_ylabel('Valore Feature')
            ax.set_title('Distribuzione Features (Boxplot)')
            ax.grid(True, alpha=0.3)

            # 2) Heatmap differenze medie
            ax = axes[0, 1]
            diffs, fnames = [], []
            for feature in distinctive_features:
                if feature in attack_data.columns and feature in benign_data.columns:
                    a = pd.to_numeric(attack_data[feature], errors='coerce').dropna()
                    b = pd.to_numeric(benign_data[feature], errors='coerce').dropna()
                    if len(a) > 0 and len(b) > 0:
                        diffs.append(a.mean() - b.mean())
                        fnames.append(feature)
            if diffs:
                arr = (np.array(diffs) / max(1e-12, np.max(np.abs(diffs)))).reshape(1, -1)
                im = ax.imshow(arr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax.set_yticks([0])
                ax.set_yticklabels(['Attack - Benign'])
                ax.set_xticks(range(len(fnames)))
                ax.set_xticklabels(fnames, rotation=45, ha='right')
                plt.colorbar(im, ax=ax, label='Differenza Normalizzata')
                for i, d in enumerate(diffs):
                    ax.text(i, 0, f'{d:.2f}', ha='center', va='center', fontweight='bold')
                ax.set_title('Differenze Medie (Attack - Benign)')

            # 3) Coefficiente di variazione
            ax = axes[1, 0]
            cv_a, cv_b, cv_f = [], [], []
            for feature in distinctive_features:
                if feature in attack_data.columns and feature in benign_data.columns:
                    a = pd.to_numeric(attack_data[feature], errors='coerce').dropna()
                    b = pd.to_numeric(benign_data[feature], errors='coerce').dropna()
                    if len(a) > 0 and len(b) > 0:
                        cv_a.append(a.std() / (abs(a.mean()) + 1e-12))
                        cv_b.append(b.std() / (abs(b.mean()) + 1e-12))
                        cv_f.append(feature)
            if cv_a:
                x = np.arange(len(cv_f))
                width = 0.35
                ax.bar(x - width/2, cv_a, width, label='Attack', color='red', alpha=0.7)
                ax.bar(x + width/2, cv_b, width, label='Benign', color='blue', alpha=0.7)
                ax.set_xlabel('Features')
                ax.set_ylabel('CV')
                ax.set_title('Variabilità per Label (CV)')
                ax.set_xticks(x)
                ax.set_xticklabels(cv_f, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 4) Score discriminativo
            ax = axes[1, 1]
            scores = []
            for feature in distinctive_features:
                if feature in attack_data.columns and feature in benign_data.columns:
                    a = pd.to_numeric(attack_data[feature], errors='coerce').dropna()
                    b = pd.to_numeric(benign_data[feature], errors='coerce').dropna()
                    if len(a) > 0 and len(b) > 0:
                        pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
                        scores.append((feature, (abs(a.mean() - b.mean()) / (pooled_std + 1e-12))))
            scores.sort(key=lambda x: x[1], reverse=True)
            if scores:
                feats, vals = zip(*scores[:10])
                ax.barh(range(len(feats)), vals, color='green', alpha=0.7)
                ax.set_yticks(range(len(feats)))
                ax.set_yticklabels(feats)
                ax.set_xlabel('Score Discriminativo')
                ax.set_title('Top 10 Features Più Discriminanti')
                ax.grid(True, alpha=0.3)
                for i, v in enumerate(vals):
                    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')

            plt.savefig('label_specific_features_analysis.png', dpi=300, bbox_inches='tight')
            print("Analisi features per label salvata: label_specific_features_analysis.png")
            plt.close(fig)

        except ImportError:
            print("matplotlib/seaborn non disponibili")
        except Exception as e:
            print(f"Errore nella creazione del grafico per label: {e}")

    # ===========================
    # Orchestrazione
    # ===========================

    def run_complete_pca_analysis(self,
                                  attack_only=False,
                                  max_samples_per_label=10000,
                                  max_components=10,
                                  features_per_component=10,
                                  min_pca_score=0.05,
                                  correlation_threshold=0.95,
                                  target_variance=0.95):
        print("PCA ANALYZER")
        if attack_only:
            print("ANALISI SOLO ATTACCHI")
        else:
            print(f"ANALISI UNIFORME (max {max_samples_per_label:,} per label totali)")
        print("=" * 70)

        # Carica dati (uniforme)
        df = self.load_data_for_pca(max_samples_per_label=max_samples_per_label, attack_only=attack_only)

        # Prepara features
        pca_data, numeric_cols = self.prepare_features_for_pca(df)

        # Correlazioni
        self.correlation_threshold = correlation_threshold
        correlation_matrix, high_corr_pairs = self.analyze_feature_correlations(pca_data)

        # PCA
        pca_configs, pca_data_scaled = self.perform_comprehensive_pca(pca_data, df.get('attack_label', pd.Series([1]*len(df))))

        # Features distintive
        distinctive_features, feature_scores = self.identify_distinctive_features(
            pca_configs, correlation_matrix, high_corr_pairs,
            max_components=max_components,
            features_per_component=features_per_component,
            min_pca_score=min_pca_score,
            correlation_threshold=correlation_threshold,
            target_variance=target_variance
        )

        # Salvataggi CSV
        self.save_pca_results_csv(pca_configs, distinctive_features, feature_scores, correlation_matrix, high_corr_pairs)

        # Grafici PCA (usa le vere label testuali)
        self.create_comprehensive_plots(pca_configs, pca_data_scaled, df['Label'], distinctive_features, correlation_matrix)

        # Analisi per label
        self.create_label_specific_analysis(pca_data, df['Label'], distinctive_features, pca_configs)

        # Scatter Features vs Labels
        self.create_features_vs_labels_scatter(df, distinctive_features, pca_configs)

        print("\nANALISI PCA COMPLETATA!")
        print(f"Features analizzate: {len(numeric_cols)}")
        print(f"Config. PCA testate: {len(pca_configs)}")
        print(f"Features distintive identificate: {len(distinctive_features)}")
        print(f"Correlazioni trovate: {len(high_corr_pairs)}")
        print(f"File CSV generati: 5")
        print(f"Grafici generati: 2")
        return {
            'pca_configs': pca_configs,
            'distinctive_features': distinctive_features,
            'feature_scores': feature_scores,
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs
        }


def main():
    analyzer = PCAAnalyzer()
    # Configurazione standard (10 features distintive)
    analyzer.run_complete_pca_analysis(
        attack_only=False,
        max_samples_per_label=10000,
        max_components=10,
        features_per_component=10,
        min_pca_score=0.05,
        correlation_threshold=0.95,
        target_variance=0.95
    )


if __name__ == "__main__":
    main()