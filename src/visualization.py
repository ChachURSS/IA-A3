"""
Module de visualisation.

Ce module contient les fonctions pour créer des graphiques
pour l'analyse exploratoire et l'évaluation des modèles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

# Configuration du style par défaut
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def set_plot_style():
    """Configure le style global des graphiques."""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_target_distribution(
    y: pd.Series,
    title: str = "Distribution de la Variable Cible (Attrition)",
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Affiche la distribution de la variable cible.
    
    Args:
        y: Série contenant la variable cible.
        title: Titre du graphique.
        figsize: Taille de la figure.
        
    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(['No Attrition', 'Attrition'], counts.values, color=colors)
    
    # Ajouter les pourcentages
    total = len(y)
    for bar, count in zip(bars, counts.values):
        percentage = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{percentage:.1f}%\n({count})', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Nombre d\'employés')
    
    plt.tight_layout()
    return fig


def plot_numeric_distributions(
    df: pd.DataFrame,
    columns: List[str] = None,
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Affiche les distributions des variables numériques.
    
    Args:
        df: DataFrame avec les données.
        columns: Liste des colonnes à afficher. Si None, toutes les numériques.
        ncols: Nombre de colonnes dans la grille.
        figsize: Taille de base de la figure (par ligne).
        
    Returns:
        Figure matplotlib.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclure les IDs et la cible
        columns = [c for c in columns if c not in ['EmployeeID', 'EmployeeId', 'Attrition']]
    
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(df[col], ax=ax, kde=True, color='steelblue')
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('')
    
    # Masquer les axes vides
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution des Variables Numériques', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    target_col: str = 'Attrition',
    columns: List[str] = None,
    ncols: int = 2,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Affiche les distributions des variables catégorielles par rapport à la cible.
    
    Args:
        df: DataFrame avec les données.
        target_col: Nom de la colonne cible.
        columns: Liste des colonnes à afficher.
        ncols: Nombre de colonnes dans la grille.
        figsize: Taille de base de la figure.
        
    Returns:
        Figure matplotlib.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        columns = [c for c in columns if c != target_col]
    
    if len(columns) == 0:
        print("Aucune colonne catégorielle à afficher")
        return None
    
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        if target_col in df.columns:
            # Barplot avec hue pour la cible
            data = df.groupby([col, target_col]).size().unstack(fill_value=0)
            data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
            ax.legend(['Stay', 'Leave'], title='')
        else:
            df[col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    # Masquer les axes vides
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution des Variables Catégorielles', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    mask_upper: bool = True
) -> plt.Figure:
    """
    Affiche la matrice de corrélation.
    
    Args:
        df: DataFrame avec les données.
        figsize: Taille de la figure.
        annot: Si True, affiche les valeurs.
        mask_upper: Si True, masque le triangle supérieur.
        
    Returns:
        Figure matplotlib.
    """
    # Garder seulement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculer la matrice de corrélation
    corr = numeric_df.corr()
    
    # Créer le masque pour le triangle supérieur
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 8}
    )
    
    ax.set_title('Matrice de Corrélation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_correlation_with_target(
    df: pd.DataFrame,
    target_col: str = 'Attrition',
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Affiche les corrélations avec la variable cible.
    
    Args:
        df: DataFrame avec les données.
        target_col: Nom de la colonne cible.
        top_n: Nombre de features à afficher.
        figsize: Taille de la figure.
        
    Returns:
        Figure matplotlib.
    """
    if target_col not in df.columns:
        print(f"Colonne cible '{target_col}' non trouvée")
        return None
    
    # Garder seulement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculer les corrélations avec la cible
    correlations = numeric_df.corr()[target_col].drop(target_col)
    correlations = correlations.sort_values(key=abs, ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in correlations]
    
    correlations.plot(kind='barh', ax=ax, color=colors)
    
    ax.set_title(f'Top {top_n} Corrélations avec {target_col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coefficient de Corrélation')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    model_name: str = ""
) -> plt.Figure:
    """
    Affiche la matrice de confusion.
    
    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        labels: Labels des classes.
        figsize: Taille de la figure.
        model_name: Nom du modèle pour le titre.
        
    Returns:
        Figure matplotlib.
    """
    if labels is None:
        labels = ['Stay', 'Leave']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    title = 'Matrice de Confusion'
    if model_name:
        title += f' - {model_name}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réalité')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Affiche les courbes ROC pour plusieurs modèles.
    
    Args:
        models: Dictionnaire {nom: modèle entraîné}.
        X_test: Features de test.
        y_test: Labels de test.
        figsize: Taille de la figure.
        
    Returns:
        Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test).astype(float)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Ligne de référence
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de Faux Positifs (FPR)')
    ax.set_ylabel('Taux de Vrais Positifs (TPR)')
    ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    model_name: str = ""
) -> plt.Figure:
    """
    Affiche l'importance des features.
    
    Args:
        importance_df: DataFrame avec colonnes 'Feature' et 'Importance'.
        top_n: Nombre de features à afficher.
        figsize: Taille de la figure.
        model_name: Nom du modèle pour le titre.
        
    Returns:
        Figure matplotlib.
    """
    # Prendre les top features
    top_features = importance_df.head(top_n).sort_values('Importance')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
    
    ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
    
    title = f'Top {top_n} Features Importantes'
    if model_name:
        title += f' - {model_name}'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Affiche la comparaison des performances des modèles.
    
    Args:
        results_df: DataFrame des résultats avec modèles en index.
        metrics: Liste des métriques à afficher.
        figsize: Taille de la figure.
        
    Returns:
        Figure matplotlib.
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Filtrer les métriques disponibles
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=metric)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des Performances des Modèles', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_boxplots_by_target(
    df: pd.DataFrame,
    columns: List[str],
    target_col: str = 'Attrition',
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Affiche des boxplots des variables numériques par rapport à la cible.
    
    Args:
        df: DataFrame avec les données.
        columns: Liste des colonnes à afficher.
        target_col: Nom de la colonne cible.
        ncols: Nombre de colonnes dans la grille.
        figsize: Taille de base de la figure.
        
    Returns:
        Figure matplotlib.
    """
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        if target_col in df.columns:
            # Créer les données pour le boxplot
            stay = df[df[target_col] == 0][col].dropna()
            leave = df[df[target_col] == 1][col].dropna()
            
            bp = ax.boxplot([stay, leave], labels=['Stay', 'Leave'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
        else:
            ax.boxplot(df[col].dropna())
        
        ax.set_title(col, fontsize=10)
    
    # Masquer les axes vides
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution par Statut d\'Attrition', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Sauvegarde une figure matplotlib.
    
    Args:
        fig: Figure à sauvegarder.
        filename: Nom du fichier.
        dpi: Résolution.
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"✅ Figure sauvegardée: {filename}")
