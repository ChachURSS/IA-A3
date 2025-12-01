"""
Module de prétraitement des données.

Ce module contient les fonctions pour le nettoyage des données,
la gestion des valeurs manquantes, l'encodage des variables catégorielles,
et la normalisation des features numériques.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    categorical_strategy: str = 'mode'
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le DataFrame.
    
    Args:
        df: DataFrame à traiter.
        strategy: Stratégie pour les colonnes numériques ('mean', 'median', 'mode').
        categorical_strategy: Stratégie pour les colonnes catégorielles ('mode', 'constant').
        
    Returns:
        DataFrame avec les valeurs manquantes traitées.
    """
    df_clean = df.copy()
    
    print("🔧 TRAITEMENT DES VALEURS MANQUANTES")
    print("-" * 40)
    
    # Colonnes numériques
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Colonnes catégorielles
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Traiter les colonnes numériques
    for col in numeric_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            else:  # mode
                fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 0
            
            df_clean[col].fillna(fill_value, inplace=True)
            print(f"  • {col}: {missing} NA → remplacés par {strategy} ({fill_value:.2f})")
    
    # Traiter les colonnes catégorielles
    for col in categorical_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            if categorical_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            else:
                fill_value = 'Unknown'
            
            df_clean[col].fillna(fill_value, inplace=True)
            print(f"  • {col}: {missing} NA → remplacés par '{fill_value}'")
    
    remaining_na = df_clean.isnull().sum().sum()
    print(f"\n✅ Valeurs manquantes restantes: {remaining_na}")
    
    return df_clean


def encode_target_variable(df: pd.DataFrame, target_col: str = 'Attrition') -> pd.DataFrame:
    """
    Encode la variable cible (Attrition) en valeurs numériques.
    
    Args:
        df: DataFrame contenant la variable cible.
        target_col: Nom de la colonne cible.
        
    Returns:
        DataFrame avec la variable cible encodée.
    """
    df_encoded = df.copy()
    
    if target_col not in df_encoded.columns:
        print(f"⚠️ Colonne '{target_col}' non trouvée")
        return df_encoded
    
    # Encoder Yes/No en 1/0
    if df_encoded[target_col].dtype == 'object':
        df_encoded[target_col] = df_encoded[target_col].map({'Yes': 1, 'No': 0})
        print(f"✅ Variable cible '{target_col}' encodée: Yes=1, No=0")
    
    return df_encoded


def encode_categorical_features(
    df: pd.DataFrame,
    method: str = 'label',
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode les variables catégorielles.
    
    Args:
        df: DataFrame à encoder.
        method: Méthode d'encodage ('label' ou 'onehot').
        exclude_cols: Colonnes à exclure de l'encodage.
        
    Returns:
        Tuple (DataFrame encodé, dictionnaire des encodeurs).
    """
    if exclude_cols is None:
        exclude_cols = ['EmployeeID', 'EmployeeId']
    
    df_encoded = df.copy()
    encoders = {}
    
    # Identifier les colonnes catégorielles
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    print(f"\n🏷️ ENCODAGE DES VARIABLES CATÉGORIELLES ({method})")
    print("-" * 40)
    
    if method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            print(f"  • {col}: {len(le.classes_)} classes → {list(le.classes_)[:5]}...")
    
    elif method == 'onehot':
        for col in categorical_cols:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            print(f"  • {col}: {len(dummies.columns)} nouvelles colonnes créées")
    
    print(f"\n✅ {len(categorical_cols)} colonnes catégorielles encodées")
    
    return df_encoded, encoders


def scale_numeric_features(
    df: pd.DataFrame,
    method: str = 'standard',
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, object]:
    """
    Normalise/Standardise les features numériques.
    
    Args:
        df: DataFrame à normaliser.
        method: Méthode de normalisation ('standard' ou 'minmax').
        exclude_cols: Colonnes à exclure de la normalisation.
        
    Returns:
        Tuple (DataFrame normalisé, scaler utilisé).
    """
    if exclude_cols is None:
        exclude_cols = ['EmployeeID', 'EmployeeId', 'Attrition']
    
    df_scaled = df.copy()
    
    # Identifier les colonnes numériques à normaliser
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\n📐 NORMALISATION DES FEATURES ({method})")
    print("-" * 40)
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    if numeric_cols:
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        print(f"  • {len(numeric_cols)} colonnes normalisées")
        print(f"  • Colonnes: {numeric_cols[:5]}..." if len(numeric_cols) > 5 else f"  • Colonnes: {numeric_cols}")
    
    return df_scaled, scaler


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'Attrition',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prépare les ensembles d'entraînement et de test.
    
    Args:
        df: DataFrame préparé.
        target_col: Nom de la colonne cible.
        test_size: Proportion de l'ensemble de test.
        random_state: Graine aléatoire pour la reproductibilité.
        stratify: Si True, stratifie sur la variable cible.
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    print(f"\n✂️ SPLIT TRAIN/TEST")
    print("-" * 40)
    
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' non trouvée dans le DataFrame")
    
    # Séparer features et target
    X = df.drop(columns=[target_col, 'EmployeeID'] if 'EmployeeID' in df.columns else [target_col])
    y = df[target_col]
    
    # Split
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"  • Ensemble d'entraînement: {len(X_train)} échantillons ({(1-test_size)*100:.0f}%)")
    print(f"  • Ensemble de test: {len(X_test)} échantillons ({test_size*100:.0f}%)")
    
    if stratify:
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        print(f"  • Taux d'attrition (train): {train_ratio*100:.1f}%")
        print(f"  • Taux d'attrition (test): {test_ratio*100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str = 'Attrition',
    missing_strategy: str = 'median',
    encoding_method: str = 'label',
    scaling_method: str = 'standard',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Pipeline complet de prétraitement des données.
    
    Args:
        df: DataFrame brut.
        target_col: Nom de la colonne cible.
        missing_strategy: Stratégie pour les valeurs manquantes.
        encoding_method: Méthode d'encodage des catégories.
        scaling_method: Méthode de normalisation.
        test_size: Proportion de l'ensemble de test.
        random_state: Graine aléatoire.
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test, metadata).
    """
    print("=" * 60)
    print("🔄 PIPELINE DE PRÉTRAITEMENT")
    print("=" * 60)
    
    metadata = {}
    
    # 1. Gestion des valeurs manquantes
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    
    # 2. Encodage de la variable cible
    if target_col in df_clean.columns:
        df_clean = encode_target_variable(df_clean, target_col)
    
    # 3. Encodage des variables catégorielles
    df_encoded, encoders = encode_categorical_features(df_clean, method=encoding_method)
    metadata['encoders'] = encoders
    
    # 4. Normalisation (optionnel, fait avant le split pour avoir les stats)
    df_scaled, scaler = scale_numeric_features(df_encoded, method=scaling_method)
    metadata['scaler'] = scaler
    
    # 5. Split train/test
    if target_col in df_scaled.columns:
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df_scaled, target_col, test_size, random_state
        )
    else:
        print(f"⚠️ Variable cible '{target_col}' non trouvée. Retour du dataset complet.")
        return df_scaled, None, None, None, metadata
    
    print("\n" + "=" * 60)
    print("✅ PRÉTRAITEMENT TERMINÉ")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, metadata


def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identifie les types de features dans le DataFrame.
    
    Args:
        df: DataFrame à analyser.
        
    Returns:
        Dictionnaire avec les listes de colonnes par type.
    """
    return {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'boolean': df.select_dtypes(include=['bool']).columns.tolist()
    }
