"""
Module de feature engineering.

Ce module contient les fonctions pour créer de nouvelles features
à partir des données existantes, notamment les données de badgeage.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def create_time_features(in_out_time: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features à partir des données de badgeage.
    
    Args:
        in_out_time: DataFrame contenant les données de badgeage.
        
    Returns:
        DataFrame avec les features agrégées par employé.
    """
    if in_out_time is None or len(in_out_time) == 0:
        print("⚠️ Pas de données de badgeage disponibles")
        return None
    
    print("⏰ CRÉATION DES FEATURES DE TEMPS")
    print("-" * 40)
    
    df = in_out_time.copy()
    
    # Standardiser les noms de colonnes
    column_mapping = {
        'EmployeeId': 'EmployeeID',
        'employee_id': 'EmployeeID',
        'In Time': 'InTime',
        'Out Time': 'OutTime',
        'in_time': 'InTime',
        'out_time': 'OutTime'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Vérifier les colonnes nécessaires
    required_cols = ['EmployeeID', 'InTime', 'OutTime']
    if not all(col in df.columns or col.lower() in [c.lower() for c in df.columns] for col in required_cols):
        print(f"⚠️ Colonnes manquantes. Disponibles: {list(df.columns)}")
        return None
    
    # Convertir les heures en datetime
    df['InTime'] = pd.to_datetime(df['InTime'], errors='coerce')
    df['OutTime'] = pd.to_datetime(df['OutTime'], errors='coerce')
    
    # Calculer les heures travaillées par jour
    df['HoursWorked'] = (df['OutTime'] - df['InTime']).dt.total_seconds() / 3600
    
    # Calculer l'heure d'arrivée (en heures depuis minuit)
    df['ArrivalHour'] = df['InTime'].dt.hour + df['InTime'].dt.minute / 60
    
    # Calculer l'heure de départ
    df['DepartureHour'] = df['OutTime'].dt.hour + df['OutTime'].dt.minute / 60
    
    # Jour de la semaine (0=Lundi, 6=Dimanche)
    df['DayOfWeek'] = df['InTime'].dt.dayofweek
    
    # Identifier les retards (arrivée après 9h30)
    df['IsLate'] = (df['ArrivalHour'] > 9.5).astype(int)
    
    # Identifier les départs tardifs (après 18h)
    df['IsLateLeave'] = (df['DepartureHour'] > 18).astype(int)
    
    # Identifier les heures supplémentaires (>8h travaillées)
    df['IsOvertime'] = (df['HoursWorked'] > 8).astype(int)
    
    # Agréger par employé
    agg_features = df.groupby('EmployeeID').agg({
        'HoursWorked': ['mean', 'std', 'min', 'max', 'sum'],
        'ArrivalHour': ['mean', 'std'],
        'DepartureHour': ['mean', 'std'],
        'IsLate': ['sum', 'mean'],
        'IsLateLeave': ['sum', 'mean'],
        'IsOvertime': ['sum', 'mean'],
        'DayOfWeek': 'count'  # Nombre de jours travaillés
    }).reset_index()
    
    # Aplatir les noms de colonnes
    agg_features.columns = ['EmployeeID',
                            'AvgHoursWorked', 'StdHoursWorked', 'MinHoursWorked', 'MaxHoursWorked', 'TotalHoursWorked',
                            'AvgArrivalHour', 'StdArrivalHour',
                            'AvgDepartureHour', 'StdDepartureHour',
                            'TotalLateDays', 'LateDaysRatio',
                            'TotalLateLeaveDays', 'LateLeaveDaysRatio',
                            'TotalOvertimeDays', 'OvertimeDaysRatio',
                            'TotalDaysWorked']
    
    print(f"  • {len(agg_features)} employés avec features de temps")
    print(f"  • {len(agg_features.columns) - 1} nouvelles features créées")
    
    return agg_features


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features dérivées à partir des données existantes.
    
    Args:
        df: DataFrame avec les données des employés.
        
    Returns:
        DataFrame enrichi avec les nouvelles features.
    """
    print("\n📊 CRÉATION DES FEATURES DÉRIVÉES")
    print("-" * 40)
    
    df_enriched = df.copy()
    new_features = []
    
    # Ratio ancienneté dans l'entreprise / expérience totale
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        df_enriched['TenureRatio'] = df_enriched['YearsAtCompany'] / (df_enriched['TotalWorkingYears'] + 1)
        new_features.append('TenureRatio')
    
    # Années depuis la dernière promotion par rapport à l'ancienneté
    if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
        df_enriched['PromotionStagnation'] = df_enriched['YearsSinceLastPromotion'] / (df_enriched['YearsAtCompany'] + 1)
        new_features.append('PromotionStagnation')
    
    # Stabilité du manager
    if 'YearsWithCurrentManager' in df.columns and 'YearsAtCompany' in df.columns:
        df_enriched['ManagerStability'] = df_enriched['YearsWithCurrentManager'] / (df_enriched['YearsAtCompany'] + 1)
        new_features.append('ManagerStability')
    
    # Revenu par année d'expérience
    if 'MonthlyIncome' in df.columns and 'TotalWorkingYears' in df.columns:
        df_enriched['IncomePerYearExp'] = df_enriched['MonthlyIncome'] / (df_enriched['TotalWorkingYears'] + 1)
        new_features.append('IncomePerYearExp')
    
    # Score de satisfaction globale
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
    available_sat_cols = [col for col in satisfaction_cols if col in df.columns]
    if available_sat_cols:
        df_enriched['OverallSatisfaction'] = df_enriched[available_sat_cols].mean(axis=1)
        new_features.append('OverallSatisfaction')
    
    # Indicateur de mobilité (nombre d'entreprises précédentes élevé)
    if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
        df_enriched['JobHopping'] = df_enriched['NumCompaniesWorked'] / (df_enriched['TotalWorkingYears'] + 1)
        new_features.append('JobHopping')
    
    # Indicateur de distance importante
    if 'DistanceFromHome' in df.columns:
        df_enriched['IsLongCommute'] = (df_enriched['DistanceFromHome'] > 20).astype(int)
        new_features.append('IsLongCommute')
    
    # Indicateur de voyage fréquent
    if 'BusinessTravel' in df.columns:
        df_enriched['FrequentTraveler'] = (df_enriched['BusinessTravel'] == 'Travel_Frequently').astype(int)
        new_features.append('FrequentTraveler')
    
    # Catégorie d'âge
    if 'Age' in df.columns:
        df_enriched['AgeGroup'] = pd.cut(
            df_enriched['Age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Junior', 'Young', 'Mid', 'Senior', 'Expert']
        )
        new_features.append('AgeGroup')
    
    # Catégorie de revenu
    if 'MonthlyIncome' in df.columns:
        income_quantiles = df_enriched['MonthlyIncome'].quantile([0.25, 0.5, 0.75])
        df_enriched['IncomeCategory'] = pd.cut(
            df_enriched['MonthlyIncome'],
            bins=[0, income_quantiles[0.25], income_quantiles[0.5], income_quantiles[0.75], float('inf')],
            labels=['Low', 'Medium', 'High', 'VeryHigh']
        )
        new_features.append('IncomeCategory')
    
    print(f"  • {len(new_features)} nouvelles features créées:")
    for feat in new_features:
        print(f"    - {feat}")
    
    return df_enriched


def select_features_by_correlation(
    df: pd.DataFrame,
    target_col: str = 'Attrition',
    threshold: float = 0.05
) -> List[str]:
    """
    Sélectionne les features ayant une corrélation significative avec la cible.
    
    Args:
        df: DataFrame avec les données.
        target_col: Nom de la colonne cible.
        threshold: Seuil minimum de corrélation absolue.
        
    Returns:
        Liste des colonnes sélectionnées.
    """
    print(f"\n🔍 SÉLECTION DE FEATURES (corrélation > {threshold})")
    print("-" * 40)
    
    if target_col not in df.columns:
        print(f"⚠️ Colonne cible '{target_col}' non trouvée")
        return list(df.columns)
    
    # Calculer les corrélations avec la cible (uniquement colonnes numériques)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        print(f"⚠️ La colonne cible doit être numérique")
        return list(df.columns)
    
    correlations = numeric_df.corr()[target_col].abs()
    
    # Filtrer les features significatives
    selected = correlations[correlations > threshold].index.tolist()
    selected = [col for col in selected if col != target_col]
    
    print(f"  • {len(selected)} features sélectionnées sur {len(numeric_df.columns) - 1}")
    
    # Afficher les top corrélations
    top_corr = correlations.drop(target_col).sort_values(ascending=False).head(10)
    print(f"\n  Top 10 corrélations avec {target_col}:")
    for col, corr in top_corr.items():
        print(f"    - {col}: {corr:.3f}")
    
    return selected


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 20
) -> List[str]:
    """
    Sélectionne les features par importance (Random Forest).
    
    Args:
        X: Features.
        y: Variable cible.
        n_features: Nombre de features à sélectionner.
        
    Returns:
        Liste des colonnes sélectionnées.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    print(f"\n🌲 SÉLECTION DE FEATURES (importance RF, top {n_features})")
    print("-" * 40)
    
    # Entraîner un Random Forest rapide
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Obtenir les importances
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    # Sélectionner les top features
    selected = importances.head(n_features).index.tolist()
    
    print(f"  • Top {n_features} features:")
    for col in selected[:10]:
        print(f"    - {col}: {importances[col]:.4f}")
    if len(selected) > 10:
        print(f"    ... et {len(selected) - 10} autres")
    
    return selected


def merge_time_features(
    df: pd.DataFrame,
    time_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Fusionne les features de temps avec le dataset principal.
    
    Args:
        df: DataFrame principal.
        time_features: DataFrame avec les features de temps.
        
    Returns:
        DataFrame fusionné.
    """
    if time_features is None:
        return df
    
    print("\n🔗 FUSION DES FEATURES DE TEMPS")
    print("-" * 40)
    
    # Vérifier la colonne ID
    id_col = 'EmployeeID' if 'EmployeeID' in df.columns else 'EmployeeId'
    
    merged = df.merge(time_features, on='EmployeeID', how='left')
    
    # Compter les lignes avec des données de temps
    time_cols = [col for col in time_features.columns if col != 'EmployeeID']
    has_time_data = merged[time_cols[0]].notna().sum() if time_cols else 0
    
    print(f"  • {has_time_data}/{len(merged)} employés avec données de temps")
    
    return merged


def feature_engineering_pipeline(
    df: pd.DataFrame,
    in_out_time: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering.
    
    Args:
        df: DataFrame avec les données des employés.
        in_out_time: DataFrame avec les données de badgeage (optionnel).
        
    Returns:
        DataFrame enrichi avec toutes les nouvelles features.
    """
    print("=" * 60)
    print("🛠️ PIPELINE DE FEATURE ENGINEERING")
    print("=" * 60)
    
    # 1. Créer les features de temps si disponibles
    if in_out_time is not None:
        time_features = create_time_features(in_out_time)
        if time_features is not None:
            df = merge_time_features(df, time_features)
    else:
        print("⚠️ Pas de données de badgeage disponibles")
    
    # 2. Créer les features dérivées
    df = create_derived_features(df)
    
    print("\n" + "=" * 60)
    print(f"✅ FEATURE ENGINEERING TERMINÉ")
    print(f"   Dataset final: {len(df)} lignes × {len(df.columns)} colonnes")
    print("=" * 60)
    
    return df
