"""
Module de chargement et fusion des données.

Ce module contient les fonctions pour charger tous les fichiers CSV
et les fusionner sur EmployeeID.
"""

import os
import pandas as pd
from typing import Optional, Tuple, Dict
import warnings

# Suppression des warnings pour une sortie plus propre
warnings.filterwarnings('ignore')


def load_employee_survey(data_path: str) -> Optional[pd.DataFrame]:
    """
    Charge le fichier employee_survey_data.csv.
    
    Args:
        data_path: Chemin vers le répertoire data.
        
    Returns:
        DataFrame contenant les données de l'enquête employés, ou None si fichier absent.
    """
    filepath = os.path.join(data_path, 'employee_survey_data.csv')
    
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier '{filepath}' non trouvé.")
        return None
    
    try:
        df = pd.read_csv(filepath, na_values=['NA', 'na', 'N/A', ''])
        print(f"✅ Chargé: employee_survey_data.csv ({len(df)} lignes, {len(df.columns)} colonnes)")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement de employee_survey_data.csv: {e}")
        return None


def load_manager_survey(data_path: str) -> Optional[pd.DataFrame]:
    """
    Charge le fichier manager_survey_data.csv.
    
    Args:
        data_path: Chemin vers le répertoire data.
        
    Returns:
        DataFrame contenant les données de l'évaluation manager, ou None si fichier absent.
    """
    filepath = os.path.join(data_path, 'manager_survey_data.csv')
    
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier '{filepath}' non trouvé.")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Chargé: manager_survey_data.csv ({len(df)} lignes, {len(df.columns)} colonnes)")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement de manager_survey_data.csv: {e}")
        return None


def load_general_data(data_path: str) -> Optional[pd.DataFrame]:
    """
    Charge le fichier general_data.csv.
    
    Args:
        data_path: Chemin vers le répertoire data.
        
    Returns:
        DataFrame contenant les données générales des employés, ou None si fichier absent.
    """
    filepath = os.path.join(data_path, 'general_data.csv')
    
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier '{filepath}' non trouvé.")
        print("   Ce fichier doit être ajouté manuellement (trop volumineux pour le dépôt).")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Chargé: general_data.csv ({len(df)} lignes, {len(df.columns)} colonnes)")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement de general_data.csv: {e}")
        return None


def load_in_out_time(data_path: str) -> Optional[pd.DataFrame]:
    """
    Charge les données de badgeage depuis in_out_time.zip ou le dossier extrait.
    
    Args:
        data_path: Chemin vers le répertoire data.
        
    Returns:
        DataFrame contenant les données de badgeage, ou None si fichier absent.
    """
    import zipfile
    import glob
    
    zip_filepath = os.path.join(data_path, 'in_out_time.zip')
    folder_path = os.path.join(data_path, 'in_out_time')
    
    # Vérifier si le dossier extrait existe
    if os.path.exists(folder_path):
        try:
            all_files = glob.glob(os.path.join(folder_path, '*.csv'))
            if all_files:
                dfs = []
                for f in all_files:
                    df_temp = pd.read_csv(f)
                    dfs.append(df_temp)
                df = pd.concat(dfs, ignore_index=True)
                print(f"✅ Chargé: in_out_time ({len(df)} lignes depuis {len(all_files)} fichiers)")
                return df
        except Exception as e:
            print(f"❌ Erreur lors du chargement de in_out_time: {e}")
            return None
    
    # Vérifier si le fichier zip existe
    if os.path.exists(zip_filepath):
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if csv_files:
                    dfs = []
                    for csv_file in csv_files:
                        with z.open(csv_file) as f:
                            df_temp = pd.read_csv(f)
                            dfs.append(df_temp)
                    df = pd.concat(dfs, ignore_index=True)
                    print(f"✅ Chargé: in_out_time.zip ({len(df)} lignes depuis {len(csv_files)} fichiers)")
                    return df
        except Exception as e:
            print(f"❌ Erreur lors du chargement de in_out_time.zip: {e}")
            return None
    
    print(f"⚠️ Fichier 'in_out_time.zip' ou dossier 'in_out_time' non trouvé.")
    print("   Ces données doivent être ajoutées manuellement (trop volumineuses pour le dépôt).")
    return None


def load_all_data(data_path: str = None) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Charge tous les fichiers de données disponibles.
    
    Args:
        data_path: Chemin vers le répertoire data. Si None, utilise le répertoire par défaut.
        
    Returns:
        Dictionnaire contenant tous les DataFrames chargés.
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, 'data')
    
    print("=" * 60)
    print("📊 CHARGEMENT DES DONNÉES")
    print("=" * 60)
    
    data = {
        'employee_survey': load_employee_survey(data_path),
        'manager_survey': load_manager_survey(data_path),
        'general_data': load_general_data(data_path),
        'in_out_time': load_in_out_time(data_path)
    }
    
    print("=" * 60)
    available = sum(1 for v in data.values() if v is not None)
    print(f"📈 {available}/4 sources de données disponibles")
    print("=" * 60)
    
    return data


def merge_datasets(
    employee_survey: Optional[pd.DataFrame],
    manager_survey: Optional[pd.DataFrame],
    general_data: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Fusionne les datasets sur EmployeeID.
    
    Args:
        employee_survey: DataFrame de l'enquête employés.
        manager_survey: DataFrame de l'évaluation manager.
        general_data: DataFrame des données générales.
        
    Returns:
        DataFrame fusionné, ou None si aucune donnée disponible.
    """
    print("\n🔗 FUSION DES DATASETS")
    print("-" * 40)
    
    dfs_to_merge = []
    id_column = None
    
    # Identifier la colonne ID (peut être EmployeeID ou EmployeeId)
    for df, name in [(employee_survey, 'employee_survey'), 
                      (manager_survey, 'manager_survey'),
                      (general_data, 'general_data')]:
        if df is not None:
            if 'EmployeeID' in df.columns:
                id_column = 'EmployeeID'
            elif 'EmployeeId' in df.columns:
                id_column = 'EmployeeId'
            break
    
    if id_column is None:
        print("❌ Aucune colonne d'identification trouvée")
        return None
    
    # Standardiser le nom de la colonne ID
    for df in [employee_survey, manager_survey, general_data]:
        if df is not None:
            if 'EmployeeId' in df.columns and 'EmployeeID' not in df.columns:
                df.rename(columns={'EmployeeId': 'EmployeeID'}, inplace=True)
    
    # Commencer avec general_data si disponible (contient Attrition)
    if general_data is not None:
        merged = general_data.copy()
        print(f"  • Base: general_data ({len(merged)} employés)")
    elif employee_survey is not None:
        merged = employee_survey.copy()
        print(f"  • Base: employee_survey ({len(merged)} employés)")
    elif manager_survey is not None:
        merged = manager_survey.copy()
        print(f"  • Base: manager_survey ({len(merged)} employés)")
    else:
        print("❌ Aucune donnée disponible pour la fusion")
        return None
    
    # Fusionner les autres datasets
    if general_data is not None and employee_survey is not None:
        merged = merged.merge(employee_survey, on='EmployeeID', how='left')
        print(f"  • Fusionné: employee_survey")
    
    if general_data is not None and manager_survey is not None:
        merged = merged.merge(manager_survey, on='EmployeeID', how='left')
        print(f"  • Fusionné: manager_survey")
    elif employee_survey is not None and manager_survey is not None and general_data is None:
        merged = merged.merge(manager_survey, on='EmployeeID', how='outer')
        print(f"  • Fusionné: manager_survey")
    
    print(f"\n✅ Dataset final: {len(merged)} lignes, {len(merged.columns)} colonnes")
    
    return merged


def get_dataset_info(df: pd.DataFrame) -> Dict[str, any]:
    """
    Retourne des informations sur un DataFrame.
    
    Args:
        df: DataFrame à analyser.
        
    Returns:
        Dictionnaire contenant les informations du dataset.
    """
    info = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return info


def display_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Affiche un résumé du dataset.
    
    Args:
        df: DataFrame à résumer.
        name: Nom du dataset pour l'affichage.
    """
    print(f"\n📋 RÉSUMÉ: {name}")
    print("=" * 50)
    print(f"Dimensions: {len(df)} lignes × {len(df.columns)} colonnes")
    print(f"\nColonnes ({len(df.columns)}):")
    
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        missing_pct = missing / len(df) * 100
        
        if missing > 0:
            print(f"  {i:2}. {col:<30} ({dtype}) - ⚠️ {missing} NA ({missing_pct:.1f}%)")
        else:
            print(f"  {i:2}. {col:<30} ({dtype})")
    
    print("=" * 50)
