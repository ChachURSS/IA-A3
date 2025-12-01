"""
Module d'implémentation des modèles de Machine Learning.

Ce module contient les classes et fonctions pour entraîner,
évaluer et comparer différents modèles de classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import warnings

warnings.filterwarnings('ignore')

# Importer XGBoost si disponible
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost non disponible. Installez-le avec: pip install xgboost")


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Retourne un dictionnaire des modèles à entraîner.
    
    Args:
        random_state: Graine aléatoire pour la reproductibilité.
        
    Returns:
        Dictionnaire {nom: instance du modèle}.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        ),
        'SVM': SVC(
            random_state=random_state,
            class_weight='balanced',
            probability=True
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=5  # Pour déséquilibre de classes
        )
    
    return models


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> Any:
    """
    Entraîne un modèle sur les données.
    
    Args:
        model: Instance du modèle à entraîner.
        X_train: Features d'entraînement.
        y_train: Labels d'entraînement.
        model_name: Nom du modèle pour l'affichage.
        
    Returns:
        Modèle entraîné.
    """
    print(f"  🏋️ Entraînement {model_name}...", end=" ")
    model.fit(X_train, y_train)
    print("✅")
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Évalue un modèle sur les données de test.
    
    Args:
        model: Modèle entraîné.
        X_test: Features de test.
        y_test: Labels de test.
        model_name: Nom du modèle pour l'affichage.
        
    Returns:
        Dictionnaire des métriques.
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculer les métriques
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_test, y_proba)
        metrics['Average Precision'] = average_precision_score(y_test, y_proba)
    
    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'f1'
) -> Dict[str, float]:
    """
    Effectue une validation croisée sur le modèle.
    
    Args:
        model: Instance du modèle.
        X: Features.
        y: Labels.
        cv: Nombre de folds.
        scoring: Métrique de scoring.
        
    Returns:
        Dictionnaire avec les scores de CV.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }


def get_hyperparameter_grids() -> Dict[str, Dict]:
    """
    Retourne les grilles d'hyperparamètres pour chaque modèle.
    
    Returns:
        Dictionnaire des grilles par modèle.
    """
    grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    
    if XGBOOST_AVAILABLE:
        grids['XGBoost'] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    
    return grids


def hyperparameter_tuning(
    model: Any,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    scoring: str = 'f1',
    model_name: str = "Model"
) -> Tuple[Any, Dict]:
    """
    Effectue une recherche d'hyperparamètres avec GridSearchCV.
    
    Args:
        model: Instance du modèle.
        param_grid: Grille d'hyperparamètres.
        X_train: Features d'entraînement.
        y_train: Labels d'entraînement.
        cv: Nombre de folds.
        scoring: Métrique de scoring.
        model_name: Nom du modèle.
        
    Returns:
        Tuple (meilleur modèle, meilleurs paramètres).
    """
    print(f"  🔧 Tuning {model_name}...", end=" ")
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def train_and_evaluate_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tune_hyperparameters: bool = False
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Entraîne et évalue tous les modèles.
    
    Args:
        X_train: Features d'entraînement.
        X_test: Features de test.
        y_train: Labels d'entraînement.
        y_test: Labels de test.
        tune_hyperparameters: Si True, effectue le tuning des hyperparamètres.
        
    Returns:
        Tuple (dictionnaire des modèles, DataFrame des résultats).
    """
    print("=" * 60)
    print("🚀 ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
    print("=" * 60)
    
    models = get_models()
    trained_models = {}
    results = []
    
    for name, model in models.items():
        print(f"\n📊 {name}")
        print("-" * 40)
        
        # Tuning des hyperparamètres si demandé
        if tune_hyperparameters:
            grids = get_hyperparameter_grids()
            if name in grids:
                model, best_params = hyperparameter_tuning(
                    model, grids[name], X_train, y_train, model_name=name
                )
                print(f"     Meilleurs params: {best_params}")
        
        # Entraînement
        model = train_model(model, X_train, y_train, name)
        trained_models[name] = model
        
        # Évaluation
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics['Model'] = name
        results.append(metrics)
        
        # Afficher les métriques
        print(f"     Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"     Precision: {metrics['Precision']:.4f}")
        print(f"     Recall:    {metrics['Recall']:.4f}")
        print(f"     F1-Score:  {metrics['F1-Score']:.4f}")
        if 'AUC-ROC' in metrics:
            print(f"     AUC-ROC:   {metrics['AUC-ROC']:.4f}")
    
    # Créer le DataFrame des résultats
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('Model')
    
    # Trier par F1-Score
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + "=" * 60)
    print("📈 COMPARAISON DES MODÈLES (trié par F1-Score)")
    print("=" * 60)
    print(results_df.round(4).to_string())
    
    return trained_models, results_df


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model"
) -> Optional[pd.DataFrame]:
    """
    Extrait l'importance des features d'un modèle.
    
    Args:
        model: Modèle entraîné.
        feature_names: Liste des noms de features.
        model_name: Nom du modèle.
        
    Returns:
        DataFrame avec les importances ou None.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def get_predictions_with_probabilities(
    model: Any,
    X: pd.DataFrame,
    employee_ids: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Obtient les prédictions avec les probabilités.
    
    Args:
        model: Modèle entraîné.
        X: Features.
        employee_ids: IDs des employés (optionnel).
        
    Returns:
        DataFrame avec les prédictions.
    """
    predictions = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = predictions.astype(float)
    
    result = pd.DataFrame({
        'Prediction': predictions,
        'Probability': probabilities,
        'Risk_Level': pd.cut(
            probabilities,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
    })
    
    if employee_ids is not None:
        result.insert(0, 'EmployeeID', employee_ids.values)
    
    return result


def get_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> str:
    """
    Génère un rapport de classification complet.
    
    Args:
        model: Modèle entraîné.
        X_test: Features de test.
        y_test: Labels de test.
        
    Returns:
        Rapport de classification en string.
    """
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=['Stay', 'Leave'])


def get_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> np.ndarray:
    """
    Calcule la matrice de confusion.
    
    Args:
        model: Modèle entraîné.
        X_test: Features de test.
        y_test: Labels de test.
        
    Returns:
        Matrice de confusion.
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_roc_curve_data(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les données de la courbe ROC.
    
    Args:
        model: Modèle entraîné.
        X_test: Features de test.
        y_test: Labels de test.
        
    Returns:
        Tuple (fpr, tpr, thresholds).
    """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test).astype(float)
    
    return roc_curve(y_test, y_proba)
