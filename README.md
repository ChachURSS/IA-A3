# Bloc-IA-A3 - Analyse du Taux d'Attrition HumanForYou

## Contexte

L'entreprise pharmaceutique **HumanForYou** (basée en Inde, ~4000 employés) connaît un taux de rotation d'environ 15% par an. La direction souhaite identifier les facteurs influençant ce taux et proposer des pistes d'amélioration pour fidéliser les employés.

## Objectifs du Projet

1. 📊 Explorer et analyser les données des employés
2. 🔍 Identifier les facteurs clés d'attrition
3. 🤖 Construire des modèles prédictifs (Logistic Regression, Random Forest, XGBoost, SVM)
4. 📈 Évaluer et comparer les performances
5. 💡 Proposer des recommandations

## Structure du Projet

```
├── livrables/
│   ├── 01_ethique.md                    # Document éthique (7 exigences UE)
│   ├── 02_bibliographie.md              # Références académiques et techniques
│   └── 03_presentation_notebook.ipynb   # Notebook Jupyter avec benchmarks
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Chargement et fusion des données
│   ├── data_preprocessing.py            # Nettoyage, gestion des NA, encodage
│   ├── feature_engineering.py           # Création de nouvelles features
│   ├── models.py                        # Implémentation des modèles ML
│   └── visualization.py                 # Fonctions de visualisation
├── lib/
│   ├── __init__.py
│   └── utils.py                         # Fonctions utilitaires
├── data/
│   ├── employee_survey_data.csv         # Enquête qualité de vie
│   ├── manager_survey_data.csv          # Évaluation manager
│   └── .gitkeep                         # Pour les fichiers volumineux
├── requirements.txt
└── README.md
```

## Données Disponibles

### Fichiers inclus dans le dépôt
- **employee_survey_data.csv** : Enquête qualité de vie (EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance)
- **manager_survey_data.csv** : Évaluation manager (JobInvolvement, PerformanceRating)

### Fichiers à ajouter manuellement (trop volumineux)
- **general_data.csv** : Données générales des employés (Age, Attrition, MonthlyIncome, etc.)
- **in_out_time.zip** : Données de badgeage (entrées/sorties 2015)

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ChachURSS/Bloc-IA-A3.git
cd Bloc-IA-A3

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Exécuter le notebook
```bash
cd livrables
jupyter notebook 03_presentation_notebook.ipynb
```

### Utiliser les modules Python
```python
from src.data_loader import load_all_data, merge_datasets
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.models import train_and_evaluate_all_models
from src.visualization import plot_roc_curves, plot_feature_importance
```

## Livrables

### 1. Document Éthique (01_ethique.md)
Analyse selon les 7 exigences de la Commission Européenne pour une IA digne de confiance :
- Respect de l'autonomie humaine
- Robustesse technique et sécurité
- Confidentialité et gouvernance des données
- Transparence
- Diversité, non-discrimination et équité
- Bien-être environnemental et sociétal
- Responsabilité

### 2. Bibliographie (02_bibliographie.md)
Références classées par thématique :
- Sources méthodologiques et théoriques
- Sources techniques (ML, Python, etc.)
- Sources éthiques et réglementaires
- Sources spécifiques au projet RH/Attrition

### 3. Notebook de Présentation (03_presentation_notebook.ipynb)
Analyse complète incluant :
- Chargement et exploration des données
- Analyse exploratoire (EDA) avec visualisations
- Prétraitement des données
- Feature engineering
- Entraînement de modèles (Logistic Regression, Random Forest, XGBoost, SVM)
- Benchmarks et métriques (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Interprétabilité (Feature Importance, SHAP)
- Conclusions et recommandations

## Modèles Implémentés

| Modèle | Description |
|--------|-------------|
| Logistic Regression | Modèle de base, interprétable |
| Random Forest | Ensemble de décision, robuste |
| XGBoost | Gradient boosting, performant |
| SVM | Support Vector Machine |

## Métriques d'Évaluation

- **Accuracy** : Précision globale
- **Precision** : Proportion de vrais positifs parmi les prédictions positives
- **Recall** : Proportion de vrais positifs détectés
- **F1-Score** : Moyenne harmonique Precision/Recall
- **AUC-ROC** : Aire sous la courbe ROC

## Dépendances

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
shap>=0.41.0
imbalanced-learn>=0.10.0
```

## Auteurs

Projet réalisé dans le cadre du Bloc IA A3.

## License

MIT License
