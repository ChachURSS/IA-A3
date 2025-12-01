"""
Module src - Analyse du taux d'attrition des employés.

Ce module contient les composants principaux pour l'analyse :
- data_loader: Chargement et fusion des données
- data_preprocessing: Nettoyage et encodage des données
- feature_engineering: Création de nouvelles features
- models: Implémentation des modèles ML
- visualization: Fonctions de visualisation
"""

from . import data_loader
from . import data_preprocessing
from . import feature_engineering
from . import models
from . import visualization

__all__ = [
    'data_loader',
    'data_preprocessing', 
    'feature_engineering',
    'models',
    'visualization'
]
