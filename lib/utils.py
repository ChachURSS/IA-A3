"""
Module utilitaires pour le projet d'analyse d'attrition.

Ce module contient des fonctions génériques réutilisables
dans l'ensemble du projet.
"""

import os
import logging
from typing import Any, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_file_exists(filepath: str) -> bool:
    """
    Vérifie si un fichier existe.
    
    Args:
        filepath: Chemin vers le fichier à vérifier.
        
    Returns:
        True si le fichier existe, False sinon.
    """
    exists = os.path.exists(filepath)
    if not exists:
        logger.warning(f"Le fichier '{filepath}' n'existe pas.")
    return exists


def get_data_directory() -> str:
    """
    Retourne le chemin vers le répertoire data du projet.
    
    Returns:
        Chemin absolu vers le répertoire data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, 'data')


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formate une valeur décimale en pourcentage.
    
    Args:
        value: Valeur à formater (0.15 pour 15%).
        decimals: Nombre de décimales à afficher.
        
    Returns:
        Chaîne formatée avec le symbole %.
    """
    return f"{value * 100:.{decimals}f}%"


def log_info(message: str) -> None:
    """
    Affiche un message d'information.
    
    Args:
        message: Message à afficher.
    """
    logger.info(message)


def log_warning(message: str) -> None:
    """
    Affiche un message d'avertissement.
    
    Args:
        message: Message à afficher.
    """
    logger.warning(message)


def log_error(message: str) -> None:
    """
    Affiche un message d'erreur.
    
    Args:
        message: Message à afficher.
    """
    logger.error(message)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division sécurisée qui retourne une valeur par défaut si le dénominateur est zéro.
    
    Args:
        numerator: Numérateur.
        denominator: Dénominateur.
        default: Valeur par défaut si division par zéro.
        
    Returns:
        Résultat de la division ou valeur par défaut.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def ensure_directory_exists(directory: str) -> None:
    """
    Crée un répertoire s'il n'existe pas.
    
    Args:
        directory: Chemin du répertoire à créer.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Répertoire créé : {directory}")
