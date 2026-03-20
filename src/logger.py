import logging
import sys

def setup_logger(name="battery_prediction", level=logging.INFO):
    """
    Configure un logger standard pour le projet.
    
    Args:
        name (str): Nom du logger.
        level (int): Niveau de logging (ex: logging.INFO).
        
    Returns:
        logging.Logger: Le logger configuré.
    """
    logger = logging.getLogger(name)
    
    # Éviter d'ajouter plusieurs handlers si le logger est déjà configuré
    if not logger.handlers:
        logger.setLevel(level)
        
        # Format du message de log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler pour la sortie console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
