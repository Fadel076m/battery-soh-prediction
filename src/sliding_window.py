import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_windows_for_cycle(cycle_data: pd.DataFrame, feature_cols: list, target_col: str, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Crée des fenêtres temporelles pour un cycle donné d'une batterie.
    """
    data_values = cycle_data[feature_cols].values
    target_value = cycle_data[target_col].iloc[0] # Le SoH est constant pour tout le cycle
    
    X, y = [], []
    
    # S'il y a moins de mesures dans ce cycle que la taille de fenêtre, on l'ignore
    if len(data_values) < window_size:
        return np.array([]), np.array([])
        
    # Découpage séquentiel
    for i in range(len(data_values) - window_size + 1):
        window = data_values[i:(i + window_size)]
        X.append(window)
        y.append(target_value)
        
    return np.array(X), np.array(y)

def build_sliding_windows(df: pd.DataFrame, window_size: int, feature_cols: list, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parcourt chaque batterie et chaque cycle pour générer les tenseurs X (séquences) et y (cibles).
    """
    logging.info(f"Création des séquences glissantes (window_size={window_size})...")
    X_list, y_list = [], []
    
    # On groupe par batterie et par cycle pour isoler chaque série temporelle
    groups = df.groupby(['battery_id', 'cycle_number'])
    
    # Barre de progression
    for (battery_id, cycle_number), group in tqdm(groups, desc="Découpage par cycle", unit="cycle"):
        X_cycle, y_cycle = create_windows_for_cycle(group, feature_cols, target_col, window_size)
        
        if len(X_cycle) > 0:
            X_list.append(X_cycle)
            y_list.append(y_cycle)
            
    # Concaténation finale pour créer les tenseurs 3D (X) et 1D (y)
    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)
    
    logging.info(f"Shape final du tenseur X (samples, window_size, features) : {X_final.shape}")
    logging.info(f"Shape final du vecteur y (samples) : {y_final.shape}")
    
    return X_final, y_final
