import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_by_battery(df: pd.DataFrame, test_battery_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Séparation LOB (Leave One Battery). On réserve une batterie complète pour le test.
    """
    logging.info(f"Séparation des données : Batterie test => {test_battery_id}")
    
    test_df = df[df['battery_id'] == test_battery_id].copy()
    train_df = df[df['battery_id'] != test_battery_id].copy()
    
    return train_df, test_df

def temporal_train_val_split(train_df: pd.DataFrame, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare temporellement par batterie. Pour chaque batterie d'entraînement,
    les (1-val_ratio)% premiers cycles vont au train, les derniers vont au val.
    """
    val_dfs = []
    train_dfs = []
    
    batteries = train_df['battery_id'].unique()
    
    for b_id in batteries:
        b_data = train_df[train_df['battery_id'] == b_id]
        
        # Leurs cycles sont déjà triés grâce à preprocessing.py
        total_cycles = b_data['cycle_number'].nunique()
        split_idx = int(total_cycles * (1 - val_ratio))
        
        # Trouver le numéro de cycle correspondant au split
        cycles = sorted(b_data['cycle_number'].unique())
        split_cycle = cycles[split_idx]
        
        # Sépare chronologiquement la batterie
        b_train = b_data[b_data['cycle_number'] <= split_cycle]
        b_val = b_data[b_data['cycle_number'] > split_cycle]
        
        train_dfs.append(b_train)
        val_dfs.append(b_val)
        
    logging.info(f"Création validation temporelle avec un ratio de la fin des cycles : {val_ratio}")
    return pd.concat(train_dfs), pd.concat(val_dfs)
