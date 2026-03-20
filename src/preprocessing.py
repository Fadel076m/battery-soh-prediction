import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_and_sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et trie les données chronologiquement par batterie et par cycle.
    Le tri est fondamental pour ne pas mélanger les séquences temporelles.
    """
    logging.info("Tri des données par battery_id et cycle_number...")
    df_sorted = df.sort_values(by=['battery_id', 'cycle_number']).reset_index(drop=True)
    
    initial_len = len(df_sorted)
    df_sorted = df_sorted.dropna()
    dropped = initial_len - len(df_sorted)
    if dropped > 0:
        logging.warning(f"{dropped} lignes (contenant des NaN) supprimées.")
        
    return df_sorted

def scale_features(df: pd.DataFrame, feature_cols: list, scaler_path: str = None) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalise les variables d'entrée avec un MinMaxScaler (de 0 à 1) 
    pour faciliter et stabiliser l'apprentissage du LSTM.
    """
    logging.info(f"Normalisation des features via MinMaxScaler : {feature_cols}")
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    
    # Ajustement et transformation des colonnes spécifiées
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler sauvegardé dans {scaler_path}")
        
    return df_scaled, scaler
