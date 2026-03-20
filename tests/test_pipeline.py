import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocessing import clean_and_sort_data, scale_features
from dataset_builder import split_by_battery, temporal_train_val_split
from sliding_window import build_sliding_windows

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'battery_dataset.csv')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    print("--- 1. Chargement et Nettoyage ---")
    df = load_data(data_path)
    df = clean_and_sort_data(df)
    
    print("\n--- 2. Séparation Test (Leave One Battery Out) ---")
    # On isole B0018 pour le test final post-entraînement
    train_val_df, test_df = split_by_battery(df, test_battery_id='B0018')
    print(f"Lignes Train/Val: {len(train_val_df)} | Lignes Test: {len(test_df)}")
    
    print("\n--- 3. Séparation Train / Validation (Temporelle) ---")
    # Sur les batteries d'entraînement, on garde les 20 derniers % de chaque batterie pour la validation
    train_df, val_df = temporal_train_val_split(train_val_df, val_ratio=0.2)
    print(f"Lignes Train: {len(train_df)} | Lignes Val: {len(val_df)}")
    
    print("\n--- 4. Normalisation ---")
    # On scale d'abord sur le set d'entraînement ! Pour éviter un data leakage indirect
    # Le scaler fit sur `train_df`, puis transforme les trois sets
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
    
    train_df_scaled, scaler = scale_features(train_df, feature_cols=features, scaler_path=scaler_path)
    # Transformation uniquement pour val et test (sans fit)
    val_df_scaled = val_df.copy()
    val_df_scaled[features] = scaler.transform(val_df[features])
    test_df_scaled = test_df.copy()
    test_df_scaled[features] = scaler.transform(test_df[features])
    
    
    print("\n--- 5. Sliding Windows ---")
    window_size = 5 # Fenêtre un peu plus longue pour avoir plus de contexte temporel
    target = 'SoH'
    
    X_train, y_train = build_sliding_windows(train_df_scaled, window_size, features, target)
    X_val, y_val = build_sliding_windows(val_df_scaled, window_size, features, target)
    X_test, y_test = build_sliding_windows(test_df_scaled, window_size, features, target)
    
    print("\n=== Résumé des Tenseurs pour le LSTM ===")
    print(f"Train : X={X_train.shape}, y={y_train.shape}")
    print(f"Val   : X={X_val.shape}, y={y_val.shape}")
    print(f"Test  : X={X_test.shape}, y={y_test.shape}")

if __name__ == '__main__':
    main()
