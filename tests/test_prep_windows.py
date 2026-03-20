import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocessing import clean_and_sort_data, scale_features
from sliding_window import build_sliding_windows

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'battery_dataset.csv')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    print("--- 1. Chargement ---")
    df = load_data(data_path)
    
    print("\n--- 2. Nettoyage ---")
    df = clean_and_sort_data(df)
    
    print("\n--- 3. Normalisation ---")
    # On inclut 'cycle_number' dans les features car c'est une indication forte pour le modèle
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
    target = 'SoH'
    
    df_scaled, scaler = scale_features(df, feature_cols=features, scaler_path=scaler_path)
    print(f"Features scalées. Exemple de taille : {df_scaled.shape}")
    
    print("\n--- 4. Sliding Windows ---")
    window_size = 3
    X, y = build_sliding_windows(df_scaled, window_size=window_size, feature_cols=features, target_col=target)
    
    print("\n--- Test réussi ---")
    print(f"La dimension de X devrait être (samples, 3, 5). Résultat: {X.shape}")
    print(f"La dimension de y devrait être (samples,). Résultat: {y.shape}")

if __name__ == '__main__':
    main()
