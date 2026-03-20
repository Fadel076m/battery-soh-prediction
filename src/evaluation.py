import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocessing import clean_and_sort_data, scale_features
from dataset_builder import split_by_battery
from sliding_window import build_sliding_windows

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_theme(style='whitegrid')

def evaluate_metrics(y_true, y_pred) -> dict:
    """
    Calcule MAE, RMSE et R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_evaluation(y_true, y_pred, save_dir: str):
    """
    Trace le scatter plot (réel vs prédit) et l'évolution de l'erreur.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Scatter Plot : Réel vs Prédit
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='royalblue')
    
    # Ligne idéale y = x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Idéal (y=x)')
    
    plt.title('Prédictions SoH vs Réalité (Test Set)')
    plt.xlabel('SoH Réel')
    plt.ylabel('SoH Prédit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_real_vs_pred.png'))
    plt.close()
    
    # 2. Évolution temporelle (SoH réel vs SoH prédit par cycle)
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='SoH Réel', color='green', lw=2)
    plt.plot(y_pred, label='SoH Prédit', color='orange', alpha=0.8, lw=2)
    plt.title('Suivi de la dégradation : Réel vs Prédit')
    plt.xlabel('Séquences Temporelles (Fenêtres)')
    plt.ylabel('SoH')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_real_vs_pred.png'))
    plt.close()
    
    # 3. Distribution des erreurs
    errors = y_true - y_pred.flatten()
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=50, kde=True, color='crimson')
    plt.title('Distribution des résidus (Erreurs)')
    plt.xlabel('Erreur (Réel - Prédit)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'battery_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'trained_model.keras')
    results_dir = os.path.join(base_dir, 'app', 'assets') # pour le dashboard futur ou ici
    if not os.path.exists(results_dir):
        results_dir = os.path.join(base_dir, 'models', 'evaluation_results')
        
    logging.info("--- Chargement des données de Test ---")
    df = load_data(data_path)
    df = clean_and_sort_data(df)
    
    # On ressort B0018 (notre set de test)
    _, test_df = split_by_battery(df, test_battery_id='B0018')
    
    # Scaler
    if not os.path.exists(scaler_path):
        logging.error("Scaler introuvable !, Veuillez lancer l'entraînement d'abord.")
        return
        
    scaler = joblib.load(scaler_path)
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
    
    test_df_scaled = test_df.copy()
    test_df_scaled[features] = scaler.transform(test_df[features])
    
    window_size = 5
    target = 'SoH'
    X_test, y_test = build_sliding_windows(test_df_scaled, window_size, features, target)
    
    logging.info(f"Shape du set de Test : X={X_test.shape}, y={y_test.shape}")
    
    logging.info("--- Chargement du Modèle LSTM ---")
    if not os.path.exists(model_path):
        logging.error("Modèle H5 introuvable !")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    logging.info("--- Prédiction sur le Test Set (B0018) ---")
    y_pred = model.predict(X_test)
    
    logging.info("--- Calcul des Métriques ---")
    # Aplatir les arrays pour sklearn
    y_test_f = y_test.flatten()
    y_pred_f = y_pred.flatten()
    
    metrics = evaluate_metrics(y_test_f, y_pred_f)
    print("\n" + "="*40)
    print("🏆 RÉSULTATS DE L'ÉVALUATION (B0018) 🏆")
    print("="*40)
    print(f"-> MAE  : {metrics['MAE']:.4f}")
    print(f"-> RMSE : {metrics['RMSE']:.4f}")
    print(f"-> R²   : {metrics['R2']:.4f}")
    print("="*40 + "\n")
    
    logging.info("--- Génération des Graphiques ---")
    plot_evaluation(y_test_f, y_pred_f, save_dir=results_dir)
    logging.info(f"Graphiques sauvegardés dans {results_dir}")

if __name__ == '__main__':
    main()
