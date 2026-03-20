import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocessing import clean_and_sort_data, scale_features
from dataset_builder import split_by_battery, temporal_train_val_split
from sliding_window import build_sliding_windows
from model_lstm import build_lstm_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_history(history, save_path: str):
    """
    Trace les courbes d'apprentissage et les sauvegarde.
    """
    plt.figure(figsize=(12, 5))
    
    # Courbe de perte (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Courbe d'erreur (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Courbes d'apprentissage sauvegardées dans {save_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'battery_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_save_path = os.path.join(models_dir, 'trained_model.keras')
    plot_save_path = os.path.join(models_dir, 'training_history.png')
    
    os.makedirs(models_dir, exist_ok=True)
    
    # ==========================
    # 1. Pipeline de Données
    # ==========================
    logging.info("--- Préparation des données ---")
    df = load_data(data_path)
    df = clean_and_sort_data(df)
    
    train_val_df, test_df = split_by_battery(df, test_battery_id='B0018')
    train_df, val_df = temporal_train_val_split(train_val_df, val_ratio=0.2)
    
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
    train_df_scaled, scaler = scale_features(train_df, feature_cols=features, scaler_path=scaler_path)
    
    val_df_scaled = val_df.copy()
    val_df_scaled[features] = scaler.transform(val_df[features])
    test_df_scaled = test_df.copy()
    test_df_scaled[features] = scaler.transform(test_df[features]) # on garde le test_df sous la main pour l'évaluation future
    
    window_size = 5
    target = 'SoH'
    
    X_train, y_train = build_sliding_windows(train_df_scaled, window_size, features, target)
    X_val, y_val = build_sliding_windows(val_df_scaled, window_size, features, target)
    
    # ==========================
    # 2. Construction du Modèle
    # ==========================
    logging.info("--- Construction du modèle LSTM ---")
    input_shape = (X_train.shape[1], X_train.shape[2]) # (window_size, features)
    model = build_lstm_model(input_shape)
    
    # ==========================
    # 3. Callbacks d'Entraînement
    # ==========================
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-5,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # ==========================
    # 4. Entraînement
    # ==========================
    logging.info("--- Début de l'entraînement ---")
    epochs = 50
    batch_size = 64
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # ==========================
    # 5. Sauvegarde et Visualisation
    # ==========================
    plot_history(history, plot_save_path)
    logging.info("--- Entraînement terminé ---")

if __name__ == '__main__':
    main()
