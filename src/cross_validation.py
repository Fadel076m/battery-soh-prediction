import pandas as pd
import numpy as np
import logging
import os
import tensorflow as tf
from src.dataset_builder import split_by_battery, temporal_train_val_split
from src.preprocessing import scale_features
from src.sliding_window import build_sliding_windows
from src.model_lstm import build_lstm_model
from src.evaluation import evaluate_metrics

def run_lobo_cv(df, features, target, window_size=5, epochs=10, batch_size=64, subset_limit=None):
    """
    Exécute une validation croisée Leave-One-Battery-Out (LOBO).
    """
    batteries = df['battery_id'].unique()
    if subset_limit:
        batteries = batteries[:subset_limit]
        logging.info(f"Limitation de la CV à {subset_limit} batteries pour gagner du temps.")

    results = []
    
    for test_b_id in batteries:
        logging.info(f"\n>>> ITERATION CV : Test sur la batterie {test_b_id} <<<")
        
        # 1. Split par batterie
        train_val_df, test_df = split_by_battery(df, test_battery_id=test_b_id)
        
        # 2. Split temporel (train/val) sur le reste
        train_df, val_df = temporal_train_val_split(train_val_df, val_ratio=0.2)
        
        # 3. Normalisation
        train_df_scaled, scaler = scale_features(train_df, feature_cols=features)
        val_df_scaled = val_df.copy()
        val_df_scaled[features] = scaler.transform(val_df[features])
        test_df_scaled = test_df.copy()
        test_df_scaled[features] = scaler.transform(test_df[features])
        
        # 4. Fenêtrage
        X_train, y_train = build_sliding_windows(train_df_scaled, window_size, features, target)
        X_val,   y_val   = build_sliding_windows(val_df_scaled,   window_size, features, target)
        X_test,  y_test  = build_sliding_windows(test_df_scaled,  window_size, features, target)
        
        # 5. Modélisation
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        
        # 6. Entraînement (silencieux pour la CV)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # 7. Évaluation
        y_pred = model.predict(X_test, verbose=0)
        metrics = evaluate_metrics(y_test.flatten(), y_pred.flatten())
        metrics['battery_id'] = test_b_id
        results.append(metrics)
        logging.info(f"Résultats pour {test_b_id} : MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
        
    return pd.DataFrame(results)

def train_final_model(df, features, target, window_size=5, epochs=50, batch_size=64, model_path="models/trained_model.keras", scaler_path="models/scaler.pkl"):
    """
    Entraîne le modèle final et le sauvegarde.
    """
    logging.info("--- Entraînement du modèle FINAL ---")
    
    # Entraînement sur tout sauf B0018 (pour la visualisation ultérieure)
    train_val_df, _ = split_by_battery(df, test_battery_id='B0018')
    train_df, val_df = temporal_train_val_split(train_val_df, val_ratio=0.2)
    
    train_df_scaled, scaler = scale_features(train_df, feature_cols=features, scaler_path=scaler_path)
    val_df_scaled = val_df.copy()
    val_df_scaled[features] = scaler.transform(val_df[features])
    
    X_train, y_train = build_sliding_windows(train_df_scaled, window_size, features, target)
    X_val,   y_val   = build_sliding_windows(val_df_scaled,   window_size, features, target)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
        ]
    )
    logging.info(f"Modèle final sauvegardé dans {model_path}")
    return model
