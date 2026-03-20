import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Construit l'architecture du réseau LSTM pour la prédiction de la dégradation.
    """
    logging.info(f"Construction du modèle orienté sur une input_shape = {input_shape}")
    
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(rate=0.2), # Prévention de l'overfitting
        LSTM(units=32, return_sequences=False),
        Dense(units=16, activation='relu'),
        Dense(units=1) # Sortie linéaire unique : le SoH prédit
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse', # Erreur Quadratique Moyenne (Régression)
        metrics=['mae'] # Erreur Absolue Moyenne
    )
    
    model.summary(print_fn=logging.info)
    return model
