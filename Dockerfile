FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le projet
COPY . .

# Hugging Face utilise le port 7860 par défaut
EXPOSE 7860

# Commande de démarrage avec Gunicorn (timeout augmenté pour le chargement du modèle)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app.dashboard:server"]
