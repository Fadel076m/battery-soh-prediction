# Prédiction du State of Health (SoH) des Batteries avec Deep Learning

Bienvenue dans ce projet de Data Science industriel visant à prédire l'état de santé (SoH) des batteries à l'aide d'un réseau de neurones récurrent (LSTM).

## 🚀 Objectif

Le but est d'estimer de manière fiable le SoH d'une batterie Lithium-Ion à partir de données de cycles de charge et de décharge (Voltage, Courant, Température, et SoC). Ces données sont séquentielles et traitées sous la forme de fenêtres temporelles (sliding windows).

## 📂 Structure du Projet

```
battery_soh_prediction/
├── app/                  # Application Streamlit
├── data/                 # Datasets bruts et traités
├── models/               # Modèles entraînés (.h5) et scalers (.pkl)
├── notebooks/            # Notebooks d'exploration (EDA)
├── src/                  # Code source (chargement, préprocessing, LSTM)
├── README.md             # Documentation
└── requirements.txt      # Dépendances Python
```

## ⚙️ Workflow et Avancement

- [x] **Étape 1 :** Configuration de l'environnement et création de l'arborescence
- [x] **Étape 2 :** Chargement et rapport de qualité des données
- [x] **Étape 3 :** Exploration des données (EDA)
- [x] **Étape 4 :** Nettoyage des données
- [x] **Étape 5 :** Fenêtres temporelles (Sliding Windows)
- [x] **Étape 6 :** Séparation Train/Test
- [x] **Étape 7 :** Modèle LSTM (Architecture & Entraînement)
- [x] **Étape 8 :** Évaluation des performances
- [x] **Étape 9 :** Déploiement du Dashboard Dash/Flask (UI Premium)

## 📊 Résultats Clés de l'Analyse Exploratoire (EDA)

- **SoH (State of Health) :** On observe une diminution continue du SoH au fil des cycles, confirmant la dégradation chimique des batteries avec le temps.
- **Corrélations :** Le State of Health est très fortement, et négativement, corrélé avec le nombre de cycles (sans surprise). D'autres variables de fonctionnement (Voltage, Température) affichent des comportements cycliques tout au long de la vie de la batterie.
- **Qualité :** Le jeu de données est exceptionnellement propre, sans aucune valeur aberrante majeure ni données manquantes nécessitant un pre-processing de nettoyage lourd.

## 🔧 Installation

```bash
# Activer l'environnement virtuel et installer les dépendances (y compris ipykernel pour Jupyter)
call .venv\Scripts\activate
pip install -r requirements.txt
pip install ipykernel jupyter
```

## 🧠 Méthodologie Anti-Leaking (Séparation des Données)
Dans l'entraînement de séries temporelles, un fractionnement aléatoire classique (ex : `train_test_split` de SciKit-Learn) provoque une perte de cohérence temporelle.
Pour prédire le SoH en conditions réelles, le modèle ne doit **jamais** s'entraîner sur la fin de vie d'une batterie pour prédire le début de vie d'une autre batterie similaire.
**Stratégie retenue dans ce projet :**
- Validation croisée ou séparation sur la base de la batterie entière (`Leave-One-Battery-Out`).
- Les données des batteries B0005, B0006 et B0007 forment le Training/Validation set. La B0018 est conservée comme Set de Test final jamais vu.
