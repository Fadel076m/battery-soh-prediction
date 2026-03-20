import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔋 Exploration des données (EDA) - Prédiction du SoH\n",
    "\n",
    "Ce notebook contient l'analyse exploratoire de notre jeu de données sur les batteries. L'objectif est de comprendre le comportement des différentes variables au fil des cycles de charge/décharge."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import plotly.express as px\n",
    "\n",
    "# Configuration de l'affichage\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "plt.rcParams['figure.figsize'] = (12, 6)\n",
    "\n",
    "# Chargement via notre module\n",
    "sys.path.append(os.path.abspath('../src'))\n",
    "from data_loader import load_data\n",
    "\n",
    "df = load_data('../data/battery_dataset.csv')\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Distribution des variables continues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
    "fig.suptitle('Distribution des variables continues', fontsize=16)\n",
    "\n",
    "vars_to_plot = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'SoH']\n",
    "for ax, var in zip(axes.flatten()[:5], vars_to_plot):\n",
    "    sns.histplot(df[var], bins=50, kde=True, ax=ax, color='steelblue')\n",
    "    ax.set_title(f'Distribution de {var}')\n",
    "\n",
    "axes.flatten()[5].axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Évolution du SoH par cycle\n",
    "Le SoH est notre variable cible. Voyons comment il se dégrade au fil des cycles pour chaque batterie."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = px.line(df, x='cycle_number', y='SoH', color='battery_id', \n",
    "              title='Évolution de l\\'État de Santé (SoH) au fil des cycles',\n",
    "              labels={'cycle_number': 'Nombre de cycles', 'SoH': 'State of Health (SoH)'})\n",
    "fig.update_layout(hovermode='x unified')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Évolution des grandeurs physiques par cycle\n",
    "Regardons comment évoluent la **Tension**, la **Température**, le **Courant** et le **SoC** sur la durée de vie de la batterie."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "batteries = df['battery_id'].unique()\n",
    "\n",
    "fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)\n",
    "fig.suptitle('Évolution des variables par cycle pour chaque batterie', fontsize=16)\n",
    "\n",
    "vars_to_plot = ['Voltage_measured', 'Temperature_measured', 'Current_measured', 'SoC']\n",
    "colors = sns.color_palette('husl', len(batteries))\n",
    "\n",
    "for ax, var in zip(axes, vars_to_plot):\n",
    "    sns.lineplot(data=df, x='cycle_number', y=var, hue='battery_id', ax=ax, palette=colors, alpha=0.7)\n",
    "    ax.set_title(f'{var} vs Cycle')\n",
    "    ax.set_ylabel(var)\n",
    "\n",
    "axes[-1].set_xlabel('Cycle Number')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Analyse des corrélations\n",
    "Quelles sont les variables les plus corrélées à la dégradation du SoH ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "corr_matrix = df.drop(columns=['battery_id']).corr()\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, square=True)\n",
    "plt.title('Heatmap des Corrélations', fontsize=16)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Zoom sur le SoH :\n",
    "Isolons les variables qui impactent le plus notre cible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "soh_corr = corr_matrix[['SoH']].sort_values(by='SoH', ascending=False)\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.barplot(x=soh_corr['SoH'], y=soh_corr.index, palette='viridis')\n",
    "plt.title('Corrélation des variables avec le SoH')\n",
    "plt.xlabel('Coefficient de corrélation de Pearson')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion de l'EDA\n",
    "- Le SoH diminue de manière continue, confirmant la dégradation.\n",
    "- La Tension, Température et le Courant subissent des variations liées au vieillissement de la batterie.\n",
    "- Le *cycle_number* est fortement corrélé négativement au SoH, ce qui est logique : plus on avance dans les cycles, plus la santé décline.\n",
    "\n",
    "-> Prochaine étape : **Nettoyage et Création des Sliding Windows pour préparer les entrées temporelles du modèle LSTM.**"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks', 'exploration.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook généré avec succès.")
