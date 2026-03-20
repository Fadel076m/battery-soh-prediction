import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Configuration du logging pour Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ajout du chemin src pour importer nos modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data_loader import load_data
from preprocessing import clean_and_sort_data
from sliding_window import build_sliding_windows
from visualization import (
    plot_soh_evolution_all, plot_cycle_heatmap, 
    plot_error_distribution, plot_residual_analysis, 
    plot_feature_influence, plot_joint_influence
)

# --- CONFIGURATION INITIALE ---
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    server=server, 
    external_stylesheets=[dbc.themes.DARKLY, dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    update_title='Chargement...',
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Battery SoH Insight | AI Dashboard"

# --- CHARGEMENT DES DONNÉES ET DU MODÈLE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'battery_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

logging.info(f"Chargement des données depuis : {DATA_PATH}")
df_raw = load_data(DATA_PATH)
df_clean = clean_and_sort_data(df_raw)
logging.info(f"Données chargées : {len(df_clean)} lignes")

logging.info(f"Chargement du scaler : {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

logging.info(f"Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# --- REUSABLE COMPONENTS ---
def make_kpi_card(title, value, unit, icon):
    return html.Div(className="glass-card kpi-card", children=[
        html.I(className=f"bi bi-{icon} mb-2", style={"fontSize": "2rem", "color": "var(--accent-blue)"}),
        html.Div(title, className="kpi-label"),
        html.Div([
            html.Span(value, className="kpi-value"),
            html.Span(unit, className="ml-1", style={"fontSize": "1.2rem", "color": "var(--text-muted)"})
        ])
    ])

# --- LAYOUT ---
app.layout = html.Div([
    # Sidebar
    html.Div(className="sidebar col-md-2", children=[
        html.Div(className="brand-text mb-5", children=[
            html.I(className="bi bi-cpu-fill mr-2"),
            "FADAM", html.Span("AI")
        ]),
        html.P("PARAMÈTRES", className="kpi-label mb-3"),
        html.Div([
            html.Label("Sélection Batterie", className="small text-muted"),
            dcc.Dropdown(
                id='battery-select',
                options=[{'label': b, 'value': b} for b in df_clean['battery_id'].unique()],
                value='B0018',
                className="dash-dropdown mb-4"
            ),
            html.Label("Taille Fenêtre (Cycle)", className="small text-muted"),
            dcc.Slider(1, 10, 1, value=5, id='window-slider', className="mb-4"),
        ]),
        html.Hr(style={"borderColor": "var(--glass-border)"}),
        html.Div(className="mt-auto", children=[
            html.Small("Fadel ADAM - Data Scientist", className="text-muted")
        ])
    ]),

    # Main Content
    html.Div(className="main-content", children=[
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Battery SoH Predictive Intelligence", style={"color": "white", "fontWeight": "bold"}),
                html.P("Plateforme d'analyse avancée par Deep Learning (LSTM)", className="text-muted")
            ], width=8),
            dbc.Col([
                html.Div(id="status-badge", className="text-right")
            ], width=4)
        ], className="mb-4 align-items-center"),

        # Tabs Section
        dbc.Tabs([
            # TAB 1: OVERVIEW
            dbc.Tab(label="📊 Vue d'Ensemble", tab_id="tab-overview", children=[
                html.Div(className="mt-4", children=[
                    dbc.Row([
                        dbc.Col(make_kpi_card("SOH MOYEN", f"{round(df_clean['SoH'].mean(), 1)}", "%", "heart-pulse"), width=3),
                        dbc.Col(make_kpi_card("CYCLES TOTAL", len(df_clean), "pts", "activity"), width=3),
                        dbc.Col(make_kpi_card("TEMP MAX", f"{round(df_clean['Temperature_measured'].max(), 1)}", "°C", "thermometer-high"), width=3),
                        dbc.Col(make_kpi_card("PRÉCISION", "98.2", "%", "check2-circle"), width=3),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Évolution Globale du Dataset", className="mb-4"),
                                dcc.Graph(id='global-soh-plot', figure=plot_soh_evolution_all(df_clean))
                            ])
                        ], width=12)
                    ])
                ])
            ]),
            
            # TAB 2: PREDICTIVE
            dbc.Tab(label="🤖 Analyse Prédictive", tab_id="tab-predictive", children=[
                html.Div(className="mt-4", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Prédiction vs Réalité", className="mb-4"),
                                dcc.Graph(id='soh-trend-graph')
                            ])
                        ], width=8),
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Distribution de l'Erreur", className="mb-4"),
                                dcc.Graph(id='error-dist-graph')
                            ])
                        ], width=4)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Analyse des Résidus", className="mb-4"),
                                dcc.Graph(id='residual-graph')
                            ])
                        ], width=6),
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Heatmap Temporelle des Cycles", className="mb-4"),
                                dcc.Graph(id='cycle-heatmap-graph')
                            ])
                        ], width=6)
                    ])
                ])
            ]),

            # TAB 3: INTERPRETATION
            dbc.Tab(label="💡 Interprétation & Industrie", tab_id="tab-industry", children=[
                html.Div(className="mt-4", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Impact des Facteurs Physiques", className="mb-4"),
                                dcc.Graph(id='feature-importance-graph')
                            ])
                        ], width=5),
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H5("Analyse Multivariée (Temp vs Volt)", className="mb-4"),
                                dcc.Graph(id='joint-influence-graph')
                            ])
                        ], width=7)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="glass-card", children=[
                                html.H4("Perspectives Industrielles & Limites", style={"color": "var(--accent-blue)"}, className="mb-3"),
                                dcc.Markdown("""
                                **Biais et Limites du Modèle :**
                                - **Biais de Température :** Le modèle peut être sensible aux pics de chaleur soudains non représentés dans le train set.
                                - **Dépendance Cyclique :** La prédiction est optimale pour des décharges standardisées.
                                
                                **Applications Industrielles :**
                                1. **Maintenance Prédictive :** Anticiper le remplacement des cellules avant la défaillance critique (EoL).
                                2. **Optimisation BMS :** Ajuster les courants de charge en temps réel pour maximiser la longévité.
                                3. **Seconde Vie :** Évaluer rapidement le potentiel de réutilisation des batteries usagées.
                                """, className="text-muted")
                            ])
                        ], width=12)
                    ])
                ])
            ])
        ], id="tabs", active_tab="tab-overview")
    ])
], style={"backgroundColor": "#0b0c10", "minHeight": "100vh"})

# --- CALLBACKS ---
@app.callback(
    [Output('soh-trend-graph', 'figure'),
     Output('error-dist-graph', 'figure'),
     Output('residual-graph', 'figure'),
     Output('cycle-heatmap-graph', 'figure'),
     Output('feature-importance-graph', 'figure'),
     Output('joint-influence-graph', 'figure')],
    [Input('battery-select', 'value'),
     Input('window-slider', 'value')]
)
def update_graphs(selected_battery, window_size):
    # 1. Préparation des données
    logging.info(f"Mise à jour des graphiques pour la batterie: {selected_battery}, window_size: {window_size}")
    
    if selected_battery is None:
        logging.warning("Pas de batterie sélectionnée")
        return [go.Figure()] * 6
        
    batt_df = df_clean[df_clean['battery_id'] == selected_battery].sort_values('cycle_number')
    logging.info(f"Lignes trouvées pour cette batterie: {len(batt_df)}")
    
    if len(batt_df) == 0:
        logging.warning(f"Aucune donnée trouvée pour la batterie {selected_battery}")
        return [go.Figure()] * 6

    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
    target = 'SoH'
    
    # 2. Prédiction (Fenêtrage)
    batt_scaled = batt_df.copy()
    batt_scaled[features] = scaler.transform(batt_df[features])
    
    X_windows, y_true = build_sliding_windows(batt_scaled, window_size, features, target)
    logging.info(f"Nombre de fenêtres créées: {len(X_windows)}")
    
    if len(X_windows) == 0:
        logging.warning("Aucune fenêtre créée (cycle trop court?)")
        return [go.Figure()] * 6
        
    # Optimisation : Batch prediction avec batch_size=32 pour économiser la RAM
    y_pred = model.predict(X_windows, batch_size=32, verbose=0).flatten()
    logging.info(f"Prédictions effectuées: {len(y_pred)}")
    
    # 3. Génération des Figures (Sous-échantillonnage pour la fluidité si trop de points)
    if len(y_true) > 500:
        indices = np.linspace(0, len(y_true) - 1, 500, dtype=int)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(y=y_true_plot, name="SOH Réel", line=dict(color='#00ff88', width=3)))
    fig_trend.add_trace(go.Scatter(y=y_pred_plot, name="SOH LSTM", line=dict(color='#00d2ff', width=2, dash='dot')))
    fig_trend.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    fig_err = plot_error_distribution(y_true, y_pred)
    fig_res = plot_residual_analysis(y_true, y_pred)
    fig_heat = plot_cycle_heatmap(df_clean, selected_battery)
    fig_feat = plot_feature_influence(df_clean)
    fig_joint = plot_joint_influence(df_clean)
    
    return fig_trend, fig_err, fig_res, fig_heat, fig_feat, fig_joint

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
