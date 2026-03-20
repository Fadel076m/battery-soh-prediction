import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_soh_evolution_all(df, save_dir=None):
    """
    Évolution du SoH pour toutes les batteries du dataset.
    """
    fig = px.line(df, x='cycle_number', y='SoH', color='battery_id',
                 title='📈 Évolution du State of Health (SoH) par batterie',
                 labels={'cycle_number': 'Nombre de Cycles', 'SoH': 'SoH (%)'},
                 template='plotly_dark')
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    
    if save_dir:
        fig.write_html(os.path.join(save_dir, 'soh_evolution_all.html'))
    return fig

def plot_cycle_heatmap(df, battery_id='B0005'):
    """
    Heatmap montrant l'évolution des caractéristiques par cycle.
    """
    batt_df = df[df['battery_id'] == battery_id].sort_values('cycle_number')
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC']
    
    # Normalisation pour la heatmap (0-1)
    df_norm = batt_df[features].copy()
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
    
    fig = go.Figure(data=go.Heatmap(
        z=df_norm.T.values,
        x=batt_df['cycle_number'],
        y=features,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=f'🔥 Heatmap des Cycles - Batterie {battery_id}',
        xaxis_title='Numéro de Cycle',
        yaxis_title='Variables (Normalisées)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_error_distribution(y_true, y_pred):
    """
    Distribution des erreurs de prédiction.
    """
    errors = y_true - y_pred
    fig = px.histogram(x=errors, nbins=30, marginal="rug", 
                       title="📊 Distribution des Erreurs de Prédiction",
                       labels={'x': 'Erreur (Réel - Prédit)', 'y': 'Fréquence'},
                       color_discrete_sequence=['#00d2ff'],
                       template='plotly_dark')
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_residual_analysis(y_true, y_pred):
    """
    Analyse des résidus perfectionnée.
    """
    residuals = y_true - y_pred
    fig = go.Figure()
    
    # Points de résidus
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode='markers',
        marker=dict(color='#ff9f43', size=8, opacity=0.6,
                   line=dict(width=1, color='white')),
        name='Résidus'
    ))
    
    # Ligne horizontale à zero
    fig.add_shape(type='line', x0=min(y_pred), y0=0, x1=max(y_pred), y1=0,
                 line=dict(color='#ff4757', width=2, dash='dash'))
    
    fig.update_layout(
        title='🎯 Analyse des Résidus (Erreur vs Prédiction)',
        xaxis_title='SoH Prédit (%)',
        yaxis_title='Résidus (Erreur)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_feature_influence(df):
    """
    Analyse de l'influence des variables avec radar chart ou barres.
    """
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
    target = 'SoH'
    
    corrs = df[features + [target]].corr()[target].drop(target).abs().sort_values(ascending=True)
    
    fig = px.bar(x=corrs.values, y=corrs.index, orientation='h',
                title='⚡ Impact des Facteurs Physiques sur le SoH',
                labels={'x': 'Force de Corrélation (Abs)', 'y': 'Variable'},
                color=corrs.values, color_continuous_scale='Blues',
                template='plotly_dark')
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False
    )
    return fig

def plot_joint_influence(df, var_x='Temperature_measured', var_y='Voltage_measured'):
    """
    Analyse combinée de deux variables sur le SoH.
    """
    fig = px.scatter(df, x=var_x, y=var_y, color='SoH',
                    title=f'🧬 Interaction {var_x} & {var_y}',
                    color_continuous_scale='Turbo',
                    template='plotly_dark')
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
