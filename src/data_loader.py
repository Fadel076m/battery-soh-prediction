import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset depuis le chemin spécifié.
    """
    logging.info(f"Chargement des données depuis : {filepath}")
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {str(e)}")
        raise

def get_data_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les types et les valeurs manquantes.
    """
    info_df = pd.DataFrame({
        'Type': df.dtypes,
        'Valeurs Manquantes': df.isnull().sum(),
        '% Manquant': round((df.isnull().sum() / len(df)) * 100, 2)
    })
    return info_df

def get_battery_info(df: pd.DataFrame) -> dict:
    """
    Statistiques globales sur les batteries.
    """
    if 'battery_id' not in df.columns or 'cycle_number' not in df.columns:
        logging.warning("Colonnes 'battery_id' ou 'cycle_number' manquantes.")
        return {}
    
    battery_ids = df['battery_id'].unique()
    cycles_per_battery = df.groupby('battery_id')['cycle_number'].max()
    
    return {
        'nombre_batteries': len(battery_ids),
        'batteries_ids': list(battery_ids),
        'cycles_max_par_batterie': cycles_per_battery.to_dict(),
        'nombre_total_mesures': len(df)
    }

def print_quality_report(df: pd.DataFrame):
    """
    Affiche un rapport textuel pour la console.
    """
    print("="*50)
    print("🔋 RAPPORT DE QUALITÉ DES DONNÉES 🔋")
    print("="*50)
    print(f"\n1. Dimensions générales : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print("\n2. Types et valeurs manquantes :")
    print(get_data_info(df))
    
    print("\n3. Informations sur les Batteries :")
    batt_info = get_battery_info(df)
    print(f"   - {batt_info.get('nombre_batteries', 0)} batteries identifiées : {batt_info.get('batteries_ids', [])}")
    print("   - Cycles par batterie :")
    for b_id, cycles in batt_info.get('cycles_max_par_batterie', {}).items():
        print(f"       * {b_id} : {cycles} cycles")
    
    print("\n4. Aperçu (Head) :")
    print(df.head())
    print("\n5. Statistiques descriptives :")
    print(df.describe().T)
    print("="*50)

if __name__ == '__main__':
    import os
    # Chemin adaptatif : selon d'où on lance le script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'battery_dataset.csv')
    df = load_data(data_path)
    print_quality_report(df)
