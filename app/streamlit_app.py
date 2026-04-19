# =============================================================
# AGRICLIMA360 - Application Streamlit avec données NOAA API
# Visualisations climatiques interactives AVEC ANIMATIONS
# et visualisation de données massives
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
import time
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# IMPORT DES LIBRAIRIES DE VISUALISATION MASSIVE
# =============================================================
try:
    import dask.dataframe as dd
    import dask.array as da
    from dask.diagnostics import ProgressBar
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader import reductions
    from datashader.colors import inferno, viridis
    import holoviews as hv
    hv.extension('bokeh', 'matplotlib')  # Initialiser icis
    from holoviews.operation.datashader import datashade, dynspread
    import hvplot.pandas
    import hvplot.dask
    import panel as pn
    pn.extension()  # Initialiser Panel
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256, Inferno256
    from bokeh.embed import components
    from bokeh.resources import CDN
    from bokeh.io import export_png
    hv.extension('bokeh')
    pn.extension()
    DATA_VIZ_ENABLED = True
    st.success("✅ Visualisation de données massives activée (Dask + Datashader)")
except ImportError as e:
    DATA_VIZ_ENABLED = False
    st.warning(f"⚠️ Visualisation de données massives désactivée: {e}")
def get_dataframe_length(df):
    """Retourne le nombre de lignes d'un DataFrame, compatible avec pandas et dask."""
    if df is None:
        return 0
    
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        try:
            # Pour Dask DataFrame, calculer la taille
            with ProgressBar():
                return df.shape[0].compute()
        except Exception as e:
            try:
                # Alternative: compter les lignes
                with ProgressBar():
                    return len(df.index).compute()
            except:
                # Estimation basée sur les partitions
                try:
                    if hasattr(df, 'npartitions'):
                        # Estimation grossière: 10000 lignes par partition
                        return df.npartitions * 10000
                    else:
                        return 0
                except:
                    return 0
    else:
        # Pour pandas DataFrame
        try:
            return len(df)
        except:
            try:
                return df.shape[0]
            except:
                return 0
# =============================================================
# CONFIGURATION
# =============================================================

# Configuration de la page
st.set_page_config(
    page_title="AgriClima360 - Dashboard Climatique Avancé",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# 1. CONFIGURATION API NOAA
# =============================================================

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/"

# ⚠️ IMPORTANT: Obtenez votre token gratuit sur https://www.ncdc.noaa.gov/cdo-web/token
NOAA_TOKEN = st.secrets.get("NOAA_TOKEN", "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM")

@st.cache_data(ttl=3600)
def get_noaa_data(endpoint, params=None, token=NOAA_TOKEN):
    headers = {"token": token}
    url = f"{BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return None

@st.cache_data(ttl=3600)
def get_climate_data(dataset_id="GHCND", start_date="2020-01-01", end_date="2023-12-31", 
                     location_id=None, datatypes=None, limit=1000):
    params = {
        "datasetid": dataset_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": limit,
    }
    
    if location_id:
        params["locationid"] = location_id
    
    if datatypes:
        params["datatypeid"] = ",".join(datatypes)
    
    data = get_noaa_data("data", params)
    
    if data and "results" in data:
        df = pd.DataFrame(data["results"])
        return df
    
    return pd.DataFrame()

# =============================================================
# 2. FONCTIONS DE TRAITEMENT AVANCÉES
# =============================================================

def process_climate_data(df):
    """Traite et enrichit les données climatiques avec plus de variables."""
    if df.empty:
        return generate_enhanced_sample_data()
    
    # Conversion de la date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    
    # Conversion des températures (de dixièmes de degrés Celsius)
    if 'value' in df.columns:
        # Les températures NOAA sont en dixièmes de degrés
        temp_types = ['TMAX', 'TMIN', 'TAVG']
        df.loc[df['datatype'].isin(temp_types), 'value'] = df.loc[df['datatype'].isin(temp_types), 'value'] / 10
        
        # Les précipitations sont en dixièmes de mm
        df.loc[df['datatype'] == 'PRCP', 'value'] = df.loc[df['datatype'] == 'PRCP', 'value'] / 10
    
    # Pivoter pour avoir les différents types de données en colonnes
    df_pivot = df.pivot_table(
        index=['date', 'year', 'month', 'day', 'day_of_year', 'week', 'station'],
        columns='datatype',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Renommer les colonnes
    column_mapping = {
        'TMAX': 'tmax',
        'TMIN': 'tmin',
        'TAVG': 'tavg',
        'PRCP': 'prcp',
        'SNOW': 'snow',
        'SNWD': 'snow_depth',
        'AWND': 'wind_avg',
        'WSF2': 'wind_fastest',
        'WDF2': 'wind_direction',
        'WSF5': 'wind_gust'
    }
    # Renommer uniquement les colonnes existantes
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=existing_columns)
    
    # Calculer tavg si manquant
    if 'tavg' not in df_pivot.columns and 'tmax' in df_pivot.columns and 'tmin' in df_pivot.columns:
        df_pivot['tavg'] = (df_pivot['tmax'] + df_pivot['tmin']) / 2
    
    # Ajouter des données simulées pour les visualisations avancées
    n_rows = len(df_pivot)
    df_pivot['humidity'] = np.random.uniform(30, 90, n_rows)
    df_pivot['wind_speed'] = np.random.uniform(0, 20, n_rows)
    df_pivot['solar_radiation'] = np.random.uniform(100, 800, n_rows)
    df_pivot['pressure'] = np.random.uniform(980, 1030, n_rows)
    df_pivot['evapotranspiration'] = np.random.uniform(0, 10, n_rows)
    df_pivot['soil_moisture'] = np.random.uniform(10, 80, n_rows)
    df_pivot['continent'] = np.random.choice(['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania'], n_rows)
    df_pivot['country'] = np.random.choice(['USA', 'Canada', 'France', 'Germany', 'China', 'India', 'Brazil', 'Australia'], n_rows)
    df_pivot['lat'] = 40.0 + np.random.uniform(-30, 30, n_rows)
    df_pivot['lon'] = -100.0 + np.random.uniform(-60, 60, n_rows)
    df_pivot['elevation'] = np.random.uniform(0, 3000, n_rows)
    
    # Ajouter des tendances temporelles
    df_pivot['warming_trend'] = 0.01 * (df_pivot['year'] - 2000)
    df_pivot['tavg_trend'] = df_pivot['tavg'] + df_pivot['warming_trend']
    
    return df_pivot

def generate_massive_sample_data(n_points=1000000):
    """Génère des données de démonstration massives pour tester Dask/Datashader."""
    st.info(f"🧪 Génération de {n_points:,} points de données pour les tests...")
    
    # Créer un DataFrame Dask
    n_partitions = 10
    chunk_size = n_points // n_partitions
    
    def generate_chunk(i):
        """Génère un chunk de données."""
        n = min(chunk_size, n_points - i * chunk_size)
        
        dates = pd.date_range('2000-01-01', '2023-12-31', periods=n)
        
        data = {
            'date': np.random.choice(dates, n),
            'year': np.random.randint(2000, 2024, n),
            'month': np.random.randint(1, 13, n),
            'day': np.random.randint(1, 29, n),
            'station': [f'ST{i:04d}' for i in np.random.randint(1, 1000, n)],
            'tavg': 15 + 10 * np.sin(2 * np.pi * np.random.rand(n)) + 0.03 * (np.random.rand(n) * 24),
            'tmax': 20 + 12 * np.sin(2 * np.pi * np.random.rand(n)) + 0.03 * (np.random.rand(n) * 24),
            'tmin': 10 + 8 * np.sin(2 * np.pi * np.random.rand(n)) + 0.03 * (np.random.rand(n) * 24),
            'prcp': np.random.exponential(5, n),
            'humidity': np.random.uniform(30, 90, n),
            'wind_speed': np.random.exponential(5, n),
            'solar_radiation': np.random.uniform(100, 800, n),
            'pressure': np.random.normal(1013, 10, n),
            'lat': np.random.uniform(-90, 90, n),
            'lon': np.random.uniform(-180, 180, n),
            'elevation': np.random.exponential(500, n),
            'continent': np.random.choice(['NA', 'EU', 'AS', 'AF', 'SA', 'OC'], n),
            'biome': np.random.choice(['Forest', 'Grassland', 'Desert', 'Tundra', 'Aquatic'], n)
        }
        
        return pd.DataFrame(data)
    
    # Créer un DataFrame Dask
    if DATA_VIZ_ENABLED:
        # Créer une liste de DataFrames pandas
        dfs = [generate_chunk(i) for i in range(n_partitions)]
        
        # Convertir en DataFrame Dask
        ddf = dd.from_pandas(pd.concat(dfs, ignore_index=True), npartitions=n_partitions)
        
        # Optimiser les types de données
        ddf['date'] = dd.to_datetime(ddf['date'])
        ddf['station'] = ddf['station'].astype('category')
        ddf['continent'] = ddf['continent'].astype('category')
        ddf['biome'] = ddf['biome'].astype('category')
        
        st.success(f"✅ {n_points:,} points générés avec Dask ({n_partitions} partitions)")
        return ddf
    else:
        # Version pandas (plus lente)
        df = generate_chunk(0)
        for i in range(1, n_partitions):
            df = pd.concat([df, generate_chunk(i)], ignore_index=True)
        
        st.success(f"✅ {len(df):,} points générés avec Pandas")
        return df

def generate_enhanced_sample_data(n_points=100000):
    """Génère des données de démonstration enrichies."""
    st.warning("🔧 Données de démonstration - Configurez votre token NOAA pour des données réelles.")
    
    if DATA_VIZ_ENABLED and n_points > 100000:
        return generate_massive_sample_data(min(n_points, 500000))
    
    dates = pd.date_range('2000-01-01', '2023-12-31', periods=n_points)
    
    data = {
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'day_of_year': dates.dayofyear,
        'week': dates.isocalendar().week,
        'station': [f'ST{i:04d}' for i in np.random.randint(1, 100, n_points)],
        'tavg': 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.03 * (dates.year - 2000) + np.random.normal(0, 2, n_points),
        'tmax': 20 + 12 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.03 * (dates.year - 2000) + np.random.normal(0, 2, n_points),
        'tmin': 10 + 8 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.03 * (dates.year - 2000) + np.random.normal(0, 2, n_points),
        'prcp': np.random.exponential(5, n_points),
        'humidity': np.random.uniform(30, 90, n_points),
        'wind_speed': np.random.exponential(5, n_points),
        'solar_radiation': np.random.uniform(100, 800, n_points),
        'pressure': np.random.normal(1013, 10, n_points),
        'evapotranspiration': np.random.uniform(0, 10, n_points),
        'soil_moisture': np.random.uniform(10, 80, n_points),
        'continent': np.random.choice(['North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania'], n_points),
        'country': np.random.choice(['USA', 'Canada', 'France', 'Germany', 'China', 'India', 'Brazil', 'Australia'], n_points),
        'lat': np.random.uniform(-90, 90, n_points),
        'lon': np.random.uniform(-180, 180, n_points),
        'elevation': np.random.exponential(500, n_points)
    }
    
    return pd.DataFrame(data)

def compute_kpis(df):
    """Calcule les indicateurs clés avancés."""
    kpis = {}
    
    if not df.empty:
        # Utiliser Dask pour les calculs si disponible
        if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
            with ProgressBar():
                kpis["temp_moy"] = df["tavg"].mean().compute() if "tavg" in df.columns else 0
                kpis["pluie_totale"] = df["prcp"].sum().compute() if "prcp" in df.columns else 0
                kpis["nb_annees"] = df["year"].nunique().compute()
                kpis["nb_stations"] = df["station"].nunique().compute() if "station" in df.columns else 0
                kpis["temp_max"] = df["tmax"].max().compute() if "tmax" in df.columns else 0
                kpis["temp_min"] = df["tmin"].min().compute() if "tmin" in df.columns else 0
                kpis["humidite_moy"] = df["humidity"].mean().compute() if "humidity" in df.columns else 65
                kpis["solar_avg"] = df["solar_radiation"].mean().compute() if "solar_radiation" in df.columns else 0
                kpis["wind_avg"] = df["wind_speed"].mean().compute() if "wind_speed" in df.columns else 0
                kpis["nb_points"] = len(df)
        else:
            # Version pandas
            kpis["temp_moy"] = df["tavg"].mean() if "tavg" in df.columns else 0
            kpis["pluie_totale"] = df["prcp"].sum() if "prcp" in df.columns else 0
            kpis["nb_annees"] = df["year"].nunique()
            kpis["nb_stations"] = df["station"].nunique() if "station" in df.columns else 0
            kpis["temp_max"] = df["tmax"].max() if "tmax" in df.columns else 0
            kpis["temp_min"] = df["tmin"].min() if "tmin" in df.columns else 0
            kpis["humidite_moy"] = df["humidity"].mean() if "humidity" in df.columns else 65
            kpis["solar_avg"] = df["solar_radiation"].mean() if "solar_radiation" in df.columns else 0
            kpis["wind_avg"] = df["wind_speed"].mean() if "wind_speed" in df.columns else 0
            kpis["nb_points"] = len(df)
        
        # Calcul de la tendance de température
        if "tavg" in df.columns and kpis["nb_annees"] > 1:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                yearly_avg = df.groupby('year')['tavg'].mean().compute().reset_index()
            else:
                yearly_avg = df.groupby('year')['tavg'].mean().reset_index()
            
            if len(yearly_avg) > 1:
                coeffs = np.polyfit(yearly_avg['year'], yearly_avg['tavg'], 1)
                kpis["temp_trend"] = coeffs[0] * 100  # °C par siècle
                kpis["temp_trend_decade"] = coeffs[0] * 10  # °C par décennie
            else:
                kpis["temp_trend"] = 0
                kpis["temp_trend_decade"] = 0
        else:
            kpis["temp_trend"] = 0
            kpis["temp_trend_decade"] = 0
            
        # Calcul de la variabilité
        if "tavg" in df.columns and kpis["nb_annees"] > 1:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                kpis["variability"] = df.groupby('year')['tavg'].std().mean().compute()
            else:
                kpis["variability"] = df.groupby('year')['tavg'].std().mean()
        else:
            kpis["variability"] = 0
            
        # Calcul des canicules
        if "tmax" in df.columns and kpis["nb_points"] > 0:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                heatwave_count = (df['tmax'] > 30).sum().compute()
                kpis["heatwaves"] = (heatwave_count / kpis["nb_points"]) * 100
            else:
                kpis["heatwaves"] = (df['tmax'] > 30).sum() / kpis["nb_points"] * 100
        else:
            kpis["heatwaves"] = 0
            
        # Calcul du risque de sécheresse
        if "prcp" in df.columns and kpis["nb_points"] > 0:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                drought_count = (df['prcp'] < 5).sum().compute()
                kpis["drought_risk"] = (drought_count / kpis["nb_points"]) * 100
            else:
                kpis["drought_risk"] = (df['prcp'] < 5).sum() / kpis["nb_points"] * 100
        else:
            kpis["drought_risk"] = 0
            
        # Nombre de continents
        if "continent" in df.columns:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                kpis["continents"] = df["continent"].nunique().compute()
            else:
                kpis["continents"] = df["continent"].nunique()
        else:
            kpis["continents"] = 1
    
    return kpis

# =============================================================
# 3. FONCTIONS DE VISUALISATION MASSIVES (DASK + DATASHADER)
# =============================================================
def create_radar_chart(df, year=None):
    """Crée un graphique radar pour une année spécifique."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    if year is None:
        year = df['year'].max()
    
    year_data = df[df['year'] == year]
    
    if len(year_data) == 0:
        return go.Figure()
    
    # Vérifier que toutes les colonnes nécessaires existent
    required_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']
    missing_cols = [col for col in required_cols if col not in year_data.columns]
    
    if missing_cols:
        # Créer des colonnes manquantes avec des valeurs par défaut
        for col in missing_cols:
            if col == 'prcp':
                year_data[col] = 0
            elif col in ['tavg', 'tmax', 'tmin']:
                year_data[col] = 20
            elif col == 'humidity':
                year_data[col] = 50
            elif col == 'wind_speed':
                year_data[col] = 5
    
    avg_data = year_data[required_cols].mean()
    
    # Normaliser les données pour le radar
    max_vals = df[required_cols].max()
    min_vals = df[required_cols].min()
    
    normalized_data = (avg_data - min_vals) / (max_vals - min_vals)
    
    fig = go.Figure(data=go.Scatterpolar(
        r=[
            normalized_data['tavg'],
            normalized_data['tmax'],
            normalized_data['tmin'],
            normalized_data['prcp'] / 100,  # Réduire l'échelle des précipitations
            normalized_data['humidity'] / 100,
            normalized_data['wind_speed'] / 20
        ],
        theta=['Temp Moy', 'Temp Max', 'Temp Min', 'Précip', 'Humidité', 'Vent'],
        fill='toself',
        name=f'Année {year}',
        line_color='blue',
        opacity=0.8
    ))
    
    # Ajouter des données de référence (moyenne historique)
    ref_data = df[required_cols].mean()
    normalized_ref = (ref_data - min_vals) / (max_vals - min_vals)
    
    fig.add_trace(go.Scatterpolar(
        r=[
            normalized_ref['tavg'],
            normalized_ref['tmax'],
            normalized_ref['tmin'],
            normalized_ref['prcp'] / 100,
            normalized_ref['humidity'] / 100,
            normalized_ref['wind_speed'] / 20
        ],
        theta=['Temp Moy', 'Temp Max', 'Temp Min', 'Précip', 'Humidité', 'Vent'],
        fill='toself',
        name='Moyenne historique',
        line_color='gray',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            angularaxis=dict(
                direction="clockwise"
            )
        ),
        showlegend=True,
        title=f'📊 Profil Climatique {year} (Graphique Radar)',
        height=500
    )
    
    return fig

def create_parallel_coordinates(df, selected_years=None):
    """Crée un diagramme de coordonnées parallèles."""
    if df.empty:
        return go.Figure()
    
    if selected_years:
        plot_df = df[df['year'].isin(selected_years)]
    else:
        plot_df = df.sample(min(500, len(df)))
    
    required_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed', 'year']
    available_cols = [col for col in required_cols if col in plot_df.columns]
    
    if 'year' not in available_cols:
        available_cols.append('year')
    
    fig = px.parallel_coordinates(plot_df,
                                 dimensions=available_cols[:-1],  # Exclure 'year' des dimensions
                                 color='year',
                                 labels={'tavg': 'Temp Moy', 'tmax': 'Temp Max',
                                        'tmin': 'Temp Min', 'prcp': 'Précip',
                                        'humidity': 'Humidité', 'wind_speed': 'Vent'},
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                 title='📈 Coordonnées Parallèles des Variables Climatiques',
                                 height=500)
    
    return fig

def create_stream_graph(df):
    """Crée un graphique stream (courbes empilées)."""
    if df.empty or 'year' not in df.columns or 'month' not in df.columns:
        return go.Figure()
    
    monthly_data = df.groupby(['year', 'month']).agg({
        'tavg': 'mean',
        'prcp': 'sum'
    }).reset_index()
    
    # Pivoter pour le format stream
    stream_data = monthly_data.pivot(index='month', columns='year', values='tavg')
    
    fig = go.Figure()
    
    for year in stream_data.columns:
        fig.add_trace(go.Scatter(
            x=stream_data.index,
            y=stream_data[year],
            mode='lines',
            stackgroup='one',
            name=str(year),
            hoverinfo='x+y+name',
            line=dict(width=0.5),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title='🌊 Évolution des Températures (Graphique Stream)',
        xaxis_title='Mois',
        yaxis_title='Température Moyenne (°C)',
        showlegend=True,
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_datashader_plot(df, x_col='lon', y_col='lat', color_col='tavg', 
                          title='Carte Thermique avec Datashader', width=800, height=600):
    """Crée une visualisation Datashader pour des millions de points."""
    if not DATA_VIZ_ENABLED:
        st.warning("Datashader non disponible")
        return None
    
    try:
        # Convertir en DataFrame pandas si c'est un Dask DataFrame
        if isinstance(df, dd.DataFrame):
            # Échantillonner pour Datashader
            sample_size = min(1000000, len(df))
            df_sample = df.sample(frac=sample_size/len(df)).compute() if len(df) > sample_size else df.compute()
        else:
            df_sample = df.sample(min(1000000, len(df))) if len(df) > 1000000 else df
        
        # Créer le canvas Datashader
        canvas = ds.Canvas(plot_width=width, plot_height=height)
        
        # Agréger les points
        agg = canvas.points(df_sample, x_col, y_col, ds.mean(color_col))
        
        # Appliquer la colormap
        img = tf.shade(agg, cmap=viridis, how='log')
        img = tf.set_background(img, 'black')
        
        # Convertir en image
        img_pil = img.to_pil()
        
        return img_pil
        
    except Exception as e:
        st.error(f"Erreur Datashader: {e}")
        return None

def create_holoviews_datashader(df, x_col='date', y_col='tavg', color_col='prcp',
                               title='Time Series avec Datashader'):
    """Crée une visualisation HoloViews avec Datashader."""
    if not DATA_VIZ_ENABLED:
        st.warning("HoloViews/Datashader non disponible")
        return None
    
    try:
        # Échantillonner si nécessaire
        if isinstance(df, dd.DataFrame):
            sample_size = min(50000, get_dataframe_length(df))
            if sample_size < get_dataframe_length(df):
                df_plot = df.sample(frac=sample_size/get_dataframe_length(df)).compute()
            else:
                df_plot = df.compute()
        else:
            df_plot = df.copy()
        
        # S'assurer que les colonnes existent
        if x_col not in df_plot.columns or y_col not in df_plot.columns:
            st.error(f"Colonnes {x_col} ou {y_col} non trouvées")
            return None
        
        # Convertir la date si nécessaire
        if x_col == 'date' and not pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            df_plot[x_col] = pd.to_datetime(df_plot[x_col])
        
        # Créer le scatter plot
        scatter = hv.Scatter(df_plot, x_col, y_col).opts(
            width=800,
            height=400,
            title=title,
            color=color_col if color_col in df_plot.columns else hv.Cycle('Category20'),
            cmap='viridis' if color_col in df_plot.columns else None,
            colorbar=True if color_col in df_plot.columns else False,
            tools=['hover', 'pan', 'wheel_zoom', 'reset'],
            alpha=0.6,
            size=5
        )
        
        # Appliquer Datashader pour les grandes données
        if len(df_plot) > 10000:
            shaded = dynspread(datashade(scatter, cmap=viridis, width=800, height=400))
            return shaded
        else:
            return scatter
        
    except Exception as e:
        st.error(f"Erreur HoloViews: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
def create_dask_histogram(df, column='tavg', bins=100, title='Distribution avec Dask'):
    """Crée un histogramme avec Dask pour de grandes données."""
    if not DATA_VIZ_ENABLED or not isinstance(df, dd.DataFrame):
        # Version pandas
        fig = px.histogram(df, x=column, nbins=bins, title=title)
        return fig
    
    try:
        with ProgressBar():
            # Calculer l'histogramme avec Dask
            hist, edges = da.histogram(df[column].to_dask_array(), bins=bins, range=[df[column].min().compute(), df[column].max().compute()])
            hist_values = hist.compute()
            edges_values = edges.compute()
        
        # Créer le graphique
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=edges_values[:-1],
            y=hist_values,
            width=np.diff(edges_values),
            marker_color='royalblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title='Fréquence',
            bargap=0.05,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erreur Dask histogram: {e}")
        return None

def create_massive_scatter(df, x_col='tavg', y_col='prcp', color_col='year',
                          title='Scatter Plot Massif', point_size=2):
    """Crée un scatter plot pour des millions de points."""
    if len(df) > 100000 and DATA_VIZ_ENABLED:
        # Utiliser Datashader pour les grandes données
        if isinstance(df, dd.DataFrame):
            df_sample = df.sample(frac=0.1).compute() if len(df) > 1000000 else df.compute()
        else:
            df_sample = df.sample(min(100000, len(df)))
        
        # Créer avec Datashader
        canvas = ds.Canvas(plot_width=800, plot_height=500)
        agg = canvas.points(df_sample, x_col, y_col, ds.mean(color_col) if color_col else ds.count())
        img = tf.shade(agg, cmap=viridis)
        
        # Convertir en figure Plotly
        img_array = np.array(img.to_pil())
        fig = px.imshow(img_array, title=f"{title} (Datashader - {len(df):,} points)")
        return fig
    else:
        # Version Plotly normale
        sample_size = min(10000, len(df))
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
        
        fig = px.scatter(df_sample, x=x_col, y=y_col, color=color_col,
                        title=f"{title} ({len(df_sample):,} points échantillonnés)",
                        opacity=0.6,
                        hover_data=['date', 'station'] if 'date' in df.columns and 'station' in df.columns else None)
        fig.update_traces(marker=dict(size=point_size))
        return fig

def create_spatial_heatmap(df, title='Carte de Chaleur Spatiale'):
    """Crée une carte de chaleur spatiale avec Datashader."""
    if not DATA_VIZ_ENABLED or 'lat' not in df.columns or 'lon' not in df.columns:
        return None
    
    try:
        # Préparer les données
        if isinstance(df, dd.DataFrame):
            df_spatial = df[['lat', 'lon', 'tavg']].dropna().compute()
        else:
            df_spatial = df[['lat', 'lon', 'tavg']].dropna()
        
        # Créer le canvas
        canvas = ds.Canvas(plot_width=800, plot_height=400)
        
        # Agréger
        agg = canvas.points(df_spatial, 'lon', 'lat', ds.mean('tavg'))
        
        # Créer l'image
        img = tf.shade(agg, cmap=inferno, how='log')
        img = tf.set_background(img, 'white')
        
        return img.to_pil()
        
    except Exception as e:
        st.error(f"Erreur carte de chaleur: {e}")
        return None

def create_time_series_aggregation(df, time_col='date', value_col='tavg', 
                                  freq='M', title='Série Temporelle Agrégée'):
    """Crée une série temporelle agrégée avec Dask."""
    
    # Vérifier si les colonnes existent
    if time_col not in df.columns or value_col not in df.columns:
        st.warning(f"Colonnes {time_col} ou {value_col} non trouvées dans les données")
        return None
    
    try:
        if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
            # Version Dask
            with ProgressBar():
                # Sélectionner uniquement les colonnes nécessaires
                df_temp = df[[time_col, value_col]].copy()
                
                # Convertir en pandas pour le traitement
                df_pd = df_temp.compute()
                
                # Convertir la colonne de temps en datetime
                df_pd['datetime'] = pd.to_datetime(df_pd[time_col], errors='coerce')
                
                # Vérifier et convertir la colonne de valeur en numérique
                if not pd.api.types.is_numeric_dtype(df_pd[value_col]):
                    df_pd[value_col] = pd.to_numeric(df_pd[value_col], errors='coerce')
                
                # Supprimer les valeurs NaN
                df_pd = df_pd.dropna(subset=['datetime', value_col])
                
                if len(df_pd) == 0:
                    st.warning(f"Aucune donnée numérique valide pour {value_col}")
                    return None
                
                # Définir l'index datetime
                df_pd.set_index('datetime', inplace=True)
                
                # Resample avec gestion des colonnes non-numériques
                df_resampled = df_pd[value_col].resample(freq).mean()
                
                # Créer la figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_resampled.index,
                    y=df_resampled.values,
                    mode='lines+markers',
                    name=value_col,
                    line=dict(width=2, color='royalblue'),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title=f"{title} (Dask - {get_dataframe_length(df):,} points)",
                    xaxis_title='Date',
                    yaxis_title=value_col,
                    height=400,
                    hovermode='x unified'
                )
                
                return fig
                
        else:
            # Version pandas
            df_pd = df.copy()
            
            # Convertir la colonne de temps en datetime
            df_pd['datetime'] = pd.to_datetime(df_pd[time_col], errors='coerce')
            
            # Vérifier et convertir la colonne de valeur en numérique
            if not pd.api.types.is_numeric_dtype(df_pd[value_col]):
                df_pd[value_col] = pd.to_numeric(df_pd[value_col], errors='coerce')
            
            # Supprimer les valeurs NaN
            df_pd = df_pd.dropna(subset=['datetime', value_col])
            
            if len(df_pd) == 0:
                st.warning(f"Aucune donnée numérique valide pour {value_col}")
                return None
            
            # Définir l'index datetime
            df_pd.set_index('datetime', inplace=True)
            
            # Resample
            df_resampled = df_pd[value_col].resample(freq).mean()
            
            # Créer la figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_resampled.index,
                y=df_resampled.values,
                mode='lines+markers',
                name=value_col,
                line=dict(width=2, color='royalblue'),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f"{title} ({get_dataframe_length(df):,} points)",
                xaxis_title='Date',
                yaxis_title=value_col,
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
    except Exception as e:
        st.error(f"Erreur lors de la création de la série temporelle: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
def get_numeric_columns_safe(df):
    """Retourne les colonnes numériques d'un DataFrame de manière sécurisée."""
    numeric_cols = []
    
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        # Pour Dask DataFrame
        with ProgressBar():
            # Prendre un échantillon pour vérifier les types
            sample = df.head(1000).compute()
            
            for col in df.columns:
                try:
                    if col in sample.columns:
                        if pd.api.types.is_numeric_dtype(sample[col]):
                            numeric_cols.append(col)
                        else:
                            # Essayer de convertir
                            converted = pd.to_numeric(sample[col], errors='coerce')
                            if not converted.isna().all():
                                numeric_cols.append(col)
                except:
                    continue
    else:
        # Pour pandas DataFrame
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                else:
                    # Essayer de convertir
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if not converted.isna().all():
                        numeric_cols.append(col)
            except:
                continue
    
    return numeric_cols   
# =============================================================
# 4. FONCTIONS DE VISUALISATION STANDARD
# =============================================================

def create_temperature_evolution(df):
    """Crée le graphique d'évolution des températures avec animation."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    # Utiliser Dask pour les calculs si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            yearly_data = df.groupby('year').agg({
                'tavg': 'mean',
                'tmax': 'max',
                'tmin': 'min'
            }).compute().reset_index()
    else:
        yearly_data = df.groupby('year').agg({
            'tavg': 'mean',
            'tmax': 'max',
            'tmin': 'min'
        }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tmax'],
        name='Température Max',
        mode='lines+markers',
        line=dict(color='red', width=3),
        hovertemplate='<b>Année</b>: %{x}<br><b>Temp Max</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tavg'],
        name='Température Moyenne',
        mode='lines+markers',
        line=dict(color='orange', width=3),
        hovertemplate='<b>Année</b>: %{x}<br><b>Temp Moy</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_data['year'],
        y=yearly_data['tmin'],
        name='Température Min',
        mode='lines+markers',
        line=dict(color='blue', width=3),
        hovertemplate='<b>Année</b>: %{x}<br><b>Temp Min</b>: %{y:.1f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title='📈 Évolution des Températures (Interactive)',
        xaxis_title='Année',
        yaxis_title='Température (°C)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_precipitation_chart(df):
    """Crée le graphique des précipitations avec interactivité."""
    if df.empty or 'prcp' not in df.columns:
        return go.Figure()
    
    # Utiliser Dask si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            monthly_prcp = df.groupby(['year', 'month'])['prcp'].sum().compute().reset_index()
    else:
        monthly_prcp = df.groupby(['year', 'month'])['prcp'].sum().reset_index()
    
    fig = px.bar(
        monthly_prcp,
        x='month',
        y='prcp',
        color='year',
        title='💧 Précipitations Mensuelles (Animées)',
        labels={'month': 'Mois', 'prcp': 'Précipitations (mm)', 'year': 'Année'},
        height=500,
        animation_frame='year',
        range_y=[0, monthly_prcp['prcp'].max() * 1.1] if not monthly_prcp.empty else [0, 100]
    )
    
    return fig

def create_animated_temperature_map(df):
    """Crée une carte animée des températures."""
    if df.empty or 'year' not in df.columns:
        return go.Figure()
    
    # Utiliser Dask si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            yearly_avg = df.groupby(['year', 'continent']).agg({
                'tavg': 'mean',
                'tmax': 'max',
                'tmin': 'min',
                'prcp': 'sum',
                'lat': 'mean',
                'lon': 'mean'
            }).compute().reset_index()
    else:
        yearly_avg = df.groupby(['year', 'continent']).agg({
            'tavg': 'mean',
            'tmax': 'max',
            'tmin': 'min',
            'prcp': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
    
    fig = px.scatter_geo(yearly_avg,
                        lat='lat',
                        lon='lon',
                        color='tavg',
                        size='prcp',
                        animation_frame='year',
                        hover_name='continent',
                        hover_data=['tavg', 'tmax', 'tmin', 'prcp'],
                        color_continuous_scale=px.colors.sequential.Viridis,
                        projection='natural earth',
                        title='🌡️ Évolution Mondiale des Températures (Animée)',
                        height=600)
    
    fig.update_layout(geo=dict(showland=True, landcolor="lightgray"))
    
    return fig

def create_3d_scatter_plot(df):
    """Crée un graphique 3D interactif."""
    if df.empty:
        return go.Figure()
    
    sample_size = min(5000, len(df))
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        df_sample = df.sample(frac=sample_size/len(df)).compute() if len(df) > sample_size else df.compute()
    else:
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
    
    fig = px.scatter_3d(df_sample,
                       x='tavg',
                       y='prcp',
                       z='humidity',
                       color='continent',
                       size='solar_radiation' if 'solar_radiation' in df.columns else 'wind_speed',
                       hover_name='station',
                       title='🌐 Visualisation 3D Interactive des Variables Climatiques',
                       height=600)
    
    fig.update_layout(scene=dict(
        xaxis_title='Température Moyenne (°C)',
        yaxis_title='Précipitations (mm)',
        zaxis_title='Humidité (%)'
    ))
    
    return fig

def create_interactive_heatmap(df):
    """Crée une heatmap interactive avec zoom."""
    if df.empty or 'tavg' not in df.columns:
        return go.Figure()
    
    # Utiliser Dask si disponible
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        with ProgressBar():
            pivot_data = df.pivot_table(index='month', columns='year', values='tavg', aggfunc='mean').compute()
    else:
        pivot_data = df.pivot_table(index='month', columns='year', values='tavg', aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        colorscale='Viridis',
        colorbar=dict(title="Température (°C)"),
        hoverongaps=False,
        hovertemplate='Année: %{x}<br>Mois: %{y}<br>Température: %{z:.1f}°C<extra></extra>',
        zsmooth='best'
    ))
    
    fig.update_layout(
        title='📅 Heatmap Interactive des Températures',
        xaxis_title="Année",
        yaxis_title="Mois",
        height=500,
        dragmode='zoom',
        hovermode='closest'
    )
    
    return fig

# =============================================================
# 5. INTERFACE STREAMLIT AVANCÉE
# =============================================================

def main():
    # Sidebar - Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=100)
        st.title("🌾 AgriClima360")
        st.markdown("### Dashboard Climatique Avancé")
        st.markdown("*Visualisation de données massives*" if DATA_VIZ_ENABLED else "*Mode standard*")
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Sélection de la source de données
        data_source = st.radio(
            "Source de données:",
            ["API NOAA (Réelles)", "Démonstration", "Données Massives (Test)"]
        )
        
        if data_source == "API NOAA (Réelles)":
            st.info("ℹ️ Token NOAA requis")
            
            # Configuration des paramètres NOAA
            with st.expander("📡 Paramètres API NOAA"):
                dataset = st.selectbox(
                    "Dataset:",
                    ["GHCND", "GSOM", "GSOY"],
                    help="GHCND = Données quotidiennes, GSOM = Mensuelles, GSOY = Annuelles"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Date début:",
                        value=datetime(2000, 1, 1),
                        min_value=datetime(1900, 1, 1)
                    )
                
                with col2:
                    end_date = st.date_input(
                        "Date fin:",
                        value=datetime(2024, 12, 31),
                        max_value=datetime.now()
                    )
                
                location = st.text_input(
                    "Localisation (optionnel):",
                    placeholder="ex: FIPS:US"
                )
                
                datatypes = st.multiselect(
                    "Types de données:",
                    ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND", "WSF2"],
                    default=["TMAX", "TMIN", "PRCP", "AWND"]
                )
                
                limit = st.slider("Nombre de résultats:", 1000, 100000, 10000,)
        
        st.markdown("---")
        
        # Navigation
        st.header("📊 Navigation")
        pages = [
            "🏠 Vue d'ensemble", 
            "📈 Analyses Animées", 
            "🌐 Visualisations 3D", 
            "🗺️ Carte Animée",
            "🚀 Données Massives",
            "🔬 Avancé", 
            "🎯 Radar & Parallèles"
        ]
        
        page = st.radio("Sections:", pages)
        
        st.markdown("---")
        
        # Filtres
        st.header("🎛️ Filtres")
        
        # Ces filtres seront appliqués après chargement
        year_filter = st.empty()
        continent_filter = st.empty()
        data_size_filter = st.empty()
        
        st.markdown("---")
        
        # Contrôles d'animation
        st.header("🎬 Contrôles d'Animation")
        animation_speed = st.slider("Vitesse d'animation:", 100, 2000, 500, 100)
        auto_play = st.checkbox("Lecture automatique", value=True)
        
        # Options de visualisation massive
        if DATA_VIZ_ENABLED:
            st.header("🚀 Options Massives")
            enable_dask = st.checkbox("Utiliser Dask", value=True)
            sample_size = st.selectbox(
                "Taille de l'échantillon:",
                ["1K", "10K", "100K", "1M", "10M", "Complet"],
                index=2
            )
        
        # Boutons d'export
        st.header("💾 Export")
        export_format = st.selectbox("Format d'export:", ["CSV", "Parquet", "JSON", "Excel"])
    
    # Chargement des données
    with st.spinner("⏳ Chargement des données..."):
        if data_source == "API NOAA (Réelles)":
            if NOAA_TOKEN == "YOUR_TOKEN_HERE" or NOAA_TOKEN == "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM":
                #st.error("❌ Token NOAA non configuré. Créez un fichier `.streamlit/secrets.toml` avec:\n```toml\nNOAA_TOKEN = 'votre_token'\n```")
                df = generate_enhanced_sample_data(50000)
            else:
                raw_data = get_climate_data(
                    dataset_id=dataset,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    location_id=location if location else None,
                    datatypes=datatypes if datatypes else None,
                    limit=limit
                )
                df = process_climate_data(raw_data)
                
        elif data_source == "Démonstration":
            df = generate_enhanced_sample_data(50000)
            
        else:  # Données Massives
            if DATA_VIZ_ENABLED:
                # Définir la taille des données
                size_map = {"1K": 1000, "10K": 10000, "100K": 100000, "1M": 1000000, "10M": 10000000}
                n_points = size_map.get(sample_size, 100000)
                
                if sample_size == "Complet" and enable_dask:
                    # Générer des données massives avec Dask
                    df = generate_massive_sample_data(1000000)
                else:
                    df = generate_enhanced_sample_data(n_points)
                    
                if enable_dask and not isinstance(df, dd.DataFrame):
                    # Convertir en Dask DataFrame
                    n_partitions = max(1, len(df) // 100000)
                    df = dd.from_pandas(df, npartitions=n_partitions)
                    st.success(f"✅ Converti en Dask DataFrame ({n_partitions} partitions)")
            else:
                df = generate_enhanced_sample_data(100000)
    
    # Vérification des données
    if df.empty:
        st.error("❌ Aucune donnée disponible. Vérifiez vos paramètres.")
        return
    
    # Afficher les informations sur les données
    if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
        st.sidebar.info(f"📊 Données Dask: {len(df):,} points, {df.npartitions} partitions")
    else:
        st.sidebar.info(f"📊 Données: {len(df):,} points")
    
    # Calcul des KPIs
    with st.spinner("📊 Calcul des indicateurs..."):
        kpis = compute_kpis(df)
    
    # Filtres dans la sidebar
    with st.sidebar:
        if 'year' in df.columns:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                with ProgressBar():
                    years = sorted(df['year'].unique().compute())
            else:
                years = sorted(df['year'].unique())
                
            if len(years) > 0:
                selected_years = year_filter.slider(
                    "Période:",
                    int(min(years)),
                    int(max(years)),
                    (int(min(years)), int(max(years)))
                )
                # Appliquer le filtre
                if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                    df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
                else:
                    df = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]
        
        if 'continent' in df.columns:
            if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                with ProgressBar():
                    continents = ['Tous'] + sorted(df['continent'].unique().compute().tolist())
            else:
                continents = ['Tous'] + sorted(df['continent'].unique().tolist())
                
            selected_continent = continent_filter.selectbox(
                "Continent:",
                continents
            )
            if selected_continent != 'Tous':
                if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                    df = df[df['continent'] == selected_continent]
                else:
                    df = df[df['continent'] == selected_continent]
        
        # Filtre de taille de données pour les démos
        if data_source == "Données Massives (Test)":
            viz_size = data_size_filter.slider(
                "Points à visualiser:",
                1000, 1000000, 100000, 1000,
                help="Réduire pour améliorer les performances"
            )
            if len(df) > viz_size:
                if DATA_VIZ_ENABLED and isinstance(df, dd.DataFrame):
                    df = df.sample(frac=viz_size/len(df))
                else:
                    df = df.sample(min(viz_size, len(df)))
    
    # Vérifier à nouveau si le dataframe n'est pas vide après filtrage
    if len(df) == 0:
        st.error("❌ Aucune donnée disponible après filtrage. Ajustez vos critères.")
        return
    
    # =============================================================
    # PAGES AVEC VISUALISATIONS
    # =============================================================
    
    if page == "🏠 Vue d'ensemble":
        st.title("🌍 AgriClima360 - Dashboard Climatique Avancé")
        st.markdown("### Visualisations interactives avec animations")
        
        if DATA_VIZ_ENABLED:
            st.info(f"🚀 Mode données massives activé: {len(df):,} points de données")
        
        # KPIs en ligne
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🌡️ Température Moy.",
                f"{kpis.get('temp_moy', 0):.1f}°C",
                f"{kpis.get('temp_trend_decade', 0):+.2f}°C/décennie"
            )
        
        with col2:
            st.metric(
                "💧 Précipitations",
                f"{kpis.get('pluie_totale', 0):,.0f} mm",
                f"{kpis.get('nb_points', 0):,} points"
            )
        
        with col3:
            st.metric(
                "⚠️ Canicules",
                f"{kpis.get('heatwaves', 0):.1f}%",
                f"Stations: {kpis.get('nb_stations', 0)}"
            )
        
        with col4:
            st.metric(
                "🌞 Radiation Solaire",
                f"{kpis.get('solar_avg', 0):.0f} W/m²",
                f"Vent: {kpis.get('wind_avg', 0):.1f} m/s"
            )
        
        with col5:
            if "continents" in kpis:
                st.metric("🌐 Continents", f"{kpis.get('continents', 1)}", f"Années: {kpis.get('nb_annees', 1)}")
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Évolution Temporelle")
            st.plotly_chart(
                create_temperature_evolution(df),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 💧 Précipitations")
            st.plotly_chart(
                create_precipitation_chart(df),
                use_container_width=True
            )
        
        # Visualisation massive si activée
        if DATA_VIZ_ENABLED and len(df) > 10000000:
            st.markdown("---")
            st.markdown("#### 🚀 Visualisation Massive")
            
            tab1, tab2 = st.tabs(["Carte Thermique", "Distribution"])
            
            with tab1:
                st.markdown("##### Carte Thermique avec Datashader")
                if 'lat' in df.columns and 'lon' in df.columns:
                    img = create_datashader_plot(df, title=f"Carte Thermique ({len(df):,} points)")
                    if img:
                        st.image(img, caption="Carte thermique des températures moyennes", use_column_width=True)
                else:
                    st.warning("Données spatiales nécessaires pour la carte thermique")
            
            with tab2:
                st.markdown("##### Distribution des Températures")
                fig_hist = create_dask_histogram(df, column='tavg', title=f"Distribution des Températures ({len(df):,} points)")
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    elif page == "🚀 Données Massives":
        st.title("🚀 Visualisation de Données Massives")
        
        if not DATA_VIZ_ENABLED:
            st.error("❌ Les packages de visualisation massive ne sont pas installés.")
            st.info("Installez-les avec: `pip install dask datashader holoviews hvplot panel bokeh`")
            return
        
        st.info(f"📊 Traitement de {len(df):,} points de données avec Dask et Datashader")
        
        # Sélection de visualisation
        viz_type = st.selectbox(
            "Type de visualisation:",
            ["Scatter Plot Massif", "Carte Thermique Spatiale", "Série Temporelle Agrégée", 
             "Distribution Dask", "Holoviews + Datashader", "Comparaison de Performances"]
        )
        
        if viz_type == "Scatter Plot Massif":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_var = st.selectbox("Variable X:", ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed'])
            with col2:
                y_var = st.selectbox("Variable Y:", ['prcp', 'tavg', 'humidity', 'wind_speed', 'solar_radiation'])
            with col3:
                color_var = st.selectbox("Couleur:", ['year', 'month', 'continent', None])
            
            point_size = st.slider("Taille des points:", 1, 10, 2)
            
            fig = create_massive_scatter(df, x_col=x_var, y_col=y_var, color_col=color_var,
                                       title=f"Scatter Plot: {y_var} vs {x_var}", point_size=point_size)
            st.plotly_chart(fig, use_container_width=True)
            
            # Informations sur les performances
            if isinstance(df, dd.DataFrame):
                st.info(f"✅ Graphique généré à partir de {len(df):,} points avec Dask")
        
        elif viz_type == "Carte Thermique Spatiale":
            st.markdown("#### 🌍 Carte de Chaleur Spatiale")
            
            if 'lat' in df.columns and 'lon' in df.columns:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    img = create_spatial_heatmap(df)
                    if img:
                        st.image(img, caption="Carte de chaleur spatiale des températures", use_column_width=True)
                
                with col2:
                    st.markdown("**Paramètres :**")
                    st.metric("Points", f"{len(df):,}")
                    if isinstance(df, dd.DataFrame):
                        st.metric("Partitions", df.npartitions)
                    
                    # Options d'affichage
                    color_map = st.selectbox("Colormap:", ['inferno', 'viridis', 'plasma', 'magma'])
                    point_size = st.slider("Taille:", 1, 20, 5)
                    
                    if st.button("🔄 Regénérer"):
                        st.rerun()
            else:
                st.warning("Les colonnes 'lat' et 'lon' sont nécessaires pour la carte spatiale")
        
        elif viz_type == "Série Temporelle Agrégée":
            st.markdown("#### 📈 Série Temporelle Agrégée")
            
            col1, col2 = st.columns(2)
            
            with col1:
                value_col = st.selectbox("Variable:", ['tavg', 'tmax', 'tmin', 'prcp', 'humidity'])
            with col2:
                freq = st.selectbox("Fréquence:", ['D', 'W', 'M', 'Q', 'Y'])
            
            fig = create_time_series_aggregation(df, value_col=value_col, freq=freq,
                                                title=f"Série Temporelle de {value_col}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques temporelles
            if 'date' in df.columns:
                if isinstance(df, dd.DataFrame):
                    with ProgressBar():
                        date_range = df['date'].min().compute(), df['date'].max().compute()
                        days = (date_range[1] - date_range[0]).days
                else:
                    date_range = df['date'].min(), df['date'].max()
                    days = (date_range[1] - date_range[0]).days
                
                st.metric("Période", f"{date_range[0].date()} à {date_range[1].date()}")
                st.metric("Durée", f"{days} jours")
        
        elif viz_type == "Distribution Dask":
            st.markdown("#### 📊 Distributions avec Dask")
            
            col1, col2 = st.columns(2)
            
            with col1:
                variable = st.selectbox("Variable à analyser:", 
                                       ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed'])
            with col2:
                bins = st.slider("Nombre de bins:", 10, 500, 100)
            
            fig = create_dask_histogram(df, column=variable, bins=bins,
                                       title=f"Distribution de {variable}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques descriptives
            if isinstance(df, dd.DataFrame):
                with ProgressBar():
                    stats = df[variable].describe().compute()
                st.dataframe(stats, use_container_width=True)
        
        elif viz_type == "Holoviews + Datashader":
                st.markdown("#### 🎨 HoloViews avec Datashader")
                
                # Afficher les colonnes disponibles
                available_cols = list(df.columns)
                if isinstance(df, dd.DataFrame):
                    with ProgressBar():
                        df_sample = df.head(100).compute()
                else:
                    df_sample = df.head(100)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_var = st.selectbox("Axe X:", available_cols, index=available_cols.index('date') if 'date' in available_cols else 0)
                with col2:
                    y_var = st.selectbox("Axe Y:", available_cols, index=available_cols.index('tavg') if 'tavg' in available_cols else 1)
                with col3:
                    color_options = [None] + available_cols
                    color_var = st.selectbox("Couleur:", color_options, index=0)
                
                # Options supplémentaires
                with st.expander("Options avancées"):
                    plot_type = st.selectbox("Type de plot:", ["Scatter", "Line", "Points"])
                    cmap_type = st.selectbox("Colormap:", ["viridis", "plasma", "inferno", "magma", "coolwarm"])
                    point_size = st.slider("Taille des points:", 1, 20, 5)
                    alpha = st.slider("Transparence:", 0.1, 1.0, 0.6)
                
                if st.button("🔄 Générer la visualisation"):
                    with st.spinner("Création de la visualisation HoloViews..."):
                        # CORRECTION ICI : utiliser y_var au lieu de y_col
                        plot = create_holoviews_datashader(
                            df, 
                            x_col=x_var, 
                            y_col=y_var,  # Changé de y_col à y_var
                            color_col=color_var,
                            title=f"{y_var} vs {x_var}"
                        )
                        
                        if plot:
                            # Afficher le plot
                            display_holoviews_in_streamlit(plot, title="Visualisation HoloViews")
                            
                            # Afficher des informations
                            st.info(f"**Points de données:** {get_dataframe_length(df):,}")
                            if color_var:
                                st.info(f"**Variable de couleur:** {color_var}")
                            
                            # Aperçu des données
                            with st.expander("📊 Aperçu des données"):
                                st.dataframe(df_sample, use_container_width=True)
                        else:
                            st.error("Impossible de créer la visualisation HoloViews")
                            
                            # Alternative: créer un simple scatter avec Plotly
                            st.info("Tentative avec Plotly comme alternative...")
                            sample_size = min(10000, get_dataframe_length(df))
                            if isinstance(df, dd.DataFrame):
                                df_viz = df.sample(frac=sample_size/get_dataframe_length(df)).compute()
                            else:
                                df_viz = df.sample(min(sample_size, len(df)))
                            
                            fig = px.scatter(df_viz, x=x_var, y=y_var, color=color_var if color_var else None,
                                            title=f"{y_var} vs {x_var} (Alternative Plotly)",
                                            opacity=alpha)
                            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Comparaison de Performances":
            st.markdown("#### ⚡ Comparaison de Performances")
            
            # Test de performance
            st.markdown("**Benchmark des opérations :**")
            
            if st.button("🚀 Lancer le benchmark"):
                with st.spinner("Exécution des tests..."):
                    # Test 1: GroupBy
                    start = time.time()
                    if isinstance(df, dd.DataFrame):
                        with ProgressBar():
                            result1 = df.groupby('year')['tavg'].mean().compute()
                    else:
                        result1 = df.groupby('year')['tavg'].mean()
                    time1 = time.time() - start
                    
                    # Test 2: Filtre
                    start = time.time()
                    if isinstance(df, dd.DataFrame):
                        with ProgressBar():
                            result2 = df[df['tavg'] > 20].compute()
                    else:
                        result2 = df[df['tavg'] > 20]
                    time2 = time.time() - start
                    
                    # Test 3: Statistiques
                    start = time.time()
                    if isinstance(df, dd.DataFrame):
                        with ProgressBar():
                            result3 = df['tavg'].describe().compute()
                    else:
                        result3 = df['tavg'].describe()
                    time3 = time.time() - start
                
                # Afficher les résultats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("GroupBy", f"{time1:.2f}s", f"{len(result1)} groupes")
                with col2:
                    st.metric("Filtre", f"{time2:.2f}s", f"{len(result2):,} points")
                with col3:
                    st.metric("Statistiques", f"{time3:.2f}s", "8 métriques")
                
                # Recommandations
                st.markdown("**Recommandations :**")
                if isinstance(df, dd.DataFrame):
                    st.success("✅ Dask est activé - Optimisé pour les grandes données")
                    st.info(f"Partitions: {df.npartitions}, Points: {len(df):,}")
                else:
                    st.warning("⚠️ Pandas seul - Pensez à activer Dask pour +100K points")
    
    elif page == "📈 Analyses Animées":
        st.title("📊 Analyses avec Animations")
        
        tab1, tab2, tab3 = st.tabs(["🌡️ Températures", "💧 Précipitations", "🔗 Corrélations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Évolution Temporelle Animée")
                fig_temp = create_temperature_evolution(df)
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                st.markdown("#### Heatmap Interactive")
                st.plotly_chart(
                    create_interactive_heatmap(df),
                    use_container_width=True
                )
        
        with tab2:
            st.markdown("#### Précipitations Animées")
            fig_prcp = create_precipitation_chart(df)
            st.plotly_chart(fig_prcp, use_container_width=True)
        
        with tab3:
            # Matrice de corrélation
            st.markdown("#### Matrice de Corrélation")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                if isinstance(df, dd.DataFrame):
                    with ProgressBar():
                        corr_matrix = df[numeric_cols].corr().compute()
                else:
                    corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               color_continuous_scale='RdBu', range_color=[-1, 1])
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🌐 Visualisations 3D":
        st.title("🌐 Visualisations 3D Interactives")
        
        fig_3d = create_3d_scatter_plot(df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    elif page == "🗺️ Carte Animée":
        st.title("🗺️ Carte Climatique Animée")
        
        if 'lat' in df.columns and 'lon' in df.columns:
            fig_map = create_animated_temperature_map(df)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Les données géographiques ne sont pas disponibles")
    
    elif page == "🔬 Avancé":
        st.title("🔬 Analyses Avancées et Export")
        
        tab1, tab2, tab3 = st.tabs(["📊 Créateur de Visualisations", "📈 Analyses Temporelles", "💾 Export des Données"])
        
        with tab1:
            st.markdown("#### 🎨 Créateur de Visualisations Personnalisées")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Type de graphique:",
                    ["Ligne Interactive", "Barre Empilée", "Scatter Animé", "Box Plot", "Violon", "Densité"]
                )
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    x_var = st.selectbox("Variable X:", numeric_cols)
                else:
                    x_var = None
            
            with col3:
                if numeric_cols and len(numeric_cols) > 1:
                    y_var = st.selectbox("Variable Y:", numeric_cols, 
                                       index=1 if len(numeric_cols) > 1 else 0)
                else:
                    y_var = None
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                color_options = [None] + categorical_cols
                color_var = st.selectbox(
                    "Couleur par:",
                    color_options
                )
            else:
                color_var = None
            
            if 'year' in df.columns:
                animation_options = [None, 'year', 'month']
                if 'continent' in df.columns:
                    animation_options.append('continent')
                animation_var = st.selectbox(
                    "Animation par:",
                    animation_options
                )
            else:
                animation_var = None
            
            # Options avancées
            with st.expander("⚙️ Options avancées"):
                trendline = st.checkbox("Ajouter une ligne de tendance")
                smoothing = st.checkbox("Lissage des courbes")
                log_scale = st.selectbox("Échelle logarithmique:", [None, "X", "Y", "Les deux"])
            
            if st.button("🔄 Générer la visualisation") and x_var and y_var:
                # Créer le graphique personnalisé
                if chart_type == "Ligne Interactive":
                    fig = px.line(df, x=x_var, y=y_var, color=color_var, 
                                 animation_frame=animation_var,
                                 title=f"{y_var} vs {x_var}")
                    if smoothing:
                        fig.update_traces(line_shape="spline")
                
                elif chart_type == "Barre Empilée":
                    fig = px.bar(df, x=x_var, y=y_var, color=color_var,
                                title=f"{y_var} par {x_var}")
                
                elif chart_type == "Scatter Animé":
                    fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                                    animation_frame=animation_var,
                                    size='prcp' if 'prcp' in df.columns else None,
                                    title=f"Scatter Plot Animé")
                
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_var, y=y_var, color=color_var,
                                title=f"Distribution de {y_var}")
                
                elif chart_type == "Violon":
                    fig = px.violin(df, x=x_var, y=y_var, color=color_var,
                                   title=f"Distribution Densité de {y_var}")
                
                else:  # Densité
                    fig = px.density_heatmap(df, x=x_var, y=y_var,
                                            title=f"Densité {x_var} vs {y_var}")
                
                # Appliquer les options avancées
                if trendline and chart_type in ["Ligne Interactive", "Scatter Animé"]:
                    fig.update_traces(mode='lines+markers')
                
                if log_scale == "X" or log_scale == "Les deux":
                    fig.update_xaxes(type="log")
                if log_scale == "Y" or log_scale == "Les deux":
                    fig.update_yaxes(type="log")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 📈 Analyses Temporelles Avancées")
            
            # Analyse de tendance
            st.markdown("##### Analyse de Tendance")
            
            if 'tavg' in df.columns and 'year' in df.columns:
                # Regression linéaire
                yearly_avg = df.groupby('year')['tavg'].mean().reset_index()
                if len(yearly_avg) > 1:
                    coeffs = np.polyfit(yearly_avg['year'], yearly_avg['tavg'], 1)
                    trend_line = np.poly1d(coeffs)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Pente de tendance", f"{coeffs[0]*10:.3f}°C/décennie")
                        st.metric("Intercept", f"{coeffs[1]:.2f}°C")
                    
                    with col2:
                        correlation = yearly_avg['year'].corr(yearly_avg['tavg'])
                        st.metric("Corrélation", f"{correlation:.3f}")
                        st.metric("R²", f"{correlation**2:.3f}")
                    
                    # Graphique de tendance
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=yearly_avg['year'],
                        y=yearly_avg['tavg'],
                        mode='markers',
                        name='Données',
                        marker=dict(size=10)
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=yearly_avg['year'],
                        y=trend_line(yearly_avg['year']),
                        mode='lines',
                        name=f'Tendance ({coeffs[0]*10:.2f}°C/décennie)',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig_trend.update_layout(
                        title='📈 Analyse de Tendance Linéaire',
                        xaxis_title='Année',
                        yaxis_title='Température Moyenne (°C)',
                        height=400
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            # Analyse saisonnière
            st.markdown("##### Analyse Saisonnière")
            
            if 'month' in df.columns and 'tavg' in df.columns:
                seasonal_avg = df.groupby('month')['tavg'].mean().reset_index()
                
                fig_seasonal = px.line_polar(seasonal_avg, r='tavg', theta='month',
                                            line_close=True,
                                            title='🔄 Variation Saisonnière des Températures')
                fig_seasonal.update_traces(fill='toself')
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with tab3:
            st.markdown("#### 💾 Export des Données et Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Export des Données")
                
                # Prévisualisation des données
                st.markdown("**Aperçu des données :**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Options d'export
                export_format = st.selectbox(
                    "Format d'export:",
                    ["CSV", "JSON", "Excel", "Parquet"]
                )
                
                if export_format == "CSV":
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Télécharger CSV",
                        csv,
                        "climate_data_advanced.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Télécharger JSON",
                        json_str,
                        "climate_data_advanced.json",
                        "application/json",
                        key='download-json'
                    )
                
                elif export_format == "Excel":
                    # Pour Excel, on utilise un buffer
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='ClimateData')
                        # Ajouter un onglet avec les statistiques
                        df.describe().to_excel(writer, sheet_name='Statistics')
                    
                    st.download_button(
                        "📥 Télécharger Excel",
                        output.getvalue(),
                        "climate_data_advanced.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='download-excel'
                    )
                
                else:  # Parquet
                    # Pour Parquet, on utilise un buffer temporaire
                    import tempfile
                    try:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                            table = pa.Table.from_pandas(df)
                            pq.write_table(table, tmp.name)
                            
                            with open(tmp.name, 'rb') as f:
                                parquet_data = f.read()
                            
                            st.download_button(
                                "📥 Télécharger Parquet",
                                parquet_data,
                                "climate_data_advanced.parquet",
                                "application/octet-stream",
                                key='download-parquet'
                            )
                    except ImportError:
                        st.error("La bibliothèque pyarrow est requise pour l'export Parquet. Installez-la avec `pip install pyarrow`")
            
            with col2:
                st.markdown("##### Export des Visualisations")
                
                # Options pour exporter les graphiques
                chart_to_export = st.selectbox(
                    "Graphique à exporter:",
                    ["Évolution des Températures", "Carte Animée", "Graphique 3D", 
                     "Radar Chart", "Matrice de Corrélation"]
                )
                
                format_img = st.selectbox(
                    "Format d'image:",
                    ["PNG", "JPEG", "SVG", "PDF"]
                )
                
                if st.button("🖼️ Générer l'image"):
                    # Créer le graphique sélectionné
                    if chart_to_export == "Évolution des Températures":
                        fig = create_temperature_evolution(df)
                    elif chart_to_export == "Carte Animée":
                        fig = create_animated_temperature_map(df)
                    elif chart_to_export == "Graphique 3D":
                        fig = create_3d_scatter_plot(df)
                    elif chart_to_export == "Radar Chart":
                        fig = create_radar_chart(df, df['year'].max() if 'year' in df.columns else None)
                    else:
                        fig = create_correlation_matrix_interactive(df)
                    
                    # Afficher le graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bouton de téléchargement (note: Plotly ne permet pas le téléchargement direct en SVG/PDF)
                    st.info(f"Pour sauvegarder en {format_img}, utilisez l'icône de capture dans la barre d'outils du graphique.")
                
                st.markdown("---")
                st.markdown("##### Rapport Automatique")
                
                if st.button("📄 Générer un rapport PDF"):
                    with st.spinner("Génération du rapport..."):
                        # Ici vous pourriez intégrer une librairie comme reportlab ou weasyprint
                        # Pour l'exemple, on montre juste un message
                        st.success("Fonctionnalité de rapport PDF à implémenter avec reportlab ou weasyprint")
                        st.markdown("""
                        **Contenu du rapport :**
                        1. Résumé exécutif
                        2. KPIs principaux
                        3. Visualisations clés
                        4. Analyses statistiques
                        5. Recommandations
                        """)
    
    
    elif page == "🎯 Radar & Parallèles":
        st.title("🎯 Visualisations Avancées")
        
        tab1, tab2, tab3 = st.tabs(["📊 Graphiques Radar", "📈 Coordonnées Parallèles", "🌊 Graphiques Stream"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Graphique Radar des Variables Climatiques")
                if 'year' in df.columns:
                    years = sorted(df['year'].unique())
                    if len(years) > 0:
                        selected_year = st.slider(
                            "Sélectionner l'année:",
                            min_value=int(min(years)),
                            max_value=int(max(years)),
                            value=int(max(years))
                        )
                        
                        radar_fig = create_radar_chart(df, selected_year)
                        st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("La colonne 'year' n'est pas disponible dans les données.")
            
            with col2:
                st.markdown("#### 📋 Légende Radar")
                st.markdown("""
                **Axe radial** : Valeurs normalisées (0-1)
                
                **Variables :**
                - **Temp Moy** : Température moyenne
                - **Temp Max** : Température maximale
                - **Temp Min** : Température minimale
                - **Précip** : Précipitations (échelle réduite)
                - **Humidité** : Humidité relative
                - **Vent** : Vitesse du vent
                
                **Interprétation :**
                - Plus l'aire est grande, plus les valeurs sont élevées
                - Comparaison avec la moyenne historique (gris)
                """)
                
                # Comparaison entre années
                st.markdown("#### Comparer deux années")
                if 'year' in df.columns:
                    available_years = sorted(df['year'].unique())
                    if len(available_years) >= 2:
                        # Calculer les indices pour les deux dernières années
                        year1_idx = max(0, len(available_years) - 2)
                        year2_idx = max(0, len(available_years) - 1)
                        
                        year1 = st.selectbox("Année 1", available_years, index=year1_idx)
                        year2 = st.selectbox("Année 2", available_years, index=year2_idx)
                        
                        if year1 != year2:
                            # Créer un radar comparatif
                            fig_compare = go.Figure()
                            
                            for year, color in zip([year1, year2], ['blue', 'red']):
                                year_data = df[df['year'] == year]
                                if len(year_data) > 0:
                                    required_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'humidity', 'wind_speed']
                                    # Vérifier les colonnes manquantes
                                    for col in required_cols:
                                        if col not in year_data.columns:
                                            if col == 'prcp':
                                                year_data[col] = 0
                                            elif col in ['tavg', 'tmax', 'tmin']:
                                                year_data[col] = 20
                                            elif col == 'humidity':
                                                year_data[col] = 50
                                            elif col == 'wind_speed':
                                                year_data[col] = 5
                                    
                                    avg_data = year_data[required_cols].mean()
                                    max_vals = df[required_cols].max()
                                    min_vals = df[required_cols].min()
                                    normalized_data = (avg_data - min_vals) / (max_vals - min_vals)
                                    
                                    fig_compare.add_trace(go.Scatterpolar(
                                        r=[
                                            normalized_data['tavg'],
                                            normalized_data['tmax'],
                                            normalized_data['tmin'],
                                            normalized_data['prcp'] / 100,
                                            normalized_data['humidity'] / 100,
                                            normalized_data['wind_speed'] / 20
                                        ],
                                        theta=['Temp Moy', 'Temp Max', 'Temp Min', 'Précip', 'Humidité', 'Vent'],
                                        fill='toself',
                                        name=f'Année {year}',
                                        line_color=color,
                                        opacity=0.5
                                    ))
                            
                            fig_compare.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                title=f'📊 Comparaison {year1} vs {year2}',
                                height=400
                            )
                            
                            st.plotly_chart(fig_compare, use_container_width=True)
        
        with tab2:
            st.markdown("#### Diagramme de Coordonnées Parallèles")
            
            if 'year' in df.columns:
                available_years = sorted(df['year'].unique())
                if available_years:
                    selected_years = st.multiselect(
                        "Sélectionner les années à comparer:",
                        available_years,
                        default=available_years[-min(3, len(available_years)):]  # Maximum 3 dernières années
                    )
                    
                    if selected_years:
                        parallel_fig = create_parallel_coordinates(df, selected_years)
                        st.plotly_chart(parallel_fig, use_container_width=True)
                        
                        st.markdown("**Comment interpréter :**")
                        st.markdown("""
                        - Chaque ligne représente une observation
                        - Les axes verticaux représentent les différentes variables
                        - La couleur montre la valeur de l'année
                        - Les lignes parallèles indiquent des corrélations positives
                        - Les lignes qui se croisent indiquent des corrélations négatives
                        """)
                    else:
                        st.warning("Veuillez sélectionner au moins une année.")
                else:
                    st.warning("Aucune année disponible dans les données.")
            else:
                st.warning("La colonne 'year' n'est pas disponible dans les données.")
        
        with tab3:
            st.markdown("#### Graphique Stream (Courbes Empilées)")
            
            stream_fig = create_stream_graph(df)
            st.plotly_chart(stream_fig, use_container_width=True)
            
            st.markdown("**Explication :**")
            st.markdown("""
            Le graphique stream montre l'évolution des températures moyennes par mois,
            empilées par année. Cela permet de voir :
            
            1. **Tendances saisonnières** : Pattern répétitif chaque année
            2. **Évolution temporelle** : Comment chaque année se compare
            3. **Variabilité** : Largeur de la bande à chaque point
            
            **Utilisations :**
            - Identifier des années exceptionnelles
            - Voir les changements saisonniers
            - Comparer visuellement plusieurs années
            """)
    
    
    # Footer
    st.markdown("---")
    tech_stack = "Dask + Datashader + HoloViews" if DATA_VIZ_ENABLED else "Pandas + Plotly"
    st.markdown(f"""
    <div style='text-align: center'>
        <p>🌍 AgriClima360 - Dashboard Climatique Avancé</p>
        <p style='font-size: 0.8em; color: gray;'>
            Tech: {tech_stack} | 
            Données: {len(df):,} points | 
            NOAA API
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
