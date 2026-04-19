# =============================================================
# AGRICLIMA360 - Application Streamlit avec NOAA API + FAOSTAT + ML
# Visualisations climatiques interactives AVEC ANIMATIONS
# et visualisation de données massives
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
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
    from datashader.colors import inferno, viridis
    import holoviews as hv
    hv.extension('bokeh')
    from holoviews.operation.datashader import datashade, dynspread
    import hvplot.pandas
    import panel as pn
    pn.extension()
    DATA_VIZ_ENABLED = True
    st.success("✅ Visualisation de données massives activée (Dask + Datashader)")
except ImportError:
    DATA_VIZ_ENABLED = False
    st.warning("⚠️ Visualisation de données massives désactivée")

# =============================================================
# IMPORT ML & FAOSTAT
# =============================================================
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, r2_score, silhouette_score
    import xgboost as xgb
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Bibliothèques ML non disponibles")

# =============================================================
# CONFIGURATION
# =============================================================
st.set_page_config(
    page_title="AgriClima360 - Dashboard Climatique Avancé + ML",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/"
try:
    NOAA_TOKEN = st.secrets.get("NOAA_TOKEN", "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM")
except:
    NOAA_TOKEN = "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM"

# =============================================================
# FONCTIONS DE CHARGEMENT ET TRAITEMENT DES DONNÉES
# =============================================================
@st.cache_data(ttl=3600)
def get_noaa_data(endpoint, params=None):
    headers = {"token": NOAA_TOKEN}
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur NOAA: {e}")
        return None

def generate_enhanced_sample_data(n_points=100000):
    """Données de démonstration réalistes."""
    dates = pd.date_range('2000-01-01', '2023-12-31', periods=n_points)
    data = {
        'date': dates, 'year': dates.year, 'month': dates.month,
        'station': [f'ST{i:04d}' for i in np.random.randint(1, 100, n_points)],
        'tavg': 15 + 10*np.sin(2*np.pi*dates.dayofyear/365) + 0.03*(dates.year-2000) + np.random.normal(0,2,n_points),
        'tmax': 20 + 12*np.sin(2*np.pi*dates.dayofyear/365) + 0.03*(dates.year-2000) + np.random.normal(0,2,n_points),
        'tmin': 10 + 8*np.sin(2*np.pi*dates.dayofyear/365) + 0.03*(dates.year-2000) + np.random.normal(0,2,n_points),
        'prcp': np.random.exponential(5, n_points),
        'humidity': np.random.uniform(30,90,n_points),
        'wind_speed': np.random.exponential(5,n_points),
        'solar_radiation': np.random.uniform(100,800,n_points),
        'lat': np.random.uniform(-90,90,n_points),
        'lon': np.random.uniform(-180,180,n_points),
        'continent': np.random.choice(['North America','Europe','Asia','Africa','South America','Oceania'], n_points),
        'country': np.random.choice(['USA','Canada','France','Germany','China','India','Brazil','Australia'], n_points)
    }
    return pd.DataFrame(data)

def compute_kpis(df):
    kpis = {}
    if not df.empty:
        kpis["temp_moy"] = df["tavg"].mean()
        kpis["pluie_totale"] = df["prcp"].sum()
        kpis["nb_annees"] = df["year"].nunique()
        kpis["nb_stations"] = df["station"].nunique() if "station" in df.columns else 0
        kpis["heatwaves"] = (df['tmax']>30).sum()/len(df)*100 if 'tmax' in df else 0
        kpis["continents"] = df["continent"].nunique() if "continent" in df.columns else 1
        # tendance
        if 'year' in df and kpis["nb_annees"]>1:
            yearly = df.groupby('year')['tavg'].mean().reset_index()
            coeffs = np.polyfit(yearly['year'], yearly['tavg'], 1)
            kpis["temp_trend_decade"] = coeffs[0]*10
        else:
            kpis["temp_trend_decade"] = 0
    return kpis

# =============================================================
# FONCTIONS DE VISUALISATION (version simplifiée mais fonctionnelle)
# =============================================================
def create_temperature_evolution(df):
    yearly = df.groupby('year')[['tavg','tmax','tmin']].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['tmax'], name='Tmax', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['tavg'], name='Tavg', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['tmin'], name='Tmin', line=dict(color='blue')))
    fig.update_layout(title='Évolution des températures', xaxis_title='Année', yaxis_title='°C', height=500)
    return fig

def create_precipitation_chart(df):
    monthly = df.groupby(['year','month'])['prcp'].sum().reset_index()
    fig = px.bar(monthly, x='month', y='prcp', color='year', animation_frame='year',
                 title='Précipitations mensuelles', labels={'prcp':'mm','month':'Mois'})
    return fig

def create_animated_temperature_map(df):
    yearly = df.groupby(['year','continent','lat','lon'])[['tavg','prcp']].mean().reset_index()
    fig = px.scatter_geo(yearly, lat='lat', lon='lon', color='tavg', size='prcp',
                         animation_frame='year', hover_name='continent',
                         projection='natural earth', title='Carte animée des températures')
    return fig

def create_3d_scatter_plot(df):
    sample = df.sample(min(5000, len(df)))
    fig = px.scatter_3d(sample, x='tavg', y='prcp', z='humidity', color='continent',
                        title='Visualisation 3D', height=600)
    return fig

def create_interactive_heatmap(df):
    pivot = df.pivot_table(index='month', columns='year', values='tavg', aggfunc='mean')
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                    colorscale='Viridis', colorbar_title='°C'))
    fig.update_layout(title='Heatmap des températures', xaxis_title='Année', yaxis_title='Mois')
    return fig

def create_radar_chart(df, year=None):
    # version simplifiée
    if year is None: year = df['year'].max()
    yearly = df[df['year']==year].mean(numeric_only=True)
    categories = ['tavg','tmax','tmin','prcp','humidity','wind_speed']
    values = [yearly.get(c,0) for c in categories]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig.update_layout(title=f'Profil climatique {year}', polar=dict(radialaxis=dict(visible=True)))
    return fig

def create_parallel_coordinates(df, selected_years=None):
    sample = df[df['year'].isin(selected_years)] if selected_years else df
    sample = sample.sample(min(500, len(sample)))
    fig = px.parallel_coordinates(sample, dimensions=['tavg','tmax','tmin','prcp','humidity'],
                                  color='year', color_continuous_scale=px.colors.diverging.Tealrose)
    return fig

def create_stream_graph(df):
    monthly = df.groupby(['year','month'])['tavg'].mean().reset_index()
    pivot = monthly.pivot(index='month', columns='year', values='tavg')
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col], mode='lines', stackgroup='one', name=str(col)))
    fig.update_layout(title='Stream graph des températures', xaxis_title='Mois')
    return fig

def create_datashader_plot(df, title="", width=800, height=600):
    if not DATA_VIZ_ENABLED: return None
    try:
        if isinstance(df, dd.DataFrame):
            df = df.head(1000000).compute()
        canvas = ds.Canvas(plot_width=width, plot_height=height)
        agg = canvas.points(df, 'lon', 'lat', ds.mean('tavg'))
        img = tf.shade(agg, cmap=viridis)
        return img.to_pil()
    except:
        return None

def create_dask_histogram(df, column='tavg', bins=100, title=''):
    fig = px.histogram(df.head(10000), x=column, nbins=bins, title=title)
    return fig

# =============================================================
# CLIENT FAOSTAT
# =============================================================
class FAOSTATClient:
    BASE_URL = "https://fenixservices.fao.org/faostat/api/v1/fr"
    @staticmethod
    @st.cache_data(ttl=86400)
    def get_yield_data(area_code=None, item_code=None, start_year=2000, end_year=2025):
        params = {"area_code": area_code or "", "item_code": item_code or "",
                  "element_code": "5417", "year": f"{start_year},{end_year}", "show_codes": "true"}
        url = f"{FAOSTATClient.BASE_URL}/data/QV"
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                df = pd.DataFrame(data["data"])
                df = df.rename(columns={"Year":"year","Value":"yield_t_ha","Area":"country","Item":"crop"})
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                df['yield_t_ha'] = pd.to_numeric(df['yield_t_ha'], errors='coerce')
                return df.dropna(subset=['year','yield_t_ha'])[['year','country','crop','yield_t_ha']]
        except: pass
        return pd.DataFrame()
    @staticmethod
    def get_available_countries():
        try:
            r = requests.get(f"{FAOSTATClient.BASE_URL}/areas/QV", timeout=30)
            data = r.json()
            df = pd.DataFrame(data["data"])
            return df[df['AreaCode']!=5000][['AreaCode','Area']].drop_duplicates().sort_values('Area')
        except: return pd.DataFrame()
    @staticmethod
    def get_available_crops():
        try:
            r = requests.get(f"{FAOSTATClient.BASE_URL}/items/QV", timeout=30)
            data = r.json()
            return pd.DataFrame(data["data"])[['ItemCode','Item']].drop_duplicates().sort_values('Item')
        except: return pd.DataFrame()

def merge_climate_with_yields(climate_df, yields_df, country_col='country'):
    if climate_df is None or yields_df.empty: return None
    if DATA_VIZ_ENABLED and isinstance(climate_df, dd.DataFrame):
        climate_agg = climate_df.groupby([country_col,'year']).agg({'tavg':'mean','tmax':'mean','tmin':'mean','prcp':'sum','humidity':'mean'}).compute().reset_index()
    else:
        climate_agg = climate_df.groupby([country_col,'year']).agg({'tavg':'mean','tmax':'mean','tmin':'mean','prcp':'sum','humidity':'mean'}).reset_index()
    return pd.merge(climate_agg, yields_df, left_on=[country_col,'year'], right_on=['country','year'], how='inner')

# =============================================================
# MODÈLES ML (fonctions déjà définies plus haut)
# =============================================================
def train_classification_model(df, target_col, feature_cols):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mode()[0] if not df[target_col].mode().empty else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

def train_regression_model(df, target_col, feature_cols):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    except:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

def train_clustering_model(df, feature_cols, n_clusters=3):
    X = df[feature_cols].fillna(df[feature_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters) if len(set(clusters))>1 else 0
    return kmeans, score, clusters, scaler

# =============================================================
# INTERFACE PRINCIPALE
# =============================================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=80)
        st.title("🌾 AgriClima360")
        st.markdown("### Dashboard Climatique + ML")
        st.markdown("---")
        data_source = st.radio("Source", ["Démonstration", "API NOAA (Réelles)", "Données Massives (Test)"])
        st.markdown("---")
        pages = [
            "🏠 Vue d'ensemble", "data explorer", "feature analysis",
            "model training", "prediction", "impact analysis",
            "📈 Analyses Animées", "🌐 Visualisations 3D", "🗺️ Carte Animée",
            "🚀 Données Massives", "🔬 Avancé", "🎯 Radar & Parallèles",
            "🌾 Données FAOSTAT", "🤖 Modèles ML"
        ]
        page = st.radio("Navigation", pages)
        st.markdown("---")
        # Filtres simplifiés
        year_filter = st.empty()
        if 'year' in st.session_state.get('df', pd.DataFrame()).columns:
            years = sorted(st.session_state.df['year'].unique())
            if years:
                selected_years = year_filter.slider("Années", min(years), max(years), (min(years), max(years)))
                st.session_state.df = st.session_state.df[(st.session_state.df['year']>=selected_years[0]) & (st.session_state.df['year']<=selected_years[1])]

    # Chargement des données
    if 'df' not in st.session_state:
        with st.spinner("Chargement..."):
            if data_source == "Démonstration":
                df = generate_enhanced_sample_data(50000)
            elif data_source == "API NOAA (Réelles)":
                df = generate_enhanced_sample_data(50000)  # fallback
            else:
                df = generate_enhanced_sample_data(100000)
            st.session_state.df = df
    df = st.session_state.df

    if df.empty:
        st.error("Aucune donnée")
        return

    kpis = compute_kpis(df)

    # ========== PAGES ==========
    if page == "🏠 Vue d'ensemble":
        st.title("🌍 AgriClima360 - Vue d'ensemble")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🌡️ Temp. moy.", f"{kpis.get('temp_moy',0):.1f}°C", f"{kpis.get('temp_trend_decade',0):+.2f}°C/décennie")
        col2.metric("💧 Précipitations", f"{kpis.get('pluie_totale',0):,.0f} mm", f"{len(df):,} points")
        col3.metric("⚠️ Canicules", f"{kpis.get('heatwaves',0):.1f}%", f"{kpis.get('nb_stations',0)} stations")
        col4.metric("🌞 Radiation", f"{df['solar_radiation'].mean():.0f} W/m²", f"Vent: {df['wind_speed'].mean():.1f} m/s")
        col5.metric("🌐 Continents", kpis.get('continents',1), f"{kpis.get('nb_annees',1)} années")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_temperature_evolution(df), use_container_width=True)
        with c2: st.plotly_chart(create_precipitation_chart(df), use_container_width=True)

    elif page == "data explorer":
        st.title("📊 Explorateur de données")
        df_disp = df.head(1000)
        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df_disp:
                years = st.multiselect("Années", sorted(df_disp['year'].unique()), default=sorted(df_disp['year'].unique())[:3])
                if years: df_disp = df_disp[df_disp['year'].isin(years)]
        with col2:
            if 'station' in df_disp:
                stations = st.multiselect("Stations", df_disp['station'].unique(), default=df_disp['station'].unique()[:2])
                if stations: df_disp = df_disp[df_disp['station'].isin(stations)]
        st.dataframe(df_disp, use_container_width=True)
        st.subheader("Statistiques")
        st.dataframe(df_disp.describe(), use_container_width=True)
        var = st.selectbox("Variable", df_disp.select_dtypes(include=[np.number]).columns)
        fig = px.histogram(df_disp, x=var, title=f"Distribution de {var}")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "feature analysis":
        st.title("🔬 Analyse des features")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        st.subheader("Matrice de corrélation")
        fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu', range_color=[-1,1])
        st.plotly_chart(fig, use_container_width=True)
        if 'tavg' in corr:
            st.subheader("Corrélations avec tavg")
            corr_tavg = corr['tavg'].drop('tavg').sort_values(ascending=False)
            fig = px.bar(x=corr_tavg.values, y=corr_tavg.index, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        # PCA
        st.subheader("ACP")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = numeric_df.dropna()
        if len(X) > 1000: X = X.sample(1000)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(2)
        pca_res = pca.fit_transform(X_scaled)
        fig = px.scatter(x=pca_res[:,0], y=pca_res[:,1], title="ACP")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "model training":
        st.title("🤖 Entraînement de modèles")
        if not ML_AVAILABLE:
            st.error("ML non disponible")
        else:
            ml_df = df.select_dtypes(include=[np.number]).dropna()
            target = st.selectbox("Variable cible", ml_df.columns)
            features = st.multiselect("Features", [c for c in ml_df.columns if c != target], default=[c for c in ml_df.columns if c != target][:3])
            if features:
                if st.button("Entraîner RandomForest (régression)"):
                    model, r2 = train_regression_model(ml_df, target, features)
                    st.success(f"R² = {r2:.3f}")
                    st.session_state.ml_model = model
                    st.session_state.ml_features = features
                    st.session_state.ml_type = 'regression'

    elif page == "prediction":
        st.title("🔮 Prédiction")
        if 'ml_model' not in st.session_state:
            st.warning("Aucun modèle entraîné. Allez dans 'model training'.")
        else:
            st.success(f"Modèle {st.session_state.ml_type} chargé")
            inputs = {}
            for f in st.session_state.ml_features:
                inputs[f] = st.number_input(f, value=0.0)
            if st.button("Prédire"):
                X = pd.DataFrame([inputs])
                pred = st.session_state.ml_model.predict(X)[0]
                st.write(f"### Prédiction : {pred:.2f}" if st.session_state.ml_type=='regression' else f"### Classe : {pred}")

    elif page == "impact analysis":
        st.title("🌍 Impact climatique sur les rendements")
        if 'merged_data' in st.session_state:
            df_impact = st.session_state.merged_data
            st.success("Données réelles FAOSTAT")
        else:
            df_impact = df.copy()
            df_impact['yield_simulated'] = 5 + 0.2*df_impact['tavg'] - 0.01*df_impact['prcp'] + np.random.normal(0,0.5,len(df_impact))
            target = 'yield_simulated'
            st.info("Rendement synthétique (simulation)")
        if 'year' in df_impact:
            annual = df_impact.groupby('year').agg({'tavg':'mean','prcp':'sum', target:'mean'}).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=annual['year'], y=annual['tavg'], name="Température"), secondary_y=False)
            fig.add_trace(go.Scatter(x=annual['year'], y=annual[target], name="Rendement"), secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(annual.corr())

    # Autres pages existantes (conservées)
    elif page == "📈 Analyses Animées":
        st.plotly_chart(create_temperature_evolution(df), use_container_width=True)
        st.plotly_chart(create_precipitation_chart(df), use_container_width=True)
    elif page == "🌐 Visualisations 3D":
        st.plotly_chart(create_3d_scatter_plot(df), use_container_width=True)
    elif page == "🗺️ Carte Animée":
        st.plotly_chart(create_animated_temperature_map(df), use_container_width=True)
    elif page == "🚀 Données Massives":
        st.info("Page dédiée aux visualisations massives (Datashader)")
        if DATA_VIZ_ENABLED:
            img = create_datashader_plot(df, title="Carte thermique")
            if img: st.image(img)
    elif page == "🔬 Avancé":
        st.info("Outils avancés : exports, analyses supplémentaires")
    elif page == "🎯 Radar & Parallèles":
        st.plotly_chart(create_radar_chart(df), use_container_width=True)
        st.plotly_chart(create_parallel_coordinates(df, selected_years=[2020,2021,2022]), use_container_width=True)
    elif page == "🌾 Données FAOSTAT":
        st.title("FAOSTAT - Rendements agricoles")
        countries = FAOSTATClient.get_available_countries()
        crops = FAOSTATClient.get_available_crops()
        if not countries.empty:
            selected_country = st.selectbox("Pays", countries['Area'])
            area_code = countries[countries['Area']==selected_country]['AreaCode'].iloc[0]
        if not crops.empty:
            selected_crop = st.selectbox("Culture", crops['Item'])
            item_code = crops[crops['Item']==selected_crop]['ItemCode'].iloc[0]
        year_range = st.slider("Période", 2000,2025,(2000,2025))
        if st.button("Récupérer"):
            yields = FAOSTATClient.get_yield_data(area_code, item_code, year_range[0], year_range[1])
            if not yields.empty:
                st.session_state.yields_df = yields
                st.dataframe(yields)
                fig = px.line(yields, x='year', y='yield_t_ha', title=f"Rendement {selected_crop} - {selected_country}")
                st.plotly_chart(fig)
        if st.button("Fusionner avec climat"):
            if 'yields_df' in st.session_state and 'country' in df.columns:
                merged = merge_climate_with_yields(df, st.session_state.yields_df)
                if merged is not None:
                    st.session_state.merged_data = merged
                    st.success("Fusion réussie")
                    st.dataframe(merged.head())
    elif page == "🤖 Modèles ML":
        st.title("Modèles ML avancés")
        if not ML_AVAILABLE:
            st.error("ML non disponible")
        else:
            ml_df = df.select_dtypes(include=[np.number]).dropna()
            model_type = st.selectbox("Type", ["Classification","Clustering"])
            if model_type == "Classification":
                if 'prcp' in ml_df:
                    seuil = st.slider("Seuil précipitations (sécheresse)", 100,1000,500)
                    ml_df['target'] = (ml_df.groupby('year')['prcp'].transform('sum') < seuil).astype(int)
                    features = st.multiselect("Features", ml_df.columns, default=['tavg','prcp'])
                    if st.button("Entraîner"):
                        model, acc = train_classification_model(ml_df, 'target', features)
                        st.success(f"Accuracy: {acc:.2f}")
            else:
                features = st.multiselect("Features", ml_df.columns, default=ml_df.columns[:3])
                n_clusters = st.slider("Nb clusters", 2,10,3)
                if st.button("Cluster"):
                    model, score, clusters, _ = train_clustering_model(ml_df, features, n_clusters)
                    st.success(f"Silhouette: {score:.3f}")
                    ml_df['cluster'] = clusters
                    st.scatter_chart(ml_df, x=features[0], y=features[1], color='cluster')

    st.markdown("---")
    st.markdown(f"<div style='text-align:center'>🌾 AgriClima360 - Données: {len(df):,} points | Tech: {'Dask+ML' if DATA_VIZ_ENABLED else 'Pandas'}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
