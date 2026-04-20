# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.ml_pipeline import MLPipeline
from src.utils import load_css, plot_theme
from config.settings import TUNISIA_STATIONS, FAO_CROPS

# =============================================================
# CONFIGURATION STREAMLIT
# =============================================================
st.set_page_config(
    page_title="AgriClima360 — MSE Hack 1.0",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# =============================================================
# INITIALISATION SESSION STATE
# =============================================================
if "climate_raw" not in st.session_state:
    st.session_state["climate_raw"] = None
if "yields_raw" not in st.session_state:
    st.session_state["yields_raw"] = None
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None
if "selected_crop" not in st.session_state:
    st.session_state["selected_crop"] = "Blé dur"

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("### 🌾 AgriClima360")
    st.markdown('<div class="tag">MSE Hack 1.0</div>', unsafe_allow_html=True)
    
    page = st.radio("Navigation", [
        "🏠 Vue d'ensemble",
        "🗺️ GeoAI — Zones de risque",
        "👨‍🌾 Scénario Bechir",
        "📥 1. Collecte données",
        "⚙️ 2. Feature Engineering",
        "🤖 3. Entraînement ML",
        "🚀 4. Mise en production",
        "🏆 Dashboard Impact"
    ], label_visibility="collapsed")
        # =============================================================
    # NOUVEAU : LIEN VERS L'APPLICATION DE VISUALISATION
    # =============================================================
    st.markdown("### Visualisation ")
    st.markdown("""
    <div style="background: #fffff4; border-radius: 8px; padding: 12px; margin: 8px 0; text-align: center;">
        <a href="https://agriclima360-f.streamlit.app/" target="_blank" style="
            color: #4ade80; 
            text-decoration: none; 
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.70rem;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        ">
             Découvrir AgriClima360-Vis
            <span style="font-size: 0.7rem;">↗</span>
        </a>
        <p style="font-size: 10px; color: #6b9e7a; margin-top: 6px;">
            Visualisation de données massives<br>Datashader + HoloViews + Dask
        </p>
    </div>
    """, unsafe_allow_html=True)
    # NOUVEAU : LIEN VERS L'APPLICATION DE VISUALISATION
    # =============================================================
    st.markdown("###  AgriClima360 World ")
    st.markdown("""
    <div style="background: #fffff9; border-radius: 8px; padding: 12px; margin: 8px 0; text-align: center;">
        <a href="https://adnane-dev-climat-imapct-agricole-appstreamlit-app-tcnmcu.streamlit.app/" target="_blank" style="
            color: #4ade80; 
            text-decoration: none; 
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.70rem;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        ">
            🌍 AgriClima360
            <span style="font-size: 0.7rem;">↗</span>
        </a>
        <p style="font-size: 10px; color: #6b9e7a; margin-top: 6px;">
            Analyse Climatique Intelligente
        </p>
    </div>
    """, unsafe_allow_html=True)    
    st.markdown("---")
    st.markdown("**Équipe**")
    st.markdown("Adnane · Radhia · Abdallah")
    st.markdown("---")
    
    # Indicateurs d'avancement
    steps_status = {
        "Données": st.session_state["climate_raw"] is not None,
        "Features": st.session_state["dataset"] is not None,
        "Modèles": st.session_state["pipeline"] is not None
    }
    for step, done in steps_status.items():
        icon = "✅" if done else "⏳"
        st.markdown(f"{icon} {step}")



# =============================================================
# PAGE 1 : VUE D'ENSEMBLE
# =============================================================
def page_overview():
    st.markdown("# 🌾 AgriClima360")
    st.markdown('<div class="tag">Pipeline CRISP-DM complet · GeoAI · Tunisie</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **AgriClima360** est une plateforme prédictive qui combine :
    - **GeoAI** : cartographie des risques climatiques (stress hydrique, sécheresse)
    - **Machine Learning** : alertes précoces et prédiction des rendements
    - **Pipeline CRISP-DM** complet (NOAA + FAO → features → modèles → dashboard)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Hausse des températures", "+1.2°C", "en 25 ans")
    col2.metric("💧 Stress hydrique", "x2", "zones critiques depuis 2010")
    col3.metric("🌾 Pertes évitables", "−45%", "avec irrigation adaptée")
    col4.metric("📍 Zones couvertes", "3", "Nord / Centre / Sud Tunisie")
    
    st.markdown("---")
    st.markdown("## Pipeline CRISP-DM")
    
    steps = [
        ("📥 1. Data Collection", "NOAA GHCN + FAO FAOSTAT", st.session_state["climate_raw"] is not None),
        ("⚙️ 2. Feature Engineering", "GDD, WDI, canicule, interactions", st.session_state["dataset"] is not None),
        ("🤖 3. Modeling", "Random Forest + K-Means", st.session_state["pipeline"] is not None),
        ("🚀 4. Deployment", "Dashboard Streamlit + prédictions", True)
    ]
    
    for title, desc, done in steps:
        icon = "✅" if done else "⏳"
        st.markdown(f"""
        <div class="step-box">
            <h4>{icon} {title}</h4>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## ")
    st.markdown("**Impact in Ecosystem Restoration**")


# =============================================================
# PAGE 2 : GeoAI — ZONES DE RISQUE
# =============================================================
def page_geoai():
    st.markdown("# 🗺️ GeoAI — Cartographie des risques")
    st.markdown('<div class="tag">Analyse spatiale · Tunisie</div>', unsafe_allow_html=True)
    
    st.markdown("### Zonage agro-climatique de la Tunisie")
    
    col1, col2, col3 = st.columns(3)
    zones = [
        ("🟢 Nord — Zone favorable", "Béja, Jendouba, Bizerte", "620 mm/an · 17.2°C", "Potentiel maximal · Céréales, vignes"),
        ("🟡 Centre — Zone vulnérable", "Kairouan, Sfax, Kasserine", "310 mm/an · 20.1°C", "Irrigation nécessaire · Oliviers, orge"),
        ("🔴 Sud — Zone critique", "Gabès, Médenine, Tataouine", "160 mm/an · 23.8°C", "Cultures résistantes uniquement")
    ]
    for col, (title, region, climat, usage) in zip([col1, col2, col3], zones):
        with col:
            st.markdown(f"""
            <div class="step-box">
                <h4>{title}</h4>
                <p><b>{region}</b><br>{climat}<br><br>{usage}</p>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Carte des stations NOAA en Tunisie")
    stations_df = pd.DataFrame([{"Nom": k, "Latitude": v["lat"], "Longitude": v["lon"], "Région": v["region"]}
                                 for k, v in TUNISIA_STATIONS.items()])
    fig = px.scatter_mapbox(stations_df, lat="Latitude", lon="Longitude", color="Région",
                             size_max=12, zoom=6, height=450,
                             mapbox_style="carto-positron",
                             title="Stations météo GHCN – Tunisie",
                             hover_name="Nom")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Indicateurs de risque par zone")
    risk_data = pd.DataFrame({
        "Zone": ["Nord", "Centre", "Sud"],
        "Stress hydrique (WDI)": [0.18, 0.42, 0.71],
        "Rendement moy. (t/ha)": [4.2, 2.8, 1.5],
        "Alerte sécheresse": ["Faible", "Modérée", "Élevée"]
    })
    st.dataframe(risk_data, use_container_width=True)
    
    fig_bar = px.bar(risk_data, x="Zone", y="Stress hydrique (WDI)", color="Zone",
                      color_discrete_sequence=["#2d6a4f", "#f4a261", "#ef4444"],
                      title="Indice de stress hydrique par zone (WDI)")
    st.plotly_chart(plot_theme(fig_bar), use_container_width=True)


# =============================================================
# PAGE 3 : SCÉNARIO BECHIR
# =============================================================
def page_scenario():
    st.markdown("# 👨‍🌾 Scénario terrain — Bechir, agriculteur à Béja")
    st.markdown('<div class="tag">Cas réel · Impact humain</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-box">
        <h4>👨‍🌾 Bechir Trabelsi – Céréalier à Béja (Nord-Ouest Tunisie)</h4>
        <p>
        <b>Culture :</b> Blé dur · <b>Surface :</b> 18 ha · <b>Saison :</b> Oct 2024 – Juin 2025<br><br>
        Bechir utilise AgriClima360 depuis le début de la campagne. Le dashboard lui a envoyé une alerte :
        les données NOAA des 3 dernières semaines indiquent un <b>stress hydrique sévère</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Température moy.", "18.9°C", "+1.7°C vs normale")
    col2.metric("💧 Précipitations", "312 mm", "-38% vs normale")
    col3.metric("📉 Indice WDI", "0.51", "Stress sévère")
    col4.metric("🌡️ Jours canicule", "7 jours", "+5 vs historique")
    
    st.markdown("""
    <div class="warn-box">
        <p>⚠️ <b>ALERTE SÉCHERESSE SÉVÈRE</b><br>
        Rendement prédit : <b>2.1 t/ha</b> vs moyenne historique 3.8 t/ha<br>
        Perte estimée : <b>45%</b> (environ 30 tonnes sur 18 ha)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Recommandations générées par AgriClima360")
    recos = [
        "Activer l'irrigation d'appoint : 3 arrosages x 30mm entre mars et mai",
        "Récolte anticipée recommandée : avant le 15 juin (risque canicule)",
        "Surface à réduire à 12 ha pour la prochaine campagne si tendance confirmée",
        "Contacter CRDA Béja pour subvention semences résistantes à la chaleur"
    ]
    for i, rec in enumerate(recos, 1):
        st.markdown(f"""
        <div class="step-box" style="padding:12px 18px;">
            <p><b>{i:02d}.</b> {rec}</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Évolution du rendement – Béja")
    years = np.arange(2000, 2026)
    historical = [3.8 - 0.03 * (y - 2000) + 0.2 * np.random.normal() for y in years[:-1]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years[:-1], y=historical, mode="lines+markers", name="Historique", line=dict(color="#2d6a4f")))
    fig.add_trace(go.Scatter(x=[2024, 2025], y=[2.1, 2.3], mode="markers", name="Prédiction", marker=dict(size=12, color="#f4a261", symbol="star")))
    fig.add_hline(y=3.8, line_dash="dot", line_color="#f4a261", annotation_text="Moyenne historique (3.8 t/ha)")
    fig.update_layout(title="Rendement blé dur – Béja", xaxis_title="Année", yaxis_title="t/ha")
    st.plotly_chart(plot_theme(fig), use_container_width=True)


# =============================================================
# PAGE 4 : COLLECTE DONNÉES
# =============================================================
def page_data_collection():
    st.markdown("# 📥 1. Collecte des données")
    st.markdown('<div class="tag">Étape 1 — CRISP-DM : Data Understanding</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Année début", min_value=1990, max_value=2020, value=2000)
        end_year = st.number_input("Année fin", min_value=2005, max_value=2025, value=2024)
    with col2:
        selected_crop = st.selectbox("Culture principale", list(FAO_CROPS.keys()), index=0)
        st.session_state["selected_crop"] = selected_crop
    
    if st.button("📥 Collecter les données", type="primary"):
        with st.spinner("Collecte des données climatiques..."):
            climate_df = DataCollector.fetch_climate_data(start_year, end_year)
            st.session_state["climate_raw"] = climate_df
        
        with st.spinner("Collecte des données de rendement..."):
            yields_df = DataCollector.fetch_yield_data(selected_crop, start_year, end_year)
            st.session_state["yields_raw"] = yields_df
        
        st.success(f"✅ Données collectées : {len(climate_df)} enregistrements climatiques, {len(yields_df)} années de rendement")
    
    if st.session_state["climate_raw"] is not None:
        st.markdown("---")
        st.markdown("## Aperçu des données")
        
        tab1, tab2 = st.tabs(["🌡️ Données climatiques (NOAA)", "🌾 Rendements (FAO)"])
        with tab1:
            st.dataframe(st.session_state["climate_raw"].head(24), use_container_width=True)
            st.caption(f"Total : {len(st.session_state['climate_raw'])} lignes")
        with tab2:
            if st.session_state["yields_raw"] is not None:
                st.dataframe(st.session_state["yields_raw"], use_container_width=True)


# =============================================================
# PAGE 5 : FEATURE ENGINEERING
# =============================================================
def page_feature_engineering():
    st.markdown("# ⚙️ 2. Feature Engineering")
    st.markdown('<div class="tag">Étape 2 — CRISP-DM : Data Preparation</div>', unsafe_allow_html=True)
    
    if st.session_state["climate_raw"] is None:
        st.info("👈 Commencez par collecter les données (page 1).")
        return
    
    normal_prcp = st.slider("Précipitation normale de référence (mm/an)", 200, 800, 500,
                            help="Référence régionale pour le calcul du stress hydrique (WDI)")
    
    if st.button("🔧 Construire le dataset", type="primary"):
        with st.spinner("Construction des features..."):
            dataset = FeatureEngineer.build_dataset(
                st.session_state["climate_raw"],
                st.session_state.get("yields_raw", pd.DataFrame()),
                normal_prcp
            )
            st.session_state["dataset"] = dataset
        
        st.success(f"✅ Dataset construit : {len(dataset)} observations × {len(dataset.columns)} colonnes")
        
        # Affichage des statistiques
        st.markdown("---")
        st.markdown("## Statistiques du dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Années couvertes", f"{dataset['year'].min()} → {dataset['year'].max()}")
        col2.metric("Alertes sécheresse", f"{dataset['drought_alert'].sum()} années")
        col3.metric("Rendement moyen", f"{dataset['yield_t_ha'].mean():.2f} t/ha")
    
    if st.session_state["dataset"] is not None:
        st.markdown("---")
        st.markdown("## Aperçu du dataset")
        st.dataframe(st.session_state["dataset"], use_container_width=True)
        
        csv = st.session_state["dataset"].to_csv(index=False)
        st.download_button("⬇️ Télécharger dataset (CSV)", csv, "agriclima360_dataset.csv", "text/csv")


# =============================================================
# PAGE 6 : ENTRAÎNEMENT ML
# =============================================================
def page_training():
    st.markdown("# 🤖 3. Entraînement ML")
    st.markdown('<div class="tag">Étape 3 — CRISP-DM : Modeling</div>', unsafe_allow_html=True)
    
    if st.session_state["dataset"] is None:
        st.info("👈 Construisez d'abord le dataset (page 2).")
        return
    
    dataset = st.session_state["dataset"]
    
    col1, col2 = st.columns(2)
    with col1:
        train_clf = st.checkbox("Classifier (alerte sécheresse)", value=True)
    with col2:
        train_reg = st.checkbox("Régresseur (rendement)", value=True)
    
    if st.button("🚀 Lancer l'entraînement", type="primary"):
        pipeline = MLPipeline()
        
        if train_clf and "drought_alert" in dataset.columns:
            with st.spinner("Entraînement du classifier..."):
                res_clf = pipeline.train_classifier(dataset)
                st.success(f"✅ Classifier : Accuracy={res_clf['accuracy']:.1%}, F1={res_clf['f1']:.3f}")
        
        if train_reg and "yield_t_ha" in dataset.columns:
            with st.spinner("Entraînement du régresseur..."):
                res_reg = pipeline.train_regressor(dataset.dropna(subset=["yield_t_ha"]))
                st.success(f"✅ Régresseur : R²={res_reg['r2']:.3f}, RMSE={res_reg['rmse']:.3f} t/ha")
        
        with st.spinner("Clustering K-Means..."):
            res_km = pipeline.train_clustering(dataset, n_clusters=3)
            st.success(f"✅ K-Means : Silhouette={res_km['silhouette']:.3f}")
        
        st.session_state["pipeline"] = pipeline
        
        st.markdown("---")
        st.markdown("## Résultats détaillés")
        
        if pipeline.results_clf:
            st.subheader("📊 Classification sécheresse")
            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy", f"{pipeline.results_clf['accuracy']:.1%}")
            col_b.metric("F1-Score", f"{pipeline.results_clf['f1']:.3f}")
            
            imp = pd.Series(pipeline.results_clf["importances"]).sort_values(ascending=True)
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="Importance des features")
            st.plotly_chart(plot_theme(fig), use_container_width=True)
        
        if pipeline.results_reg:
            st.subheader("📈 Régression rendement")
            col_a, col_b = st.columns(2)
            col_a.metric("R²", f"{pipeline.results_reg['r2']:.3f}")
            col_b.metric("RMSE", f"{pipeline.results_reg['rmse']:.3f} t/ha")
        
        if pipeline.results_kmeans:
            st.subheader("🔵 Clustering K-Means")
            st.metric("Silhouette Score", f"{pipeline.results_kmeans['silhouette']:.3f}")


# =============================================================
# PAGE 7 : MISE EN PRODUCTION
# =============================================================
def page_production():
    st.markdown("# 🚀 4. Mise en production")
    st.markdown('<div class="tag">Étape 4 — CRISP-DM : Deployment</div>', unsafe_allow_html=True)
    
    if st.session_state["pipeline"] is None:
        st.info("👈 Entraînez les modèles d'abord (page 3).")
        return
    
    pipeline = st.session_state["pipeline"]
    
    st.markdown("## 🔮 Simulation en temps réel")
    st.markdown("Ajustez les paramètres climatiques pour prédire le rendement et l'alerte sécheresse.")
    
    col1, col2 = st.columns(2)
    with col1:
        tavg = st.slider("🌡️ Température moyenne (°C)", 10.0, 35.0, 19.5, 0.1)
        prcp = st.slider("💧 Précipitations annuelles (mm)", 100, 800, 400, 10)
        gdd = st.slider("🌱 GDD total (°C·jours)", 500, 2500, 1600, 50)
    with col2:
        wdi = st.slider("📉 Water Deficit Index (0-1)", 0.0, 1.0, 0.35, 0.01)
        heatwave = st.slider("🔥 Jours canicule (Tmax > 35°C)", 0, 60, 5)
        diurnal = st.slider("🌙 Amplitude thermique diurne (°C)", 5.0, 20.0, 10.0, 0.5)
    
    scenario = {
        "tavg_mean": tavg,
        "tmax_mean": tavg + 7,
        "prcp_total": prcp,
        "gdd_total": gdd,
        "wdi_mean": wdi,
        "heatwave_days": heatwave,
        "diurnal_range": diurnal,
        "temp_x_prcp": tavg * prcp / 100
    }
    
    if st.button("🔮 Prédire", type="primary"):
        result = pipeline.predict_scenario(scenario)
        if result:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🌾 Rendement prédit", f"{result['yield_pred']:.2f} t/ha")
            col_b.metric("⚠️ Alerte sécheresse", "OUI" if result["drought_alert"] else "NON")
            col_c.metric("📊 Probabilité", f"{result['drought_prob']:.1%}")
            
            if result["drought_alert"]:
                st.markdown("""
                <div class="warn-box">
                    <p>⚠️ <b>Sécheresse probable</b> — Recommandations : irrigation d'appoint, récolte anticipée, contacter le CRDA.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="ok-box">
                    <p>✅ <b>Conditions favorables</b> — Campagne normale attendue.</p>
                </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📦 Export des modèles")
    
    model_bytes = pipeline.export_bundle()
    st.download_button("⬇️ Télécharger le bundle des modèles (.pkl)", model_bytes, "agriclima360_model_bundle.pkl", "application/octet-stream")
    
    if st.button("💾 Sauvegarder les modèles sur le serveur"):
        path = pipeline.save_models()
        st.success(f"✅ Modèles sauvegardés dans {path}")


# =============================================================
# PAGE 8 : DASHBOARD IMPACT
# =============================================================
def page_impact():
    st.markdown("# 🏆 Dashboard Impact")
    st.markdown('<div class="tag">Synthèse · Jury Special Prize</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div class="kpi"><div class="kpi-v">85 000</div><div class="kpi-l">hectares couverts</div></div>', unsafe_allow_html=True)
    col2.markdown('<div class="kpi"><div class="kpi-v">12 400</div><div class="kpi-l">agriculteurs ciblés</div></div>', unsafe_allow_html=True)
    col3.markdown('<div class="kpi"><div class="kpi-v">21 j</div><div class="kpi-l">alerte précoce</div></div>', unsafe_allow_html=True)
    col4.markdown('<div class="kpi"><div class="kpi-v">-45%</div><div class="kpi-l">pertes évitables</div></div>', unsafe_allow_html=True)
    
    if st.session_state["pipeline"]:
        st.markdown("---")
        st.markdown("## Performances des modèles")
        pipeline = st.session_state["pipeline"]
        
        col_a, col_b, col_c = st.columns(3)
        if pipeline.results_clf:
            col_a.metric("Accuracy sécheresse", f"{pipeline.results_clf['accuracy']:.1%}")
        if pipeline.results_reg:
            col_b.metric("R² rendement", f"{pipeline.results_reg['r2']:.3f}")
        if pipeline.results_kmeans:
            col_c.metric("Silhouette K-Means", f"{pipeline.results_kmeans['silhouette']:.3f}")
    
    st.markdown("---")
    st.markdown("## Partenaires institutionnels")
    col_a, col_b, col_c = st.columns(3)
    partners = [
        ("GDA Sidi Amor", "Partenaire officiel MSE Hack", "Déploiement pilote — groupements agriculteurs Nord Tunisie"),
        ("CRDA Béja/Jendouba", "Direction régionale agriculture", "Intégration alertes sécheresse dans les systèmes officiels"),
        ("IRESA / INGREF", "Recherche agronomique Tunisie", "Validation scientifique — extension cultures & régions")
    ]
    for col, (name, role, desc) in zip([col_a, col_b, col_c], partners):
        with col:
            st.markdown(f"""
            <div class="step-box">
                <h4>{name}</h4>
                <p><b>{role}</b><br><br>{desc}</p>
            </div>""", unsafe_allow_html=True)


# =============================================================
# ROUTAGE DES PAGES
# =============================================================
if page == "🏠 Vue d'ensemble":
    page_overview()
elif page == "🗺️ GeoAI — Zones de risque":
    page_geoai()
elif page == "👨‍🌾 Scénario Bechir":
    page_scenario()
elif page == "📥 1. Collecte données":
    page_data_collection()
elif page == "⚙️ 2. Feature Engineering":
    page_feature_engineering()
elif page == "🤖 3. Entraînement ML":
    page_training()
elif page == "🚀 4. Mise en production":
    page_production()
elif page == "🏆 Dashboard Impact":
    page_impact()

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown("""<div style="text-align:center;font-size:11px;color:#a0c4a0">
AgriClima360 v3.0 · Pipeline CRISP-DM · NOAA + FAO · GeoAI Tunisie · MSE Hack 1.0</div>""", unsafe_allow_html=True)
