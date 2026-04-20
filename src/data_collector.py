# src/data_collector.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from config.settings import NOAA_TOKEN, TUNISIA_STATIONS, FAO_CROPS

class DataCollector:
    """Collecte des données climatiques (NOAA) et agricoles (FAO)."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Collecte des données NOAA...")
    def fetch_climate_data(start_year=2000, end_year=2024):
        """Génère des données climatiques simulées pour la Tunisie."""
        np.random.seed(42)
        years = np.arange(start_year, end_year + 1)
        months = np.arange(1, 13)
        records = []
        
        drought_years = {2002, 2008, 2012, 2016, 2021, 2023}
        
        for yr in years:
            warming = 0.04 * (yr - 2000)  # +0.4°C par décennie
            prcp_factor = 0.55 if yr in drought_years else np.random.uniform(0.80, 1.25)
            
            for mo in months:
                season = -2 * np.cos(2 * np.pi * (mo - 3) / 12)
                tavg = 17.5 + season + warming + np.random.normal(0, 0.7)
                tmax = tavg + 7 + np.random.normal(0, 0.5)
                tmin = tavg - 6 + np.random.normal(0, 0.5)
                prcp = max(0, 45 + 35 * np.sin((mo - 1) * np.pi / 6) + np.random.exponential(12))
                prcp = prcp * prcp_factor
                
                records.append({
                    "year": yr, "month": mo,
                    "tavg": round(tavg, 2), "tmax": round(tmax, 2),
                    "tmin": round(tmin, 2), "prcp": round(prcp, 1)
                })
        
        return pd.DataFrame(records)
    
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner="Collecte des données FAO...")
    def fetch_yield_data(crop="Blé dur", start_year=2000, end_year=2024):
        """Génère des données de rendement simulées pour la Tunisie."""
        np.random.seed(42)
        years = np.arange(start_year, end_year + 1)
        
        # Paramètres par culture
        crop_params = {
            "Blé dur": {"base": 3.5, "trend": 0.02},
            "Blé tendre": {"base": 2.8, "trend": 0.018},
            "Orge": {"base": 1.8, "trend": 0.015},
            "Maïs": {"base": 5.5, "trend": 0.025},
            "Tomate": {"base": 35.0, "trend": 0.01},
            "Olive": {"base": 2.1, "trend": 0.02}
        }
        params = crop_params.get(crop, {"base": 3.0, "trend": 0.02})
        
        drought_years = {2002, 2008, 2012, 2016, 2021, 2023}
        records = []
        
        for yr in years:
            drought_effect = 0.65 if yr in drought_years else 1.0
            yield_val = params["base"] * (1 + params["trend"] * (yr - 2000))
            yield_val = yield_val * drought_effect + np.random.normal(0, 0.12)
            records.append({"year": yr, "yield_t_ha": round(max(0.4, yield_val), 2)})
        
        return pd.DataFrame(records)