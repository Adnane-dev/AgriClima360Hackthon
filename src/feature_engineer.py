# src/feature_engineer.py
import streamlit as st
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Construction des features agro-climatiques."""
    
    GDD_BASE = 10  # °C pour le blé dur
    HEAT_THRESHOLD = 35  # °C
    
    @classmethod
    def build_dataset(cls, climate_df, yields_df, normal_prcp=500):
        """
        Construit le dataset complet avec features agricoles.
        
        Args:
            climate_df: DataFrame avec colonnes year, month, tavg, tmax, tmin, prcp
            yields_df: DataFrame avec colonnes year, yield_t_ha
            normal_prcp: Précipitation normale annuelle de référence (mm)
        
        Returns:
            DataFrame avec features agrégées par année
        """
        if climate_df.empty:
            return pd.DataFrame()
        
        df = climate_df.copy()
        
        # 1. Features mensuelles
        df["gdd"] = (df["tavg"] - cls.GDD_BASE).clip(lower=0)
        df["heatwave_day"] = (df["tmax"] > cls.HEAT_THRESHOLD).astype(int)
        monthly_normal_prcp = normal_prcp / 12
        df["wdi"] = ((monthly_normal_prcp - df["prcp"]) / monthly_normal_prcp).clip(0, 1)
        df["temp_x_prcp"] = df["tavg"] * df["prcp"]
        df["diurnal_range"] = df["tmax"] - df["tmin"]
        
        # 2. Agrégation annuelle
        annual = df.groupby("year").agg(
            tavg_mean=("tavg", "mean"),
            tmax_mean=("tmax", "mean"),
            tmin_mean=("tmin", "mean"),
            prcp_total=("prcp", "sum"),
            gdd_total=("gdd", "sum"),
            heatwave_days=("heatwave_day", "sum"),
            wdi_mean=("wdi", "mean"),
            diurnal_range=("diurnal_range", "mean"),
            temp_x_prcp=("temp_x_prcp", "mean")
        ).reset_index()
        
        # 3. Features temporelles (rolling, tendances)
        annual["prcp_total_trend"] = annual["prcp_total"].diff().fillna(0)
        annual["tavg_mean_trend"] = annual["tavg_mean"].diff().fillna(0)
        annual["prcp_total_ma3"] = annual["prcp_total"].rolling(3, min_periods=1).mean()
        annual["tavg_mean_ma3"] = annual["tavg_mean"].rolling(3, min_periods=1).mean()
        
        # 4. Alerte sécheresse (cible pour classification)
        annual["drought_alert"] = (annual["wdi_mean"] > 0.35).astype(int)
        
        # 5. Fusion avec les rendements
        if not yields_df.empty and "yield_t_ha" in yields_df.columns:
            merged = annual.merge(yields_df[["year", "yield_t_ha"]], on="year", how="left")
        else:
            merged = annual.copy()
            merged["yield_t_ha"] = np.nan
        
        return merged.sort_values("year").reset_index(drop=True)
    
    @staticmethod
    def get_feature_descriptions():
        """Retourne les descriptions des features pour l'interface."""
        return {
            "tavg_mean": "Température moyenne annuelle (°C)",
            "tmax_mean": "Température maximale annuelle (°C)",
            "tmin_mean": "Température minimale annuelle (°C)",
            "prcp_total": "Précipitation totale annuelle (mm)",
            "gdd_total": "Growing Degree Days (accumulation de chaleur)",
            "heatwave_days": "Nombre de jours avec Tmax > 35°C",
            "wdi_mean": "Water Deficit Index (0=OK, 1=stress critique)",
            "diurnal_range": "Amplitude thermique diurne (°C)",
            "temp_x_prcp": "Interaction température × précipitation",
            "prcp_total_trend": "Tendance des précipitations (Δmm/an)",
            "tavg_mean_trend": "Tendance des températures (Δ°C/an)",
            "drought_alert": "Alerte sécheresse (0=Non, 1=Oui)"
        }