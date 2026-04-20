# =============================================================
# Pipeline Machine Learning : Classification, Régression, Clustering
# =============================================================

import pandas as pd
import numpy as np
import pickle
import joblib
import streamlit as st
from datetime import datetime
from config.settings import FEATURES_CLASSIFICATION, FEATURES_REGRESSION

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, silhouette_score
    from sklearn.model_selection import train_test_split
    ML_OK = True
except ImportError:
    ML_OK = False


class MLPipeline:
    """Pipeline ML complet pour l'agriculture."""
    
    def __init__(self):
        self.clf_model = None
        self.reg_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.results_clf = {}
        self.results_reg = {}
        self.results_kmeans = {}
    
    def train_classifier(self, df: pd.DataFrame, target="drought_alert", features=None):
        """Entraîne un classifieur RandomForest pour la prédiction de sécheresse."""
        if not ML_OK:
            return {"error": "Scikit-learn non disponible"}
        
        features = features or FEATURES_CLASSIFICATION
        available = [f for f in features if f in df.columns]
        df_clean = df.dropna(subset=[target])
        
        if len(df_clean) < 10:
            return {"error": "Données insuffisantes"}
        
        X = df_clean[available].fillna(0)
        y = df_clean[target]
        
        # Vérifier si les deux classes sont présentes
        if y.nunique() < 2:
            return {"error": f"Une seule classe présente: {y.unique()}"}
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        self.clf_model = model
        self.results_clf = {
            "accuracy": accuracy_score(y_te, y_pred),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "model": model,
            "features": available,
            "importances": dict(zip(available, model.feature_importances_))
        }
        return self.results_clf
    
    def train_regressor(self, df: pd.DataFrame, target="yield_t_ha", features=None):
        """Entraîne un régresseur RandomForest pour la prédiction de rendement."""
        if not ML_OK:
            return {"error": "Scikit-learn non disponible"}
        
        features = features or FEATURES_REGRESSION
        available = [f for f in features if f in df.columns]
        df_clean = df.dropna(subset=[target])
        
        if len(df_clean) < 10:
            return {"error": "Données insuffisantes"}
        
        X = df_clean[available].fillna(0)
        y = df_clean[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        self.reg_model = model
        self.results_reg = {
            "r2": r2_score(y_te, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
            "model": model,
            "features": available,
            "importances": dict(zip(available, model.feature_importances_))
        }
        return self.results_reg
    
    def train_clustering(self, df: pd.DataFrame, n_clusters=3, features=None):
        """Entraîne un modèle K-Means pour le zonage agro-climatique."""
        if not ML_OK:
            return {"error": "Scikit-learn non disponible"}
        
        features = features or FEATURES_REGRESSION
        available = [f for f in features if f in df.columns]
        X = df[available].fillna(0)
        
        if len(X) < n_clusters:
            return {"error": "Données insuffisantes"}
        
        Xs = self.scaler.fit_transform(X)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = model.fit_predict(Xs)
        
        self.kmeans_model = model
        self.results_kmeans = {
            "silhouette": silhouette_score(Xs, labels) if len(set(labels)) > 1 else 0,
            "n_clusters": n_clusters,
            "labels": labels,
            "features": available
        }
        return self.results_kmeans
    
    def predict_scenario(self, scenario: dict) -> dict:
        """Prédiction en temps réel avec gestion robuste des erreurs."""
        result = {}
        
        # =========================================================
        # 1. CLASSIFICATION (Alerte sécheresse)
        # =========================================================
        if self.clf_model:
            try:
                features = self.results_clf.get("features", [])
                if not features:
                    features = ["tavg_mean", "tmax_mean", "prcp_total", "gdd_total", "heatwave_days", "wdi_mean"]
                
                df = pd.DataFrame([{k: scenario.get(k, 0) for k in features}])
                
                if hasattr(self.clf_model, "predict_proba"):
                    proba = self.clf_model.predict_proba(df)
                    # Gestion robuste du cas où une seule classe est présente
                    if proba.shape[1] >= 2:
                        result["drought_prob"] = float(proba[0][1])
                        result["drought_alert"] = int(proba[0][1] > 0.5)
                    else:
                        result["drought_prob"] = float(proba[0][0])
                        result["drought_alert"] = int(proba[0][0] > 0.5)
                else:
                    pred = self.clf_model.predict(df)[0]
                    result["drought_alert"] = int(pred)
                    result["drought_prob"] = 0.9 if pred == 1 else 0.1
                    
            except Exception as e:
                print(f"Erreur classification: {e}")
                result["drought_prob"] = 0.5
                result["drought_alert"] = 0
        else:
            # Modèle non entraîné → prédiction basée sur WDI
            wdi = scenario.get("wdi_mean", 0.35)
            result["drought_prob"] = min(1.0, max(0.0, wdi))
            result["drought_alert"] = 1 if wdi > 0.35 else 0
        
        # =========================================================
        # 2. RÉGRESSION (Prédiction rendement)
        # =========================================================
        if self.reg_model:
            try:
                features = self.results_reg.get("features", [])
                if not features:
                    features = ["tavg_mean", "tmax_mean", "prcp_total", "gdd_total", 
                               "heatwave_days", "wdi_mean", "diurnal_range", "temp_x_prcp"]
                
                df = pd.DataFrame([{k: scenario.get(k, 0) for k in features}])
                result["yield_pred"] = float(self.reg_model.predict(df)[0])
                
            except Exception as e:
                print(f"Erreur régression: {e}")
                # Fallback: formule empirique
                tavg = scenario.get("tavg_mean", 19.5)
                prcp = scenario.get("prcp_total", 400)
                wdi = scenario.get("wdi_mean", 0.35)
                result["yield_pred"] = max(0.5, min(6.0, 5.0 - 0.1 * (tavg - 18) - 2.0 * wdi + 0.002 * prcp))
        else:
            # Modèle non entraîné → formule empirique
            tavg = scenario.get("tavg_mean", 19.5)
            prcp = scenario.get("prcp_total", 400)
            wdi = scenario.get("wdi_mean", 0.35)
            result["yield_pred"] = max(0.5, min(6.0, 5.0 - 0.1 * (tavg - 18) - 2.0 * wdi + 0.002 * prcp))
        
        # =========================================================
        # 3. CLUSTERING (Zone agro-climatique)
        # =========================================================
        if self.kmeans_model:
            try:
                features = self.results_kmeans.get("features", [])
                if not features:
                    features = ["tavg_mean", "prcp_total", "gdd_total", "wdi_mean"]
                
                df = pd.DataFrame([{k: scenario.get(k, 0) for k in features[:4]}])
                
                if hasattr(self, 'scaler') and self.scaler is not None:
                    X_scaled = self.scaler.transform(df)
                else:
                    X_scaled = df.values
                
                result["cluster"] = int(self.kmeans_model.predict(X_scaled)[0])
                
            except Exception as e:
                print(f"Erreur clustering: {e}")
                # Fallback: clustering basé sur WDI
                wdi = scenario.get("wdi_mean", 0.35)
                if wdi < 0.25:
                    result["cluster"] = 0
                elif wdi < 0.55:
                    result["cluster"] = 1
                else:
                    result["cluster"] = 2
        else:
            # Modèle non entraîné → clustering basé sur WDI
            wdi = scenario.get("wdi_mean", 0.35)
            if wdi < 0.25:
                result["cluster"] = 0
            elif wdi < 0.55:
                result["cluster"] = 1
            else:
                result["cluster"] = 2
        
        return result
    
    def export_bundle(self) -> bytes:
        """Exporte tous les modèles en un bundle sérialisé."""
        bundle = {
            "clf_model": self.clf_model,
            "reg_model": self.reg_model,
            "kmeans_model": self.kmeans_model,
            "scaler": self.scaler,
            "features_clf": self.results_clf.get("features", []),
            "features_reg": self.results_reg.get("features", []),
            "features_kmeans": self.results_kmeans.get("features", []),
            "metrics": {
                "accuracy": self.results_clf.get("accuracy"),
                "f1": self.results_clf.get("f1"),
                "r2": self.results_reg.get("r2"),
                "rmse": self.results_reg.get("rmse"),
                "silhouette": self.results_kmeans.get("silhouette")
            },
            "metadata": {
                "date": datetime.now().isoformat(),
                "version": "3.0"
            }
        }
        return pickle.dumps(bundle)
    
    def save_models(self, path="models/"):
        """Sauvegarde les modèles individuellement sur le disque."""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.clf_model:
            joblib.dump(self.clf_model, f"{path}/classifier.pkl")
        if self.reg_model:
            joblib.dump(self.reg_model, f"{path}/regressor.pkl")
        if self.kmeans_model:
            joblib.dump(self.kmeans_model, f"{path}/kmeans.pkl")
            joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        return path