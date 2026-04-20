# src/ml_pipeline.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, silhouette_score

class MLPipeline:
    """Pipeline complet d'entraînement et prédiction ML."""
    
    FEATURES_CLASSIFICATION = ["tavg_mean", "tmax_mean", "prcp_total", "gdd_total", "heatwave_days", "wdi_mean"]
    FEATURES_REGRESSION = ["tavg_mean", "tmax_mean", "prcp_total", "gdd_total", "heatwave_days", "wdi_mean", "diurnal_range", "temp_x_prcp"]
    
    def __init__(self):
        self.clf_model = None
        self.reg_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.results_clf = {}
        self.results_reg = {}
        self.results_kmeans = {}
    
    def train_classifier(self, df, target="drought_alert", features=None, test_size=0.2):
        """Entraîne un Random Forest pour la classification (alerte sécheresse)."""
        features = features or self.FEATURES_CLASSIFICATION
        available = [f for f in features if f in df.columns]
        
        X = df[available].fillna(0)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        
        model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.clf_model = model
        self.results_clf = {
            "model": model,
            "features": available,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "cv_f1": round(float(cross_val_score(model, X, y, cv=min(5, len(X)//4), scoring="f1").mean()), 4),
            "importances": dict(zip(available, model.feature_importances_))
        }
        return self.results_clf
    
    def train_regressor(self, df, target="yield_t_ha", features=None, test_size=0.2):
        """Entraîne un Random Forest pour la régression (prédiction rendement)."""
        features = features or self.FEATURES_REGRESSION
        available = [f for f in features if f in df.columns]
        
        df_valid = df.dropna(subset=[target])
        if len(df_valid) < 10:
            return {"error": "Données insuffisantes"}
        
        X = df_valid[available].fillna(0)
        y = df_valid[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.reg_model = model
        self.results_reg = {
            "model": model,
            "features": available,
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "cv_r2": round(float(cross_val_score(model, X, y, cv=min(5, len(X)//4), scoring="r2").mean()), 4),
            "importances": dict(zip(available, model.feature_importances_)),
            "y_test": y_test.values,
            "y_pred": y_pred
        }
        return self.results_reg
    
    def train_clustering(self, df, n_clusters=3, features=None):
        """Entraîne un K-Means pour le zonage agro-climatique."""
        features = features or self.FEATURES_REGRESSION
        available = [f for f in features if f in df.columns]
        
        X = df[available].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        
        sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0.0
        
        self.kmeans_model = model
        self.results_kmeans = {
            "model": model,
            "scaler": self.scaler,
            "features": available,
            "labels": labels,
            "n_clusters": n_clusters,
            "silhouette": round(float(sil_score), 4)
        }
        return self.results_kmeans
    
    def predict_scenario(self, scenario_input):
        """Prédiction temps réel à partir d'un scénario utilisateur."""
        if self.clf_model is None or self.reg_model is None:
            return {}
        
        # Préparation des DataFrames
        clf_features = self.results_clf.get("features", [])
        reg_features = self.results_reg.get("features", [])
        
        clf_df = pd.DataFrame([{k: scenario_input.get(k, 0) for k in clf_features}])
        reg_df = pd.DataFrame([{k: scenario_input.get(k, 0) for k in reg_features}])
        
        # Prédictions
        clf_proba = self.clf_model.predict_proba(clf_df)[0][1] if hasattr(self.clf_model, "predict_proba") else 0
        clf_label = self.clf_model.predict(clf_df)[0]
        reg_pred = self.reg_model.predict(reg_df)[0]
        
        return {
            "drought_prob": round(float(clf_proba), 3),
            "drought_alert": int(clf_label),
            "yield_pred": round(float(reg_pred), 2)
        }
    
    def save_models(self, path="models/"):
        """Sauvegarde les modèles au format .pkl et .joblib."""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.clf_model:
            joblib.dump(self.clf_model, f"{path}/classifier.pkl")
        if self.reg_model:
            joblib.dump(self.reg_model, f"{path}/regressor.pkl")
        if self.kmeans_model:
            joblib.dump(self.kmeans_model, f"{path}/kmeans.pkl")
            joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        return f"Modèles sauvegardés dans {path}"
    
    def export_bundle(self):
        """Exporte tous les modèles dans un bundle pickle unique."""
        bundle = {
            "clf_model": self.clf_model,
            "reg_model": self.reg_model,
            "kmeans_model": self.kmeans_model,
            "scaler": self.scaler,
            "results_clf": {k: v for k, v in self.results_clf.items() if k != "model"},
            "results_reg": {k: v for k, v in self.results_reg.items() if k != "model"},
            "version": "3.0"
        }
        return pickle.dumps(bundle)