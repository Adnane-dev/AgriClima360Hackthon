# 🌾 AgriClima360 – Analyse d'Impact Climatique sur les Rendements Agricoles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MSE Hack 1.0](https://img.shields.io/badge/MSE%20Hack%201.0-Finalist-FF6B6B)](https://mse.rnu.tn/MSE-hack1/)

> **Plateforme intelligente d'analyse des impacts climatiques sur l'agriculture**  
> *Pipeline CRISP-DM complet | GeoAI | Prédiction des rendements | Cartographie des risques*

---

## 📋 Table des Matières

- [🎯 Contexte & Cadre du Projet](#-contexte--cadre-du-projet)
- [👥 Équipe](#-équipe)
- [🎯 Vue d'Ensemble](#-vue-densemble)
- [✨ Fonctionnalités](#-fonctionnalités)
- [📊 Sources de Données](#-sources-de-données)
- [🏗️ Architecture du Pipeline](#️-architecture-du-pipeline)
- [🤖 Modèles ML Implémentés](#-modèles-ml-implémentés)
- [🚀 Installation Rapide](#-installation-rapide)
- [💻 Utilisation](#-utilisation)
- [📈 Visualisations & Dashboard](#-visualisations--dashboard)
- [🌍 Impact Attendu](#-impact-attendu)
- [🔮 Perspectives](#-perspectives)
- [📧 Contact](#-contact)

---

## 🎯 Contexte & Cadre du Projet

**AgriClima360** a été développé dans le cadre du **MSE Hack 1.0 : GeoAI & Serious Games for Ecosystem Restoration**, organisé par la **Manouba School of Engineering** en partenariat avec l'UNESCO, la Hanns Seidel Foundation, et ENoLL.

**Problématique :**  
Les agriculteurs tunisiens et méditerranéens subissent des pertes de rendement croissantes dues au changement climatique (sécheresses, vagues de chaleur, stress hydrique). Le manque d'outils prédictifs accessibles limite leur capacité d'adaptation.

**Solution :**  
AgriClima360 propose un pipeline **CRISP-DM** complet combinant :
- **GeoAI** : analyse spatiale des risques climatiques (cartes de stress hydrique, zonage agro-climatique)
- **Machine Learning** : prédiction des rendements agricoles et détection précoce des sécheresses
- **Dashboard interactif** : interface Streamlit pour l'exploration et la simulation

**Prix visé :** 🏆 *Special Jury Prize – Impact in Ecosystem Restoration*

---

## 👥 Équipe

| Rôle | Nom | Compétences |
|------|-----|-------------|
| **Team Leader & Data Science** | Adnane Mahamadou Saadou | Pipeline CRISP-DM, modélisation ML, backend |
| **Data Visualization & UX** | Radhia Darghoothi | Dashboard Streamlit, graphiques interactifs, UX |
| **GeoAI & Analyse Spatiale** | Abdallah Garba Nassamou | Cartographie, SIG, analyse géotechnique, cas terrain Tunisie |

---

## 🎯 Vue d'Ensemble

**AgriClima360** transforme les données climatiques brutes (NOAA) et les rendements agricoles (FAO) en features pertinentes pour des modèles de Machine Learning performants, accessibles via un dashboard intuitif.

### Problématique Résolue
- ❌ Données climatiques complexes et volumineuses (64 Go)
- ❌ Manque de features spécifiques à l'agriculture
- ❌ Modèles génériques peu adaptés au contexte local
- ❌ Difficulté à quantifier l'impact climatique sur les rendements

### Solution Apportée
- ✅ Feature engineering spécialisé (GDD, stress hydrique, indices agro-climatiques)
- ✅ Pipeline automatisé de traitement (NOAA + FAO)
- ✅ Modèles optimisés par tâche (régression, classification, clustering)
- ✅ Dashboard interactif d'analyse d'impact et de simulation

---

## ✨ Fonctionnalités

### 📊 Traitement des Données
- **Import automatique** des données NOAA (température, précipitations) et FAO (rendements)
- **Nettoyage intelligent** (détection outliers, imputation valeurs manquantes)
- **Agrégations temporelles** (journalier → mensuel → saisonnier → annuel)
- **Géocodage** des stations météo → pays (pour fusion avec FAO)

### 🔬 Feature Engineering Avancé
| Catégorie | Features Créées | Utilité |
|-----------|----------------|---------|
| 🌡️ **Thermiques** | GDD, nuits froides, vagues de chaleur | Cycle de croissance |
| 💧 **Hydriques** | Indice de stress hydrique (water deficit) | Stress hydrique |
| 📈 **Statistiques** | Moyennes glissantes, CV, tendances | Variabilité climatique |
| 🔄 **Interaction** | Précipitation × Température | Effets combinés |

### 🤖 Modèles Machine Learning

| Type | Modèle | Performance | Usage |
|------|--------|-------------|-------|
| **Régression** | Random Forest Regressor | RMSE = 2.17°C | Prédiction TMAX |
| **Classification** | Random Forest Classifier | Accuracy 92%, F1=0.87 | Alerte sécheresse |
| **Clustering** | K-Means (3 clusters) | Silhouette = 0.72 | Zonage agro-climatique |

### 📈 Visualisations Interactives
- Cartes choroplèthes des zones à risque (stress hydrique, sécheresse)
- Évolution temporelle des indicateurs climatiques (2000-2025)
- Graphiques dynamiques des tendances par région (Nord/Centre/Sud)
- Dashboard de simulation utilisateur (choix irrigation → impact rendement)

---

## 📊 Sources de Données

### 1. NOAA GHCN (Global Historical Climatology Network)
- **Variables** : TMAX, TMIN, PRCP, TAVG
- **Période** : 2000-2025 (25 ans)
- **Volume** : 64 Go (Big Data)
- **Couverture** : Mondiale (coordonnées des stations)

### 2. FAO (Food and Agriculture Organization) – FAOSTAT
- **Indicateur** : Rendement agricole (tonnes/ha) – culture principale : maïs
- **Période** : 2000-2025
- **API** : Requêtes automatiques par pays
- **Fallback** : Données synthétiques si API indisponible

### 3. Données géospatiales (GeoAI)
- **Zonage manuel** : Tunisie → Nord (humide), Centre (semi-aride), Sud (aride)
- **Cartes** : Stations NOAA, stress hydrique, propension agricole

---

## 🏗️ Architecture du Pipeline
