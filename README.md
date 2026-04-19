# 🌾 AgriClima360 - Analyse d'Impact Climatique sur les Rendements Agricoles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MLflow](https://img.shields.io/badge/MLflow-2.7+-0194E2.svg)](https://mlflow.org)

> **Plateforme intelligente d'analyse des impacts climatiques sur l'agriculture**  
> Transformation des données brutes NOAA en modèles prédictifs performants (classification, clustering, régression)

## 📋 Table des Matières

- [🎯 Vue d'Ensemble](#-vue-densemble)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🏗️ Architecture du Projet](#️-architecture-du-projet)
- [🚀 Installation Rapide](#-installation-rapide)
- [📊 Pipeline de Traitement](#-pipeline-de-traitement)
- [🤖 Modèles ML Implémentés](#-modèles-ml-implémentés)
- [📈 Visualisations & Dashboard](#-visualisations--dashboard)
- [💻 Utilisation](#-utilisation)
- [🔬 Feature Engineering](#-feature-engineering)
- [📊 Métriques & Évaluation](#-métriques--évaluation)
- [📚 Documentation](#-documentation)
- [🤝 Contribution](#-contribution)
- [📧 Contact](#-contact)

## 🎯 Vue d'Ensemble

**AgriClima360** est une solution complète d'analyse de données climatiques appliquées à l'agriculture. Le projet transforme les données brutes de la NOAA (National Oceanic and Atmospheric Administration) en features agricoles pertinentes pour construire des modèles de Machine Learning performants.

### Problématique Résolue
- ❌ Données climatiques brutes complexes et volumineuses
- ❌ Manque de features spécifiques à l'agriculture
- ❌ Modèles génériques peu adaptés au contexte agricole
- ❌ Difficulté à quantifier l'impact climatique sur les rendements

### Solution Apportée
- ✅ Feature engineering spécialisé (GDD, stress hydrique, indices agro-climatiques)
- ✅ Pipeline automatisé de traitement des données NOAA
- ✅ Modèles optimisés par tâche (classification/ clustering/ régression)
- ✅ Dashboard interactif d'analyse d'impact

## ✨ Fonctionnalités

### 📊 Traitement des Données
- **Import automatique** des données NOAA (température, précipitations, vent, humidité)
- **Nettoyage intelligent** (détection outliers, imputation valeurs manquantes)
- **Agrégations temporelles** (journalier → mensuel → saisonnier → annuel)
- **Validation qualité** avec rapports automatisés

### 🔬 Feature Engineering Avancé
| Catégorie | Features Créées | Utilité |
|-----------|----------------|---------|
| 🌡️ **Thermiques** | GDD, nuits froides, vagues de chaleur | Cycle de croissance |
| 💧 **Hydriques** | SPI, SPEI, déficit précipitation | Stress hydrique |
| 🌾 **Phénologiques** | Fenêtres de semis, risques récolte | Planning cultural |
| 📈 **Statistiques** | Moyennes glissantes, CV, tendances | Variabilité climatique |
| 🔄 **Interaction** | Précipitation × Température | Effets combinés |

### 🤖 Modèles Machine Learning

#### Classification
- Prédiction du risque de sécheresse (faible/moyen/élevé)
- Détection d'événements extrêmes (gel, canicule, inondation)
- Classification des zones de vulnérabilité agricole

#### Clustering
- Délimitation automatique des zones agro-climatiques
- Regroupement des régions par similarité climatique
- Identification des patterns saisonniers

#### Régression
- Prédiction des rendements agricoles (tonnes/ha)
- Estimation de l'impact des variables climatiques
- Forecasting des séries temporelles météo

### 📈 Visualisations Interactives
- Cartes choroplèthes des zones à risque
- Évolution temporelle des indicateurs clés
- Matrices de corrélation features-rendement
- Dashboard d'importance des features (SHAP)

## 🏗️ Architecture du Projet
