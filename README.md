# 🔋 Optaro - Prédiction de Consommation Énergétique

## 📋 Description

Optaro est un projet d'analyse et de prédiction de consommation énergétique utilisant des techniques d'apprentissage automatique avancées. Le système combine des données de consommation historiques avec des informations météorologiques et contextuelles pour prédire avec précision la consommation énergétique future.

## 🎯 Objectifs

- **Prédiction précise** : Utiliser XGBoost pour prédire la consommation énergétique quotidienne
- **Analyse temporelle** : Comprendre les patterns saisonniers et cycliques de consommation
- **Intégration météo** : Incorporer les données météorologiques pour améliorer les prédictions
- **Visualisation** : Fournir des tableaux de bord interactifs pour l'analyse des résultats

## 🏗️ Architecture du Projet

```
optaro/
├── 📊 Extraction et Nettoyage des Données
│   ├── extraction_data.py          # Extraction depuis l'API
│   ├── clean_data.py               # Nettoyage des données brutes
│   └── get_meteo.py                # Récupération données météo
│
├── 🔧 Préparation et Enrichissement
│   ├── fusion_data_context.py      # Fusion données + contexte
│   ├── fix_holidays.py             # Correction jours fériés
│   └── data_with_context_fixed.csv # Dataset final enrichi
│
├── 🤖 Modélisation et Prédiction
│   ├── model_xgboost.py            # Modèle principal XGBoost
│   ├── predict_hybrid.py           # Prédictions hybrides
│   ├── predict_realistic.py        # Prédictions réalistes
│   └── xgboost_2023_predictions.py # Prédictions 2023
│
├── 📈 Analyse et Visualisation
│   ├── analyze_errors.py           # Analyse des erreurs
│   ├── create_detailed_charts.py   # Graphiques détaillés
│   ├── create_summary_dashboard.py # Dashboard de synthèse
│   └── visualize_hybrid_predictions.py # Visualisation prédictions
│
├── 🔍 Diagnostic et Optimisation
│   ├── diagnostic_precision.py     # Diagnostic de précision
│   ├── check_training_methodology.py # Vérification méthodologie
│   └── analyse_causes_erreurs.py   # Analyse causes d'erreurs
│
└── 📁 Données et Modèles
    ├── data_7_years.csv            # Données brutes 7 ans
    ├── weather_haifa.csv           # Données météo Haïfa
    ├── jours_feries.csv            # Jours fériés
    └── xgboost_energy_model.pkl    # Modèle entraîné
```

## 🚀 Installation et Configuration

### Prérequis
- Python 3.8+
- pip ou conda

### Installation des dépendances
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn requests
```

### Configuration de l'API
Modifier l'URL de l'API dans `extraction_data.py` :
```python
API_URL = "http://votre-serveur:8000/query"
```

## 📊 Pipeline de Données

### 1. Extraction des Données
```bash
python extraction_data.py
```
- Extraction des données de consommation des 7 dernières années
- Source : API interne avec requête SQL optimisée
- Format : CSV avec colonnes Day, SourceID, QuantityID, DailyAverage

### 2. Nettoyage des Données
```bash
python clean_data.py
```
- Suppression des valeurs aberrantes et manquantes
- Normalisation des formats de dates
- Élimination des doublons

### 3. Enrichissement Contextuel
```bash
python get_meteo.py
python fusion_data_context.py
python fix_holidays.py
```
- Ajout des données météorologiques (température, précipitations, vent)
- Intégration des jours fériés et contexte temporel
- Création du dataset final enrichi

## 🤖 Modélisation

### Modèle Principal : XGBoost
Le modèle utilise **XGBoost** avec les caractéristiques suivantes :

#### Features Engineering
- **Temporelles** : année, mois, jour de la semaine, trimestre
- **Cycliques** : encodage sinusoïdal/cosinusoïdal pour capturer la saisonnalité
- **Météorologiques** : température moyenne, min, max, précipitations, vitesse du vent
- **Dérivées** : écart de température, indicateurs de froid/chaud, pluie
- **Historiques** : lags de 1, 2, 3, 7 jours
- **Moyennes mobiles** : fenêtres de 3, 7, 14 jours
- **Interactions** : température × jours fériés, weekend × température

#### Optimisation des Hyperparamètres
- **Time Series Cross Validation** : 3 splits temporels
- **Grid Search** sur les paramètres clés :
  - `n_estimators`: [300, 500]
  - `max_depth`: [6, 8]
  - `learning_rate`: [0.05, 0.1]
  - `subsample`: [0.8, 0.9]
  - `colsample_bytree`: [0.8, 0.9]

### Entraînement du Modèle
```bash
python model_xgboost.py
```

#### Résultats Typiques
- **MAE** : ~15,000-25,000 kWh
- **RMSE** : ~20,000-35,000 kWh
- **R²** : 0.85-0.92
- **MAPE** : 8-15%

## 📈 Analyse et Visualisation

### Génération des Graphiques
```bash
# Dashboard principal
python create_summary_dashboard.py

# Analyse détaillée
python create_detailed_charts.py

# Visualisation des prédictions hybrides
python visualize_hybrid_predictions.py
```

### Types de Visualisations
- **Série temporelle** : Comparaison prédictions vs réalité
- **Scatter plots** : Corrélation prédite/observée
- **Importance des variables** : Top features contributeurs
- **Analyse des erreurs** : Distribution et patterns d'erreurs
- **Calendrier mensuel** : Visualisation par mois/jour
- **Décomposition saisonnière** : Tendances et cycles

## 🔍 Diagnostic et Optimisation

### Analyse des Erreurs
```bash
python analyze_errors.py
python analyse_causes_erreurs.py
```

### Métriques de Performance
- **Précision globale** : Évaluation sur ensemble de test
- **Précision temporelle** : Performance par période
- **Analyse résiduelle** : Distribution des erreurs
- **Détection d'anomalies** : Identification des outliers

## 📊 Prédictions

### Prédictions Standard
```bash
python predict_realistic.py
```

### Prédictions Hybrides
```bash
python predict_hybrid.py
```

### Prédictions Spécialisées 2023
```bash
python xgboost_2023_predictions.py
```

## 🔧 Configuration Avancée

### Paramètres du Modèle
Modifier dans `model_xgboost.py` :
```python
param_grid = {
    'n_estimators': [300, 500, 800],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    # ... autres paramètres
}
```

### Fenêtres Temporelles
Ajuster les lags et moyennes mobiles :
```python
# Lags
for lag in [1, 2, 3, 7, 14]:  # Ajouter lag 14 jours
    df[f'consumption_lag_{lag}'] = df['DailyAverage'].shift(lag)

# Moyennes mobiles
for window in [3, 7, 14, 30]:  # Ajouter fenêtre 30 jours
    df[f'consumption_ma_{window}'] = df['DailyAverage'].rolling(window).mean()
```

## 📁 Structure des Données

### Fichiers Principaux
- `data_7_years.csv` : Données brutes de consommation (7 ans)
- `data_cleaned.csv` : Données nettoyées
- `data_with_context_fixed.csv` : Dataset final avec contexte
- `weather_haifa.csv` : Données météorologiques Haïfa
- `jours_feries.csv` : Calendrier des jours fériés

### Format des Données
```csv
Day,SourceID,QuantityID,SourceTypeName,DailyAverage,TempAvg,TempMin,TempMax,Precip,WindSpeed,is_holiday_full,is_holiday_half
2017-01-01,123,129,EnergyMeter,45632.5,18.2,12.1,24.3,0.0,15.2,1,0
```

## 🚀 Utilisation

### Workflow Complet
```bash
# 1. Extraction des données
python extraction_data.py

# 2. Nettoyage
python clean_data.py

# 3. Enrichissement
python get_meteo.py
python fusion_data_context.py
python fix_holidays.py

# 4. Entraînement du modèle
python model_xgboost.py

# 5. Prédictions
python predict_hybrid.py

# 6. Visualisations
python create_summary_dashboard.py
```

### Prédiction sur Nouvelles Données
```python
import pickle
import pandas as pd

# Charger le modèle
with open('xgboost_energy_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Préparer les nouvelles données (même format que l'entraînement)
new_data = pd.read_csv('nouvelles_donnees.csv')
# ... feature engineering ...

# Prédire
predictions = model.predict(new_data)
```

## 🔧 Maintenance et Monitoring

### Mise à Jour du Modèle
- **Fréquence recommandée** : Mensuelle ou trimestrielle
- **Données nouvelles** : Intégrer les dernières observations
- **Réentraînement** : Utiliser la même méthodologie avec données étendues

### Monitoring des Performances
- Surveiller la dérive des performances (concept drift)
- Analyser les nouvelles erreurs et patterns
- Ajuster les hyperparamètres si nécessaire

## 📞 Support et Contribution

### Issues et Bugs
- Ouvrir une issue sur GitHub avec description détaillée
- Inclure logs d'erreur et contexte d'exécution

### Contributions
- Fork le repository
- Créer une branche feature
- Soumettre une pull request avec description

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🏆 Crédits

Développé par l'équipe Optaro pour l'optimisation énergétique intelligente.

---

**🔋 Optaro - Prédire l'énergie de demain, aujourd'hui** 