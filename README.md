# OPTARO - Système de Prédiction Énergétique Industrielle Optimisé

## 🎯 Vue d'Ensemble du Projet

**OPTARO** est un système intelligent de prédiction de consommation énergétique industrielle utilisant un **modèle optimisé avec lags** atteignant des performances exceptionnelles. Le projet inclut des outils de diagnostic avancé et de comparaison de périodes.

### 🏭 Contexte Industriel
- **Données** : 1,114 jours de consommation énergétique (2022-2025)
- **Variables météo** : Température (Min/Max/Moy), Précipitations, Vent, Pression
- **Jours fériés** : 65 patterns détectés automatiquement
- **Performance finale** : MAE 3,889 kWh, R² 0.941
- **Amélioration** : +32.7% vs modèle baseline

---

## 🏆 Résultats Finaux - Modèle Optimisé avec Lags

### 📊 Performance Exceptionnelle
```
🥇 MODÈLE FINAL (modele_optimise_avec_lags.pkl)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Test MAE     : 3,889 kWh    (🔥 Amélioration 32.7%)
📈 Test R²      : 0.941        (🔥 Excellent)
⚖️ Overfitting  : -0.034       (🔥 Parfait contrôle)
📊 Features     : 40 variables (incluant lags critiques)
🚀 Baseline     : 5,774 kWh → 3,889 kWh (-1,885 kWh)
```

### 🔧 Diagnostic Complet Effectué

#### 🌊 S-Curve Analysis (Non-linéarité détectée)
- **Coefficient cubique** : -7.44e+00 (significatif)
- **Recommandation** : GBM/XGBoost confirmé nécessaire
- **Pattern** : Relation non-linéaire température-consommation

#### 🎯 Top 10 Worst Days Analysis
- **5/10 en décembre** : Pattern fin d'année identifié
- **3/10 en été** : Canicules et pics climatisation
- **0/10 jours fériés** : Détection holidays efficace
- **Pattern principal** : Sous-estimation charges faibles hivernales

#### 📊 Quartile Analysis (Erreurs par charge)
- **Q1 (faibles)** : 7.3% erreur (acceptable)
- **Q2-Q3** : 4.0-5.7% erreur (excellent)
- **Q4 (fortes)** : 3.8% erreur (optimal)
- **Conclusion** : Pas besoin de transformation log

#### ⚡ 1-Hour Test Results
- **Meilleur α Ridge** : 10.0
- **Lags J-1, J-7** : +33.0% amélioration ✨
- **LightGBM+Lags** : +26.6% amélioration
- **Feature critique** : `consumption_lag_7` (7,227 importance)

### 🥇 Top Features - Modèle Final
1. **`consumption_lag_7`** (7,227) - Consommation J-7 (critique)
2. **`consumption_lag_1`** (6,437) - Consommation J-1 (critique)
3. **`temp_squared`** (1,852) - Effet quadratique température
4. **`heating_needs`** (1,591) - Besoins chauffage
5. **`is_winter`** (1,427) - Effet saisonnier hiver
6. **`temp_ma_7`** (1,392) - Moyenne mobile température
7. **`day_of_year_sin`** (1,256) - Cycle saisonnier
8. **`is_december`** (1,095) - Effet fin d'année

### 🎯 Améliorations Spécialisées
- **Features end-of-year** : `is_december`, `days_to_new_year`, `is_end_of_year`
- **Lags optimisés** : J-1 et J-7 (données consommation réelle)
- **Gestion hivernale** : Meilleure prédiction des charges faibles
- **Robustesse** : Généralisation excellente (-0.034 overfitting)

---

## 🛠️ Outils Disponibles

### 🤖 1. Modèle Principal Optimisé
```bash
python modele_optimise_avec_lags.py
```
**Le modèle de référence** avec performance industrielle
- **40 features optimisées** incluant lags critiques
- **Entraînement Ridge** avec α=10.0 optimal
- **Validation temporelle** 70/30 split
- **Diagnostic automatique** complet intégré
- **Export** : `modele_optimise_avec_lags.pkl` prêt production

**Outputs** :
- Modèle sauvé avec scaler et métadonnées
- Graphiques de validation détaillés
- Analyse des features et importance
- Métriques de performance complètes

### 📊 2. Comparateur de Périodes Complet
```bash
python comparateur_periodes.py
```
**Interface interactive** pour comparer deux périodes historiques
- **Saisie flexible** : YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY
- **4 graphiques automatiques** : évolution, distributions, température, jour semaine
- **Analyses complètes** : statistiques, impact financier, recommandations
- **Export optionnel** : CSV et PNG haute résolution

**Exemple d'utilisation** :
```
📅 Période 1: 01/07/2024 → 31/07/2024 (Été 2024)
📅 Période 2: 01/12/2024 → 31/12/2024 (Hiver 2024)
🎯 Résultat: Été +79.8% vs Hiver (+179,944€)
🌡️ Facteur: +14.7°C température moyenne
📊 Graphiques: 4 analyses automatiques générées
```

### ⚡ 3. Comparateur Simple et Robuste
```bash
python comparateur_simple.py
```
**Version rapide** avec exemples prédéfinis
- **4 comparaisons prêtes** : Été vs Été, Hiver vs Été, mensuel, etc.
- **Traitement simplifié** : pas de crash, résultats immédiats
- **Mode texte** : statistiques claires sans graphiques
- **Export simple** : CSV de synthèse

**Comparaisons disponibles** :
1. 🌞 Été 2024 vs Été 2023
2. ❄️ Hiver vs Été 2024  
3. 🗓️ Juin 2024 vs Juin 2023
4. 🔥 Août vs Septembre 2024
5. ✏️ Saisie manuelle

### 🔮 4. Prédicteur de Consommation Future
```bash
python predicteur_futur.py
```
**Prédictions futures** avec simulation météo réaliste
- **Météo intelligente** : basée sur historique réel (93 jours juillet)
- **Températures réalistes** : 28.6°C ± 1.2°C (vs 38.4°C corrigé)
- **Lags simulés** : continuation intelligente des patterns récents
- **Comparaison historique** : vs même période années précédentes
- **Visualisations** : 4 graphiques de prédiction

**Exemple Juillet 2025** :
```
🔮 Période: 2025-07-01 → 2025-07-31
🌡️ Température simulée: 28.6°C (réaliste)
⚡ Consommation prévue: 2,640,665 kWh
💰 Coût estimé: 396,100€
📉 vs Juillet 2024: -22.2% (-752,946 kWh)
```

### 🚨 5. Système d'Alerte Production
```bash
python alerte_usine_final.py
```
**Détection d'anomalies** avec modèle optimisé
- **4 niveaux d'alerte** : Normal, Attention, Alerte, Critique
- **Seuils basés sur MAE** : 3,889 kWh de référence
- **Calcul probabilité** : anomalie basée sur modèle 40-features
- **Historique** : conservation et analyse des alertes

---

## 📁 Structure Finale du Projet

```
optaro-main/
├── 📊 DONNÉES
│   └── data_with_context_fixed.csv              # Dataset principal (1,114 jours)
│
├── 🤖 MODÈLE OPTIMISÉ (PRODUCTION)
│   ├── modele_optimise_avec_lags.py             # Script modèle final
│   ├── modele_optimise_avec_lags.pkl            # Modèle + scaler + métadonnées
│   └── modele_optimise_avec_lags_validation.png # Graphiques validation
│
├── 📊 OUTILS DE COMPARAISON
│   ├── comparateur_periodes.py                  # Comparateur complet interactif
│   ├── comparateur_simple.py                    # Comparateur rapide et robuste
│   └── predicteur_futur.py                      # Prédictions futures réalistes
│
├── 🔍 DIAGNOSTIC ET MONITORING
│   ├── diagnostic_quartiles.png                 # Analyse erreurs par quartile
│   ├── diagnostic_residus_temperature.png       # S-curve et non-linéarité
│   └── alerte_usine_final.py                    # Système alertes production
│
└── 📖 DOCUMENTATION
    ├── README.md                                # Ce fichier (actualisé)
    └── .gitignore                               # Configuration Git
```

---

## 🚀 Installation et Démarrage Rapide

### Prérequis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pickle
```

### Configuration Express
1. **Vérifier** la présence de `data_with_context_fixed.csv`
2. **Tester le modèle** : `python modele_optimise_avec_lags.py`
3. **Comparer des périodes** : `python comparateur_simple.py`
4. **Prédire le futur** : `python predicteur_futur.py`

---

## 📊 Utilisation en Production

### 🎯 Chargement du Modèle Optimisé
```python
import pickle
import pandas as pd

# Charger le modèle complet
with open('modele_optimise_avec_lags.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler'] 
features = model_data['features']
performance = model_data['performance']

print(f"Performance: MAE {performance['test_mae']:.0f} kWh, R² {performance['test_r2']:.3f}")
# Output: Performance: MAE 3889 kWh, R² 0.941
```

### 🔮 Prédiction Simple
```python
# Préparer des données avec les 40 features requises
new_data = create_features(raw_data)  # Fonction de preprocessing
X_scaled = scaler.transform(new_data[features])
prediction = model.predict(X_scaled)

print(f"Consommation prédite: {prediction[0]:,.0f} kWh")
```

### 📊 Comparaison de Périodes
```python
# Exemple comparaison été vs hiver
python comparateur_simple.py
# Choix: 2 (Hiver vs Été 2024)
# Résultat: Différence, coût, facteurs explicatifs
```

---

## 🎯 Points Clés Techniques

### ✅ Ce qui Marche Exceptionnellement
- **Lags J-1, J-7** : +33% d'amélioration critique
- **Ridge α=10.0** : Optimisation parfaite régularisation
- **40 features équilibrées** : Pas de sur-engineering
- **Validation temporelle** : Généralisation robuste
- **Features fin d'année** : Gestion patterns décembre

### 🔄 Améliorations Futures Possibles
- **XGBoost/LightGBM** : Test avec modèle non-linéaire
- **Features Rolling** : Moyennes mobiles 14j, 30j étendues  
- **Interaction avancées** : Plus de variables météo croisées
- **Ensembling** : Combinaison Ridge + Tree models
- **Features géographiques** : Si données localisation disponibles

### ⚠️ Limitations Connues
- **Lags nécessaires** : J-1, J-7 requis (pas de cold start)
- **Données météo** : Simulation future basée sur historique
- **Changements structurels** : Réentraînement si modification installation
- **Horizons longs** : Précision décroissante au-delà de 1 mois

---

## 📈 Résumé des Performances

| Métrique | Baseline | Modèle Final | Amélioration |
|----------|----------|--------------|--------------|
| **MAE Test** | 5,774 kWh | **3,889 kWh** | **🔥 +32.7%** |
| **R² Test** | 0.798 | **0.941** | **🔥 +17.9%** |
| **Overfitting** | -0.085 | **-0.034** | **🔥 +60.0%** |
| **Features** | 35 | **40** | **🔥 Lags critiques** |
| **Stabilité** | Variable | **Robuste** | **🔥 Production-ready** |

**🏆 CONCLUSION : Modèle industriel fiable avec performance exceptionnelle et outils complets de gestion énergétique.**

---

*Dernière mise à jour : Projet optimisé avec diagnostic complet et outils de comparaison avancés*