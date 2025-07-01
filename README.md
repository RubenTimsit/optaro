# OPTARO - Système de Prédiction Énergétique Industrielle Optimisé pour Israël 🇮🇱

## 🎯 Vue d'Ensemble du Projet

**OPTARO** est un système intelligent de prédiction de consommation énergétique industrielle utilisant un **modèle israélien optimisé** atteignant des performances exceptionnelles. Le projet inclut des outils de diagnostic avancé et de comparaison de périodes, spécialement adaptés au **contexte culturel et opérationnel israélien**.

### 🏭 Contexte Industriel Israélien
- **Données** : 1,114 jours de consommation énergétique (2022-2025)
- **Variables météo** : Température (Min/Max/Moy), Précipitations, Vent, Pression
- **Calendrier israélien** : Week-ends Vendredi-Samedi, Dimanches ouvrables
- **Performance finale** : MAE 3,150 kWh, R² 0.962
- **Amélioration vs modèle classique** : +19.0% de précision

---

## 🇮🇱 Spécificités du Modèle Israélien

### 📅 Système de Week-ends Israélien
- **Week-ends** : Vendredi-Samedi (au lieu de Samedi-Dimanche)
- **Jours ouvrables** : Dimanche-Jeudi
- **Impact majeur** : 11,000 kWh de différence entre Samedi (71,925 kWh) vs Dimanche (82,889 kWh)
- **Précision week-ends** : Vendredi 3.8%, Samedi 5.5%, Dimanche 4.4% d'erreur relative

### 🎯 Features Spécialisées Israéliennes
- **Jours individuels** : `is_friday`, `is_saturday`, `is_sunday`, etc.
- **Week-end israélien** : `is_weekend_israel` (Vendredi-Samedi)
- **Interactions température** : `temp_x_weekend_israel`, `temp_x_friday`, `temp_x_saturday`
- **Système culturel** : Adaptation complète aux patterns opérationnels israéliens

---

## 🏆 Résultats Finaux - Modèle Israélien Optimisé

### 📊 Performance Exceptionnelle
```
🇮🇱 MODÈLE ISRAÉLIEN FINAL (modele_optimise_israel.pkl)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Test MAE       : 3,150 kWh    (🔥 Amélioration 19.0% vs modèle classique)
📈 Test R²        : 0.962        (🔥 Excellent - 96.2% variance expliquée)
📊 MAPE           : 4.3%         (🔥 Précision industrielle < 5%)
⚖️ Overfitting    : Contrôlé     (🔥 Généralisation robuste)
📊 Features       : 57 variables (spécialisées Israël)
🇮🇱 Calendrier    : Vendredi-Samedi week-ends
```

### 🎯 Analyse de Précision par Context

#### 📊 Précision par Quartiles de Consommation
- **Q1 (faibles)** : 5.8% MAPE (bon)
- **Q2 (moyennes basses)** : 4.6% MAPE (excellent) 
- **Q3 (moyennes hautes)** : 3.9% MAPE (excellent)
- **Q4 (fortes)** : 2.8% MAPE (optimal)

#### 🗓️ Précision par Saisons
- **Printemps** : 4.7% MAPE
- **Été** : 2.7% MAPE (meilleure)
- **Automne** : 4.5% MAPE
- **Hiver** : 5.5% MAPE

#### 🇮🇱 Précision Week-ends Israéliens (Problème Résolu !)
- **Vendredi** : 3.8% erreur relative (excellent)
- **Samedi** : 5.5% erreur relative (bon)
- **Dimanche** : 4.4% erreur relative (excellent - jour ouvrable)

### 🥇 Top Features - Modèle Israélien
1. **`consumption_lag_1`** (6,792) - Consommation J-1 (critique)
2. **`temp_squared`** (2,590) - Effet quadratique température
3. **`is_friday`** (1,685) - Vendredi (début week-end israélien)
4. **`heating_needs`** (1,635) - Besoins chauffage
5. **`temp_x_weekend_israel`** (1,502) - Interaction température × week-end israélien
6. **`is_saturday`** (1,445) - Samedi (week-end israélien)
7. **`consumption_lag_7`** (1,318) - Consommation J-7
8. **`temp_x_friday`** (1,285) - Interaction température × vendredi

### 🎯 Diagnostic des Problèmes Week-ends (RÉSOLUS)
**Problème initial identifié** : Le modèle classique ne représentait pas bien les week-ends et jours fériés en Israël

**Analyse diagnostique** :
- 11,000 kWh de différence Samedi vs Dimanche
- 19,500 kWh de différence jours "pont" vs normaux
- Interaction température-weekend variant de -7% à -2%

**Solution israélienne** :
- ✅ Modèle adapté au calendrier Vendredi-Samedi
- ✅ Variables spécialisées par jour de la semaine
- ✅ Interactions météo × contexte israélien
- ✅ Précision week-ends : 3.8% à 5.5% (vs >10% avant)

---

## 🛠️ Outils Disponibles (Mis à Jour Israël)

### 🇮🇱 1. Modèle Principal Israélien Optimisé
```bash
python modele_optimise_israel.py
```
**Le modèle de référence** adapté au contexte israélien
- **57 features spécialisées** incluant système israélien
- **Week-ends Vendredi-Samedi** parfaitement gérés
- **Validation temporelle** avec patterns israéliens
- **Diagnostic automatique** week-ends et jours fériés
- **Export** : `modele_optimise_israel.pkl` prêt production

**Outputs** :
- Modèle sauvé avec métadonnées israéliennes
- Graphiques de validation week-ends
- Analyse précision par jour de semaine
- Métriques de performance culturellement adaptées

### 📊 2. Comparateur de Périodes (Version Israélienne)
```bash
python comparateur_periodes.py
```
**Interface interactive** pour comparer deux périodes avec calendrier israélien
- **Saisie flexible** : YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY
- **Calendrier israélien** : Week-ends Vendredi-Samedi automatiquement détectés
- **4 graphiques automatiques** : évolution, distributions, température, jour semaine israélien
- **Analyses spécialisées** : impact week-ends israéliens, patterns culturels
- **Export optionnel** : CSV et PNG haute résolution

**Exemple d'utilisation israélienne** :
```
📅 Période 1: 01/07/2024 → 31/07/2024 (Été 2024)
📅 Période 2: 01/12/2024 → 31/12/2024 (Hiver 2024)
🇮🇱 Week-ends détectés: 8 Vendredis-Samedis vs 9 en hiver
🎯 Résultat: Été +79.8% vs Hiver (+179,944€)
🌡️ Facteur: +14.7°C température moyenne
📊 Graphiques: 4 analyses israéliennes générées
```

### ⚡ 3. Comparateur Simple Israélien
```bash
python comparateur_simple.py
```
**Version rapide** avec exemples adaptés au contexte israélien
- **4 comparaisons prêtes** : Été vs Été, Hiver vs Été, mensuel, etc.
- **Calendrier israélien** : Détection automatique week-ends Vendredi-Samedi
- **Statistiques culturelles** : Vendredis, Samedis, Dimanches séparés
- **Mode texte** : statistiques claires avec contexte israélien
- **Export simple** : CSV avec breakdown israélien

**Comparaisons disponibles** :
1. 🌞 Été 2024 vs Été 2023
2. ❄️ Hiver vs Été 2024  
3. 🗓️ Juin 2024 vs Juin 2023
4. 🔥 Août vs Septembre 2024
5. ✏️ Saisie manuelle

**Nouvelles statistiques israéliennes** :
```
Week-ends (Ven-Sam)     |            8 |           9 |        -1
Vendredis               |            4 |           4 |         0
Samedis                 |            4 |           5 |        -1
Dimanches (ouvrable)    |            4 |           4 |         0
```

### 🔮 4. Prédicteur de Consommation Future (Israélien)
```bash
python predicteur_futur.py
```
**Prédictions futures** avec simulation météo et calendrier israélien
- **Météo intelligente** : basée sur historique réel israélien
- **Calendrier israélien** : Week-ends Vendredi-Samedi dans simulations
- **Lags adaptés** : continuation patterns israéliens
- **Comparaison historique** : vs mêmes périodes avec calendrier correct
- **Visualisations** : 4 graphiques adaptés contexte israélien

**Exemple Juillet 2025 - Version Israélienne** :
```
🇮🇱 Période: 2025-07-01 → 2025-07-31
🌡️ Température simulée: 28.6°C (réaliste)
⚡ Consommation prévue: 2,640,665 kWh
💰 Coût estimé: 396,100€
📅 Week-ends (Ven-Sam): 8 jours détectés
💼 Dimanches ouvrables: 4 jours
📉 vs Juillet 2024: -22.2% (-752,946 kWh)
```

### 🔍 5. Diagnostic Week-ends Israéliens
```bash
python diagnostic_weekends_feries_israel.py
```
**Analyse spécialisée** des patterns week-ends israéliens
- **Comparaison Samedi vs Dimanche** : Différences quantifiées
- **Impact température** : Interactions week-end israélien
- **Jours "pont"** : Analyse patterns culturels
- **Validation calendrier** : Vérification adaptation israélienne

---

## 📁 Structure Finale du Projet (Israël)

```
optaro-main/
├── 📊 DONNÉES
│   ├── data_with_context_fixed.csv              # Dataset principal (1,114 jours)
│   └── data_with_israel_temporal_features.csv   # Dataset avec features israéliennes
│
├── 🇮🇱 MODÈLE ISRAÉLIEN OPTIMISÉ (PRODUCTION)
│   ├── modele_optimise_israel.py                # Script modèle israélien final
│   ├── modele_optimise_israel.pkl               # Modèle + scaler + métadonnées
│   └── modele_optimise_israel_validation.png    # Graphiques validation israélienne
│
├── 📊 OUTILS DE COMPARAISON (ISRAËL)
│   ├── comparateur_periodes.py                  # Comparateur israélien interactif
│   ├── comparateur_simple.py                    # Comparateur rapide israélien
│   └── predicteur_futur.py                      # Prédictions futures israéliennes
│
├── 🔍 DIAGNOSTIC ISRAÉLIEN
│   ├── diagnostic_weekends_feries_israel.py     # Diagnostic week-ends israéliens
│   ├── diagnostic_weekends_israel.png           # Analyse patterns week-ends
│   └── analyse_precision_modele_israel.png      # Analyse précision complète
│
├── 📊 ANALYSES PRÉCISION
│   ├── analyse_precision_modele.py              # Analyse précision complète
│   └── predictions_futures_20250712_20250815.png # Exemples prédictions
│
└── 📖 DOCUMENTATION
    ├── README.md                                # Ce fichier (français - Israël)
    ├── README_EN.md                             # Version anglaise (Israël)
    └── .gitignore                               # Configuration Git
```

---

## 🚀 Installation et Démarrage Rapide (Israël)

### Prérequis
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pickle
```

### Configuration Express Israélienne
1. **Vérifier** la présence de `data_with_context_fixed.csv`
2. **Tester le modèle israélien** : `python modele_optimise_israel.py`
3. **Diagnostiquer week-ends** : `python diagnostic_weekends_feries_israel.py`
4. **Comparer des périodes** : `python comparateur_simple.py`
5. **Prédire le futur** : `python predicteur_futur.py`

---

## 📊 Utilisation en Production (Israël)

### 🇮🇱 Chargement du Modèle Israélien
```python
import pickle
import pandas as pd

# Charger le modèle israélien complet
with open('modele_optimise_israel.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler'] 
features = model_data['features']  # 57 features israéliennes
performance = model_data['performance']

print(f"Performance: MAE {performance['test_mae']:.0f} kWh, R² {performance['test_r2']:.3f}")
# Output: Performance: MAE 3150 kWh, R² 0.962
print("🇮🇱 Week-ends: Vendredi-Samedi | Jours ouvrables: Dimanche-Jeudi")
```

### 🔮 Prédiction avec Calendrier Israélien
```python
# Préparer des données avec les 57 features israéliennes requises
new_data = create_features_israel(raw_data)  # Fonction avec calendrier israélien
X_scaled = scaler.transform(new_data[features])
prediction = model.predict(X_scaled)

print(f"Consommation prédite: {prediction[0]:,.0f} kWh")

# Vérifier le type de jour
if new_data['is_weekend_israel'].iloc[0]:
    print("🇮🇱 Week-end israélien (Vendredi ou Samedi)")
elif new_data['is_sunday'].iloc[0]:
    print("💼 Dimanche - Jour ouvrable en Israël")
```

### 📊 Comparaison Calendrier Classique vs Israélien
```python
# Comparaison des deux approches
print("AVANT (Modèle classique):")
print("- Week-ends: Samedi-Dimanche")
print("- MAE: 3,889 kWh")
print("- Erreurs week-ends: >10%")

print("\nAPRÈS (Modèle israélien):")
print("- Week-ends: Vendredi-Samedi")  
print("- MAE: 3,150 kWh (-19%)")
print("- Erreurs week-ends: 3.8-5.5%")
print("🇮🇱 Adaptation culturelle réussie !")
```

---

## 🎯 Points Clés Techniques (Israël)

### ✅ Ce qui Marche Exceptionnellement
- **Calendrier israélien** : Week-ends Vendredi-Samedi parfaitement gérés
- **Variables jour spécifiques** : `is_friday`, `is_saturday`, `is_sunday` critiques
- **Interactions culturelles** : `temp_x_weekend_israel` très performante
- **Précision week-ends** : Erreur divisée par 2-3 vs modèle classique
- **Adaptation contextuelle** : 57 features spécialisées Israël

### 🇮🇱 Avantages Spécifiques Israéliens
- **Résolution problème week-ends** : Enfin des prédictions justes !
- **Dimanche ouvrable** : Correctement traité comme jour de travail
- **Patterns culturels** : Consommation vendredi vs samedi différenciée
- **Interactions météo** : Adaptées aux patterns opérationnels israéliens
- **Validation culturelle** : Métriques adaptées au contexte local

### 🔄 Améliorations Futures Possibles
- **Jours fériés israéliens** : Intégration calendrier religieux complet
- **Heures de pointe** : Adaptation horaires travail israéliens
- **Saisons régionales** : Features climatiques spécifiques région
- **Événements culturels** : Patterns Rosh Hashana, Yom Kippour, etc.
- **Ensembling culturel** : Combinaison modèles régionaux

### ⚠️ Limitations Spécifiques
- **Données lags** : J-1, J-7 toujours requis
- **Changements calendrier** : Réentraînement si modifications culturelles
- **Transfert géographique** : Modèle spécifique contexte israélien
- **Jours fériés** : Simplifiés dans version actuelle

---

## 📈 Résumé des Performances (Classique vs Israélien)

| Métrique | Modèle Classique | Modèle Israélien | Amélioration |
|----------|------------------|------------------|--------------|
| **MAE Test** | 3,889 kWh | **3,150 kWh** | **🇮🇱 +19.0%** |
| **R² Test** | 0.941 | **0.962** | **🇮🇱 +2.2%** |
| **MAPE** | 5.3% | **4.3%** | **🇮🇱 +18.9%** |
| **Week-ends** | >10% erreur | **3.8-5.5%** | **🇮🇱 +50-60%** |
| **Calendrier** | Samedi-Dimanche | **Vendredi-Samedi** | **🇮🇱 Culturel** |
| **Features** | 40 | **57** | **🇮🇱 Spécialisées** |
| **Contexte** | Générique | **Israélien** | **🇮🇱 Adapté** |

## 🎯 Impact Métier (Israël)

### 💰 Impact Financier
- **Précision améliorée** : 19% de réduction d'erreur = économies opérationnelles
- **Planification week-ends** : Prédictions justes Vendredi-Samedi
- **Optimisation calendrier** : Respect patterns culturels locaux
- **Réduction gaspillage** : Meilleure anticipation pic/creux

### 🏭 Impact Opérationnel  
- **Planification maintenance** : Calendrier israélien respecté
- **Gestion équipes** : Anticipation charges Dimanche (ouvrable)
- **Approvisionnement** : Prévisions justes week-ends israéliens
- **Tableaux de bord** : Métriques adaptées contexte local

**🏆 CONCLUSION : Modèle industriel israélien fiable avec adaptation culturelle complète et performance exceptionnelle pour le contexte opérationnel local.**

---

*Dernière mise à jour : Modèle optimisé pour Israël avec diagnostic week-ends complet et outils culturellement adaptés*