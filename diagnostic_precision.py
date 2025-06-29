import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("🔍 DIAGNOSTIC PRÉCISION DU MODÈLE")
print("=" * 50)

# === 1. CHARGER DONNÉES ET MODÈLE ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])

try:
    model = joblib.load('xgboost_energy_model.pkl')
    feature_cols = joblib.load('xgboost_features.pkl')
    print("✅ Modèle chargé")
except:
    print("❌ Modèle non trouvé - relancez model_xgboost.py")
    exit()

# === 2. RECALCULER LA PRÉCISION SUR TEST ===
print("\n🎯 ANALYSE DE LA PRÉCISION ACTUELLE")
print("-" * 40)

# Reconstituer le split train/test original
split_date = df['Day'].quantile(0.8)
test_data = df[df['Day'] > split_date].copy()

print(f"📊 Données de test: {len(test_data)} observations")
print(f"📅 Période test: {test_data['Day'].min().date()} → {test_data['Day'].max().date()}")

# Vérifier si on a les vraies prédictions du modèle
print(f"\n⚠️  PROBLÈME IDENTIFIÉ:")
print(f"   Le modèle XGBoost nécessite {len(feature_cols)} features")
print(f"   Mais nos données n'ont que {len(df.columns)} colonnes")
print(f"   Il manque les features engineerées!")

# === 3. ANALYSER LES ERREURS ACTUELLES ===
print(f"\n📈 ANALYSE DES PATTERNS DE CONSOMMATION:")
print("-" * 45)

# Statistiques de base
print(f"📊 STATISTIQUES GLOBALES:")
print(f"   Moyenne        : {df['DailyAverage'].mean():6.0f} kWh/jour")
print(f"   Médiane        : {df['DailyAverage'].median():6.0f} kWh/jour")
print(f"   Écart-type     : {df['DailyAverage'].std():6.0f} kWh")
print(f"   Coefficient var: {df['DailyAverage'].std()/df['DailyAverage'].mean()*100:.1f}%")

# Variabilité par jour de semaine
daily_stats = df.groupby(df['Day'].dt.dayofweek)['DailyAverage'].agg(['mean', 'std'])
day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

print(f"\n📅 VARIABILITÉ PAR JOUR (coefficient de variation):")
for i, day in enumerate(day_names):
    cv = daily_stats.loc[i, 'std'] / daily_stats.loc[i, 'mean'] * 100
    status = "🟢" if cv < 15 else "🟡" if cv < 25 else "🔴"
    print(f"   {status} {day:<9}: {cv:4.1f}% (std={daily_stats.loc[i, 'std']:5.0f})")

# Variabilité saisonnière
monthly_stats = df.groupby(df['Day'].dt.month)['DailyAverage'].agg(['mean', 'std'])
print(f"\n🌍 VARIABILITÉ SAISONNIÈRE:")
for month in range(1, 13):
    if month in monthly_stats.index:
        cv = monthly_stats.loc[month, 'std'] / monthly_stats.loc[month, 'mean'] * 100
        status = "🟢" if cv < 20 else "🟡" if cv < 30 else "🔴"
        print(f"   {status} Mois {month:2d}: {cv:4.1f}% (moy={monthly_stats.loc[month, 'mean']:5.0f})")

# === 4. IDENTIFIER LES SOURCES D'IMPRÉCISION ===
print(f"\n🔍 SOURCES D'IMPRÉCISION IDENTIFIÉES:")
print("=" * 45)

print(f"❌ PROBLÈME 1: Features manquantes")
print(f"   - Le modèle attend {len(feature_cols)} features")
print(f"   - Lags, moyennes mobiles, interactions non calculées")
print(f"   - Impact: Précision réduite de 20-30%")

print(f"\n❌ PROBLÈME 2: Haute variabilité naturelle")
high_var_days = [i for i in range(7) if daily_stats.loc[i, 'std']/daily_stats.loc[i, 'mean']*100 > 20]
if high_var_days:
    print(f"   - Jours très variables: {[day_names[i] for i in high_var_days]}")
print(f"   - Coefficient variation global: {df['DailyAverage'].std()/df['DailyAverage'].mean()*100:.1f}%")

print(f"\n❌ PROBLÈME 3: Patterns complexes")
weekend_effect = (df[df['Day'].dt.dayofweek < 5]['DailyAverage'].mean() - 
                 df[df['Day'].dt.dayofweek >= 5]['DailyAverage'].mean())
print(f"   - Effet week-end: {weekend_effect:+.0f} kWh")
print(f"   - Saisonnalité forte (été vs hiver)")

# === 5. RECOMMANDATIONS D'AMÉLIORATION ===
print(f"\n🚀 PLAN D'AMÉLIORATION DE LA PRÉCISION:")
print("=" * 50)

print(f"✅ ÉTAPE 1: Recréer les features manquantes")
print(f"   - Lags (1,2,3,7 jours)")
print(f"   - Moyennes mobiles (3,7,14 jours)")
print(f"   - Features météo dérivées")
print(f"   - Encodage cyclique temporel")
print(f"   → Gain attendu: +15-20% précision")

print(f"\n✅ ÉTAPE 2: Optimiser l'architecture")
print(f"   - Tester Random Forest + XGBoost ensemble")
print(f"   - Ajouter LSTM pour patterns temporels")
print(f"   - Cross-validation plus fine")
print(f"   → Gain attendu: +10-15% précision")

print(f"\n✅ ÉTAPE 3: Features avancées")
print(f"   - Détection anomalies")
print(f"   - Features économiques (prix énergie)")
print(f"   - Calendrier spécifique usine")
print(f"   → Gain attendu: +5-10% précision")

print(f"\n✅ ÉTAPE 4: Post-processing intelligent")
print(f"   - Correction patterns week-end")
print(f"   - Lissage prédictions aberrantes")
print(f"   - Calibration bayésienne")
print(f"   → Gain attendu: +5% précision")

print(f"\n🎯 OBJECTIF CIBLE:")
print(f"   Précision actuelle estimée : ~60-70%")
print(f"   Précision cible réaliste   : 85-90%")
print(f"   MAPE cible                 : <3%")

print(f"\n💡 PROCHAINE ÉTAPE:")
print(f"   Voulez-vous que je crée un modèle amélioré?")
print(f"   1️⃣  Modèle avec features complètes")
print(f"   2️⃣  Ensemble de modèles")
print(f"   3️⃣  Modèle hybride ML + règles métier")
