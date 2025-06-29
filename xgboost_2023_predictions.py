import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 PRÉDICTIONS XGBOOST vs RÉALITÉ 2023")
print("=" * 50)

# === 1. CHARGER DONNÉES AVEC FEATURES ===
# Recharger les données avec features du modèle original
try:
    # Essayer de charger les données avec features si disponibles
    df_features = pd.read_csv("data_with_features.csv")
    df_features['Day'] = pd.to_datetime(df_features['Day'])
    print("✅ Données avec features chargées")
except:
    print("⚠️  Fichier features non trouvé, utilisation données de base")
    df_features = pd.read_csv("data_with_context_fixed.csv")
    df_features['Day'] = pd.to_datetime(df_features['Day'])

# Charger modèle
try:
    model = joblib.load('xgboost_energy_model.pkl')
    feature_cols = joblib.load('xgboost_features.pkl')
    print("✅ Modèle XGBoost chargé")
except:
    print("❌ Modèle non trouvé")
    exit()

# === 2. DONNÉES 2023 ===
data_2023 = df_features[(df_features['Day'].dt.year == 2023)].copy()
print(f"✅ Données 2023: {len(data_2023)} observations")

if len(data_2023) == 0:
    print("❌ Pas de données 2023")
    exit()

# === 3. PRÉDICTIONS AVEC GESTION DES FEATURES MANQUANTES ===
print("🔧 Préparation des features pour prédiction...")

# Identifier features disponibles
available_features = [col for col in feature_cols if col in data_2023.columns]
missing_features = [col for col in feature_cols if col not in data_2023.columns]

print(f"✅ Features disponibles: {len(available_features)}/{len(feature_cols)}")
if missing_features:
    print(f"⚠️  Features manquantes: {len(missing_features)}")

# Créer dataset avec toutes les features nécessaires
X_2023 = pd.DataFrame()

# Copier features disponibles
for feat in available_features:
    X_2023[feat] = data_2023[feat]

# Remplir features manquantes avec des valeurs par défaut
for feat in missing_features:
    if 'temp' in feat.lower():
        X_2023[feat] = data_2023.get('TempAvg', 20).fillna(20)
    elif 'lag' in feat:
        X_2023[feat] = data_2023.get('DailyAverage', 80000).fillna(80000)
    elif 'ma' in feat:
        X_2023[feat] = data_2023.get('DailyAverage', 80000).fillna(80000)
    elif 'sin' in feat or 'cos' in feat:
        X_2023[feat] = 0
    elif 'is_' in feat:
        X_2023[feat] = 0
    else:
        X_2023[feat] = 0

# Réorganiser colonnes dans l'ordre attendu
X_2023 = X_2023[feature_cols]

# Gérer valeurs manquantes
X_2023 = X_2023.fillna(X_2023.mean())

print("🎯 Génération des prédictions...")
predictions_2023 = model.predict(X_2023)

# === 4. GRAPHIQUE PRINCIPAL ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('🎯 PRÉDICTIONS XGBoost vs RÉALITÉ - ANNÉE 2023', fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Série temporelle
ax1 = axes[0, 0]
ax1.plot(data_2023['Day'], data_2023['DailyAverage'], 
         label='Consommation réelle', color='blue', linewidth=2.5, alpha=0.8)
ax1.plot(data_2023['Day'], predictions_2023, 
         label='Prédictions XGBoost', color='red', linewidth=2, alpha=0.8)

ax1.set_title('📈 Évolution Temporelle 2023', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Consommation (kWh)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# GRAPHIQUE 2: Scatter plot précision
ax2 = axes[0, 1]
ax2.scatter(data_2023['DailyAverage'], predictions_2023, alpha=0.6, color='green', s=20)
ax2.plot([data_2023['DailyAverage'].min(), data_2023['DailyAverage'].max()], 
         [data_2023['DailyAverage'].min(), data_2023['DailyAverage'].max()], 
         'r--', linewidth=2, label='Prédiction parfaite')

# Métriques
r2_2023 = r2_score(data_2023['DailyAverage'], predictions_2023)
mae_2023 = mean_absolute_error(data_2023['DailyAverage'], predictions_2023)
mape_2023 = np.mean(np.abs((data_2023['DailyAverage'] - predictions_2023) / data_2023['DailyAverage'])) * 100

ax2.set_title(f'🎯 Précision du Modèle\nR² = {r2_2023:.3f} | MAPE = {mape_2023:.1f}%', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Consommation Réelle (kWh)')
ax2.set_ylabel('Consommation Prédite (kWh)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# GRAPHIQUE 3: Erreurs mensuelles
ax3 = axes[1, 0]
data_2023['month'] = data_2023['Day'].dt.month
data_2023['predictions'] = predictions_2023
data_2023['error_pct'] = ((predictions_2023 - data_2023['DailyAverage']) / data_2023['DailyAverage']) * 100

monthly_errors = data_2023.groupby('month')['error_pct'].mean()
months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

colors = ['green' if abs(err) < 5 else 'orange' if abs(err) < 10 else 'red' for err in monthly_errors]
bars = ax3.bar(range(1, 13), monthly_errors, color=colors, alpha=0.7)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_title('📊 Erreurs Moyennes par Mois', fontsize=14, fontweight='bold')
ax3.set_xlabel('Mois')
ax3.set_ylabel('Erreur Moyenne (%)')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(months)
ax3.grid(True, alpha=0.3, axis='y')

# Ajouter valeurs sur barres
for i, (bar, err) in enumerate(zip(bars, monthly_errors)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if err > 0 else -1), 
             f'{err:.1f}%', ha='center', va='bottom' if err > 0 else 'top', fontweight='bold')

# GRAPHIQUE 4: Analyse résidus
ax4 = axes[1, 1]
residuals = data_2023['DailyAverage'] - predictions_2023
ax4.scatter(predictions_2023, residuals, alpha=0.6, color='purple', s=20)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_title('🔍 Analyse des Résidus', fontsize=14, fontweight='bold')
ax4.set_xlabel('Prédictions (kWh)')
ax4.set_ylabel('Résidus (Réel - Prédit)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_2023_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# === 5. STATISTIQUES DÉTAILLÉES ===
print(f"\n📊 RÉSULTATS DÉTAILLÉS - PRÉDICTIONS 2023")
print("=" * 55)

print(f"🎯 MÉTRIQUES DE PERFORMANCE:")
print(f"   R² (coefficient détermination): {r2_2023:.3f}")
print(f"   MAE (erreur absolue moyenne)  : {mae_2023:6.0f} kWh")
print(f"   MAPE (erreur % moyenne)       : {mape_2023:5.1f}%")
print(f"   RMSE (erreur quadratique)     : {np.sqrt(np.mean((data_2023['DailyAverage'] - predictions_2023)**2)):6.0f} kWh")

print(f"\n📈 COMPARAISON MOYENNES:")
print(f"   Consommation réelle moyenne   : {data_2023['DailyAverage'].mean():6.0f} kWh/jour")
print(f"   Prédictions moyennes          : {predictions_2023.mean():6.0f} kWh/jour")
print(f"   Biais (différence)            : {predictions_2023.mean() - data_2023['DailyAverage'].mean():+6.0f} kWh/jour")

print(f"\n🎯 QUALITÉ DES PRÉDICTIONS:")
excellent = sum(abs(data_2023['error_pct']) < 5)
good = sum((abs(data_2023['error_pct']) >= 5) & (abs(data_2023['error_pct']) < 10))
average = sum((abs(data_2023['error_pct']) >= 10) & (abs(data_2023['error_pct']) < 20))
poor = sum(abs(data_2023['error_pct']) >= 20)

print(f"   Excellentes (<5% erreur)      : {excellent:3d} jours ({excellent/len(data_2023)*100:4.1f}%)")
print(f"   Bonnes (5-10% erreur)         : {good:3d} jours ({good/len(data_2023)*100:4.1f}%)")
print(f"   Moyennes (10-20% erreur)      : {average:3d} jours ({average/len(data_2023)*100:4.1f}%)")
print(f"   Faibles (>20% erreur)         : {poor:3d} jours ({poor/len(data_2023)*100:4.1f}%)")

print(f"\n💾 Graphique sauvé: xgboost_2023_predictions.png")
print(f"🎉 Analyse terminée avec succès!")
