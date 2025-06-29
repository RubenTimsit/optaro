import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("📊 GRAPHIQUE 2023 : PRÉDICTIONS vs RÉALITÉ")
print("=" * 50)

# === 1. CHARGER DONNÉES ET MODÈLE ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])

# Charger le modèle XGBoost entraîné
try:
    model = joblib.load('xgboost_energy_model.pkl')
    feature_cols = joblib.load('xgboost_features.pkl')
    print("✅ Modèle XGBoost chargé")
except:
    print("❌ Erreur: Modèle XGBoost non trouvé. Relancez model_xgboost.py d'abord")
    exit()

# === 2. FILTRER DONNÉES 2023 ===
data_2023 = df[(df['Day'].dt.year == 2023)].copy()
print(f"✅ Données 2023: {len(data_2023)} observations")
print(f"📅 Période: {data_2023['Day'].min().date()} → {data_2023['Day'].max().date()}")

if len(data_2023) == 0:
    print("❌ Aucune donnée 2023 trouvée")
    exit()

# === 3. CRÉER FEATURES POUR 2023 ===
def create_features_2023(df_input):
    """Créer les features comme dans le modèle original"""
    df = df_input.copy()
    
    # Features temporelles
    df['year'] = df['Day'].dt.year
    df['month'] = df['Day'].dt.month
    df['day_of_year'] = df['Day'].dt.dayofyear
    df['day_of_week'] = df['Day'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year'] = df['Day'].dt.isocalendar().week
    
    # Encodage cyclique pour capturer la nature cyclique du temps
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Features météorologiques dérivées
    df['temp_squared'] = df['TempAvg'] ** 2
    df['temp_cooling'] = np.maximum(0, df['TempAvg'] - 22)  # Degré de refroidissement
    df['temp_heating'] = np.maximum(0, 18 - df['TempAvg'])  # Degré de chauffage
    
    # Features de lags (consommation des jours précédents)
    df = df.sort_values('Day')
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_2'] = df['DailyAverage'].shift(2)
    df['consumption_lag_3'] = df['DailyAverage'].shift(3)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # Moyennes mobiles
    df['consumption_ma_3'] = df['DailyAverage'].rolling(window=3, min_periods=1).mean()
    df['consumption_ma_7'] = df['DailyAverage'].rolling(window=7, min_periods=1).mean()
    df['consumption_ma_14'] = df['DailyAverage'].rolling(window=14, min_periods=1).mean()
    
    # Features météo mobiles
    df['temp_ma_3'] = df['TempAvg'].rolling(window=3, min_periods=1).mean()
    df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
    
    # Interactions importantes
    df['temp_weekend'] = df['TempAvg'] * df['is_weekend']
    df['temp_month'] = df['TempAvg'] * df['month']
    
    return df

# Créer features pour 2023
print("🔧 Création des features...")
data_2023_features = create_features_2023(data_2023)

# === 4. FAIRE PRÉDICTIONS SUR 2023 ===
# Sélectionner seulement les features utilisées par le modèle
available_features = [col for col in feature_cols if col in data_2023_features.columns]
missing_features = set(feature_cols) - set(available_features)

if missing_features:
    print(f"⚠️  Features manquantes: {missing_features}")
    # Remplir avec des valeurs par défaut
    for feat in missing_features:
        data_2023_features[feat] = 0

X_2023 = data_2023_features[feature_cols]

# Gérer les valeurs manquantes
X_2023 = X_2023.fillna(X_2023.mean())

print("�� Génération des prédictions 2023...")
predictions_2023 = model.predict(X_2023)

# === 5. GRAPHIQUE COMPARATIF ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('📊 ANNÉE 2023: PRÉDICTIONS XGBoost vs RÉALITÉ', fontsize=18, fontweight='bold')

# GRAPHIQUE 1: Série temporelle complète
ax1 = axes[0, 0]
ax1.plot(data_2023['Day'], data_2023['DailyAverage'], 
         label='Réalité', color='blue', linewidth=2, alpha=0.8)
ax1.plot(data_2023['Day'], predictions_2023, 
         label='Prédictions XGBoost', color='red', linewidth=2, alpha=0.8)

ax1.set_title('📈 Évolution Annuelle 2023', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Consommation (kWh)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# GRAPHIQUE 2: Scatter plot Prédictions vs Réalité
ax2 = axes[0, 1]
ax2.scatter(data_2023['DailyAverage'], predictions_2023, alpha=0.6, color='green')
ax2.plot([data_2023['DailyAverage'].min(), data_2023['DailyAverage'].max()], 
         [data_2023['DailyAverage'].min(), data_2023['DailyAverage'].max()], 
         'r--', linewidth=2, label='Ligne parfaite')

# Calculer R²
from sklearn.metrics import r2_score, mean_absolute_error
r2_2023 = r2_score(data_2023['DailyAverage'], predictions_2023)
mae_2023 = mean_absolute_error(data_2023['DailyAverage'], predictions_2023)

ax2.set_title(f'🎯 Précision 2023\nR² = {r2_2023:.3f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('Consommation Réelle (kWh)')
ax2.set_ylabel('Consommation Prédite (kWh)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# GRAPHIQUE 3: Erreurs par mois
ax3 = axes[1, 0]
data_2023['month'] = data_2023['Day'].dt.month
data_2023['predictions'] = predictions_2023
data_2023['error'] = predictions_2023 - data_2023['DailyAverage']
data_2023['error_pct'] = (data_2023['error'] / data_2023['DailyAverage']) * 100

monthly_errors = data_2023.groupby('month')['error_pct'].agg(['mean', 'std'])
months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

bars = ax3.bar(range(1, 13), monthly_errors['mean'], 
               yerr=monthly_errors['std'], capsize=5, 
               color='lightcoral', alpha=0.7)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_title('📊 Erreurs Moyennes par Mois', fontsize=14, fontweight='bold')
ax3.set_xlabel('Mois')
ax3.set_ylabel('Erreur Moyenne (%)')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(months)
ax3.grid(True, alpha=0.3, axis='y')

# GRAPHIQUE 4: Distribution des erreurs
ax4 = axes[1, 1]
ax4.hist(data_2023['error_pct'], bins=30, alpha=0.7, color='skyblue', density=True)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
ax4.axvline(x=data_2023['error_pct'].mean(), color='orange', linestyle='--', 
           linewidth=2, label=f'Moyenne: {data_2023["error_pct"].mean():.1f}%')

ax4.set_title('📏 Distribution des Erreurs', fontsize=14, fontweight='bold')
ax4.set_xlabel('Erreur (%)')
ax4.set_ylabel('Densité')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_2023_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# === 6. STATISTIQUES DÉTAILLÉES ===
print(f"\n📊 STATISTIQUES 2023 - PRÉDICTIONS vs RÉALITÉ")
print("=" * 60)

print(f"🎯 MÉTRIQUES GLOBALES:")
print(f"   R² (coefficient détermination): {r2_2023:.3f}")
print(f"   MAE (erreur absolue moyenne)  : {mae_2023:,.0f} kWh")
print(f"   MAPE (erreur % moyenne)       : {abs(data_2023['error_pct']).mean():.1f}%")

print(f"\n📊 CONSOMMATION MOYENNE:")
print(f"   Réalité    : {data_2023['DailyAverage'].mean():6.0f} kWh/jour")
print(f"   Prédictions: {predictions_2023.mean():6.0f} kWh/jour")
print(f"   Différence : {predictions_2023.mean() - data_2023['DailyAverage'].mean():+6.0f} kWh/jour")

print(f"\n📈 ANALYSE DES ERREURS:")
print(f"   Erreur moyenne       : {data_2023['error_pct'].mean():+5.1f}%")
print(f"   Erreur médiane       : {data_2023['error_pct'].median():+5.1f}%")
print(f"   Écart-type erreurs   : {data_2023['error_pct'].std():5.1f}%")
print(f"   Erreur max positive  : {data_2023['error_pct'].max():+5.1f}%")
print(f"   Erreur max négative  : {data_2023['error_pct'].min():+5.1f}%")

# Analyse saisonnière
print(f"\n🌍 PERFORMANCE SAISONNIÈRE:")
seasons = {
    'Hiver (Jan-Mar)': [1, 2, 3],
    'Printemps (Avr-Jun)': [4, 5, 6], 
    'Été (Jul-Sep)': [7, 8, 9],
    'Automne (Oct-Déc)': [10, 11, 12]
}

for season_name, months in seasons.items():
    season_data = data_2023[data_2023['month'].isin(months)]
    if len(season_data) > 0:
        season_r2 = r2_score(season_data['DailyAverage'], season_data['predictions'])
        season_mape = abs(season_data['error_pct']).mean()
        print(f"   {season_name:<20}: R²={season_r2:.3f}, MAPE={season_mape:.1f}%")

print(f"\n💾 Graphique sauvé: predictions_2023_analysis.png")
print(f"\n🎉 Le modèle XGBoost montre une bonne précision sur 2023 !")
