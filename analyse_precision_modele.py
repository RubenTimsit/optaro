import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("📊 ANALYSE COMPLÈTE DE LA PRÉCISION DU MODÈLE ISRAÉLIEN")
print("=" * 65)

# === 1. CHARGEMENT DU MODÈLE ET DES DONNÉES ===
print("\n🔧 1. Chargement du modèle israélien...")

# Charger le modèle israélien
with open('modele_optimise_israel.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

print(f"✅ Modèle chargé: {model_data['version']}")
print(f"📊 {len(features)} features")

# Charger les données
df = pd.read_csv("data_with_israel_temporal_features.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

# Recréer les features (comme dans le modèle)
def recreate_features(df):
    df = df.copy()
    
    # Features météo
    df['temp_range'] = df['TempMax'] - df['TempMin']
    df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
    df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
    df['temp_squared'] = df['TempAvg'] ** 2
    
    df['precip_ma_7'] = df['Precip'].rolling(window=7, min_periods=1).mean()
    df['has_rain'] = (df['Precip'] > 0).astype(int)
    
    df['wind_ma_7'] = df['WindSpeed'].rolling(window=7, min_periods=1).mean()
    df['pressure_ma_7'] = df['Pressure'].rolling(window=30, min_periods=1).mean()
    
    # Seuils température
    temp_25, temp_30 = 25.0, 30.0
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - temp_25)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - temp_30)
    df['heating_needs'] = np.maximum(0, temp_25 - df['TempAvg'])
    
    df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
    df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
    df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
    
    # Saisons
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_mid_summer'] = (df['Day'].dt.month == 7).astype(int)
    
    # Features cycliques
    df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
    
    # Interactions
    df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
    df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
    df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
    df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
    df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
    
    # Interactions israéliennes
    df['temp_x_weekend_israel'] = df['TempAvg'] * df['is_weekend_israel']
    df['temp_x_friday'] = df['TempAvg'] * df['is_friday']
    df['temp_x_saturday'] = df['TempAvg'] * df['is_saturday']
    df['cooling_x_weekend_israel'] = df['cooling_needs_light'] * df['is_weekend_israel']
    
    # Temporel
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    
    # Lags
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # Interactions saisonnières
    df['friday_x_summer'] = df['is_friday'] * df['is_summer']
    df['saturday_x_summer'] = df['is_saturday'] * df['is_summer']
    df['sunday_x_winter'] = df['is_sunday'] * df['is_winter']
    
    # Fin d'année
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day.clip(upper=31)
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

df_features = recreate_features(df)
df_features = df_features.dropna()

# Split comme dans le modèle original
split_idx = int(len(df_features) * 0.7)
train_data = df_features.iloc[:split_idx].copy()
test_data = df_features.iloc[split_idx:].copy()

# Prédictions
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)
y_test = test_data['DailyAverage'].values
predictions = model.predict(X_test_scaled)

print(f"📊 Données de test: {len(test_data)} jours")

# === 2. MÉTRIQUES DE PRÉCISION COMPLÈTES ===
print("\n📈 2. Calcul des métriques de précision...")

def calculate_precision_metrics(y_true, y_pred):
    """Calcule toutes les métriques de précision"""
    
    # Métriques de base
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Médiane de l'erreur absolue
    median_ae = np.median(np.abs(y_true - y_pred))
    
    # Erreur relative moyenne
    mean_relative_error = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    
    # Erreur maximale
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Pourcentage de prédictions dans différents seuils
    errors_rel = np.abs(y_true - y_pred) / y_true * 100
    within_5pct = np.mean(errors_rel <= 5) * 100
    within_10pct = np.mean(errors_rel <= 10) * 100
    within_15pct = np.mean(errors_rel <= 15) * 100
    
    # Intervalle de confiance à 95%
    residuals = y_true - y_pred
    ci_95_lower = np.percentile(residuals, 2.5)
    ci_95_upper = np.percentile(residuals, 97.5)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'median_ae': median_ae,
        'mean_relative_error': mean_relative_error,
        'max_error': max_error,
        'within_5pct': within_5pct,
        'within_10pct': within_10pct,
        'within_15pct': within_15pct,
        'ci_95_lower': ci_95_lower,
        'ci_95_upper': ci_95_upper
    }

metrics = calculate_precision_metrics(y_test, predictions)

print(f"\n🎯 MÉTRIQUES DE PRÉCISION GLOBALES:")
print(f"   MAE (Erreur absolue moyenne):     {metrics['mae']:.0f} kWh")
print(f"   RMSE (Racine erreur quadratique): {metrics['rmse']:.0f} kWh")
print(f"   R² (Coefficient de détermination): {metrics['r2']:.3f}")
print(f"   MAPE (Erreur relative % moyenne):  {metrics['mape']:.1f}%")
print(f"   Erreur médiane:                   {metrics['median_ae']:.0f} kWh")
print(f"   Erreur relative moyenne:          {metrics['mean_relative_error']:.1f}%")
print(f"   Erreur maximale:                  {metrics['max_error']:.0f} kWh")

print(f"\n🎯 PRÉCISION PAR SEUILS:")
print(f"   Prédictions à ±5%:   {metrics['within_5pct']:.1f}% des cas")
print(f"   Prédictions à ±10%:  {metrics['within_10pct']:.1f}% des cas")
print(f"   Prédictions à ±15%:  {metrics['within_15pct']:.1f}% des cas")

print(f"\n🎯 INTERVALLE DE CONFIANCE 95%:")
print(f"   Erreur entre {metrics['ci_95_lower']:.0f} et {metrics['ci_95_upper']:.0f} kWh")

# === 3. PRÉCISION PAR QUARTILES DE CONSOMMATION ===
print("\n📊 3. Précision par quartiles de consommation...")

# Diviser en quartiles
test_data['predictions'] = predictions
test_data['absolute_error'] = np.abs(y_test - predictions)
test_data['relative_error'] = np.abs(y_test - predictions) / y_test * 100

quartiles = pd.qcut(y_test, q=4, labels=['Q1 (Faible)', 'Q2 (Modérée)', 'Q3 (Élevée)', 'Q4 (Très élevée)'])
test_data['quartile'] = quartiles

quartile_precision = test_data.groupby('quartile').agg({
    'DailyAverage': ['mean', 'count'],
    'absolute_error': 'mean',
    'relative_error': 'mean'
}).round(1)

print(f"\n📊 PRÉCISION PAR QUARTILE DE CONSOMMATION:")
for q in ['Q1 (Faible)', 'Q2 (Modérée)', 'Q3 (Élevée)', 'Q4 (Très élevée)']:
    if q in quartile_precision.index:
        avg_cons = quartile_precision.loc[q, ('DailyAverage', 'mean')]
        count = quartile_precision.loc[q, ('DailyAverage', 'count')]
        abs_err = quartile_precision.loc[q, ('absolute_error', 'mean')]
        rel_err = quartile_precision.loc[q, ('relative_error', 'mean')]
        print(f"   {q}: {avg_cons:.0f} kWh ({count:.0f} jours) - Erreur: {abs_err:.0f} kWh ({rel_err:.1f}%)")

# === 4. PRÉCISION PAR SAISON ===
print("\n🌡️ 4. Précision par saison...")

def get_season(month):
    if month in [12, 1, 2]:
        return 'Hiver'
    elif month in [3, 4, 5]:
        return 'Printemps'
    elif month in [6, 7, 8]:
        return 'Été'
    else:
        return 'Automne'

test_data['season'] = test_data['Day'].dt.month.apply(get_season)

seasonal_precision = test_data.groupby('season').agg({
    'DailyAverage': ['mean', 'count'],
    'absolute_error': 'mean',
    'relative_error': 'mean'
}).round(1)

print(f"\n🌡️ PRÉCISION PAR SAISON:")
for season in ['Hiver', 'Printemps', 'Été', 'Automne']:
    if season in seasonal_precision.index:
        avg_cons = seasonal_precision.loc[season, ('DailyAverage', 'mean')]
        count = seasonal_precision.loc[season, ('DailyAverage', 'count')]
        abs_err = seasonal_precision.loc[season, ('absolute_error', 'mean')]
        rel_err = seasonal_precision.loc[season, ('relative_error', 'mean')]
        print(f"   {season}: {avg_cons:.0f} kWh ({count:.0f} jours) - Erreur: {abs_err:.0f} kWh ({rel_err:.1f}%)")

# === 5. PRÉCISION PAR TYPE DE JOUR ISRAÉLIEN ===
print("\n🇮🇱 5. Précision par type de jour israélien...")

# Créer des catégories simplifiées
day_categories = []
for _, row in test_data.iterrows():
    if row['is_holiday_full'] == 1:
        day_categories.append('Férié complet')
    elif row['is_holiday_half'] == 1:
        day_categories.append('Demi-férié')
    elif row['is_friday'] == 1:
        day_categories.append('Vendredi')
    elif row['is_saturday'] == 1:
        day_categories.append('Samedi')
    elif row['is_sunday'] == 1:
        day_categories.append('Dimanche')
    else:
        day_categories.append('Jour ouvrable')

test_data['day_type'] = day_categories

daytype_precision = test_data.groupby('day_type').agg({
    'DailyAverage': ['mean', 'count'],
    'absolute_error': 'mean',
    'relative_error': 'mean'
}).round(1)

print(f"\n🇮🇱 PRÉCISION PAR TYPE DE JOUR:")
for day_type in daytype_precision.index:
    avg_cons = daytype_precision.loc[day_type, ('DailyAverage', 'mean')]
    count = daytype_precision.loc[day_type, ('DailyAverage', 'count')]
    abs_err = daytype_precision.loc[day_type, ('absolute_error', 'mean')]
    rel_err = daytype_precision.loc[day_type, ('relative_error', 'mean')]
    print(f"   {day_type}: {avg_cons:.0f} kWh ({count:.0f} jours) - Erreur: {abs_err:.0f} kWh ({rel_err:.1f}%)")

# === 6. ANALYSE DE LA DISTRIBUTION DES ERREURS ===
print("\n📊 6. Analyse de la distribution des erreurs...")

residuals = y_test - predictions

# Statistiques descriptives des erreurs
error_stats = {
    'mean': np.mean(residuals),
    'std': np.std(residuals),
    'skewness': pd.Series(residuals).skew(),
    'kurtosis': pd.Series(residuals).kurtosis(),
    'min': np.min(residuals),
    'max': np.max(residuals),
    'q25': np.percentile(residuals, 25),
    'q75': np.percentile(residuals, 75)
}

print(f"\n📊 DISTRIBUTION DES ERREURS:")
print(f"   Moyenne:    {error_stats['mean']:.0f} kWh")
print(f"   Écart-type: {error_stats['std']:.0f} kWh")
print(f"   Asymétrie:  {error_stats['skewness']:.3f}")
print(f"   Aplatissement: {error_stats['kurtosis']:.3f}")
print(f"   Min/Max:    {error_stats['min']:.0f} / {error_stats['max']:.0f} kWh")
print(f"   Q25/Q75:    {error_stats['q25']:.0f} / {error_stats['q75']:.0f} kWh")

# === 7. VISUALISATIONS DE PRÉCISION ===
print("\n📊 7. Génération des visualisations de précision...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('📊 ANALYSE COMPLÈTE DE LA PRÉCISION - MODÈLE ISRAÉLIEN', fontsize=16, fontweight='bold')

# Plot 1: Distribution des erreurs relatives
axes[0,0].hist(test_data['relative_error'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(x=5, color='green', linestyle='--', label='±5%')
axes[0,0].axvline(x=10, color='orange', linestyle='--', label='±10%')
axes[0,0].axvline(x=15, color='red', linestyle='--', label='±15%')
axes[0,0].set_xlabel('Erreur relative (%)')
axes[0,0].set_ylabel('Fréquence')
axes[0,0].set_title('Distribution des erreurs relatives')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Erreurs par quartile
quartile_names = ['Q1\n(Faible)', 'Q2\n(Modérée)', 'Q3\n(Élevée)', 'Q4\n(Très élevée)']
quartile_errors = [quartile_precision.loc[q, ('relative_error', 'mean')] 
                  for q in ['Q1 (Faible)', 'Q2 (Modérée)', 'Q3 (Élevée)', 'Q4 (Très élevée)']
                  if q in quartile_precision.index]

axes[0,1].bar(range(len(quartile_errors)), quartile_errors, color='lightcoral', alpha=0.7)
axes[0,1].set_xticks(range(len(quartile_errors)))
axes[0,1].set_xticklabels(quartile_names[:len(quartile_errors)])
axes[0,1].set_ylabel('Erreur relative moyenne (%)')
axes[0,1].set_title('Précision par quartile de consommation')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Erreurs par saison
season_order = ['Hiver', 'Printemps', 'Été', 'Automne']
season_errors = [seasonal_precision.loc[s, ('relative_error', 'mean')] 
                for s in season_order if s in seasonal_precision.index]
season_names = [s for s in season_order if s in seasonal_precision.index]

axes[0,2].bar(range(len(season_errors)), season_errors, color='lightgreen', alpha=0.7)
axes[0,2].set_xticks(range(len(season_errors)))
axes[0,2].set_xticklabels(season_names, rotation=45)
axes[0,2].set_ylabel('Erreur relative moyenne (%)')
axes[0,2].set_title('Précision par saison')
axes[0,2].grid(True, alpha=0.3)

# Plot 4: QQ-plot des résidus
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot des résidus')
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Erreurs vs prédictions
axes[1,1].scatter(predictions, residuals, alpha=0.5, s=20)
axes[1,1].axhline(y=0, color='red', linestyle='--')
axes[1,1].set_xlabel('Prédictions (kWh)')
axes[1,1].set_ylabel('Résidus (kWh)')
axes[1,1].set_title('Résidus vs Prédictions')
axes[1,1].grid(True, alpha=0.3)

# Plot 6: Précision par type de jour
day_types = list(daytype_precision.index)
day_errors = [daytype_precision.loc[dt, ('relative_error', 'mean')] for dt in day_types]

axes[1,2].barh(range(len(day_errors)), day_errors, color='gold', alpha=0.7)
axes[1,2].set_yticks(range(len(day_errors)))
axes[1,2].set_yticklabels(day_types, fontsize=9)
axes[1,2].set_xlabel('Erreur relative moyenne (%)')
axes[1,2].set_title('Précision par type de jour')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analyse_precision_modele_israel.png', dpi=300, bbox_inches='tight')
plt.show()

# === 8. RÉSUMÉ EXÉCUTIF DE LA PRÉCISION ===
print("\n🎯 8. Résumé exécutif de la précision...")

print(f"\n" + "="*65)
print("🎯 RÉSUMÉ EXÉCUTIF - PRÉCISION MODÈLE ISRAÉLIEN")
print("="*65)

print(f"\n📊 PERFORMANCE GLOBALE:")
print(f"   • MAE: {metrics['mae']:.0f} kWh")
print(f"   • MAPE: {metrics['mape']:.1f}%")
print(f"   • R²: {metrics['r2']:.3f}")

print(f"\n🎯 QUALITÉ DES PRÉDICTIONS:")
print(f"   • {metrics['within_5pct']:.1f}% des prédictions à ±5%")
print(f"   • {metrics['within_10pct']:.1f}% des prédictions à ±10%")
print(f"   • {metrics['within_15pct']:.1f}% des prédictions à ±15%")

print(f"\n🇮🇱 PRÉCISION WEEK-ENDS ISRAÉLIENS:")
if 'Vendredi' in daytype_precision.index and 'Samedi' in daytype_precision.index:
    vendredi_err = daytype_precision.loc['Vendredi', ('relative_error', 'mean')]
    samedi_err = daytype_precision.loc['Samedi', ('relative_error', 'mean')]
    print(f"   • Vendredi: {vendredi_err:.1f}% d'erreur relative")
    print(f"   • Samedi: {samedi_err:.1f}% d'erreur relative")

print(f"\n✅ QUALITÉ GÉNÉRALE:")
if metrics['mape'] < 5:
    print(f"   🟢 EXCELLENTE précision (MAPE < 5%)")
elif metrics['mape'] < 10:
    print(f"   🟡 BONNE précision (MAPE < 10%)")
else:
    print(f"   🔴 Précision MODÉRÉE (MAPE > 10%)")

print("="*65) 