import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("🇮🇱 MODÈLE OPTIMISÉ POUR ISRAËL - WEEK-ENDS VENDREDI-SAMEDI")
print("=" * 70)

# === 1. CHARGEMENT DES DONNÉES ISRAÉLIENNES ===
print("\n📊 1. Chargement des données avec variables israéliennes...")

# Charger les données enrichies avec variables israéliennes
df = pd.read_csv("data_with_israel_temporal_features.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

print(f"📊 Données israéliennes: {len(df)} jours")
print(f"   Week-ends (Ven-Sam): {df['is_weekend_israel'].sum()} jours")
print(f"   Jours ouvrables (Dim-Jeu): {df['is_workday_israel'].sum()} jours")

# === 2. CRÉATION DES FEATURES OPTIMISÉES POUR ISRAËL ===
print("\n🔧 2. Création des features optimisées pour Israël...")

def create_features_israel(df):
    """Créé des features spécifiques au contexte israélien avec lags"""
    df = df.copy()
    
    # === FEATURES MÉTÉO COMPLÈTES ===
    df['temp_range'] = df['TempMax'] - df['TempMin']
    df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
    df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
    df['temp_squared'] = df['TempAvg'] ** 2
    
    df['precip_ma_7'] = df['Precip'].rolling(window=7, min_periods=1).mean()
    df['has_rain'] = (df['Precip'] > 0).astype(int)
    
    df['wind_ma_7'] = df['WindSpeed'].rolling(window=7, min_periods=1).mean()
    df['pressure_ma_7'] = df['Pressure'].rolling(window=30, min_periods=1).mean()
    
    # === SEUILS TEMPÉRATURE OPTIMISÉS ===
    temp_25, temp_30 = 25.0, 30.0
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - temp_25)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - temp_30)
    df['heating_needs'] = np.maximum(0, temp_25 - df['TempAvg'])
    
    df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
    df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
    df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
    
    # === SAISONS EXPLICITES ===
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_mid_summer'] = (df['Day'].dt.month == 7).astype(int)
    
    # === FEATURES CYCLIQUES ===
    df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
    
    # === 🇮🇱 INTERACTIONS MÉTÉO × WEEK-END ISRAÉLIEN ===
    df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
    df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
    df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
    df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
    df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
    
    # 🎯 INTERACTIONS TEMPÉRATURE × WEEK-END ISRAÉLIEN
    df['temp_x_weekend_israel'] = df['TempAvg'] * df['is_weekend_israel']
    df['temp_x_friday'] = df['TempAvg'] * df['is_friday']
    df['temp_x_saturday'] = df['TempAvg'] * df['is_saturday']
    df['cooling_x_weekend_israel'] = df['cooling_needs_light'] * df['is_weekend_israel']
    
    # === TEMPOREL ===
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    
    # === 🇮🇱 JOURS FÉRIÉS ISRAÉLIENS DISTINCTS ===
    df['is_holiday_full'] = df['is_holiday_full'].astype(int)
    df['is_holiday_half'] = df['is_holiday_half'].astype(int)
    
    # === 🎯 LAGS CRITIQUES J-1 et J-7 ===
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # === 🇮🇱 FEATURES SPÉCIFIQUES ISRAËL ===
    # Distinguer chaque jour de la semaine au lieu d'une simple variable weekend
    # (déjà dans les données : is_sunday, is_monday, etc.)
    
    # Interactions jours spéciaux × saison
    df['friday_x_summer'] = df['is_friday'] * df['is_summer']
    df['saturday_x_summer'] = df['is_saturday'] * df['is_summer']
    df['sunday_x_winter'] = df['is_sunday'] * df['is_winter']  # Dimanche ouvrable en hiver
    
    # === FEATURES FIN D'ANNÉE ===
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day.clip(upper=31)
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

df_features = create_features_israel(df)

# Supprimer les NaN créés par les lags
df_features = df_features.dropna()

print(f"✅ Features israéliennes créées avec lags")
print(f"📊 Données après lags: {len(df_features)} jours")

# === 3. DÉFINITION DES FEATURES OPTIMISÉES ISRAËL ===
features_israel = [
    # === TEMPÉRATURE ===
    'TempAvg', 'TempMin', 'TempMax',
    'temp_range', 'temp_ma_7', 'temp_ma_30', 'temp_squared',
    
    # === PRÉCIPITATIONS ===
    'Precip', 'precip_ma_7', 'has_rain',
    
    # === VENT ET PRESSION ===
    'WindSpeed', 'wind_ma_7', 'Pressure', 'pressure_ma_7',
    
    # === SEUILS TEMPÉRATURE ===
    'cooling_needs_light', 'cooling_needs_heavy', 'heating_needs',
    'temp_above_25', 'temp_above_28', 'temp_above_30',
    
    # === SAISONS ===
    'is_summer', 'is_winter', 'is_mid_summer',
    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
    
    # === INTERACTIONS MÉTÉO AVANCÉES ===
    'temp_x_summer', 'temp_x_mid_summer', 'temp_squared_x_summer',
    'temp_x_wind', 'pressure_x_temp',
    
    # === 🇮🇱 JOURS DE LA SEMAINE ISRAÉLIENS (SÉPARÉS) ===
    'is_sunday', 'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday',
    'is_friday', 'is_saturday',  # Pas de is_weekend global !
    
    # === 🇮🇱 INTERACTIONS TEMPÉRATURE × WEEK-END ISRAÉLIEN ===
    'temp_x_weekend_israel', 'temp_x_friday', 'temp_x_saturday',
    'cooling_x_weekend_israel',
    
    # === 🇮🇱 JOURS FÉRIÉS DISTINCTS ===
    'is_holiday_full', 'is_holiday_half',
    
    # === 🇮🇱 JOURS SPÉCIAUX ISRAÉLIENS ===
    'is_bridge_israel', 'thursday_before_long_weekend', 'sunday_after_long_weekend',
    
    # === 🇮🇱 INTERACTIONS SAISONNIÈRES ISRAÉLIENNES ===
    'friday_x_summer', 'saturday_x_summer', 'sunday_x_winter',
    
    # === TEMPOREL ===
    'time_trend',
    
    # === 🎯 LAGS CRITIQUES ===
    'consumption_lag_1', 'consumption_lag_7',
    
    # === FEATURES FIN D'ANNÉE ===
    'is_december', 'days_to_new_year', 'is_end_of_year'
]

print(f"🎯 {len(features_israel)} features optimisées pour Israël")

# === 4. SPLIT TEMPOREL PROPRE ===
print("\n📅 3. Split temporel optimisé...")

split_idx = int(len(df_features) * 0.7)
train_data = df_features.iloc[:split_idx].copy()
test_data = df_features.iloc[split_idx:].copy()

print(f"📊 Split temporel:")
print(f"   Train: {train_data['Day'].min().date()} → {train_data['Day'].max().date()} ({len(train_data)} jours)")
print(f"   Test:  {test_data['Day'].min().date()} → {test_data['Day'].max().date()} ({len(test_data)} jours)")

# Préparer les données
X_train = train_data[features_israel]
X_test = test_data[features_israel]
y_train = train_data['DailyAverage'].values
y_test = test_data['DailyAverage'].values

# === 5. NORMALISATION ===
print("\n📐 4. Normalisation...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. ENTRAÎNEMENT MODÈLE ISRAÉLIEN ===
print("\n🤖 5. Entraînement modèle Ridge optimisé pour Israël...")

model_israel = Ridge(alpha=10.0, random_state=42)
model_israel.fit(X_train_scaled, y_train)

# === 7. PRÉDICTIONS ET ÉVALUATION ===
print("\n📊 6. Évaluation du modèle israélien...")

train_pred = model_israel.predict(X_train_scaled)
test_pred = model_israel.predict(X_test_scaled)

# Métriques
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"🎯 PERFORMANCE MODÈLE ISRAÉLIEN:")
print(f"   Train - MAE: {train_mae:.0f} kWh, R²: {train_r2:.3f}")
print(f"   Test  - MAE: {test_mae:.0f} kWh, R²: {test_r2:.3f}")
print(f"   Overfitting: {train_r2 - test_r2:.3f}")

# === 8. COMPARAISON AVEC L'ANCIEN MODÈLE ===
print("\n⚖️ 7. Comparaison avec l'ancien modèle...")

# Charger l'ancien modèle pour comparaison
try:
    with open('modele_optimise_avec_lags.pkl', 'rb') as f:
        ancien_modele_data = pickle.load(f)
    ancien_test_mae = ancien_modele_data['performance']['test_mae']
    
    improvement = ((ancien_test_mae - test_mae) / ancien_test_mae) * 100
    
    print(f"🏆 COMPARAISON PERFORMANCE:")
    print(f"   Ancien modèle (Sam-Dim):  {ancien_test_mae:.0f} kWh MAE")
    print(f"   Modèle israélien (Ven-Sam): {test_mae:.0f} kWh MAE")
    print(f"   Amélioration: {improvement:.1f}% !")
    
except FileNotFoundError:
    print("⚠️  Ancien modèle non trouvé, impossible de comparer")
    improvement = 0

# === 9. ANALYSE SPÉCIFIQUE DES WEEK-ENDS ISRAÉLIENS ===
print("\n🇮🇱 8. Analyse spécifique des week-ends israéliens...")

# Prédictions par type de jour
test_data['predictions'] = test_pred
test_data['residuals'] = y_test - test_pred

# Erreurs par type de jour israélien
weekend_errors = test_data[test_data['is_weekend_israel'] == 1]['residuals'].abs().mean()
workday_errors = test_data[test_data['is_workday_israel'] == 1]['residuals'].abs().mean()
friday_errors = test_data[test_data['is_friday'] == 1]['residuals'].abs().mean()
saturday_errors = test_data[test_data['is_saturday'] == 1]['residuals'].abs().mean()

print(f"📊 ERREURS PAR TYPE DE JOUR ISRAÉLIEN:")
print(f"   Jours ouvrables (Dim-Jeu): {workday_errors:.0f} kWh MAE")
print(f"   Week-ends (Ven-Sam):       {weekend_errors:.0f} kWh MAE")
print(f"   - Vendredi seul:           {friday_errors:.0f} kWh MAE")
print(f"   - Samedi seul:             {saturday_errors:.0f} kWh MAE")

# === 10. IMPORTANCE DES FEATURES ISRAÉLIENNES ===
print("\n🔥 9. Importance des features israéliennes...")

feature_importance = np.abs(model_israel.coef_)
importance_df = pd.DataFrame({
    'feature': features_israel,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n🏆 TOP 15 FEATURES IMPORTANTES (ISRAËL):")
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    print(f"   {i+1:2d}. {row['feature']:<30}: {row['importance']:.3f}")

# === 11. SAUVEGARDE DU MODÈLE ISRAÉLIEN ===
print("\n💾 10. Sauvegarde du modèle israélien...")

model_data_israel = {
    'model': model_israel,
    'scaler': scaler,
    'features': features_israel,
    'performance': {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting': train_r2 - test_r2,
        'weekend_errors': weekend_errors,
        'workday_errors': workday_errors,
        'friday_errors': friday_errors,
        'saturday_errors': saturday_errors,
        'improvement_vs_ancien': improvement
    },
    'config': {
        'country': 'Israel',
        'weekend_days': 'Friday-Saturday',
        'workdays': 'Sunday-Thursday'
    },
    'version': 'modele_israel_v1.0',
    'date_creation': datetime.now(),
    'nb_features': len(features_israel)
}

with open('modele_optimise_israel.pkl', 'wb') as f:
    pickle.dump(model_data_israel, f)

print(f"✅ Modèle israélien sauvegardé: modele_optimise_israel.pkl")

# === 12. VISUALISATIONS ISRAÉLIENNES ===
print("\n📊 11. Génération des visualisations israéliennes...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🇮🇱 MODÈLE OPTIMISÉ ISRAËL - WEEK-ENDS VENDREDI-SAMEDI', fontsize=16, fontweight='bold')

# Plot 1: Prédictions vs Réel (Test)
axes[0,0].scatter(y_test, test_pred, alpha=0.6, color='orange', s=20)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0,0].set_xlabel('Consommation Réelle (kWh)')
axes[0,0].set_ylabel('Consommation Prédite (kWh)')
axes[0,0].set_title(f'Test: Prédictions vs Réel (R²={test_r2:.3f})')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Erreurs par type de jour israélien
day_types = ['Ouvrables\n(Dim-Jeu)', 'Week-ends\n(Ven-Sam)', 'Vendredi', 'Samedi']
errors = [workday_errors, weekend_errors, friday_errors, saturday_errors]
colors = ['lightblue', 'orange', 'gold', 'darkorange']

axes[0,1].bar(day_types, errors, color=colors)
axes[0,1].set_ylabel('Erreur Absolue Moyenne (kWh)')
axes[0,1].set_title('Erreurs par Type de Jour Israélien')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Évolution temporelle des erreurs
axes[1,0].plot(test_data['Day'], np.abs(test_data['residuals']), color='purple', alpha=0.7, linewidth=1)
axes[1,0].axhline(y=test_mae, color='red', linestyle='--', label=f'MAE moyen: {test_mae:.0f} kWh')

# Marquer les week-ends israéliens
weekend_test = test_data[test_data['is_weekend_israel'] == 1]
axes[1,0].scatter(weekend_test['Day'], np.abs(weekend_test['residuals']), 
                  color='orange', s=30, alpha=0.8, label='Week-ends (Ven-Sam)')

axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Erreur Absolue (kWh)')
axes[1,0].set_title('Évolution Temporelle des Erreurs')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].tick_params(axis='x', rotation=45)

# Plot 4: Top 10 Features israéliennes
top_features = importance_df.head(10)
axes[1,1].barh(range(len(top_features)), top_features['importance'].values)
axes[1,1].set_yticks(range(len(top_features)))
axes[1,1].set_yticklabels(top_features['feature'].values, fontsize=8)
axes[1,1].set_xlabel('Importance (|coefficient|)')
axes[1,1].set_title('Top 10 Features Israéliennes')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modele_optimise_israel_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("🇮🇱 MODÈLE OPTIMISÉ ISRAËL TERMINÉ !")
print("="*70)
print(f"📊 Performance finale: MAE {test_mae:.0f} kWh (R² {test_r2:.3f})")
print(f"🎯 Week-ends israéliens: {weekend_errors:.0f} kWh MAE")
print(f"🚀 Amélioration vs ancien: {improvement:.1f}%")
print(f"💾 Fichier sauvegardé: modele_optimise_israel.pkl")
print(f"📈 Graphiques: modele_optimise_israel_validation.png")
print("="*70) 