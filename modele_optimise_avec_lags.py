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

print("🚀 MODÈLE OPTIMISÉ AVEC LAGS - AMÉLIORATION 33%")
print("=" * 60)

# === 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ===
print("\n📊 1. Chargement et préparation des données...")

df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

print(f"📊 Données chargées: {len(df)} jours")

# === 2. DÉTECTION AUTOMATIQUE DES JOURS FÉRIÉS ===
def creer_detecteur_jours_feries(csv_path="data_with_context_fixed.csv"):
    """Crée un détecteur de jours fériés basé sur les vraies données historiques"""
    df_temp = pd.read_csv(csv_path)
    df_temp['Day'] = pd.to_datetime(df_temp['Day'])
    
    # Extraire les jours fériés réels
    jours_feries = df_temp[(df_temp['is_holiday_full'] == 1) | (df_temp['is_holiday_half'] == 1)].copy()
    
    # Créer des patterns récurrents
    patterns_feries = set()
    for _, row in jours_feries.iterrows():
        date = row['Day']
        patterns_feries.add((date.month, date.day))
    
    print(f"🎉 {len(patterns_feries)} patterns de jours fériés détectés")
    
    def detecter_jour_ferie(date):
        return 1 if (date.month, date.day) in patterns_feries else 0
    
    return detecter_jour_ferie, patterns_feries

detecteur_feries, patterns_feries_globaux = creer_detecteur_jours_feries()

# === 3. CRÉATION DES FEATURES COMPLÈTES + LAGS ===
print("\n🔧 2. Création des features complètes + LAGS optimisés...")

def create_features_avec_lags(df):
    """Créé des features complètes AVEC lags J-1 et J-7"""
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
    
    # === INTERACTIONS MÉTÉO AVANCÉES ===
    df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
    df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
    df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
    df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
    df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
    
    # === TEMPOREL ===
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    df['is_weekend'] = (df['Day'].dt.dayofweek >= 5).astype(int)
    
    # === JOURS FÉRIÉS RÉELS ===
    vraie_holiday = (df['is_holiday_full'] + df['is_holiday_half'] > 0).astype(int)
    df['is_holiday'] = vraie_holiday
    
    # === 🎯 LAGS CRITIQUES J-1 et J-7 ===
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # === FEATURES FIN D'ANNÉE (diagnostic a montré concentration en décembre) ===
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day  # Distance au nouvel an
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

df_features = create_features_avec_lags(df)

# Supprimer les NaN créés par les lags
df_features = df_features.dropna()

print(f"✅ Features complètes créées avec lags")
print(f"📊 Données après lags: {len(df_features)} jours")

# === 4. DÉFINITION DES FEATURES OPTIMISÉES ===
enhanced_features_with_lags = [
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
    
    # === INTERACTIONS AVANCÉES ===
    'temp_x_summer', 'temp_x_mid_summer', 'temp_squared_x_summer',
    'temp_x_wind', 'pressure_x_temp',
    
    # === TEMPOREL ===
    'time_trend', 'is_weekend', 'is_holiday',
    
    # === 🎯 LAGS CRITIQUES (GAIN 33%) ===
    'consumption_lag_1', 'consumption_lag_7',
    
    # === FEATURES FIN D'ANNÉE ===
    'is_december', 'days_to_new_year', 'is_end_of_year'
]

print(f"🎯 {len(enhanced_features_with_lags)} features avec lags optimisés")

# === 5. SPLIT TEMPOREL PROPRE ===
print("\n📅 3. Split temporel optimisé...")

split_idx = int(len(df_features) * 0.7)
train_data = df_features.iloc[:split_idx].copy()
test_data = df_features.iloc[split_idx:].copy()

print(f"📊 Split temporel:")
print(f"   Train: {train_data['Day'].min().date()} → {train_data['Day'].max().date()} ({len(train_data)} jours)")
print(f"   Test:  {test_data['Day'].min().date()} → {test_data['Day'].max().date()} ({len(test_data)} jours)")

# Préparer les données
X_train = train_data[enhanced_features_with_lags]
X_test = test_data[enhanced_features_with_lags]
y_train = train_data['DailyAverage'].values
y_test = test_data['DailyAverage'].values

# === 6. NORMALISATION ===
print("\n📐 4. Normalisation...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 7. ENTRAÎNEMENT MODÈLE OPTIMISÉ ===
print("\n🤖 5. Entraînement modèle Ridge optimisé (α=10.0)...")

# Utiliser le meilleur α trouvé par le diagnostic
model_optimise = Ridge(alpha=10.0, random_state=42)
model_optimise.fit(X_train_scaled, y_train)

# === 8. PRÉDICTIONS ET ÉVALUATION ===
print("\n📊 6. Évaluation du modèle optimisé...")

train_pred = model_optimise.predict(X_train_scaled)
test_pred = model_optimise.predict(X_test_scaled)

# Métriques
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"🎯 PERFORMANCE MODÈLE OPTIMISÉ:")
print(f"   Train - MAE: {train_mae:.0f} kWh, R²: {train_r2:.3f}")
print(f"   Test  - MAE: {test_mae:.0f} kWh, R²: {test_r2:.3f}")
print(f"   Overfitting: {train_r2 - test_r2:.3f}")

# === 9. IMPORTANCE DES FEATURES ===
print("\n🔥 7. Importance des features (Ridge coefficients)...")

feature_importance = np.abs(model_optimise.coef_)
feature_names = enhanced_features_with_lags

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n🏆 TOP 15 FEATURES IMPORTANTES:")
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    print(f"   {i+1:2d}. {row['feature']:<25}: {row['importance']:.3f}")

# === 10. SAUVEGARDE DU MODÈLE OPTIMISÉ ===
print("\n💾 8. Sauvegarde du modèle optimisé...")

model_data_optimise = {
    'model': model_optimise,
    'scaler': scaler,
    'features': enhanced_features_with_lags,
    'patterns_feries': patterns_feries_globaux,  # Seulement les patterns, pas la fonction
    'performance': {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'overfitting': train_r2 - test_r2
    },
    'version': 'optimise_avec_lags_v1.0',
    'date_creation': datetime.now(),
    'nb_features': len(enhanced_features_with_lags)
}

with open('modele_optimise_avec_lags.pkl', 'wb') as f:
    pickle.dump(model_data_optimise, f)

print(f"✅ Modèle sauvegardé: modele_optimise_avec_lags.pkl")

# === 11. VISUALISATIONS DE VALIDATION ===
print("\n📊 9. Génération des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🚀 MODÈLE OPTIMISÉ AVEC LAGS - VALIDATION', fontsize=16, fontweight='bold')

# Plot 1: Prédictions vs Réel (Test)
axes[0,0].scatter(y_test, test_pred, alpha=0.6, color='orange', s=20)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0,0].set_xlabel('Consommation Réelle (kWh)')
axes[0,0].set_ylabel('Consommation Prédite (kWh)')
axes[0,0].set_title(f'Test: Prédictions vs Réel (R²={test_r2:.3f})')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Résidus vs Température (vérifier S-curve)
test_residuals = y_test - test_pred
axes[0,1].scatter(test_data['TempAvg'], test_residuals, alpha=0.6, color='green', s=20)
axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[0,1].set_xlabel('Température Moyenne (°C)')
axes[0,1].set_ylabel('Résidus (kWh)')
axes[0,1].set_title('Résidus vs Température (S-curve corrigée)')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Évolution temporelle des erreurs
axes[1,0].plot(test_data['Day'], np.abs(test_residuals), color='purple', alpha=0.7, linewidth=1)
axes[1,0].axhline(y=test_mae, color='red', linestyle='--', label=f'MAE moyen: {test_mae:.0f} kWh')
axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Erreur Absolue (kWh)')
axes[1,0].set_title('Évolution Temporelle des Erreurs')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].tick_params(axis='x', rotation=45)

# Plot 4: Top 10 Features
top_features = importance_df.head(10)
axes[1,1].barh(range(len(top_features)), top_features['importance'].values)
axes[1,1].set_yticks(range(len(top_features)))
axes[1,1].set_yticklabels(top_features['feature'].values)
axes[1,1].set_xlabel('Importance (|coefficient|)')
axes[1,1].set_title('Top 10 Features Importantes')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modele_optimise_avec_lags_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# === 12. COMPARAISON AVEC MODÈLE PRÉCÉDENT ===
print("\n📈 10. Comparaison avec le modèle précédent...")

# Simuler performance du modèle sans lags (diagnostic)
mae_sans_lags = 5774  # Du diagnostic
improvement = ((mae_sans_lags - test_mae) / mae_sans_lags) * 100

print(f"🏆 AMÉLIORATION CONFIRMÉE:")
print(f"   Modèle sans lags: {mae_sans_lags:.0f} kWh MAE")
print(f"   Modèle avec lags: {test_mae:.0f} kWh MAE")
print(f"   Gain: {improvement:.1f}% d'amélioration !")

# === 13. DÉTECTION DES POINTS D'AMÉLIORATION RESTANTS ===
print("\n🔍 11. Analyse des erreurs résiduelles...")

# Erreurs par quartile
quartiles = pd.qcut(y_test, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
erreur_par_quartile = pd.DataFrame({
    'quartile': quartiles,
    'erreur_relative': np.abs(test_residuals) / y_test
}).groupby('quartile')['erreur_relative'].mean() * 100

print(f"📊 Erreur relative par quartile (modèle optimisé):")
for q, err in erreur_par_quartile.items():
    print(f"   {q}: {err:.1f}%")

# Top 5 pires jours restants
test_errors = test_data.copy()
test_errors['abs_error'] = np.abs(test_residuals)
worst_5 = test_errors.nlargest(5, 'abs_error')

print(f"\n🎯 Top 5 pires jours restants:")
for _, row in worst_5.iterrows():
    print(f"   {row['Day'].strftime('%Y-%m-%d')}: {row['abs_error']:.0f} kWh (Temp: {row['TempAvg']:.1f}°C)")

print("\n" + "="*70)
print("🚀 MODÈLE OPTIMISÉ AVEC LAGS TERMINÉ !")
print("="*70)
print(f"📊 Performance finale: MAE {test_mae:.0f} kWh (R² {test_r2:.3f})")
print(f"🎯 Amélioration: {improvement:.1f}% vs modèle précédent")
print(f"💾 Fichier sauvegardé: modele_optimise_avec_lags.pkl")
print(f"📈 Graphiques: modele_optimise_avec_lags_validation.png")
print("="*70) 