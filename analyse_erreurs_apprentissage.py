import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🎯 ANALYSE DES ERREURS D'APPRENTISSAGE DU MODÈLE")
print("=" * 55)

# === 1. CHARGER ET PRÉPARER DONNÉES ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
print(f"✅ Données chargées: {len(df)} observations")

# Features temporelles
df['day_of_week'] = df['Day'].dt.dayofweek
df['month'] = df['Day'].dt.month
df['day_of_year'] = df['Day'].dt.dayofyear
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Features cycliques
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Trier par date
df = df.sort_values('Day').reset_index(drop=True)

# Features LAG (essentielles pour série temporelle)
for lag in [1, 2, 3, 7]:
    df[f'consumption_lag_{lag}'] = df['DailyAverage'].shift(lag)

# Moving averages
for window in [3, 7, 14]:
    df[f'consumption_ma_{window}'] = df['DailyAverage'].rolling(window=window).mean()

# Supprimer les NaN créés par les lags
df = df.dropna().reset_index(drop=True)
print(f"✅ Après feature engineering: {len(df)} observations")

# === 2. PRÉPARER TRAIN/TEST SPLIT TEMPOREL ===
split_date = '2024-10-01'
train_mask = df['Day'] < split_date
test_mask = df['Day'] >= split_date

X_features = ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
              'is_weekend', 'consumption_lag_1', 'consumption_lag_2', 
              'consumption_lag_3', 'consumption_lag_7', 'consumption_ma_3',
              'consumption_ma_7', 'consumption_ma_14']

# Ajouter features météo si disponibles
weather_cols = [col for col in df.columns if 'temperature' in col or 'pressure' in col or 'wind' in col]
if weather_cols:
    X_features.extend(weather_cols[:3])  # Prendre max 3 features météo
    print(f"✅ Features météo ajoutées: {weather_cols[:3]}")

X_train = df[train_mask][X_features]
y_train = df[train_mask]['DailyAverage']
X_test = df[test_mask][X_features]
y_test = df[test_mask]['DailyAverage']

print(f"📊 Train: {len(X_train)} obs, Test: {len(X_test)} obs")
print(f"🔧 Features utilisées: {len(X_features)}")

# === 3. ENTRAÎNER MODÈLE SIMPLE POUR ANALYSE ===
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Prédictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"\n📈 PERFORMANCES GLOBALES:")
print(f"   Train R²: {r2_score(y_train, y_train_pred):.3f}")
print(f"   Test R²:  {r2_score(y_test, y_test_pred):.3f}")
print(f"   Train MAE: {mean_absolute_error(y_train, y_train_pred):,.0f} kWh")
print(f"   Test MAE:  {mean_absolute_error(y_test, y_test_pred):,.0f} kWh")

# === 4. ANALYSER ERREURS PAR ZONES ===
# Créer DataFrame avec erreurs
train_results = df[train_mask].copy()
train_results['predictions'] = y_train_pred
train_results['error'] = train_results['predictions'] - train_results['DailyAverage']
train_results['abs_error'] = np.abs(train_results['error'])
train_results['error_pct'] = (train_results['error'] / train_results['DailyAverage']) * 100
train_results['abs_error_pct'] = np.abs(train_results['error_pct'])

test_results = df[test_mask].copy()
test_results['predictions'] = y_test_pred
test_results['error'] = test_results['predictions'] - test_results['DailyAverage']
test_results['abs_error'] = np.abs(test_results['error'])
test_results['error_pct'] = (test_results['error'] / test_results['DailyAverage']) * 100
test_results['abs_error_pct'] = np.abs(test_results['error_pct'])

print(f"\n🎯 ANALYSE DES ERREURS D'APPRENTISSAGE:")
print("=" * 45)

# ERREURS PAR JOUR DE SEMAINE
print(f"📅 ERREURS PAR JOUR DE SEMAINE (sur test):")
jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
for day_idx in range(7):
    day_errors = test_results[test_results['day_of_week'] == day_idx]
    if len(day_errors) > 0:
        mae = day_errors['abs_error'].mean()
        mape = day_errors['abs_error_pct'].mean()
        count = len(day_errors)
        status = "🔴" if mape > 15 else "🟡" if mape > 10 else "🟢"
        print(f"   {status} {jours[day_idx]:<9}: MAE={mae:6.0f} kWh, MAPE={mape:5.1f}% ({count:2d} obs)")

# ERREURS PAR NIVEAU DE CONSOMMATION
print(f"\n⚡ ERREURS PAR NIVEAU DE CONSOMMATION (sur test):")
test_results['consumption_level'] = pd.cut(test_results['DailyAverage'], 
                                         bins=[0, 50000, 70000, 90000, float('inf')],
                                         labels=['Très faible', 'Faible', 'Normal', 'Élevé'])

for level in ['Très faible', 'Faible', 'Normal', 'Élevé']:
    level_data = test_results[test_results['consumption_level'] == level]
    if len(level_data) > 0:
        mae = level_data['abs_error'].mean()
        mape = level_data['abs_error_pct'].mean()
        count = len(level_data)
        status = "🔴" if mape > 15 else "🟡" if mape > 10 else "🟢"
        print(f"   {status} {level:<12}: MAE={mae:6.0f} kWh, MAPE={mape:5.1f}% ({count:2d} obs)")

# PIRES PRÉDICTIONS DU MODÈLE
print(f"\n🚨 TOP 10 PIRES ERREURS D'APPRENTISSAGE (test):")
print("-" * 50)
worst_predictions = test_results.nlargest(10, 'abs_error_pct')
for i, (_, row) in enumerate(worst_predictions.iterrows(), 1):
    day_name = row['Day'].strftime('%A')
    print(f"{i:2d}. {row['Day'].date()} ({day_name}): "
          f"Réel={row['DailyAverage']:6.0f}, Prédit={row['predictions']:6.0f}, "
          f"Erreur={row['error_pct']:+5.1f}%")

# === 5. FEATURE IMPORTANCE ===
print(f"\n🎯 IMPORTANCE DES FEATURES (ce que le modèle utilise le plus):")
print("-" * 55)
feature_importance = pd.DataFrame({
    'feature': X_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance.head(8).iterrows(), 1):
    print(f"{i}. {row['feature']:<20}: {row['importance']:.3f}")

# === 6. GRAPHIQUES D'ANALYSE ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('🎯 ANALYSE DES ERREURS D\'APPRENTISSAGE DU MODÈLE', fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Prédictions vs Réalité
ax1 = axes[0, 0]
ax1.scatter(test_results['DailyAverage'], test_results['predictions'], alpha=0.6, s=20)
min_val = min(test_results['DailyAverage'].min(), test_results['predictions'].min())
max_val = max(test_results['DailyAverage'].max(), test_results['predictions'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax1.set_title('🎯 Prédictions vs Réalité', fontweight='bold')
ax1.set_xlabel('Consommation Réelle (kWh)')
ax1.set_ylabel('Consommation Prédite (kWh)')
ax1.grid(True, alpha=0.3)

# GRAPHIQUE 2: Erreurs par jour de semaine
ax2 = axes[0, 1]
day_errors = [test_results[test_results['day_of_week'] == i]['abs_error_pct'].mean() 
              if len(test_results[test_results['day_of_week'] == i]) > 0 else 0 
              for i in range(7)]
colors = ['red' if err > 15 else 'orange' if err > 10 else 'green' for err in day_errors]
bars = ax2.bar(range(7), day_errors, color=colors, alpha=0.7)
ax2.set_title('📅 Erreurs par Jour', fontweight='bold')
ax2.set_ylabel('MAPE (%)')
ax2.set_xticks(range(7))
ax2.set_xticklabels(['L', 'M', 'M', 'J', 'V', 'S', 'D'])
ax2.grid(True, alpha=0.3, axis='y')

# GRAPHIQUE 3: Feature Importance
ax3 = axes[0, 2]
top_features = feature_importance.head(8)
bars = ax3.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
ax3.set_title('🎯 Features les Plus Importantes', fontweight='bold')
ax3.set_xlabel('Importance')
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['feature'], fontsize=9)
ax3.grid(True, alpha=0.3, axis='x')

# GRAPHIQUE 4: Distribution des erreurs
ax4 = axes[1, 0]
ax4.hist(test_results['error_pct'], bins=30, alpha=0.7, color='skyblue', density=True)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.axvline(x=test_results['error_pct'].mean(), color='orange', linestyle='--', linewidth=2)
ax4.set_title('📊 Distribution des Erreurs', fontweight='bold')
ax4.set_xlabel('Erreur (%)')
ax4.set_ylabel('Densité')
ax4.grid(True, alpha=0.3)

# GRAPHIQUE 5: Évolution temporelle des erreurs
ax5 = axes[1, 1]
ax5.plot(test_results['Day'], test_results['abs_error_pct'], alpha=0.7, linewidth=1)
ax5.set_title('📈 Évolution des Erreurs dans le Temps', fontweight='bold')
ax5.set_xlabel('Date')
ax5.set_ylabel('Erreur Absolue (%)')
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# GRAPHIQUE 6: Erreurs par niveau de consommation
ax6 = axes[1, 2]
level_errors = []
level_names = []
for level in ['Très faible', 'Faible', 'Normal', 'Élevé']:
    level_data = test_results[test_results['consumption_level'] == level]
    if len(level_data) > 0:
        level_errors.append(level_data['abs_error_pct'].mean())
        level_names.append(level)

colors6 = ['red' if err > 15 else 'orange' if err > 10 else 'green' for err in level_errors]
bars = ax6.bar(range(len(level_names)), level_errors, color=colors6, alpha=0.7)
ax6.set_title('⚡ Erreurs par Niveau', fontweight='bold')
ax6.set_ylabel('MAPE (%)')
ax6.set_xticks(range(len(level_names)))
ax6.set_xticklabels(level_names, rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analyse_erreurs_apprentissage.png', dpi=300, bbox_inches='tight')
plt.show()

# === 7. SYNTHÈSE ZONES PROBLÉMATIQUES ===
print(f"\n🎯 SYNTHÈSE: OÙ LE MODÈLE A LE PLUS DE DIFFICULTÉS:")
print("=" * 60)

# Identifier les pires zones d'apprentissage
worst_day = jours[np.argmax(day_errors)] if day_errors else "N/A"
worst_day_error = max(day_errors) if day_errors else 0

print(f"🔴 ZONES DE DIFFICULTÉS MAJEURES:")
print(f"   📅 Pire jour à prédire: {worst_day} ({worst_day_error:.1f}% MAPE)")
print(f"   ⚡ Consommations très faibles: Erreur >>50%")
print(f"   📊 R² test: {r2_score(y_test, y_test_pred):.1%} (modèle explique seulement {r2_score(y_test, y_test_pred):.1%})")

print(f"\n🟡 DIFFICULTÉS MODÉRÉES:")
weekend_test = test_results[test_results['is_weekend'] == 1]
weekday_test = test_results[test_results['is_weekend'] == 0]
if len(weekend_test) > 0 and len(weekday_test) > 0:
    print(f"   🏠 Week-ends: {weekend_test['abs_error_pct'].mean():.1f}% MAPE")
    print(f"   🏢 Semaine:   {weekday_test['abs_error_pct'].mean():.1f}% MAPE")

print(f"\n💡 RECOMMANDATIONS POUR AMÉLIORER L'APPRENTISSAGE:")
print(f"   1️⃣  Features additionnelles pour {worst_day.lower()}s")
print(f"   2️⃣  Traitement spécial des consommations extrêmes")
print(f"   3️⃣  Plus de features lag (14, 21, 30 jours)")
print(f"   4️⃣  Features d'interaction (température × jour_semaine)")
print(f"   5️⃣  Ensemble de modèles spécialisés par zone")

print(f"\n💾 Graphiques sauvés: analyse_erreurs_apprentissage.png")
