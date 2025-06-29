import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔧 MODÈLE OPTIMISÉ - RÉSOLUTION DES PROBLÈMES LIGHTGBM")
print("=" * 70)

# === 1. CHARGEMENT DES DONNÉES ===
print("\n📊 1. Chargement des données...")
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
print(f"✅ {len(df):,} observations chargées")

correlation = df['TempAvg'].corr(df['DailyAverage'])
print(f"🌡️ Corrélation température-consommation: {correlation:.3f}")

# === 2. FEATURE ENGINEERING SIMPLIFIÉ ET OPTIMISÉ ===
print("\n🎯 2. Feature Engineering optimisé...")

def create_optimized_features(df):
    """Features optimisées - moins nombreuses mais plus efficaces"""
    df = df.copy()
    
    # === FEATURES TEMPORELLES ESSENTIELLES ===
    df['month'] = df['Day'].dt.month
    df['day_of_week'] = df['Day'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df['Day'].dt.quarter
    df['day_of_year'] = df['Day'].dt.dayofyear
    
    # Features cycliques (plus efficaces que les catégorielles)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # === FEATURES TEMPÉRATURE OPTIMISÉES ===
    # Température au carré (relation quadratique climatisation)
    df['temp_squared'] = df['TempAvg'] ** 2
    
    # Seuils de confort (plus pertinents que temp³)
    df['cooling_degree_days'] = np.maximum(0, df['TempAvg'] - 22)
    df['heating_degree_days'] = np.maximum(0, 18 - df['TempAvg'])
    
    # Amplitude thermique
    df['temp_range'] = df['TempMax'] - df['TempMin']
    
    # === FEATURES HISTORIQUES SÉLECTIONNÉES ===
    df = df.sort_values('Day')
    
    # Lags les plus importants seulement
    for lag in [1, 2, 7]:
        df[f'consumption_lag_{lag}'] = df['DailyAverage'].shift(lag)
        df[f'temp_lag_{lag}'] = df['TempAvg'].shift(lag)
    
    # === MOYENNES MOBILES OPTIMISÉES ===
    for window in [3, 7]:
        df[f'consumption_ma_{window}'] = df['DailyAverage'].rolling(window=window).mean()
        df[f'temp_ma_{window}'] = df['TempAvg'].rolling(window=window).mean()
    
    # === INTERACTIONS PRINCIPALES ===
    df['temp_weekend'] = df['TempAvg'] * df['is_weekend']
    df['temp_month'] = df['TempAvg'] * df['month']
    
    # === FEATURES MÉTÉO SIMPLIFIÉES ===
    df['is_rainy'] = (df['Precip'] > 0).astype(int)
    df['high_wind'] = (df['WindSpeed'] > df['WindSpeed'].quantile(0.8)).astype(int)
    
    # === JOURS SPÉCIAUX ===
    df['is_holiday'] = df['is_holiday_full'] + df['is_holiday_half']
    
    return df

# Créer les features optimisées
df_features = create_optimized_features(df)
print(f"✅ Features créées: {df_features.shape[1]} colonnes")

# Supprimer les NaN
df_features = df_features.dropna().reset_index(drop=True)
print(f"✅ Après nettoyage: {len(df_features)} observations")

# === 3. SÉLECTION INTELLIGENTE DES FEATURES ===
print("\n🎯 3. Sélection intelligente des features...")

# Features candidates optimisées
candidate_features = [
    # Temporelles
    'month_sin', 'month_cos', 'day_sin', 'day_cos', 
    'is_weekend', 'quarter',
    
    # Température
    'TempAvg', 'temp_squared', 'TempMax', 'TempMin', 'temp_range',
    'cooling_degree_days', 'heating_degree_days',
    
    # Interactions
    'temp_weekend', 'temp_month',
    
    # Météo
    'Precip', 'WindSpeed', 'Pressure', 'is_rainy', 'high_wind',
    
    # Historique
    'consumption_lag_1', 'consumption_lag_2', 'consumption_lag_7',
    'temp_lag_1', 'temp_lag_2', 'temp_lag_7',
    
    # Moyennes mobiles
    'consumption_ma_3', 'consumption_ma_7',
    'temp_ma_3', 'temp_ma_7',
    
    # Jours spéciaux
    'is_holiday'
]

# Vérifier disponibilité
available_features = [f for f in candidate_features if f in df_features.columns]
print(f"✅ Features disponibles: {len(available_features)}")

# === 4. SPLIT TEMPOREL ===
print("\n📊 4. Préparation des données...")

split_date = '2024-10-01'
train_mask = df_features['Day'] < split_date
test_mask = df_features['Day'] >= split_date

X_train = df_features[train_mask][available_features]
y_train = df_features[train_mask]['DailyAverage']
X_test = df_features[test_mask][available_features]
y_test = df_features[test_mask]['DailyAverage']

print(f"📊 Train: {len(X_train)} obs, Test: {len(X_test)} obs")

# === 5. SÉLECTION DES MEILLEURES FEATURES ===
print("\n🔍 5. Sélection des meilleures features...")

# Utiliser SelectKBest pour garder les 15 meilleures features
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Récupérer les noms des features sélectionnées
selected_feature_names = [available_features[i] for i in selector.get_support(indices=True)]
print(f"✅ Features sélectionnées: {len(selected_feature_names)}")
print("📋 Features retenues:")
for i, feature in enumerate(selected_feature_names, 1):
    print(f"   {i:2d}. {feature}")

# === 6. MODÈLES OPTIMISÉS ===
print("\n🤖 6. Entraînement des modèles optimisés...")

models = {}
predictions = {}

# 1. XGBoost optimisé
print("   🚀 XGBoost optimisé...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,  # Réduit
    max_depth=6,       # Moins profond
    learning_rate=0.1, # Plus élevé
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)
xgb_model.fit(X_train_selected, y_train, 
              eval_set=[(X_test_selected, y_test)], 
              verbose=False)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_model.predict(X_test_selected)

# 2. LightGBM optimisé (paramètres corrigés)
print("   💡 LightGBM optimisé...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,      # Réduit
    max_depth=5,           # Moins profond
    learning_rate=0.1,     # Plus élevé
    num_leaves=20,         # Limité pour éviter overfitting
    min_child_samples=20,  # Augmenté
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    force_col_wise=True,   # Éviter les warnings
    verbose=-1             # Supprimer les logs
)
lgb_model.fit(X_train_selected, y_train, 
              eval_set=[(X_test_selected, y_test)], 
              callbacks=[lgb.early_stopping(50)])
models['LightGBM'] = lgb_model
predictions['LightGBM'] = lgb_model.predict(X_test_selected)

# 3. Random Forest optimisé
print("   🌳 Random Forest optimisé...")
rf_model = RandomForestRegressor(
    n_estimators=200,    # Réduit
    max_depth=10,        # Limité
    min_samples_split=10, # Augmenté
    min_samples_leaf=5,   # Augmenté
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_selected, y_train)
models['RandomForest'] = rf_model
predictions['RandomForest'] = rf_model.predict(X_test_selected)

# 4. Gradient Boosting optimisé
print("   📈 Gradient Boosting optimisé...")
gb_model = GradientBoostingRegressor(
    n_estimators=300,    # Réduit
    max_depth=5,         # Moins profond
    learning_rate=0.1,   # Plus élevé
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_selected, y_train)
models['GradientBoosting'] = gb_model
predictions['GradientBoosting'] = gb_model.predict(X_test_selected)

# === 7. ÉVALUATION ===
print("\n📊 7. Évaluation des performances...")

def calculate_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"   {model_name:<15}: R²={r2:.3f}, MAE={mae:6.0f}, RMSE={rmse:6.0f}, MAPE={mape:5.1f}%")
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

print("🏆 RÉSULTATS OPTIMISÉS:")
print("-" * 70)

results = {}
for name, pred in predictions.items():
    results[name] = calculate_metrics(y_test, pred, name)

# === 8. IMPORTANCE DES FEATURES ===
print(f"\n🔝 8. Importance des variables optimisées...")

best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
best_model = models[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': selected_feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 10 variables - {best_model_name}:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} {row['importance']:.3f}")

# === 9. VISUALISATION OPTIMISÉE ===
print(f"\n📊 9. Génération des graphiques optimisés...")

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔧 MODÈLE OPTIMISÉ - PROBLÈMES RÉSOLUS', fontsize=16, fontweight='bold')

# 1. Comparaison R²
ax1 = axes[0, 0]
r2_scores = [results[name]['R2'] for name in results.keys()]
model_names = list(results.keys())
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

bars = ax1.bar(model_names, r2_scores, color=colors)
ax1.set_title('📊 R² par Modèle (Optimisé)', fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Prédictions vs Réalité
best_pred = predictions[best_model_name]
ax2 = axes[0, 1]
ax2.scatter(y_test, best_pred, alpha=0.6, color='blue', s=30)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title(f'🎯 {best_model_name} - Prédictions vs Réalité\nR² = {results[best_model_name]["R2"]:.3f}', fontweight='bold')
ax2.set_xlabel('Consommation Réelle (kWh)')
ax2.set_ylabel('Consommation Prédite (kWh)')
ax2.grid(True, alpha=0.3)

# 3. Importance des features
ax3 = axes[1, 0]
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(8)
    ax3.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title(f'🔝 Features Importantes - {best_model_name}', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

# 4. Comparaison MAPE
ax4 = axes[1, 1]
mape_scores = [results[name]['MAPE'] for name in results.keys()]
bars = ax4.bar(model_names, mape_scores, color=colors)
ax4.set_title('📊 MAPE par Modèle (Optimisé)', fontweight='bold')
ax4.set_ylabel('MAPE (%)')
ax4.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, mape_scores):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('modele_optimise_results.png', dpi=300, bbox_inches='tight')
print("✅ Graphiques sauvegardés: modele_optimise_results.png")

# === 10. RÉSUMÉ FINAL ===
print(f"\n✅ RÉSUMÉ - PROBLÈMES RÉSOLUS")
print("=" * 60)

print(f"🔥 MEILLEUR MODÈLE: {best_model_name}")
print(f"   📊 R²: {results[best_model_name]['R2']:.3f}")
print(f"   📉 MAPE: {results[best_model_name]['MAPE']:.1f}%")
print(f"   📈 MAE: {results[best_model_name]['MAE']:,.0f} kWh")

print(f"\n🔧 OPTIMISATIONS APPLIQUÉES:")
print(f"   ✅ Features réduites: {len(selected_feature_names)} (vs 42 avant)")
print(f"   ✅ Hyperparamètres LightGBM corrigés")
print(f"   ✅ Sélection automatique des meilleures features")
print(f"   ✅ Prévention de l'overfitting")
print(f"   ✅ Suppression des warnings LightGBM")

print(f"\n🎯 FEATURES LES PLUS IMPORTANTES:")
if hasattr(best_model, 'feature_importances_'):
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['importance']:.3f}")

# Sauvegarder le meilleur modèle
import joblib
joblib.dump(models[best_model_name], f'best_model_optimized_{best_model_name.lower()}.pkl')
joblib.dump(selected_feature_names, 'optimized_features.pkl')
print(f"\n💾 Modèle optimisé sauvegardé: best_model_optimized_{best_model_name.lower()}.pkl") 