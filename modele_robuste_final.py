import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 MODÈLE COMPLET - TOUTES VARIABLES MÉTÉO ET VRAIS JOURS FÉRIÉS")
print("Features complètes : Température, Précipitations, Vent, Pression")
print("=" * 70)

# === FONCTION DE DÉTECTION DES JOURS FÉRIÉS BASÉE SUR LES VRAIES DONNÉES ===
def creer_detecteur_jours_feries(csv_path="data_with_context_fixed.csv"):
    """
    Crée un détecteur de jours fériés basé sur les vraies données historiques
    """
    df = pd.read_csv(csv_path)
    df['Day'] = pd.to_datetime(df['Day'])
    
    # Extraire les jours fériés réels
    jours_feries = df[(df['is_holiday_full'] == 1) | (df['is_holiday_half'] == 1)].copy()
    
    # Créer des patterns récurrents
    patterns_feries = set()
    
    for _, row in jours_feries.iterrows():
        date = row['Day']
        # Ajouter (mois, jour) pour les fêtes fixes
        patterns_feries.add((date.month, date.day))
    
    print(f"🎉 {len(patterns_feries)} patterns de jours fériés détectés")
    
    def detecter_jour_ferie(date):
        """Détecte si une date est un jour férié"""
        return 1 if (date.month, date.day) in patterns_feries else 0
    
    return detecter_jour_ferie, patterns_feries

# Créer le détecteur global
detecteur_feries, patterns_feries_globaux = creer_detecteur_jours_feries()

# === 1. CHARGEMENT ET PRÉPARATION AVANCÉE ===
print("\n📊 1. PRÉPARATION COMPLÈTE DES DONNÉES...")

df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

print(f"📊 Données chargées: {len(df)} jours")
print(f"📊 Variables météo: TempAvg, TempMin, TempMax, Precip, WindSpeed, Pressure")

# Vérifier les données manquantes
print("🔍 Vérification données manquantes:")
for col in ['TempAvg', 'TempMin', 'TempMax', 'Precip', 'WindSpeed', 'Pressure']:
    missing = df[col].isnull().sum()
    print(f"   • {col}: {missing} valeurs manquantes")

# === 2. ENGINEERING FEATURES COMPLET ===
print("🚀 2. Création de features complètes avec TOUTES les variables météo...")

def create_features_completes(df):
    """Créé des features complètes avec TOUTES les variables météo"""
    
    df = df.copy()
    
    # Calcul de la médiane de température
    temp_median = df['TempAvg'].median()
    
    # === FEATURES TEMPÉRATURE COMPLÈTES ===
    df['temp_range'] = df['TempMax'] - df['TempMin']  # Amplitude thermique
    df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
    df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
    df['temp_squared'] = df['TempAvg'] ** 2
    
    # === FEATURES PRÉCIPITATIONS ===
    df['precip_ma_7'] = df['Precip'].rolling(window=7, min_periods=1).mean()
    df['has_rain'] = (df['Precip'] > 0).astype(int)
    
    # === FEATURES VENT ET PRESSION ===
    df['wind_ma_7'] = df['WindSpeed'].rolling(window=7, min_periods=1).mean()
    df['pressure_ma_7'] = df['Pressure'].rolling(window=30, min_periods=1).mean()
    
    # === SEUILS TEMPÉRATURE OPTIMISÉS ===
    temp_25, temp_30 = 25.0, 30.0
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - temp_25)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - temp_30)
    df['heating_needs'] = np.maximum(0, temp_25 - df['TempAvg'])
    
    # Seuils binaires
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
    
    # === JOURS SPÉCIAUX ===
    df['is_weekend'] = (df['Day'].dt.dayofweek >= 5).astype(int)
    
    # 🔥 UTILISATION DU VRAI DÉTECTEUR DE JOURS FÉRIÉS
    df['is_holiday'] = df['Day'].apply(detecteur_feries)
    
    return df, temp_median

df_features, temp_median = create_features_completes(df)

# IMPORTANT : Utiliser les vraies données de jours fériés pour l'entraînement
vraie_holiday = (df['is_holiday_full'] + df['is_holiday_half'] > 0).astype(int)
df_features['is_holiday'] = vraie_holiday

# Sélection des features complètes
enhanced_features = [
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
    'time_trend', 'is_weekend', 'is_holiday'
]

print(f"✅ {len(enhanced_features)} features complètes créées")
print(f"   🌡️ Médiane température: {temp_median:.1f}°C")
print(f"🎉 {vraie_holiday.sum()} jours fériés dans les données d'entraînement")

# === 3. SPLIT TEMPOREL PROPRE ===
print("\n📅 3. Split temporel robuste...")

# Split plus équilibré : 70% train, 30% test
split_idx = int(len(df_features) * 0.7)
train_data = df_features.iloc[:split_idx].copy()
test_data = df_features.iloc[split_idx:].copy()

print(f"📊 Split plus équilibré:")
print(f"   Train: {train_data['Day'].min().date()} → {train_data['Day'].max().date()} ({len(train_data)} jours)")
print(f"   Test:  {test_data['Day'].min().date()} → {test_data['Day'].max().date()} ({len(test_data)} jours)")
print(f"   Ratio: {len(train_data)/len(test_data):.1f}:1")

# Vérifier la répartition des jours fériés
nb_feries_train = train_data['is_holiday'].sum()
nb_feries_test = test_data['is_holiday'].sum()
print(f"📅 Jours fériés: Train={nb_feries_train}, Test={nb_feries_test}")

# === 4. NORMALISATION ===
print("📐 4. Normalisation des features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data[enhanced_features])
X_test_scaled = scaler.transform(test_data[enhanced_features])

y_train = train_data['DailyAverage'].values
y_test = test_data['DailyAverage'].values

# === 5. MODÈLES ROBUSTES AVEC RÉGULARISATION ===
print("\n🤖 5. Test de modèles robustes...")

models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random_Forest': RandomForestRegressor(
        n_estimators=100,     # Optimisé pour les nouvelles features
        max_depth=10,         # Profondeur adaptée
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
}

# === 6. VALIDATION CROISÉE TEMPORELLE ROBUSTE ===
print("🔍 6. Validation croisée temporelle...")

tscv = TimeSeriesSplit(n_splits=5)
results = {}

for name, model in models.items():
    print(f"\n   🧪 Test du modèle {name}...")
    
    cv_scores = []
    cv_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
        X_fold_train = X_train_scaled[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Ajustement des hyperparamètres pour certains modèles
        if name == 'Ridge':
            model.alpha = 10.0   # Régularisation modérée
        elif name == 'Lasso':
            model.alpha = 1.0    # Régularisation adaptée pour capturer non-linéarités
        
        model.fit(X_fold_train, y_fold_train)
        val_pred = model.predict(X_fold_val)
        
        r2 = r2_score(y_fold_val, val_pred)
        mae = mean_absolute_error(y_fold_val, val_pred)
        
        cv_scores.append(r2)
        cv_maes.append(mae)
        
        print(f"      Fold {fold+1}: R² = {r2:.3f}, MAE = {mae:.0f}")
    
    results[name] = {
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'cv_mae_mean': np.mean(cv_maes),
        'cv_mae_std': np.std(cv_maes)
    }
    
    print(f"   📊 {name} - Moyenne R²: {results[name]['cv_r2_mean']:.3f} ± {results[name]['cv_r2_std']:.3f}")

# === 7. SÉLECTION DU MEILLEUR MODÈLE ===
print(f"\n🏆 7. Sélection du meilleur modèle...")

best_model_name = max(results.keys(), key=lambda x: results[x]['cv_r2_mean'])
best_model = models[best_model_name]

print(f"🎯 Meilleur modèle: {best_model_name}")
print(f"   R² CV: {results[best_model_name]['cv_r2_mean']:.3f} ± {results[best_model_name]['cv_r2_std']:.3f}")

# === 8. ENTRAÎNEMENT FINAL ET ÉVALUATION ===
print(f"\n📈 8. Évaluation finale sur test...")

# Réentraîner sur tout le train
if best_model_name == 'Ridge':
    best_model.alpha = 10.0
elif best_model_name == 'Lasso':
    best_model.alpha = 1.0

best_model.fit(X_train_scaled, y_train)

# Prédictions
train_pred = best_model.predict(X_train_scaled)
test_pred = best_model.predict(X_test_scaled)

# Métriques finales
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

overfitting = train_r2 - test_r2

print(f"📊 RÉSULTATS FINAUX:")
print(f"   🔧 Train: R² = {train_r2:.3f}, MAE = {train_mae:.0f}")
print(f"   🧪 Test:  R² = {test_r2:.3f}, MAE = {test_mae:.0f}")
print(f"   ⚠️  Overfitting: {overfitting:.3f}")

if overfitting < 0.1:
    print(f"   ✅ OVERFITTING CONTRÔLÉ !")
elif overfitting < 0.2:
    print(f"   🟡 Overfitting acceptable")
else:
    print(f"   🔴 Overfitting encore présent")

# === 9. VISUALISATIONS ===
print(f"\n📊 9. Génération des visualisations...")

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'🚀 MODÈLE COMPLET - {best_model_name.upper()}', fontsize=16, fontweight='bold')

# 1. Timeline avec prédictions
ax1 = axes[0, 0]
ax1.plot(train_data['Day'], y_train, 'b-', alpha=0.7, label='Train Réel', linewidth=1)
ax1.plot(train_data['Day'], train_pred, 'b--', alpha=0.8, label='Train Prédit', linewidth=1)
ax1.plot(test_data['Day'], y_test, 'r-', alpha=0.8, label='Test Réel', linewidth=2)
ax1.plot(test_data['Day'], test_pred, 'r--', alpha=0.8, label='Test Prédit', linewidth=2)
ax1.set_title('📅 Timeline - Modèle Complet')
ax1.set_xlabel('Date')
ax1.set_ylabel('Consommation (kWh)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Train scatter
ax2 = axes[0, 1]
ax2.scatter(y_train, train_pred, alpha=0.6, s=15, color='blue')
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax2.set_title(f'🔧 TRAIN\nR² = {train_r2:.3f}')
ax2.set_xlabel('Réel (kWh)')
ax2.set_ylabel('Prédit (kWh)')
ax2.grid(True, alpha=0.3)

# 3. Test scatter
ax3 = axes[0, 2]
ax3.scatter(y_test, test_pred, alpha=0.6, s=15, color='red')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_title(f'🧪 TEST\nR² = {test_r2:.3f}')
ax3.set_xlabel('Réel (kWh)')
ax3.set_ylabel('Prédit (kWh)')
ax3.grid(True, alpha=0.3)

# 4. Résidus
ax4 = axes[1, 0]
train_residuals = y_train - train_pred
test_residuals = y_test - test_pred
ax4.scatter(train_pred, train_residuals, alpha=0.6, s=10, color='blue', label='Train')
ax4.scatter(test_pred, test_residuals, alpha=0.6, s=10, color='red', label='Test')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_title('📊 Résidus vs Prédictions')
ax4.set_xlabel('Prédictions (kWh)')
ax4.set_ylabel('Résidus (kWh)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Distribution des résidus
ax5 = axes[1, 1]
ax5.hist(train_residuals, bins=20, alpha=0.7, label='Train', density=True, color='blue')
ax5.hist(test_residuals, bins=20, alpha=0.7, label='Test', density=True, color='red')
ax5.set_title('📊 Distribution des Résidus')
ax5.set_xlabel('Résidus (kWh)')
ax5.set_ylabel('Densité')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Comparaison validation croisée
ax6 = axes[1, 2]
model_names = list(results.keys())
cv_means = [results[name]['cv_r2_mean'] for name in model_names]
cv_stds = [results[name]['cv_r2_std'] for name in model_names]

bars = ax6.bar(range(len(model_names)), cv_means, yerr=cv_stds, 
               color='lightblue', alpha=0.8, capsize=5)
ax6.set_title('📊 Validation Croisée')
ax6.set_xlabel('Modèles')
ax6.set_ylabel('R² (moyenne ± écart-type)')
ax6.set_xticks(range(len(model_names)))
ax6.set_xticklabels(model_names, rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

# Marquer le meilleur
best_idx = model_names.index(best_model_name)
bars[best_idx].set_color('gold')

plt.tight_layout()
plt.savefig('modele_robuste_final.png', dpi=300, bbox_inches='tight')
plt.show()

# === 10. SYSTÈME D'ALERTE AMÉLIORÉ ===
print(f"\n🚨 10. Mise à jour du système d'alerte...")

# Calculer les nouveaux paramètres d'alerte
new_train_mae = train_mae
new_train_std = np.std(train_residuals)

print(f"📊 NOUVEAUX PARAMÈTRES D'ALERTE:")
print(f"   MAE: {new_train_mae:.0f} kWh")
print(f"   STD: {new_train_std:.0f} kWh")

# === 10.5. ANALYSE DES FEATURES IMPORTANTES ===
print(f"\n🔍 10.5. Analyse des features importantes...")

if hasattr(best_model, 'coef_'):
    # Pour les modèles linéaires
    feature_importance = abs(best_model.coef_)
    feature_names = enhanced_features
    
    # Trier par importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"🎯 TOP 15 FEATURES LES PLUS IMPORTANTES:")
    for i, row in importance_df.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:8.0f}")
    
    # Features par catégorie
    categories = {
        'Météo': ['TempAvg', 'TempMin', 'TempMax', 'Precip', 'WindSpeed', 'Pressure'],
        'Saisonnier': ['is_summer', 'is_mid_summer', 'temp_x_summer', 'temp_squared_x_summer'],
        'Jours spéciaux': ['is_weekend', 'is_holiday']
    }
    
    print(f"\n📊 IMPORTANCE PAR CATÉGORIE:")
    for category, features in categories.items():
        category_importance = importance_df[importance_df['feature'].isin(features)]
        if not category_importance.empty:
            total_importance = category_importance['importance'].sum()
            print(f"   🏷️ {category}: {total_importance:,.0f}")
            for _, row in category_importance.head(3).iterrows():
                print(f"      • {row['feature']:20s}: {row['importance']:6.0f}")
    
elif hasattr(best_model, 'feature_importances_'):
    # Pour Random Forest
    feature_importance = best_model.feature_importances_
    feature_names = enhanced_features
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"🎯 TOP 15 FEATURES LES PLUS IMPORTANTES:")
    for i, row in importance_df.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")

# === 12. SAUVEGARDE DU MODÈLE COMPLET ===
print(f"\n💾 12. Sauvegarde du modèle complet...")

model_data = {
    'model': best_model,
    'scaler': scaler,
    'mae': test_mae,
    'temp_median': temp_median,
    'features': enhanced_features,
    'patterns_feries': patterns_feries_globaux,
    'performance': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfitting': overfitting
    },
    'version': 'complet_v1_meteo'
}

with open('modele_prediction_complet.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"✅ Modèle complet sauvegardé dans 'modele_prediction_complet.pkl'")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print(f"   📊 modele_robuste_final.png")
print(f"   🤖 modele_prediction_complet.pkl")
print(f"   ✨ Modèle prêt pour la production (optimisé toutes variables)")

# === 11. SYNTHÈSE FINALE ===
print(f"\n✅ SYNTHÈSE - MODÈLE COMPLET CRÉÉ")
print("=" * 60)

print(f"🎯 MODÈLE SÉLECTIONNÉ: {best_model_name}")
print(f"📊 PERFORMANCE:")
print(f"   • Test R²: {test_r2:.3f}")
print(f"   • Test MAE: {test_mae:.0f} kWh")
print(f"   • Overfitting: {overfitting:.3f}")

print(f"\n🚀 NOUVELLES FEATURES COMPLÈTES:")
print(f"   ✅ 6 variables météo : TempAvg, TempMin, TempMax, Precip, WindSpeed, Pressure")
print(f"   ✅ Features non-linéaires : temp_squared, temp_range, interactions")
print(f"   ✅ Moyennes mobiles : température (7j, 30j), pluie (7j), vent (7j), pression (30j)")
print(f"   ✅ Vraies dates de jours fériés : {len(patterns_feries_globaux)} patterns détectés")
print(f"   ✅ Interactions avancées : temp×été, temp×vent, pression×temp")
print(f"   ✅ Total : {len(enhanced_features)} features")

if overfitting < 0.15 and test_r2 > 0.3:
    print(f"\n🎉 MODÈLE COMPLET ET ROBUSTE CRÉÉ !")
    print(f"   Utilise TOUTES les variables météo disponibles")
    print(f"   Détecte automatiquement les vrais jours fériés")
    print(f"   Optimisé pour la production industrielle")
else:
    print(f"\n⚠️  Modèle amélioré mais nécessite encore du travail")
    print(f"   Considérer d'autres approches (séries temporelles)") 