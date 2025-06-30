import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

print("🔮 PRÉDICTEUR DE CONSOMMATION FUTURE")
print("=" * 60)

# === 1. CHARGEMENT DU MODÈLE ===
print("\n🤖 1. Chargement du modèle optimisé...")

try:
    with open('modele_optimise_avec_lags.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    patterns_feries = model_data['patterns_feries']
    performance = model_data['performance']
    
    print(f"✅ Modèle chargé avec succès !")
    print(f"   • Performance: MAE {performance['test_mae']:.0f} kWh (R² {performance['test_r2']:.3f})")
    
except FileNotFoundError:
    print("❌ Erreur: Modèle non trouvé.")
    exit(1)

# === 2. CHARGEMENT DES DONNÉES HISTORIQUES ===
print("\n📊 2. Chargement des données historiques...")

df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

print(f"✅ Données historiques: {len(df)} jours")
print(f"   Dernière date: {df['Day'].max().date()}")

# === 3. FONCTIONS DE SIMULATION MÉTÉO ===
def simuler_meteo_future(start_date, end_date, reference_years=[2024, 2023, 2022]):
    """Simule des données météo futures basées sur les vraies données historiques"""
    
    # Extraire toutes les données historiques pour la même période (même mois/jour)
    month = start_date.month
    start_day = start_date.day
    end_day = end_date.day
    
    # Collecter toutes les données historiques pour cette période de l'année
    historical_data = []
    
    for year in reference_years:
        try:
            ref_start = start_date.replace(year=year)
            ref_end = end_date.replace(year=year)
            
            ref_mask = (df['Day'] >= ref_start) & (df['Day'] <= ref_end)
            ref_data = df[ref_mask].copy()
            
            if len(ref_data) > 0:
                historical_data.append(ref_data)
        except:
            continue
    
    # Si pas de données exactes, utiliser tout le mois
    if len(historical_data) == 0:
        print(f"   📝 Pas de données exactes, utilisation du mois {month}")
        monthly_data = df[df['Day'].dt.month == month].copy()
        if len(monthly_data) > 0:
            historical_data = [monthly_data]
        else:
            # Dernière option: tout l'historique
            historical_data = [df]
    
    # Combiner toutes les données historiques
    all_historical = pd.concat(historical_data, ignore_index=True)
    
    # Calculer les statistiques réalistes
    stats = {
        'temp_avg_mean': all_historical['TempAvg'].mean(),
        'temp_avg_std': all_historical['TempAvg'].std(),
        'temp_min_mean': all_historical['TempMin'].mean(),
        'temp_min_std': all_historical['TempMin'].std(),
        'temp_max_mean': all_historical['TempMax'].mean(),
        'temp_max_std': all_historical['TempMax'].std(),
        'precip_mean': all_historical['Precip'].mean(),
        'precip_std': all_historical['Precip'].std(),
        'wind_mean': all_historical['WindSpeed'].mean(),
        'wind_std': all_historical['WindSpeed'].std(),
        'pressure_mean': all_historical['Pressure'].mean(),
        'pressure_std': all_historical['Pressure'].std()
    }
    
    print(f"   📊 Références: {len(all_historical)} jours historiques")
    print(f"   🌡️  Temp historique: {stats['temp_avg_mean']:.1f}°C ± {stats['temp_avg_std']:.1f}°C")
    
    # Générer les dates futures
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simuler des données réalistes
    np.random.seed(42)  # Pour la reproductibilité
    
    simulated_data = []
    for i, date in enumerate(dates):
        # Température moyenne: échantillonner autour de la moyenne historique
        temp_avg_sim = np.random.normal(stats['temp_avg_mean'], stats['temp_avg_std'] * 0.5)
        
        # Ajouter une légère variation saisonnière jour par jour
        day_variation = np.sin(2 * np.pi * i / len(dates)) * 2  # ±2°C max
        temp_avg_sim += day_variation
        
        # Température min/max cohérentes avec la moyenne
        temp_range = np.random.normal(
            stats['temp_max_mean'] - stats['temp_min_mean'], 
            (stats['temp_max_std'] + stats['temp_min_std']) * 0.5
        )
        temp_range = np.clip(temp_range, 5, 20)  # Entre 5°C et 20°C d'écart
        
        temp_min_sim = temp_avg_sim - temp_range / 2
        temp_max_sim = temp_avg_sim + temp_range / 2
        
        # Autres variables météo basées sur l'historique
        precip_sim = np.maximum(0, np.random.gamma(2, stats['precip_mean'] / 2))
        wind_sim = np.maximum(0, np.random.normal(stats['wind_mean'], stats['wind_std'] * 0.7))
        pressure_sim = np.random.normal(stats['pressure_mean'], stats['pressure_std'] * 0.5)
        
        simulated_data.append({
            'Day': date,
            'TempAvg': temp_avg_sim,
            'TempMin': temp_min_sim,
            'TempMax': temp_max_sim,
            'Precip': precip_sim,
            'WindSpeed': wind_sim,
            'Pressure': pressure_sim
        })
    
    return pd.DataFrame(simulated_data)

def detecter_jour_ferie(date):
    """Détecte si une date est un jour férié"""
    return 1 if (date.month, date.day) in patterns_feries else 0

def create_features_prediction(df_future, df_historical):
    """Créé des features pour la prédiction future"""
    df = df_future.copy()
    
    # === FEATURES MÉTÉO ===
    df['temp_range'] = df['TempMax'] - df['TempMin']
    df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
    df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
    df['temp_squared'] = df['TempAvg'] ** 2
    
    df['precip_ma_7'] = df['Precip'].rolling(window=7, min_periods=1).mean()
    df['has_rain'] = (df['Precip'] > 0).astype(int)
    df['wind_ma_7'] = df['WindSpeed'].rolling(window=7, min_periods=1).mean()
    df['pressure_ma_7'] = df['Pressure'].rolling(window=30, min_periods=1).mean()
    
    # === SEUILS TEMPÉRATURE ===
    temp_25, temp_30 = 25.0, 30.0
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - temp_25)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - temp_30)
    df['heating_needs'] = np.maximum(0, temp_25 - df['TempAvg'])
    
    df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
    df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
    df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
    
    # === SAISONS ===
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_mid_summer'] = (df['Day'].dt.month == 7).astype(int)
    
    # === FEATURES CYCLIQUES ===
    df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
    
    # === INTERACTIONS ===
    df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
    df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
    df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
    df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
    df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
    
    # === TEMPOREL ===
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    df['is_weekend'] = (df['Day'].dt.dayofweek >= 5).astype(int)
    df['is_holiday'] = df['Day'].apply(detecter_jour_ferie)
    
    # === LAGS CRITIQUES - UTILISER LES DERNIÈRES DONNÉES CONNUES ===
    # Prendre les dernières valeurs de consommation connues
    last_consumptions = df_historical.tail(10)['DailyAverage'].values
    
    # Pour les lags, utiliser une approche de continuation intelligente
    df['consumption_lag_1'] = np.nan
    df['consumption_lag_7'] = np.nan
    
    # Simuler des valeurs de lag basées sur les patterns récents
    recent_avg = last_consumptions.mean()
    recent_std = last_consumptions.std()
    
    for i in range(len(df)):
        if i == 0:
            # Premier jour: utiliser la dernière consommation connue
            df.loc[i, 'consumption_lag_1'] = last_consumptions[-1]
            df.loc[i, 'consumption_lag_7'] = last_consumptions[-7] if len(last_consumptions) >= 7 else recent_avg
        elif i < 7:
            # Utiliser les prédictions précédentes ou les données historiques
            if i == 1:
                df.loc[i, 'consumption_lag_1'] = last_consumptions[-1]
            else:
                # Pour simplifier, utiliser une estimation basée sur les patterns
                df.loc[i, 'consumption_lag_1'] = recent_avg + np.random.normal(0, recent_std * 0.1)
            
            if i + len(last_consumptions) >= 7:
                df.loc[i, 'consumption_lag_7'] = last_consumptions[-(7-i)]
            else:
                df.loc[i, 'consumption_lag_7'] = recent_avg
        else:
            # Utiliser les valeurs simulées précédentes
            df.loc[i, 'consumption_lag_1'] = recent_avg + np.random.normal(0, recent_std * 0.1)
            df.loc[i, 'consumption_lag_7'] = recent_avg + np.random.normal(0, recent_std * 0.1)
    
    # === FEATURES FIN D'ANNÉE ===
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

# === 4. INTERFACE UTILISATEUR ===
def parse_date(date_str):
    """Parse une date en format flexible"""
    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError("Format de date non reconnu")

print("\n📅 3. Sélection de la période future à prédire")
print("   Format accepté: YYYY-MM-DD, DD/MM/YYYY ou DD-MM-YYYY")
print("   📌 Note: Pour les prédictions futures, le modèle simule des données météo")

while True:
    try:
        print(f"\n🔮 PÉRIODE FUTURE À PRÉDIRE:")
        start_str = input("   Date de début: ")
        end_str = input("   Date de fin: ")
        start_date = parse_date(start_str)
        end_date = parse_date(end_str)
        
        if start_date >= end_date:
            print("❌ La date de début doit être antérieure à la date de fin")
            continue
            
        # Vérifier si c'est vraiment dans le futur
        last_date = df['Day'].max()
        if start_date <= last_date:
            print(f"⚠️  Attention: {start_date.date()} est dans vos données historiques")
            print(f"   Dernière date disponible: {last_date.date()}")
            choice = input("   Continuer quand même ? (o/n): ")
            if choice.lower() not in ['o', 'oui', 'y', 'yes']:
                continue
        
        break
        
    except ValueError as e:
        print(f"❌ Erreur: {e}")

print(f"\n✅ Période future sélectionnée:")
print(f"   {start_date.date()} → {end_date.date()} ({(end_date-start_date).days + 1} jours)")

# === 5. SIMULATION MÉTÉO FUTURE ===
print(f"\n🌤️  4. Simulation des données météo futures...")

future_weather = simuler_meteo_future(start_date, end_date)
print(f"✅ Météo simulée pour {len(future_weather)} jours")
print(f"   Température moyenne prévue: {future_weather['TempAvg'].mean():.1f}°C")
print(f"   Plage de température: {future_weather['TempAvg'].min():.1f}°C → {future_weather['TempAvg'].max():.1f}°C")

# === 6. CRÉATION DES FEATURES ===
print(f"\n🔧 5. Création des features pour prédiction...")

future_features = create_features_prediction(future_weather, df)
print(f"✅ Features créées: {len(future_features.columns)} variables")

# === 7. PRÉDICTIONS ===
print(f"\n🤖 6. Génération des prédictions futures...")

X_future = future_features[features]
X_future_scaled = scaler.transform(X_future)
predictions = model.predict(X_future_scaled)

future_results = future_features.copy()
future_results['predictions'] = predictions

print(f"✅ Prédictions générées!")

# === 8. ANALYSE DES RÉSULTATS ===
print(f"\n📊 7. Analyse des résultats de prédiction...")

stats = {
    'nb_jours': len(future_results),
    'consommation_totale': future_results['predictions'].sum(),
    'consommation_moyenne': future_results['predictions'].mean(),
    'consommation_max': future_results['predictions'].max(),
    'consommation_min': future_results['predictions'].min(),
    'temp_moyenne': future_results['TempAvg'].mean(),
    'nb_weekends': future_results['is_weekend'].sum(),
    'nb_feries': future_results['is_holiday'].sum()
}

print(f"🔮 PRÉDICTIONS POUR {start_date.date()} → {end_date.date()}")
print("=" * 70)
print(f"📊 Consommation totale prévue:  {stats['consommation_totale']:,.0f} kWh")
print(f"📈 Consommation moyenne/jour:   {stats['consommation_moyenne']:,.0f} kWh")
print(f"🔥 Pic de consommation:         {stats['consommation_max']:,.0f} kWh")
print(f"📉 Minimum de consommation:     {stats['consommation_min']:,.0f} kWh")
print(f"🌡️  Température moyenne:        {stats['temp_moyenne']:.1f}°C")
print(f"📅 Weekends dans la période:    {stats['nb_weekends']} jours")
print(f"🎉 Jours fériés:                {stats['nb_feries']} jours")

# Calcul du coût estimé
prix_kwh = 0.15
cout_estime = stats['consommation_totale'] * prix_kwh
print(f"💰 Coût estimé ({prix_kwh}€/kWh):       {cout_estime:,.0f}€")

# === 9. COMPARAISON AVEC L'HISTORIQUE ===
print(f"\n🔍 8. Comparaison avec l'historique...")

# Comparer avec la même période de l'année précédente si disponible
reference_year = start_date.year - 1
ref_start = start_date.replace(year=reference_year)
ref_end = end_date.replace(year=reference_year)

ref_mask = (df['Day'] >= ref_start) & (df['Day'] <= ref_end)
ref_data = df[ref_mask]

if len(ref_data) > 0:
    ref_consumption = ref_data['DailyAverage'].sum()
    diff_vs_ref = stats['consommation_totale'] - ref_consumption
    diff_pct = (diff_vs_ref / ref_consumption) * 100
    
    print(f"📊 Comparaison avec {reference_year}:")
    print(f"   • {reference_year}: {ref_consumption:,.0f} kWh (réel)")
    print(f"   • {start_date.year}: {stats['consommation_totale']:,.0f} kWh (prédit)")
    print(f"   • Différence: {diff_pct:+.1f}% ({diff_vs_ref:+,.0f} kWh)")
else:
    print(f"❌ Pas de données de référence pour {reference_year}")

# === 10. VISUALISATIONS ===
print(f"\n📊 9. Génération des graphiques...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'🔮 PRÉDICTIONS FUTURES: {start_date.date()} → {end_date.date()}\n'
             f'Modèle Optimisé (MAE {performance["test_mae"]:.0f} kWh)', 
             fontsize=14, fontweight='bold')

# Plot 1: Évolution temporelle des prédictions
axes[0,0].plot(future_results['Day'], future_results['predictions'], 
               color='purple', linewidth=2, alpha=0.8, label='Prédictions futures')
axes[0,0].fill_between(future_results['Day'], future_results['predictions'], 
                       alpha=0.3, color='purple')
axes[0,0].set_ylabel('Consommation Prédite (kWh)')
axes[0,0].set_title('Évolution de la Consommation Future')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# Plot 2: Distribution des prédictions
axes[0,1].hist(future_results['predictions'], bins=15, alpha=0.7, color='purple', 
               edgecolor='black')
axes[0,1].axvline(stats['consommation_moyenne'], color='red', linestyle='--', 
                  linewidth=2, label=f'Moyenne: {stats["consommation_moyenne"]:.0f} kWh')
axes[0,1].set_xlabel('Consommation Prédite (kWh)')
axes[0,1].set_ylabel('Fréquence')
axes[0,1].set_title('Distribution des Prédictions')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Température vs Consommation
axes[1,0].scatter(future_results['TempAvg'], future_results['predictions'], 
                  alpha=0.7, color='purple', s=50)

# Ligne de tendance
z = np.polyfit(future_results['TempAvg'], future_results['predictions'], 2)
p = np.poly1d(z)
temp_range = np.linspace(future_results['TempAvg'].min(), 
                        future_results['TempAvg'].max(), 100)
axes[1,0].plot(temp_range, p(temp_range), color='red', linestyle='--', linewidth=2)

axes[1,0].set_xlabel('Température Simulée (°C)')
axes[1,0].set_ylabel('Consommation Prédite (kWh)')
axes[1,0].set_title('Impact de la Température')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Prédictions par jour de la semaine
future_results['day_of_week'] = future_results['Day'].dt.day_name()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_avg = future_results.groupby('day_of_week')['predictions'].mean().reindex(days_order)

axes[1,1].bar(range(len(days_order)), daily_avg.values, alpha=0.7, color='purple', 
              edgecolor='black')
axes[1,1].set_xlabel('Jour de la Semaine')
axes[1,1].set_ylabel('Consommation Moyenne (kWh)')
axes[1,1].set_title('Prédictions par Jour')
axes[1,1].set_xticks(range(len(days_order)))
axes[1,1].set_xticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()

# Sauvegarder
filename = f"predictions_futures_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Graphique sauvegardé: {filename}")

# === 11. EXPORT DES DONNÉES ===
save_data = input("\n💾 Voulez-vous sauvegarder les prédictions détaillées ? (o/n): ")

if save_data.lower() in ['o', 'oui', 'y', 'yes']:
    export_data = future_results[['Day', 'predictions', 'TempAvg', 'TempMin', 'TempMax', 
                                 'is_weekend', 'is_holiday']].copy()
    export_data.columns = ['Date', 'Consommation_Predite_kWh', 'Temp_Moyenne', 
                          'Temp_Min', 'Temp_Max', 'Weekend', 'Jour_Ferie']
    
    csv_filename = f"predictions_futures_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    export_data.to_csv(csv_filename, index=False)
    
    print(f"✅ Prédictions sauvegardées: {csv_filename}")

print("\n" + "="*70)
print("🔮 PRÉDICTION FUTURE TERMINÉE !")
print("="*70)
print(f"📊 Période: {start_date.date()} → {end_date.date()}")
print(f"⚡ Consommation totale: {stats['consommation_totale']:,.0f} kWh")
print(f"💰 Coût estimé: {cout_estime:,.0f}€")
print(f"🎯 Fiabilité modèle: R² {performance['test_r2']:.3f}")
print(f"📁 Graphique: {filename}")
print("="*70) 