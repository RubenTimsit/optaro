import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🇮🇱 COMPARATEUR SIMPLE - PÉRIODES HISTORIQUES (MODÈLE ISRAÉLIEN)")
print("=" * 70)

# === 1. CHARGEMENT DU MODÈLE ISRAÉLIEN ===
print("\n🤖 Chargement du modèle israélien optimisé...")

try:
    with open('modele_optimise_israel.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    performance = model_data['performance']
    
    print(f"✅ Modèle israélien chargé: MAE {performance['test_mae']:.0f} kWh, R² {performance['test_r2']:.3f}")
    print(f"🇮🇱 Week-ends: Vendredi-Samedi | Jours ouvrables: Dimanche-Jeudi")
    
except Exception as e:
    print(f"❌ Erreur modèle: {e}")
    exit(1)

# === 2. CHARGEMENT DES DONNÉES ===
print("\n📊 Chargement des données...")

try:
    df = pd.read_csv("data_with_context_fixed.csv")
    df['Day'] = pd.to_datetime(df['Day'])
    df = df.sort_values('Day').reset_index(drop=True)
    
    print(f"✅ Données: {len(df)} jours ({df['Day'].min().date()} → {df['Day'].max().date()})")
    
except Exception as e:
    print(f"❌ Erreur données: {e}")
    exit(1)

# === 3. EXEMPLES DE COMPARAISONS PRÉDÉFINIES ===
comparaisons_exemples = {
    "1": {
        "nom": "🌞 Été 2024 vs Été 2023",
        "periode1": ("2024-07-01", "2024-07-31", "Été 2024"),
        "periode2": ("2023-07-01", "2023-07-31", "Été 2023")
    },
    "2": {
        "nom": "❄️ Hiver vs Été 2024",
        "periode1": ("2024-12-01", "2024-12-31", "Hiver 2024"),
        "periode2": ("2024-07-01", "2024-07-31", "Été 2024")
    },
    "3": {
        "nom": "🗓️ Juin 2024 vs Juin 2023",
        "periode1": ("2024-06-01", "2024-06-30", "Juin 2024"),
        "periode2": ("2023-06-01", "2023-06-30", "Juin 2023")
    },
    "4": {
        "nom": "🔥 Août vs Septembre 2024",
        "periode1": ("2024-08-01", "2024-08-31", "Août 2024"),
        "periode2": ("2024-09-01", "2024-09-30", "Septembre 2024")
    }
}

print("\n📅 Comparaisons disponibles:")
for key, comp in comparaisons_exemples.items():
    print(f"   {key}. {comp['nom']}")

print("   5. Saisie manuelle de périodes")

choix = input("\n🎯 Votre choix (1-5): ")

if choix in comparaisons_exemples:
    comp = comparaisons_exemples[choix]
    start1_str, end1_str, label1 = comp["periode1"]
    start2_str, end2_str, label2 = comp["periode2"]
    
    start1 = pd.to_datetime(start1_str)
    end1 = pd.to_datetime(end1_str)
    start2 = pd.to_datetime(start2_str)
    end2 = pd.to_datetime(end2_str)
    
    print(f"\n✅ Comparaison sélectionnée: {comp['nom']}")
    
elif choix == "5":
    print("\n📝 Saisie manuelle (format: YYYY-MM-DD)")
    try:
        start1_str = input("   Période 1 - Début: ")
        end1_str = input("   Période 1 - Fin: ")
        start2_str = input("   Période 2 - Début: ")
        end2_str = input("   Période 2 - Fin: ")
        
        start1 = pd.to_datetime(start1_str)
        end1 = pd.to_datetime(end1_str)
        start2 = pd.to_datetime(start2_str)
        end2 = pd.to_datetime(end2_str)
        
        label1 = f"Période 1"
        label2 = f"Période 2"
        
    except Exception as e:
        print(f"❌ Erreur format: {e}")
        exit(1)
else:
    print("❌ Choix invalide")
    exit(1)

# === 4. FONCTIONS ISRAÉLIENNES ===
def create_features_israel(df_period):
    """Création des features pour le modèle israélien optimisé"""
    df = df_period.copy()
    
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
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - 25.0)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - 30.0)
    df['heating_needs'] = np.maximum(0, 25.0 - df['TempAvg'])
    
    df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
    df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
    df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
    
    # === SAISONS ===
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_mid_summer'] = (df['Day'].dt.month == 7).astype(int)
    
    # === JOURS DE LA SEMAINE (SYSTÈME ISRAÉLIEN) ===
    df['is_sunday'] = (df['Day'].dt.dayofweek == 6).astype(int)  # Dimanche = jour ouvrable en Israël
    df['is_monday'] = (df['Day'].dt.dayofweek == 0).astype(int)
    df['is_tuesday'] = (df['Day'].dt.dayofweek == 1).astype(int)
    df['is_wednesday'] = (df['Day'].dt.dayofweek == 2).astype(int)
    df['is_thursday'] = (df['Day'].dt.dayofweek == 3).astype(int)
    df['is_friday'] = (df['Day'].dt.dayofweek == 4).astype(int)    # Vendredi = week-end en Israël
    df['is_saturday'] = (df['Day'].dt.dayofweek == 5).astype(int) # Samedi = week-end en Israël
    
    # === WEEK-ENDS ISRAÉLIENS (VENDREDI-SAMEDI) ===
    df['is_weekend_israel'] = ((df['Day'].dt.dayofweek == 4) | (df['Day'].dt.dayofweek == 5)).astype(int)
    
    # === JOURS FÉRIÉS ISRAÉLIENS ===
    df['is_holiday'] = 0  # Simplifié pour cet exemple
    
    # === FEATURES CYCLIQUES ===
    df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
    
    # === INTERACTIONS TEMPÉRATURE-WEEK-END ISRAÉLIEN ===
    df['temp_x_weekend_israel'] = df['TempAvg'] * df['is_weekend_israel']
    df['temp_x_friday'] = df['TempAvg'] * df['is_friday']
    df['temp_x_saturday'] = df['TempAvg'] * df['is_saturday']
    df['temp_x_sunday'] = df['TempAvg'] * df['is_sunday']
    
    # === AUTRES INTERACTIONS ===
    df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
    df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
    df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
    df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
    df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
    
    # === TEMPOREL ===
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    
    # === LAGS ===
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # === FEATURES FIN D'ANNÉE ===
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

# === 5. EXTRACTION ET TRAITEMENT ===
print(f"\n🔧 Traitement des périodes...")

# Période 1
mask1 = (df['Day'] >= start1) & (df['Day'] <= end1)
period1_data = df[mask1].copy()

# Période 2
mask2 = (df['Day'] >= start2) & (df['Day'] <= end2)
period2_data = df[mask2].copy()

if len(period1_data) == 0:
    print(f"❌ Aucune donnée pour période 1: {start1.date()} - {end1.date()}")
    exit(1)

if len(period2_data) == 0:
    print(f"❌ Aucune donnée pour période 2: {start2.date()} - {end2.date()}")
    exit(1)

print(f"✅ Période 1: {len(period1_data)} jours")
print(f"✅ Période 2: {len(period2_data)} jours")

# Créer les features israéliennes
period1_features = create_features_israel(period1_data)
period2_features = create_features_israel(period2_data)

# Nettoyer
period1_clean = period1_features.dropna()
period2_clean = period2_features.dropna()

print(f"📊 Après nettoyage: {len(period1_clean)} et {len(period2_clean)} jours")

# === 6. PRÉDICTIONS ===
print(f"\n🤖 Génération des prédictions...")

try:
    X1 = period1_clean[features]
    X2 = period2_clean[features]
    
    X1_scaled = scaler.transform(X1)
    X2_scaled = scaler.transform(X2)
    
    pred1 = model.predict(X1_scaled)
    pred2 = model.predict(X2_scaled)
    
    period1_clean = period1_clean.copy()
    period2_clean = period2_clean.copy()
    period1_clean['predictions'] = pred1
    period2_clean['predictions'] = pred2
    
    print(f"✅ Prédictions réussies!")
    
except Exception as e:
    print(f"❌ Erreur prédiction: {e}")
    exit(1)

# === 7. ANALYSE COMPARATIVE ===
print(f"\n📊 Analyse comparative...")

stats1 = {
    'nb_jours': len(period1_clean),
    'total': period1_clean['predictions'].sum(),
    'moyenne': period1_clean['predictions'].mean(),
    'max': period1_clean['predictions'].max(),
    'min': period1_clean['predictions'].min(),
    'temp': period1_clean['TempAvg'].mean(),
    'weekends_israel': period1_clean['is_weekend_israel'].sum(),
    'vendredi': period1_clean['is_friday'].sum(),
    'samedi': period1_clean['is_saturday'].sum(),
    'dimanche': period1_clean['is_sunday'].sum(),
    'feries': period1_clean['is_holiday'].sum()
}

stats2 = {
    'nb_jours': len(period2_clean),
    'total': period2_clean['predictions'].sum(),
    'moyenne': period2_clean['predictions'].mean(),
    'max': period2_clean['predictions'].max(),
    'min': period2_clean['predictions'].min(),
    'temp': period2_clean['TempAvg'].mean(),
    'weekends_israel': period2_clean['is_weekend_israel'].sum(),
    'vendredi': period2_clean['is_friday'].sum(),
    'samedi': period2_clean['is_saturday'].sum(),
    'dimanche': period2_clean['is_sunday'].sum(),
    'feries': period2_clean['is_holiday'].sum()
}

diff_total = stats1['total'] - stats2['total']
diff_pct = (diff_total / stats2['total']) * 100
diff_moyenne = stats1['moyenne'] - stats2['moyenne']

print(f"\n📈 RÉSULTATS COMPARATIFS")
print("=" * 70)
print(f"{'Métrique':<25} | {label1:<15} | {label2:<15} | {'Différence':<12}")
print("-" * 70)
print(f"{'Jours analysés':<25} | {stats1['nb_jours']:>13} | {stats2['nb_jours']:>13} | {stats1['nb_jours']-stats2['nb_jours']:>10}")
print(f"{'Total (kWh)':<25} | {stats1['total']:>11.0f} | {stats2['total']:>11.0f} | {diff_total:>10.0f}")
print(f"{'Moyenne/jour (kWh)':<25} | {stats1['moyenne']:>11.0f} | {stats2['moyenne']:>11.0f} | {diff_moyenne:>10.0f}")
print(f"{'Max (kWh)':<25} | {stats1['max']:>11.0f} | {stats2['max']:>11.0f} | {stats1['max']-stats2['max']:>10.0f}")
print(f"{'Min (kWh)':<25} | {stats1['min']:>11.0f} | {stats2['min']:>11.0f} | {stats1['min']-stats2['min']:>10.0f}")
print(f"{'Temp moy (°C)':<25} | {stats1['temp']:>11.1f} | {stats2['temp']:>11.1f} | {stats1['temp']-stats2['temp']:>10.1f}")
print(f"{'Week-ends (Ven-Sam)':<25} | {stats1['weekends_israel']:>13} | {stats2['weekends_israel']:>13} | {stats1['weekends_israel']-stats2['weekends_israel']:>10}")
print(f"{'Vendredis':<25} | {stats1['vendredi']:>13} | {stats2['vendredi']:>13} | {stats1['vendredi']-stats2['vendredi']:>10}")
print(f"{'Samedis':<25} | {stats1['samedi']:>13} | {stats2['samedi']:>13} | {stats1['samedi']-stats2['samedi']:>10}")
print(f"{'Dimanches (ouvrable)':<25} | {stats1['dimanche']:>13} | {stats2['dimanche']:>13} | {stats1['dimanche']-stats2['dimanche']:>10}")
print(f"{'Jours fériés':<25} | {stats1['feries']:>13} | {stats2['feries']:>13} | {stats1['feries']-stats2['feries']:>10}")

print(f"\n🎯 SYNTHÈSE:")
if diff_pct > 0:
    print(f"   📈 {label1} consomme {diff_pct:.1f}% DE PLUS que {label2}")
    print(f"   📊 Soit {diff_total:.0f} kWh supplémentaires ({diff_moyenne:.0f} kWh/jour)")
else:
    print(f"   📉 {label1} consomme {abs(diff_pct):.1f}% DE MOINS que {label2}")
    print(f"   📊 Soit {abs(diff_total):.0f} kWh économisés ({abs(diff_moyenne):.0f} kWh/jour)")

# Impact financier
prix_kwh = 0.15
cout1 = stats1['total'] * prix_kwh
cout2 = stats2['total'] * prix_kwh
economie = abs(cout1 - cout2)

print(f"\n💰 IMPACT FINANCIER (prix: {prix_kwh}€/kWh):")
print(f"   • {label1}: {cout1:,.0f}€")
print(f"   • {label2}: {cout2:,.0f}€")
print(f"   • Différence: {economie:,.0f}€")

# Facteurs explicatifs
temp_diff = stats1['temp'] - stats2['temp']
if abs(temp_diff) > 5:
    if temp_diff > 0:
        print(f"\n🌡️  FACTEUR TEMPÉRATURE: +{temp_diff:.1f}°C explique la surconsommation")
        if stats1['temp'] > 25:
            print(f"   🔥 Climatisation intensive (pic: {stats1['max']:.0f} kWh)")
        else:
            print(f"   ❄️  Chauffage nécessaire")
    else:
        print(f"\n🌡️  FACTEUR TEMPÉRATURE: {temp_diff:.1f}°C explique l'économie")

# Patterns de consommation
print(f"\n📊 PATTERNS:")
print(f"   • Variabilité {label1}: {period1_clean['predictions'].std():.0f} kWh/jour")
print(f"   • Variabilité {label2}: {period2_clean['predictions'].std():.0f} kWh/jour")

if stats1['weekends_israel'] != stats2['weekends_israel']:
    weekend_diff = stats1['weekends_israel'] - stats2['weekends_israel']
    print(f"   • Impact week-ends israéliens (Ven-Sam): {weekend_diff:+d} jours")

# === 8. EXPORT SIMPLE ===
export_simple = input("\n💾 Sauvegarder les résultats ? (o/n): ")

if export_simple.lower() in ['o', 'oui', 'y', 'yes']:
    # CSV simple
    results_df = pd.DataFrame({
        'Metrique': ['Total_kWh', 'Moyenne_kWh', 'Max_kWh', 'Min_kWh', 'Temp_C', 'Cout_EUR'],
        label1: [stats1['total'], stats1['moyenne'], stats1['max'], stats1['min'], stats1['temp'], cout1],
        label2: [stats2['total'], stats2['moyenne'], stats2['max'], stats2['min'], stats2['temp'], cout2],
        'Difference': [diff_total, diff_moyenne, stats1['max']-stats2['max'], 
                      stats1['min']-stats2['min'], temp_diff, cout1-cout2]
    })
    
    filename = f"comparaison_simple_{start1.strftime('%Y%m%d')}_{start2.strftime('%Y%m%d')}.csv"
    results_df.to_csv(filename, index=False)
    print(f"✅ Résultats sauvés: {filename}")

print("\n" + "="*70)
print("🇮🇱 COMPARAISON TERMINÉE - MODÈLE ISRAÉLIEN !")
print("="*70)
print(f"🎯 Différence principale: {diff_pct:+.1f}% ({diff_total:+.0f} kWh)")
print(f"💰 Impact financier: {economie:,.0f}€")
print(f"🤖 Modèle israélien: MAE {performance['test_mae']:.0f} kWh (R² {performance['test_r2']:.3f})")
print(f"📅 Week-ends: Vendredi-Samedi | Dimanches: Jours ouvrables")
print("="*70) 