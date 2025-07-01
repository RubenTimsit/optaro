import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

print("🇮🇱 COMPARATEUR DE PÉRIODES - MODÈLE ISRAÉLIEN OPTIMISÉ")
print("=" * 70)

# === 1. CHARGEMENT DU MODÈLE ISRAÉLIEN ===
print("\n🤖 1. Chargement du modèle israélien optimisé...")

try:
    with open('modele_optimise_israel.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    performance = model_data['performance']
    
    print(f"✅ Modèle israélien chargé avec succès !")
    print(f"   • Performance: MAE {performance['test_mae']:.0f} kWh (R² {performance['test_r2']:.3f})")
    print(f"   • Features: {len(features)} variables")
    print(f"   🇮🇱 Week-ends: Vendredi-Samedi | Jours ouvrables: Dimanche-Jeudi")
    
except FileNotFoundError:
    print("❌ Erreur: Modèle israélien non trouvé. Veuillez d'abord exécuter modele_optimise_israel.py")
    exit(1)

# === 2. FONCTIONS UTILITAIRES ISRAÉLIENNES ===
def create_features_israel(df):
    """Créé des features pour le modèle israélien (identique au modèle d'entraînement)"""
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

    # === INTERACTIONS TEMPÉRATURE-WEEK-END ISRAÉLIEN ===
    df['temp_x_weekend_israel'] = df['TempAvg'] * df['is_weekend_israel']
    df['temp_x_friday'] = df['TempAvg'] * df['is_friday']
    df['temp_x_saturday'] = df['TempAvg'] * df['is_saturday']
    df['temp_x_sunday'] = df['TempAvg'] * df['is_sunday']
    
    # === TEMPOREL ===
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    
    # === LAGS CRITIQUES ===
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # === FEATURES FIN D'ANNÉE ===
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

def parse_date(date_str):
    """Parse une date en format flexible"""
    try:
        # Essayer différents formats
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError("Format de date non reconnu")
    except:
        raise ValueError("Format de date invalide")

def get_period_data(start_date, end_date, df_source):
    """Extrait les données pour une période donnée"""
    mask = (df_source['Day'] >= start_date) & (df_source['Day'] <= end_date)
    period_data = df_source[mask].copy()
    
    if len(period_data) == 0:
        raise ValueError(f"Aucune donnée trouvée pour la période {start_date.date()} - {end_date.date()}")
    
    return period_data

# === 3. CHARGEMENT DES DONNÉES HISTORIQUES ===
print("\n📊 2. Chargement des données historiques...")

df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

print(f"✅ Données chargées: {len(df)} jours")
print(f"   Période disponible: {df['Day'].min().date()} → {df['Day'].max().date()}")

# === 4. INTERFACE UTILISATEUR ===
def get_user_input():
    """Interface pour saisir les périodes"""
    print("\n📅 3. Sélection des périodes à comparer")
    print("   Format accepté: YYYY-MM-DD, DD/MM/YYYY ou DD-MM-YYYY")
    print("   Ex: 2024-07-01, 01/07/2024, 01-07-2024")
    
    while True:
        try:
            print(f"\n🎯 PÉRIODE 1 À ESTIMER:")
            start1_str = input("   Date de début: ")
            end1_str = input("   Date de fin: ")
            start1 = parse_date(start1_str)
            end1 = parse_date(end1_str)
            
            if start1 >= end1:
                print("❌ La date de début doit être antérieure à la date de fin")
                continue
            
            print(f"\n🎯 PÉRIODE 2 À COMPARER:")
            start2_str = input("   Date de début: ")
            end2_str = input("   Date de fin: ")
            start2 = parse_date(start2_str)
            end2 = parse_date(end2_str)
            
            if start2 >= end2:
                print("❌ La date de début doit être antérieure à la date de fin")
                continue
            
            return start1, end1, start2, end2
            
        except ValueError as e:
            print(f"❌ Erreur: {e}")
            print("   Veuillez réessayer avec un format valide")

# === 5. SAISIE DES PÉRIODES ===
start1, end1, start2, end2 = get_user_input()

print(f"\n✅ Périodes sélectionnées:")
print(f"   Période 1: {start1.date()} → {end1.date()} ({(end1-start1).days + 1} jours)")
print(f"   Période 2: {start2.date()} → {end2.date()} ({(end2-start2).days + 1} jours)")

# === 6. EXTRACTION ET PRÉPARATION DES DONNÉES ===
print("\n🔧 4. Préparation des données pour prédiction...")

try:
    # Extraire les données pour chaque période
    period1_data = get_period_data(start1, end1, df)
    period2_data = get_period_data(start2, end2, df)
    
    print(f"✅ Période 1: {len(period1_data)} jours de données")
    print(f"✅ Période 2: {len(period2_data)} jours de données")
    
    # Créer les features israéliennes
    period1_features = create_features_israel(period1_data)
    period2_features = create_features_israel(period2_data)
    
    # Supprimer les NaN des lags (garder seulement les données avec lags valides)
    period1_clean = period1_features.dropna()
    period2_clean = period2_features.dropna()
    
    print(f"📊 Après nettoyage des lags:")
    print(f"   Période 1: {len(period1_clean)} jours (perdus: {len(period1_features) - len(period1_clean)})")
    print(f"   Période 2: {len(period2_clean)} jours (perdus: {len(period2_features) - len(period2_clean)})")
    
except ValueError as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# === 7. PRÉDICTIONS ===
print("\n🤖 5. Génération des prédictions...")

# Préparer les features pour le modèle
X1 = period1_clean[features]
X2 = period2_clean[features]

# Normaliser avec le scaler du modèle
X1_scaled = scaler.transform(X1)
X2_scaled = scaler.transform(X2)

# Prédictions
pred1 = model.predict(X1_scaled)
pred2 = model.predict(X2_scaled)

# Ajouter les prédictions aux DataFrames
period1_clean = period1_clean.copy()
period2_clean = period2_clean.copy()
period1_clean['predictions'] = pred1
period2_clean['predictions'] = pred2

print(f"✅ Prédictions générées!")

# === 8. CALCUL DES STATISTIQUES COMPARATIVES ===
print("\n📊 6. Calcul des statistiques comparatives...")

stats1 = {
    'periode': f"{start1.date()} → {end1.date()}",
    'nb_jours': len(period1_clean),
    'consommation_totale': period1_clean['predictions'].sum(),
    'consommation_moyenne': period1_clean['predictions'].mean(),
    'consommation_max': period1_clean['predictions'].max(),
    'consommation_min': period1_clean['predictions'].min(),
    'temp_moyenne': period1_clean['TempAvg'].mean(),
    'nb_weekends_israel': period1_clean['is_weekend_israel'].sum(),
    'nb_vendredis': period1_clean['is_friday'].sum(),
    'nb_samedis': period1_clean['is_saturday'].sum(),
    'nb_dimanches': period1_clean['is_sunday'].sum(),
    'nb_feries': period1_clean['is_holiday'].sum()
}

stats2 = {
    'periode': f"{start2.date()} → {end2.date()}",
    'nb_jours': len(period2_clean),
    'consommation_totale': period2_clean['predictions'].sum(),
    'consommation_moyenne': period2_clean['predictions'].mean(),
    'consommation_max': period2_clean['predictions'].max(),
    'consommation_min': period2_clean['predictions'].min(),
    'temp_moyenne': period2_clean['TempAvg'].mean(),
    'nb_weekends_israel': period2_clean['is_weekend_israel'].sum(),
    'nb_vendredis': period2_clean['is_friday'].sum(),
    'nb_samedis': period2_clean['is_saturday'].sum(),
    'nb_dimanches': period2_clean['is_sunday'].sum(),
    'nb_feries': period2_clean['is_holiday'].sum()
}

# Calculs comparatifs
diff_totale = stats1['consommation_totale'] - stats2['consommation_totale']
diff_moyenne = stats1['consommation_moyenne'] - stats2['consommation_moyenne']
diff_pct = (diff_totale / stats2['consommation_totale']) * 100

print(f"📈 STATISTIQUES COMPARATIVES:")
print("=" * 70)
print(f"{'Métrique':<25} | {'Période 1':<15} | {'Période 2':<15} | {'Différence':<12}")
print("-" * 70)
print(f"{'Jours':<25} | {stats1['nb_jours']:>13} | {stats2['nb_jours']:>13} | {stats1['nb_jours']-stats2['nb_jours']:>10}")
print(f"{'Total (kWh)':<25} | {stats1['consommation_totale']:>11.0f} | {stats2['consommation_totale']:>11.0f} | {diff_totale:>10.0f}")
print(f"{'Moyenne/jour (kWh)':<25} | {stats1['consommation_moyenne']:>11.0f} | {stats2['consommation_moyenne']:>11.0f} | {diff_moyenne:>10.0f}")
print(f"{'Max (kWh)':<25} | {stats1['consommation_max']:>11.0f} | {stats2['consommation_max']:>11.0f} | {stats1['consommation_max']-stats2['consommation_max']:>10.0f}")
print(f"{'Min (kWh)':<25} | {stats1['consommation_min']:>11.0f} | {stats2['consommation_min']:>11.0f} | {stats1['consommation_min']-stats2['consommation_min']:>10.0f}")
print(f"{'Temp moy (°C)':<25} | {stats1['temp_moyenne']:>11.1f} | {stats2['temp_moyenne']:>11.1f} | {stats1['temp_moyenne']-stats2['temp_moyenne']:>10.1f}")
print(f"{'Week-ends (Ven-Sam)':<25} | {stats1['nb_weekends_israel']:>13} | {stats2['nb_weekends_israel']:>13} | {stats1['nb_weekends_israel']-stats2['nb_weekends_israel']:>10}")
print(f"{'Vendredis':<25} | {stats1['nb_vendredis']:>13} | {stats2['nb_vendredis']:>13} | {stats1['nb_vendredis']-stats2['nb_vendredis']:>10}")
print(f"{'Samedis':<25} | {stats1['nb_samedis']:>13} | {stats2['nb_samedis']:>13} | {stats1['nb_samedis']-stats2['nb_samedis']:>10}")
print(f"{'Dimanches (ouvrable)':<25} | {stats1['nb_dimanches']:>13} | {stats2['nb_dimanches']:>13} | {stats1['nb_dimanches']-stats2['nb_dimanches']:>10}")
print(f"{'Jours fériés':<25} | {stats1['nb_feries']:>13} | {stats2['nb_feries']:>13} | {stats1['nb_feries']-stats2['nb_feries']:>10}")

print(f"\n🎯 RÉSUMÉ:")
if diff_pct > 0:
    print(f"   📈 Période 1 consomme {diff_pct:.1f}% DE PLUS que Période 2")
    print(f"   📊 Soit {diff_totale:.0f} kWh supplémentaires ({diff_moyenne:.0f} kWh/jour)")
else:
    print(f"   📉 Période 1 consomme {abs(diff_pct):.1f}% DE MOINS que Période 2")
    print(f"   📊 Soit {abs(diff_totale):.0f} kWh économisés ({abs(diff_moyenne):.0f} kWh/jour)")

# === 9. VISUALISATIONS COMPARATIVES ===
print("\n📊 7. Génération des graphiques comparatifs...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'🇮🇱 COMPARAISON DE PÉRIODES - MODÈLE ISRAÉLIEN\n'
             f'Période 1: {start1.date()} → {end1.date()} | '
             f'Période 2: {start2.date()} → {end2.date()} | Week-ends: Ven-Sam', 
             fontsize=14, fontweight='bold')

# Plot 1: Évolution temporelle des consommations
axes[0,0].plot(period1_clean['Day'], period1_clean['predictions'], 
               color='blue', linewidth=2, label=f'Période 1 ({start1.date()})', alpha=0.8)
axes[0,0].plot(period2_clean['Day'], period2_clean['predictions'], 
               color='red', linewidth=2, label=f'Période 2 ({start2.date()})', alpha=0.8)
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Consommation Prédite (kWh)')
axes[0,0].set_title('Évolution Temporelle des Consommations')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# Plot 2: Distributions des consommations
axes[0,1].hist(period1_clean['predictions'], bins=20, alpha=0.7, color='blue', 
               label=f'Période 1\n(μ={stats1["consommation_moyenne"]:.0f} kWh)', density=True)
axes[0,1].hist(period2_clean['predictions'], bins=20, alpha=0.7, color='red', 
               label=f'Période 2\n(μ={stats2["consommation_moyenne"]:.0f} kWh)', density=True)
axes[0,1].set_xlabel('Consommation Prédite (kWh)')
axes[0,1].set_ylabel('Densité')
axes[0,1].set_title('Distribution des Consommations')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Comparaison par type de jour (système israélien)
def categorize_day_israel(row):
    if row['is_holiday']:
        return 'Jour férié'
    elif row['is_weekend_israel']:
        return 'Week-end (Ven-Sam)'
    else:
        return 'Jour ouvrable'

period1_clean['day_type'] = period1_clean.apply(categorize_day_israel, axis=1)
period2_clean['day_type'] = period2_clean.apply(categorize_day_israel, axis=1)

day_types = ['Jour ouvrable', 'Week-end (Ven-Sam)', 'Jour férié']
means1 = [period1_clean[period1_clean['day_type'] == dt]['predictions'].mean() 
          for dt in day_types]
means2 = [period2_clean[period2_clean['day_type'] == dt]['predictions'].mean() 
          for dt in day_types]

x = np.arange(len(day_types))
width = 0.35

axes[1,0].bar(x - width/2, means1, width, label='Période 1', color='blue', alpha=0.7)
axes[1,0].bar(x + width/2, means2, width, label='Période 2', color='red', alpha=0.7)
axes[1,0].set_xlabel('Type de Jour')
axes[1,0].set_ylabel('Consommation Moyenne (kWh)')
axes[1,0].set_title('Consommation par Type de Jour')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(day_types)
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Relation Température vs Consommation
axes[1,1].scatter(period1_clean['TempAvg'], period1_clean['predictions'], 
                  alpha=0.6, color='blue', s=30, label='Période 1')
axes[1,1].scatter(period2_clean['TempAvg'], period2_clean['predictions'], 
                  alpha=0.6, color='red', s=30, label='Période 2')

# Ajouter des lignes de tendance
z1 = np.polyfit(period1_clean['TempAvg'], period1_clean['predictions'], 1)
z2 = np.polyfit(period2_clean['TempAvg'], period2_clean['predictions'], 1)
p1 = np.poly1d(z1)
p2 = np.poly1d(z2)

temp_range1 = np.linspace(period1_clean['TempAvg'].min(), period1_clean['TempAvg'].max(), 100)
temp_range2 = np.linspace(period2_clean['TempAvg'].min(), period2_clean['TempAvg'].max(), 100)

axes[1,1].plot(temp_range1, p1(temp_range1), color='blue', linestyle='--', alpha=0.8)
axes[1,1].plot(temp_range2, p2(temp_range2), color='red', linestyle='--', alpha=0.8)

axes[1,1].set_xlabel('Température Moyenne (°C)')
axes[1,1].set_ylabel('Consommation Prédite (kWh)')
axes[1,1].set_title('Consommation vs Température')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()

# Sauvegarder le graphique
filename = f"comparaison_{start1.strftime('%Y%m%d')}_{end1.strftime('%Y%m%d')}_vs_{start2.strftime('%Y%m%d')}_{end2.strftime('%Y%m%d')}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Graphique sauvegardé: {filename}")

# === 10. EXPORT DES DONNÉES (OPTIONNEL) ===
save_data = input("\n💾 Voulez-vous sauvegarder les données détaillées ? (o/n): ")

if save_data.lower() in ['o', 'oui', 'y', 'yes']:
    # Préparer les données d'export
    export_p1 = period1_clean[['Day', 'DailyAverage', 'predictions', 'TempAvg', 'is_weekend', 'is_holiday']].copy()
    export_p1['periode'] = 'Période 1'
    
    export_p2 = period2_clean[['Day', 'DailyAverage', 'predictions', 'TempAvg', 'is_weekend', 'is_holiday']].copy()
    export_p2['periode'] = 'Période 2'
    
    export_combined = pd.concat([export_p1, export_p2], ignore_index=True)
    
    csv_filename = f"donnees_comparaison_{start1.strftime('%Y%m%d')}_{end1.strftime('%Y%m%d')}_vs_{start2.strftime('%Y%m%d')}_{end2.strftime('%Y%m%d')}.csv"
    export_combined.to_csv(csv_filename, index=False)
    
    print(f"✅ Données sauvegardées: {csv_filename}")

print("\n" + "="*70)
print("🇮🇱 COMPARAISON TERMINÉE - MODÈLE ISRAÉLIEN !")
print("="*70)
print(f"📊 Période 1: {stats1['consommation_totale']:.0f} kWh total")
print(f"📊 Période 2: {stats2['consommation_totale']:.0f} kWh total")
print(f"📈 Différence: {diff_pct:+.1f}% ({diff_totale:+.0f} kWh)")
print(f"🇮🇱 Week-ends: Vendredi-Samedi | Dimanches: Jours ouvrables")
print(f"📁 Graphique: {filename}")
print("="*70) 