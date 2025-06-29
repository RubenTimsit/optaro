import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("📊 GRAPHIQUE SIMPLE 2023 : CONSOMMATION RÉELLE")
print("=" * 50)

# === 1. CHARGER DONNÉES ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])

# === 2. FILTRER 2023 ===
data_2023 = df[(df['Day'].dt.year == 2023)].copy()
print(f"✅ Données 2023: {len(data_2023)} observations")

if len(data_2023) == 0:
    print("❌ Aucune donnée 2023 trouvée")
    # Vérifier les années disponibles
    years_available = sorted(df['Day'].dt.year.unique())
    print(f"📅 Années disponibles: {years_available}")
    
    # Prendre la première année complète disponible
    if len(years_available) > 1:
        year_to_use = years_available[1]  # Deuxième année (première souvent incomplète)
        print(f"🔄 Utilisation de l'année {year_to_use} à la place")
        data_year = df[(df['Day'].dt.year == year_to_use)].copy()
    else:
        print("❌ Pas assez de données pour l'analyse")
        exit()
else:
    data_year = data_2023
    year_to_use = 2023

# === 3. CRÉER UNE PRÉDICTION SIMPLE BASÉE SUR MOYENNES ===
print("🔧 Création de prédictions basées sur patterns historiques...")

# Calculer moyennes par jour de semaine et mois
daily_patterns = data_year.groupby(data_year['Day'].dt.dayofweek)['DailyAverage'].mean()
monthly_patterns = data_year.groupby(data_year['Day'].dt.month)['DailyAverage'].mean()

# Créer prédictions simples
predictions_simple = []
for _, row in data_year.iterrows():
    day_of_week = row['Day'].dayofweek
    month = row['Day'].month
    
    # Moyenne pondérée : 70% pattern jour + 30% pattern mois
    daily_avg = daily_patterns[day_of_week]
    monthly_avg = monthly_patterns[month]
    
    prediction = 0.7 * daily_avg + 0.3 * monthly_avg
    predictions_simple.append(prediction)

data_year['predictions_simple'] = predictions_simple

# === 4. GRAPHIQUES ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'📊 ANNÉE {year_to_use}: CONSOMMATION ÉNERGÉTIQUE - ANALYSE COMPLÈTE', 
             fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Série temporelle
ax1 = axes[0, 0]
ax1.plot(data_year['Day'], data_year['DailyAverage'], 
         label='Consommation réelle', color='blue', linewidth=2)
ax1.plot(data_year['Day'], data_year['predictions_simple'], 
         label='Prédiction pattern', color='red', linewidth=2, alpha=0.7)

ax1.set_title('📈 Évolution Temporelle', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Consommation (kWh)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# GRAPHIQUE 2: Moyennes par mois
ax2 = axes[0, 1]
monthly_real = data_year.groupby(data_year['Day'].dt.month)['DailyAverage'].mean()
monthly_pred = data_year.groupby(data_year['Day'].dt.month)['predictions_simple'].mean()

months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

x = np.arange(1, 13)
width = 0.35

bars1 = ax2.bar(x - width/2, monthly_real, width, label='Réel', alpha=0.8, color='blue')
bars2 = ax2.bar(x + width/2, monthly_pred, width, label='Prédiction', alpha=0.8, color='red')

ax2.set_title('📊 Moyennes Mensuelles', fontsize=14, fontweight='bold')
ax2.set_xlabel('Mois')
ax2.set_ylabel('Consommation moyenne (kWh)')
ax2.set_xticks(x)
ax2.set_xticklabels(months)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# GRAPHIQUE 3: Patterns par jour de semaine
ax3 = axes[1, 0]
daily_real = data_year.groupby(data_year['Day'].dt.dayofweek)['DailyAverage'].mean()
daily_pred = data_year.groupby(data_year['Day'].dt.dayofweek)['predictions_simple'].mean()

day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
x_days = np.arange(7)

bars3 = ax3.bar(x_days - width/2, daily_real, width, label='Réel', alpha=0.8, color='blue')
bars4 = ax3.bar(x_days + width/2, daily_pred, width, label='Prédiction', alpha=0.8, color='red')

ax3.set_title('📅 Patterns Hebdomadaires', fontsize=14, fontweight='bold')
ax3.set_xlabel('Jour de la semaine')
ax3.set_ylabel('Consommation moyenne (kWh)')
ax3.set_xticks(x_days)
ax3.set_xticklabels(day_names)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# GRAPHIQUE 4: Distribution et boxplot
ax4 = axes[1, 1]
ax4.boxplot([data_year['DailyAverage'], data_year['predictions_simple']], 
           labels=['Réel', 'Prédiction'])
ax4.set_title('📏 Distribution des Consommations', fontsize=14, fontweight='bold')
ax4.set_ylabel('Consommation (kWh)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'analysis_{year_to_use}_consumption.png', dpi=300, bbox_inches='tight')
plt.show()

# === 5. STATISTIQUES ===
print(f"\n📊 STATISTIQUES {year_to_use}")
print("=" * 40)

print(f"📈 CONSOMMATION RÉELLE:")
print(f"   Moyenne    : {data_year['DailyAverage'].mean():6.0f} kWh/jour")
print(f"   Médiane    : {data_year['DailyAverage'].median():6.0f} kWh/jour")
print(f"   Min/Max    : {data_year['DailyAverage'].min():6.0f} / {data_year['DailyAverage'].max():6.0f} kWh")
print(f"   Écart-type : {data_year['DailyAverage'].std():6.0f} kWh")

print(f"\n🌍 ANALYSE SAISONNIÈRE:")
seasons = {
    'Hiver (Jan-Mar)': [1, 2, 3],
    'Printemps (Avr-Jun)': [4, 5, 6], 
    'Été (Jul-Sep)': [7, 8, 9],
    'Automne (Oct-Déc)': [10, 11, 12]
}

for season_name, months_list in seasons.items():
    season_data = data_year[data_year['Day'].dt.month.isin(months_list)]
    if len(season_data) > 0:
        print(f"   {season_name:<20}: {season_data['DailyAverage'].mean():6.0f} kWh/jour")

print(f"\n📅 PATTERNS HEBDOMADAIRES:")
for i, day_name in enumerate(day_names):
    day_avg = daily_real[i]
    emoji = "🏢" if i < 5 else "��"
    print(f"   {emoji} {day_name}     : {day_avg:6.0f} kWh/jour")

print(f"\n💾 Graphique sauvé: analysis_{year_to_use}_consumption.png")
