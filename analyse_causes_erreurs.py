import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔍 ANALYSE DES CAUSES D'ERREURS")
print("=" * 40)

# Charger données
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df['day_of_week'] = df['Day'].dt.dayofweek  
df['month'] = df['Day'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print(f"✅ {len(df)} observations analysées")

# === 1. ANALYSE DES CONSOMMATIONS TRÈS FAIBLES (PROBLÈME MAJEUR) ===
print(f"\n🚨 ANALYSE DES CONSOMMATIONS TRÈS FAIBLES:")
print("-" * 45)

very_low = df[df['DailyAverage'] < 50000].copy()
print(f"📊 {len(very_low)} observations avec consommation < 50,000 kWh")

if len(very_low) > 0:
    print(f"   💡 Plage: {very_low['DailyAverage'].min():,.0f} - {very_low['DailyAverage'].max():,.0f} kWh")
    print(f"   📈 Moyenne: {very_low['DailyAverage'].mean():,.0f} kWh")
    
    # Analyser les dates
    print(f"\n📅 RÉPARTITION TEMPORELLE:")
    print(f"   Années: {very_low['Day'].dt.year.value_counts().sort_index().to_dict()}")
    print(f"   Mois: {very_low['Day'].dt.month.value_counts().sort_index().to_dict()}")
    
    # Jours spéciaux?
    if 'jour_ferie' in df.columns:
        holidays_low = very_low['jour_ferie'].sum()
        print(f"   🎉 Jours fériés: {holidays_low}/{len(very_low)} ({holidays_low/len(very_low)*100:.1f}%)")
    
    # Météo extrême?
    print(f"\n🌡️ CONDITIONS MÉTÉO:")
    if 'temperature_C' in df.columns:
        temp_low = very_low['temperature_C'].describe()
        temp_normal = df[df['DailyAverage'] >= 50000]['temperature_C'].describe()
        print(f"   Température très faible: {temp_low['mean']:.1f}°C (±{temp_low['std']:.1f})")
        print(f"   Température normale:     {temp_normal['mean']:.1f}°C (±{temp_normal['std']:.1f})")

# === 2. ANALYSE DES MARDIS (PIRE JOUR) ===
print(f"\n🗓️ ANALYSE DES MARDIS (PIRE JOUR):")
print("-" * 35)

mardis = df[df['day_of_week'] == 1].copy()  # Mardi = 1
autres_jours = df[df['day_of_week'] != 1].copy()

print(f"📊 {len(mardis)} mardis analysés")
print(f"   Consommation moyenne mardis: {mardis['DailyAverage'].mean():,.0f} kWh")
print(f"   Consommation moyenne autres: {autres_jours['DailyAverage'].mean():,.0f} kWh")
print(f"   Différence: {(mardis['DailyAverage'].mean() - autres_jours['DailyAverage'].mean()):+,.0f} kWh")

# Variabilité
print(f"   Écart-type mardis: {mardis['DailyAverage'].std():,.0f} kWh")
print(f"   Écart-type autres: {autres_jours['DailyAverage'].std():,.0f} kWh")

# === 3. ANALYSE DE FÉVRIER (PIRE MOIS) ===
print(f"\n🌨️ ANALYSE DE FÉVRIER (PIRE MOIS):")
print("-" * 35)

fevrier = df[df['month'] == 2].copy()
autres_mois = df[df['month'] != 2].copy()

print(f"📊 {len(fevrier)} jours de février analysés")
print(f"   Consommation moyenne février: {fevrier['DailyAverage'].mean():,.0f} kWh")
print(f"   Consommation moyenne autres:  {autres_mois['DailyAverage'].mean():,.0f} kWh")
print(f"   Différence: {(fevrier['DailyAverage'].mean() - autres_mois['DailyAverage'].mean()):+,.0f} kWh")

if 'temperature_C' in df.columns:
    print(f"   Température février: {fevrier['temperature_C'].mean():.1f}°C")
    print(f"   Température autres:  {autres_mois['temperature_C'].mean():.1f}°C")

# === 4. PATTERNS ANOMALIES ===
print(f"\n🔍 DÉTECTION D'ANOMALIES:")
print("-" * 28)

# Calcul des quartiles pour détecter anomalies
Q1 = df['DailyAverage'].quantile(0.25)
Q3 = df['DailyAverage'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies_basses = df[df['DailyAverage'] < lower_bound]
anomalies_hautes = df[df['DailyAverage'] > upper_bound]

print(f"   📉 Anomalies basses: {len(anomalies_basses)} (< {lower_bound:,.0f} kWh)")
print(f"   📈 Anomalies hautes: {len(anomalies_hautes)} (> {upper_bound:,.0f} kWh)")

# Analyse anomalies basses
if len(anomalies_basses) > 0:
    print(f"\n   🔍 Anomalies basses par jour semaine:")
    jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    for day_idx in range(7):
        count = len(anomalies_basses[anomalies_basses['day_of_week'] == day_idx])
        print(f"      {jours[day_idx]}: {count:2d} anomalies")

# === 5. GRAPHIQUE DÉTAILLÉ ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔍 ANALYSE DÉTAILLÉE DES CAUSES D\'ERREURS', fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Distribution par niveaux de consommation
ax1 = axes[0, 0]
bins = [0, 25000, 50000, 70000, 90000, 120000]
labels = ['Très\nfaible', 'Faible', 'Normale', 'Élevée', 'Très\nélevée']
counts, _, _ = ax1.hist(df['DailyAverage'], bins=bins, alpha=0.7, 
                       color=['red', 'orange', 'green', 'blue', 'purple'])

ax1.set_title('⚡ Distribution par Niveaux de Consommation', fontweight='bold')
ax1.set_xlabel('Consommation (kWh)')
ax1.set_ylabel('Nombre d\'observations')
ax1.grid(True, alpha=0.3, axis='y')

# Ajouter nombres sur barres
for i, count in enumerate(counts):
    if count > 0:
        ax1.text(bins[i] + (bins[i+1]-bins[i])/2, count + 5, 
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')

# GRAPHIQUE 2: Mardis vs autres jours
ax2 = axes[0, 1]
data_comparison = [mardis['DailyAverage'], autres_jours['DailyAverage']]
box_plot = ax2.boxplot(data_comparison, labels=['Mardis', 'Autres jours'], 
                      patch_artist=True, showmeans=True)
box_plot['boxes'][0].set_facecolor('lightcoral')
box_plot['boxes'][1].set_facecolor('lightblue')
ax2.set_title('🗓️ Mardis vs Autres Jours', fontweight='bold')
ax2.set_ylabel('Consommation (kWh)')
ax2.grid(True, alpha=0.3, axis='y')

# GRAPHIQUE 3: Évolution temporelle des anomalies
ax3 = axes[1, 0]
df_sorted = df.sort_values('Day')
ax3.plot(df_sorted['Day'], df_sorted['DailyAverage'], alpha=0.6, color='gray', linewidth=0.5)
ax3.scatter(anomalies_basses['Day'], anomalies_basses['DailyAverage'], 
           color='red', s=30, alpha=0.8, label=f'Anomalies basses ({len(anomalies_basses)})')
ax3.scatter(anomalies_hautes['Day'], anomalies_hautes['DailyAverage'], 
           color='orange', s=30, alpha=0.8, label=f'Anomalies hautes ({len(anomalies_hautes)})')
ax3.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7)
ax3.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7)
ax3.set_title('📊 Évolution Temporelle et Anomalies', fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Consommation (kWh)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# GRAPHIQUE 4: Heatmap erreurs par mois/jour
ax4 = axes[1, 1]
pivot_data = df.groupby(['month', 'day_of_week'])['DailyAverage'].std().unstack()
im = ax4.imshow(pivot_data.values, cmap='Reds', aspect='auto')
ax4.set_title('🔥 Variabilité par Mois/Jour (Écart-type)', fontweight='bold')
ax4.set_xlabel('Jour de semaine')
ax4.set_ylabel('Mois')
ax4.set_xticks(range(7))
ax4.set_xticklabels(['L', 'M', 'M', 'J', 'V', 'S', 'D'])
ax4.set_yticks(range(12))
ax4.set_yticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                    'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
plt.colorbar(im, ax=ax4, label='Écart-type (kWh)')

plt.tight_layout()
plt.savefig('analyse_causes_erreurs.png', dpi=300, bbox_inches='tight')
plt.show()

# === 6. SYNTHÈSE FINALE ===
print(f"\n�� SYNTHÈSE DES PRINCIPALES CAUSES D'ERREURS:")
print("=" * 55)

print(f"🔴 PROBLÈME MAJEUR: Consommations très faibles")
print(f"   • {len(very_low)} observations < 50k kWh créent 103% d'erreur")
print(f"   • Probablement arrêts maintenance ou pannes")
print(f"   • Recommandation: Détection/filtrage anomalies")

print(f"\n🟡 PROBLÈMES MODÉRÉS:")
print(f"   • Mardis: +{(mardis['DailyAverage'].std() - autres_jours['DailyAverage'].std()):+.0f} kWh variabilité")
print(f"   • Février: Conditions hivernales variables")
print(f"   • Anomalies hautes: {len(anomalies_hautes)} pics inexpliqués")

print(f"\n💡 RECOMMANDATIONS PRIORITAIRES:")
print(f"   1️⃣  Système détection anomalies automatique")
print(f"   2️⃣  Modèle séparé pour consommations < 50k kWh")
print(f"   3️⃣  Features spécifiques mardis/février")
print(f"   4️⃣  Intégration planning maintenance usine")

print(f"\n💾 Graphique sauvé: analyse_causes_erreurs.png")
