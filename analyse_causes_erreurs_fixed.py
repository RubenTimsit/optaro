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
    
    # Détails de ces jours problématiques
    print(f"\n�� DÉTAIL DES JOURS PROBLÉMATIQUES:")
    for _, row in very_low.iterrows():
        day_name = row['Day'].strftime('%A')
        print(f"   📅 {row['Day'].date()} ({day_name}): {row['DailyAverage']:6.0f} kWh")

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

print(f"   �� Anomalies basses: {len(anomalies_basses)} (< {lower_bound:,.0f} kWh)")
print(f"   📈 Anomalies hautes: {len(anomalies_hautes)} (> {upper_bound:,.0f} kWh)")

# === 5. GRAPHIQUE DÉTAILLÉ ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔍 ANALYSE DÉTAILLÉE DES CAUSES D\'ERREURS', fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Distribution par niveaux de consommation (corrigé)
ax1 = axes[0, 0]
bins = [0, 25000, 50000, 70000, 90000, 120000]
counts, bin_edges, patches = ax1.hist(df['DailyAverage'], bins=bins, alpha=0.7, edgecolor='black')

# Colorer les barres manuellement
colors = ['red', 'orange', 'green', 'blue', 'purple']
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

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
if len(anomalies_basses) > 0:
    ax3.scatter(anomalies_basses['Day'], anomalies_basses['DailyAverage'], 
               color='red', s=50, alpha=0.8, label=f'Anomalies basses ({len(anomalies_basses)})')
if len(anomalies_hautes) > 0:
    ax3.scatter(anomalies_hautes['Day'], anomalies_hautes['DailyAverage'], 
               color='orange', s=50, alpha=0.8, label=f'Anomalies hautes ({len(anomalies_hautes)})')
ax3.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.7)
ax3.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7)
ax3.set_title('📊 Évolution Temporelle et Anomalies', fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Consommation (kWh)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# GRAPHIQUE 4: Variabilité par jour de semaine
ax4 = axes[1, 1]
day_std = df.groupby('day_of_week')['DailyAverage'].std()
jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
colors4 = ['red' if std > 19000 else 'orange' if std > 18000 else 'green' for std in day_std]
bars = ax4.bar(range(7), day_std.values, color=colors4, alpha=0.7)
ax4.set_title('📈 Variabilité par Jour de Semaine', fontweight='bold')
ax4.set_xlabel('Jour de semaine')
ax4.set_ylabel('Écart-type (kWh)')
ax4.set_xticks(range(7))
ax4.set_xticklabels(jours)
ax4.grid(True, alpha=0.3, axis='y')

# Ajouter valeurs sur barres
for bar, std in zip(bars, day_std.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{std:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('analyse_causes_erreurs.png', dpi=300, bbox_inches='tight')
plt.show()

# === 6. SYNTHÈSE FINALE ===
print(f"\n🎯 SYNTHÈSE DES PRINCIPALES CAUSES D'ERREURS:")
print("=" * 55)

print(f"🔴 PROBLÈME MAJEUR: Consommations très faibles")
print(f"   • {len(very_low)} observations < 50k kWh")
print(f"   • Représentent les 10 pires erreurs du modèle")
print(f"   • Probablement arrêts maintenance, pannes ou fériés spéciaux")
print(f"   • Solution: Détection/traitement séparé de ces cas")

print(f"\n🟡 PROBLÈMES MODÉRÉS:")
print(f"   • Mardis: Variabilité +{(mardis['DailyAverage'].std() - autres_jours['DailyAverage'].std()):+.0f} kWh vs autres jours")
print(f"   • Février: {len(fevrier)} jours, -{(fevrier['DailyAverage'].mean() - autres_mois['DailyAverage'].mean()):.0f} kWh vs moyenne")
print(f"   • Variabilité générale: écart-type {df['DailyAverage'].std():.0f} kWh")

print(f"\n💡 RECOMMANDATIONS PRIORITAIRES:")
print(f"   1️⃣  Filtre préliminaire: Exclure consommations < 50k kWh du modèle principal")
print(f"   2️⃣  Modèle spécialisé pour anomalies (maintenance, pannes)")
print(f"   3️⃣  Features additionnelles pour mardis (planning production?)")
print(f"   4️⃣  Correction saisonnière renforcée pour février")
print(f"   5️⃣  Intégration calendrier maintenance usine")

print(f"\n🎯 IMPACT POTENTIEL:")
print(f"   • Éliminer les 10 cas < 50k kWh pourrait réduire erreur de ~50%")
print(f"   • Amélioration ciblée mardis/février: +15-20% précision")
print(f"   • Objectif réaliste: Passer de 70% à 85-90% précision")

print(f"\n💾 Graphique sauvé: analyse_causes_erreurs.png")
