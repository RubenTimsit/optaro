import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔍 ANALYSE DES ZONES D'ERREURS DU MODÈLE")
print("=" * 55)

# === 1. CHARGER DONNÉES ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])

print(f"✅ Données chargées: {len(df)} observations")
print(f"📅 Période: {df['Day'].min().date()} → {df['Day'].max().date()}")

# === 2. CRÉER PRÉDICTIONS SIMPLES POUR ANALYSER ERREURS ===
print("\n🔧 Création d'un modèle simple pour analyser les erreurs...")

# Modèle basé sur moyennes historiques par jour/mois
df['day_of_week'] = df['Day'].dt.dayofweek
df['month'] = df['Day'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Calculer moyennes par patterns
daily_patterns = df.groupby('day_of_week')['DailyAverage'].mean()
monthly_patterns = df.groupby('month')['DailyAverage'].mean()
weekend_factor = df[df['is_weekend']==1]['DailyAverage'].mean() / df[df['is_weekend']==0]['DailyAverage'].mean()

# Prédictions simples
predictions = []
for _, row in df.iterrows():
    base_pred = daily_patterns[row['day_of_week']] * 0.6 + monthly_patterns[row['month']] * 0.4
    predictions.append(base_pred)

df['predictions'] = predictions
df['error'] = df['predictions'] - df['DailyAverage']
df['error_pct'] = (df['error'] / df['DailyAverage']) * 100
df['abs_error'] = abs(df['error'])
df['abs_error_pct'] = abs(df['error_pct'])

print(f"✅ Prédictions simples calculées")

# === 3. ANALYSE DES ERREURS PAR ZONE ===
print(f"\n📊 ANALYSE DES ERREURS PAR ZONE:")
print("=" * 40)

# Par jour de semaine
print(f"🗓️  ERREURS PAR JOUR DE SEMAINE:")
day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
day_errors = df.groupby('day_of_week')['abs_error_pct'].agg(['mean', 'std', 'count'])

for i, day in enumerate(day_names):
    if i in day_errors.index:
        mean_err = day_errors.loc[i, 'mean']
        std_err = day_errors.loc[i, 'std']
        count = day_errors.loc[i, 'count']
        status = "🔴" if mean_err > 20 else "🟡" if mean_err > 15 else "🟢"
        print(f"   {status} {day:<9}: {mean_err:5.1f}% ±{std_err:4.1f}% ({count:3d} obs)")

# Par mois
print(f"\n🌍 ERREURS PAR MOIS:")
month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
month_errors = df.groupby('month')['abs_error_pct'].agg(['mean', 'std', 'count'])

for i in range(1, 13):
    if i in month_errors.index:
        mean_err = month_errors.loc[i, 'mean']
        std_err = month_errors.loc[i, 'std']
        count = month_errors.loc[i, 'count']
        status = "🔴" if mean_err > 20 else "🟡" if mean_err > 15 else "🟢"
        print(f"   {status} {month_names[i-1]:<3}: {mean_err:5.1f}% ±{std_err:4.1f}% ({count:3d} obs)")

# Par niveau de consommation
print(f"\n⚡ ERREURS PAR NIVEAU DE CONSOMMATION:")
df['consumption_level'] = pd.cut(df['DailyAverage'], 
                                bins=[0, 50000, 70000, 90000, float('inf')],
                                labels=['Très faible', 'Faible', 'Normale', 'Élevée'])

level_errors = df.groupby('consumption_level')['abs_error_pct'].agg(['mean', 'std', 'count'])
for level in ['Très faible', 'Faible', 'Normale', 'Élevée']:
    if level in level_errors.index:
        mean_err = level_errors.loc[level, 'mean']
        std_err = level_errors.loc[level, 'std']
        count = level_errors.loc[level, 'count']
        status = "🔴" if mean_err > 20 else "🟡" if mean_err > 15 else "🟢"
        print(f"   {status} {level:<12}: {mean_err:5.1f}% ±{std_err:4.1f}% ({count:3d} obs)")

# === 4. IDENTIFIER LES PIRES ERREURS ===
print(f"\n🚨 TOP 10 PIRES ERREURS:")
print("-" * 40)
worst_errors = df.nlargest(10, 'abs_error_pct')[['Day', 'DailyAverage', 'predictions', 'error_pct']]
for i, (_, row) in enumerate(worst_errors.iterrows(), 1):
    day_name = row['Day'].strftime('%A')
    print(f"{i:2d}. {row['Day'].date()} ({day_name}): "
          f"Réel={row['DailyAverage']:6.0f}, Prédit={row['predictions']:6.0f}, "
          f"Erreur={row['error_pct']:+5.1f}%")

# === 5. GRAPHIQUES D'ANALYSE ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔍 ANALYSE DES ZONES D\'ERREURS', fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Erreurs par jour de semaine
ax1 = axes[0, 0]
day_error_means = [day_errors.loc[i, 'mean'] if i in day_errors.index else 0 for i in range(7)]
colors = ['red' if err > 20 else 'orange' if err > 15 else 'green' for err in day_error_means]
bars1 = ax1.bar(range(7), day_error_means, color=colors, alpha=0.7)
ax1.set_title('📅 Erreurs par Jour de Semaine', fontweight='bold')
ax1.set_ylabel('Erreur Moyenne (%)')
ax1.set_xticks(range(7))
ax1.set_xticklabels([name[:3] for name in day_names])
ax1.grid(True, alpha=0.3, axis='y')

# Ajouter valeurs sur barres
for bar, err in zip(bars1, day_error_means):
    if err > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{err:.1f}%', ha='center', va='bottom', fontweight='bold')

# GRAPHIQUE 2: Erreurs par mois
ax2 = axes[0, 1]
month_error_means = [month_errors.loc[i, 'mean'] if i in month_errors.index else 0 for i in range(1, 13)]
colors2 = ['red' if err > 20 else 'orange' if err > 15 else 'green' for err in month_error_means]
bars2 = ax2.bar(range(12), month_error_means, color=colors2, alpha=0.7)
ax2.set_title('🌍 Erreurs par Mois', fontweight='bold')
ax2.set_ylabel('Erreur Moyenne (%)')
ax2.set_xticks(range(12))
ax2.set_xticklabels([name for name in month_names])
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

# GRAPHIQUE 3: Distribution des erreurs
ax3 = axes[1, 0]
ax3.hist(df['error_pct'], bins=50, alpha=0.7, color='skyblue', density=True)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
ax3.axvline(x=df['error_pct'].mean(), color='orange', linestyle='--', 
           linewidth=2, label=f'Moyenne: {df["error_pct"].mean():.1f}%')
ax3.set_title('📊 Distribution des Erreurs', fontweight='bold')
ax3.set_xlabel('Erreur (%)')
ax3.set_ylabel('Densité')
ax3.legend()
ax3.grid(True, alpha=0.3)

# GRAPHIQUE 4: Erreurs vs consommation
ax4 = axes[1, 1]
scatter = ax4.scatter(df['DailyAverage'], df['abs_error_pct'], 
                     alpha=0.6, c=df['is_weekend'], cmap='coolwarm', s=20)
ax4.set_title('⚡ Erreurs vs Niveau Consommation', fontweight='bold')
ax4.set_xlabel('Consommation Réelle (kWh)')
ax4.set_ylabel('Erreur Absolue (%)')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Week-end (1) vs Semaine (0)')

plt.tight_layout()
plt.savefig('analyse_zones_erreurs.png', dpi=300, bbox_inches='tight')
plt.show()

# === 6. SYNTHÈSE DES ZONES PROBLÉMATIQUES ===
print(f"\n🎯 SYNTHÈSE DES ZONES PROBLÉMATIQUES:")
print("=" * 50)

# Identifier les pires segments
worst_day = day_names[np.argmax(day_error_means)]
worst_month = month_names[np.argmax(month_error_means)]

print(f"🔴 ZONES À FORTE ERREUR:")
print(f"   📅 Pire jour    : {worst_day} ({max(day_error_means):.1f}% erreur)")
print(f"   🌍 Pire mois    : {worst_month} ({max(month_error_means):.1f}% erreur)")

# Analyser patterns
weekend_errors = df[df['is_weekend']==1]['abs_error_pct'].mean()
weekday_errors = df[df['is_weekend']==0]['abs_error_pct'].mean()
print(f"   🏠 Week-ends    : {weekend_errors:.1f}% erreur moyenne")
print(f"   🏢 Jours semaine: {weekday_errors:.1f}% erreur moyenne")

# Saisonnalité
summer_months = [6, 7, 8]
winter_months = [12, 1, 2]
summer_errors = df[df['month'].isin(summer_months)]['abs_error_pct'].mean()
winter_errors = df[df['month'].isin(winter_months)]['abs_error_pct'].mean()
print(f"   ☀️ Été         : {summer_errors:.1f}% erreur moyenne")
print(f"   ❄️ Hiver       : {winter_errors:.1f}% erreur moyenne")

print(f"\n💡 RECOMMANDATIONS CIBLÉES:")
print(f"   1️⃣  Améliorer prédictions {worst_day.lower()}s")
print(f"   2️⃣  Modèle spécialisé pour {worst_month.lower()}")
print(f"   3️⃣  Traitement spécial week-ends")
print(f"   4️⃣  Ajustement saisonnier renforcé")

print(f"\n💾 Graphique sauvé: analyse_zones_erreurs.png")
