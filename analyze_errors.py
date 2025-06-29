import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🔍 ANALYSE DES ZONES D'ERREURS MAXIMALES")
print("=" * 55)

# === 1. CHARGER DONNÉES ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])

# Créer une prédiction simple pour analyser les erreurs
print("🔧 Création de prédictions de référence...")

# Patterns historiques moyens
daily_patterns = df.groupby(df['Day'].dt.dayofweek)['DailyAverage'].mean()
monthly_patterns = df.groupby(df['Day'].dt.month)['DailyAverage'].mean()
yearly_avg = df['DailyAverage'].mean()

# Prédictions simples basées sur patterns
predictions_simple = []
for _, row in df.iterrows():
    day_of_week = row['Day'].dayofweek
    month = row['Day'].month
    
    # Pattern jour + ajustement saisonnier
    daily_pred = daily_patterns[day_of_week]
    seasonal_factor = monthly_patterns[month] / yearly_avg
    
    prediction = daily_pred * seasonal_factor
    predictions_simple.append(prediction)

df['prediction'] = predictions_simple
df['error'] = df['prediction'] - df['DailyAverage']
df['error_abs'] = abs(df['error'])
df['error_pct'] = (df['error'] / df['DailyAverage']) * 100
df['error_abs_pct'] = abs(df['error_pct'])

print(f"✅ {len(df)} prédictions générées")

# === 2. ANALYSE PAR JOUR DE SEMAINE ===
print(f"\n📅 ERREURS PAR JOUR DE SEMAINE")
print("=" * 40)

day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
day_errors = df.groupby(df['Day'].dt.dayofweek).agg({
    'error_abs': 'mean',
    'error_abs_pct': 'mean',
    'error': 'mean'
}).round(1)

print("Jour         | Erreur Abs | Erreur %  | Biais    | Statut")
print("-" * 55)
for i, day in enumerate(day_names):
    error_abs = day_errors.loc[i, 'error_abs']
    error_pct = day_errors.loc[i, 'error_abs_pct']
    bias = day_errors.loc[i, 'error']
    
    # Déterminer statut
    if error_pct < 10:
        status = "🟢 Bon"
    elif error_pct < 20:
        status = "🟡 Moyen"
    else:
        status = "🔴 Problème"
    
    emoji = "🏢" if i < 5 else "🏠"
    print(f"{emoji} {day:<10} | {error_abs:7.0f}    | {error_pct:6.1f}%  | {bias:+7.0f} | {status}")

# === 3. ANALYSE PAR MOIS/SAISON ===
print(f"\n🌍 ERREURS PAR MOIS")
print("=" * 30)

month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

monthly_errors = df.groupby(df['Day'].dt.month).agg({
    'error_abs': 'mean',
    'error_abs_pct': 'mean',
    'error': 'mean'
}).round(1)

print("Mois | Erreur Abs | Erreur %  | Biais    | Statut")
print("-" * 50)
for month in range(1, 13):
    if month in monthly_errors.index:
        error_abs = monthly_errors.loc[month, 'error_abs']
        error_pct = monthly_errors.loc[month, 'error_abs_pct']
        bias = monthly_errors.loc[month, 'error']
        
        if error_pct < 15:
            status = "🟢 Bon"
        elif error_pct < 25:
            status = "🟡 Moyen"
        else:
            status = "🔴 Problème"
        
        print(f"{month_names[month-1]} | {error_abs:8.0f}    | {error_pct:6.1f}%  | {bias:+7.0f} | {status}")

# === 4. ANALYSE PAR PLAGE DE CONSOMMATION ===
print(f"\n⚡ ERREURS PAR NIVEAU DE CONSOMMATION")
print("=" * 45)

# Créer bins de consommation
df['consumption_bin'] = pd.cut(df['DailyAverage'], 
                              bins=[0, 50000, 70000, 90000, 120000, float('inf')],
                              labels=['Très faible', 'Faible', 'Normale', 'Élevée', 'Très élevée'])

consumption_errors = df.groupby('consumption_bin').agg({
    'error_abs': 'mean',
    'error_abs_pct': 'mean',
    'DailyAverage': ['count', 'mean']
}).round(1)

print("Niveau         | Count | Moyenne  | Erreur Abs | Erreur % | Statut")
print("-" * 65)
for bin_name in ['Très faible', 'Faible', 'Normale', 'Élevée', 'Très élevée']:
    if bin_name in consumption_errors.index:
        count = consumption_errors.loc[bin_name, ('DailyAverage', 'count')]
        mean_cons = consumption_errors.loc[bin_name, ('DailyAverage', 'mean')]
        error_abs = consumption_errors.loc[bin_name, ('error_abs', 'mean')]
        error_pct = consumption_errors.loc[bin_name, ('error_abs_pct', 'mean')]
        
        if error_pct < 12:
            status = "🟢 Bon"
        elif error_pct < 25:
            status = "�� Moyen"
        else:
            status = "🔴 Problème"
        
        print(f"{bin_name:<14} | {count:4.0f}  | {mean_cons:7.0f} | {error_abs:9.0f} | {error_pct:7.1f}% | {status}")

# === 5. IDENTIFICATION DES PIRES ERREURS ===
print(f"\n🚨 TOP 10 PIRES ERREURS")
print("=" * 35)

worst_errors = df.nlargest(10, 'error_abs_pct')[['Day', 'DailyAverage', 'prediction', 'error_abs_pct']]
print("Date        | Réel    | Prédit  | Erreur %")
print("-" * 45)
for _, row in worst_errors.iterrows():
    day_name = row['Day'].strftime('%a')
    print(f"{row['Day'].date()} {day_name} | {row['DailyAverage']:6.0f} | {row['prediction']:6.0f} | {row['error_abs_pct']:6.1f}%")

# === 6. GRAPHIQUES D'ANALYSE ===
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔍 ANALYSE DES ZONES D\'ERREURS', fontsize=16, fontweight='bold')

# GRAPHIQUE 1: Erreurs par jour de semaine
ax1 = axes[0, 0]
day_error_pcts = [day_errors.loc[i, 'error_abs_pct'] for i in range(7)]
colors = ['red' if x > 20 else 'orange' if x > 15 else 'green' for x in day_error_pcts]
bars1 = ax1.bar(range(7), day_error_pcts, color=colors, alpha=0.7)

ax1.set_title('📅 Erreurs par Jour de Semaine', fontweight='bold')
ax1.set_ylabel('Erreur Moyenne (%)')
ax1.set_xticks(range(7))
ax1.set_xticklabels([d[:3] for d in day_names])
ax1.grid(True, alpha=0.3, axis='y')

# Ajouter valeurs sur barres
for bar, val in zip(bars1, day_error_pcts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# GRAPHIQUE 2: Erreurs par mois
ax2 = axes[0, 1]
months_with_data = sorted(monthly_errors.index)
monthly_error_pcts = [monthly_errors.loc[m, 'error_abs_pct'] for m in months_with_data]
colors2 = ['red' if x > 25 else 'orange' if x > 20 else 'green' for x in monthly_error_pcts]

bars2 = ax2.bar(range(len(months_with_data)), monthly_error_pcts, color=colors2, alpha=0.7)
ax2.set_title('🌍 Erreurs par Mois', fontweight='bold')
ax2.set_ylabel('Erreur Moyenne (%)')
ax2.set_xticks(range(len(months_with_data)))
ax2.set_xticklabels([month_names[m-1] for m in months_with_data], rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# GRAPHIQUE 3: Heatmap erreurs jour/mois
ax3 = axes[1, 0]
df['month'] = df['Day'].dt.month
df['dayofweek'] = df['Day'].dt.dayofweek

# Créer matrice jour/mois des erreurs
heatmap_data = df.groupby(['month', 'dayofweek'])['error_abs_pct'].mean().unstack()
im = ax3.imshow(heatmap_data.values, cmap='Reds', aspect='auto')
ax3.set_title('🗓️ Heatmap Erreurs (Mois x Jour)', fontweight='bold')
ax3.set_ylabel('Mois')
ax3.set_xlabel('Jour semaine')
ax3.set_yticks(range(len(heatmap_data.index)))
ax3.set_yticklabels([month_names[m-1] for m in heatmap_data.index])
ax3.set_xticks(range(7))
ax3.set_xticklabels([d[:3] for d in day_names])

# GRAPHIQUE 4: Distribution des erreurs
ax4 = axes[1, 1]
ax4.hist(df['error_abs_pct'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax4.axvline(df['error_abs_pct'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Moyenne: {df["error_abs_pct"].mean():.1f}%')
ax4.axvline(df['error_abs_pct'].median(), color='orange', linestyle='--', 
           linewidth=2, label=f'Médiane: {df["error_abs_pct"].median():.1f}%')

ax4.set_title('📊 Distribution des Erreurs', fontweight='bold')
ax4.set_xlabel('Erreur Absolue (%)')
ax4.set_ylabel('Fréquence')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# === 7. RÉSUMÉ DES ZONES PROBLÉMATIQUES ===
print(f"\n🚨 ZONES PROBLÉMATIQUES IDENTIFIÉES")
print("=" * 45)

# Jours problématiques
problem_days = [i for i in range(7) if day_errors.loc[i, 'error_abs_pct'] > 20]
if problem_days:
    print(f"🔴 JOURS PROBLÉMATIQUES:")
    for day_idx in problem_days:
        print(f"   - {day_names[day_idx]}: {day_errors.loc[day_idx, 'error_abs_pct']:.1f}% erreur")

# Mois problématiques
problem_months = [m for m in monthly_errors.index if monthly_errors.loc[m, 'error_abs_pct'] > 25]
if problem_months:
    print(f"\n🔴 MOIS PROBLÉMATIQUES:")
    for month in problem_months:
        print(f"   - {month_names[month-1]}: {monthly_errors.loc[month, 'error_abs_pct']:.1f}% erreur")

# Niveaux de consommation problématiques
problem_levels = []
for bin_name in consumption_errors.index:
    if consumption_errors.loc[bin_name, ('error_abs_pct', 'mean')] > 25:
        problem_levels.append(bin_name)

if problem_levels:
    print(f"\n🔴 NIVEAUX CONSOMMATION PROBLÉMATIQUES:")
    for level in problem_levels:
        error = consumption_errors.loc[level, ('error_abs_pct', 'mean')]
        print(f"   - {level}: {error:.1f}% erreur")

print(f"\n💡 RECOMMANDATIONS CIBLÉES:")
print("=" * 35)

if problem_days:
    print(f"✅ Améliorer modélisation des jours spécifiques")
if problem_months:
    print(f"✅ Renforcer features saisonnières")
if problem_levels:
    print(f"✅ Modèles spécialisés par niveau de consommation")

print(f"\n📊 STATISTIQUES GLOBALES:")
print(f"   Erreur moyenne absolue: {df['error_abs_pct'].mean():.1f}%")
print(f"   Erreur médiane        : {df['error_abs_pct'].median():.1f}%")
print(f"   90% des erreurs < {df['error_abs_pct'].quantile(0.9):.1f}%")

print(f"\n💾 Graphique sauvé: error_analysis.png")
