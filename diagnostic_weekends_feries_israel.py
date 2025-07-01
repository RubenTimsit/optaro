import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🇮🇱 DIAGNOSTIC WEEK-ENDS ET JOURS FÉRIÉS - ISRAËL")
print("=" * 60)
print("ℹ️  WEEK-ENDS EN ISRAËL = VENDREDI + SAMEDI")
print("=" * 60)

# === 1. CHARGEMENT DES DONNÉES ===
print("\n📊 1. Chargement et analyse des données...")

df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)

print(f"📊 Données: {len(df)} jours ({df['Day'].min().date()} → {df['Day'].max().date()})")

# === 2. CRÉATION DES VARIABLES TEMPORELLES POUR ISRAËL ===
print("\n🔧 2. Création des variables temporelles pour Israël...")

# Variables de base (0=Lundi, 6=Dimanche)
df['weekday'] = df['Day'].dt.dayofweek
df['is_sunday'] = (df['weekday'] == 6).astype(int)    # Dimanche = jour ouvrable
df['is_monday'] = (df['weekday'] == 0).astype(int)    # Lundi = jour ouvrable
df['is_tuesday'] = (df['weekday'] == 1).astype(int)   # Mardi = jour ouvrable
df['is_wednesday'] = (df['weekday'] == 2).astype(int) # Mercredi = jour ouvrable
df['is_thursday'] = (df['weekday'] == 3).astype(int)  # Jeudi = jour ouvrable
df['is_friday'] = (df['weekday'] == 4).astype(int)    # Vendredi = WEEK-END
df['is_saturday'] = (df['weekday'] == 5).astype(int)  # Samedi = WEEK-END

# 🇮🇱 DÉFINITION CORRECTE DES WEEK-ENDS ISRAÉLIENS
df['is_weekend_israel'] = ((df['weekday'] == 4) | (df['weekday'] == 5)).astype(int)  # Vendredi OU Samedi
df['is_workday_israel'] = (~df['is_weekend_israel'].astype(bool)).astype(int)        # Dimanche à Jeudi

# Pour comparaison : ancien modèle (samedi-dimanche)
df['is_weekend'] = ((df['weekday'] == 5) | (df['weekday'] == 6)).astype(int)  # Samedi OU Dimanche

# Jours fériés enrichis
df['is_holiday_full'] = df['is_holiday_full'].astype(int)
df['is_holiday_half'] = df['is_holiday_half'].astype(int)
df['is_holiday_any'] = ((df['is_holiday_full'] == 1) | (df['is_holiday_half'] == 1)).astype(int)

print("✅ Variables temporelles israéliennes créées:")
print(f"   - Week-ends (Ven+Sam): {df['is_weekend_israel'].sum()} jours")
print(f"   - Jours ouvrables (Dim-Jeu): {df['is_workday_israel'].sum()} jours")

# === 3. DÉTECTION DES "PONTS" ET JOURS SPÉCIAUX ISRAÉLIENS ===
print("\n🌉 3. Détection des ponts et jours spéciaux israéliens...")

def detect_bridges_israel(df):
    """Détecte les ponts et jours spéciaux pour Israël"""
    df = df.copy()
    df = df.sort_values('Day').reset_index(drop=True)
    
    # Créer des décalages pour analyser les jours adjacents
    df['prev_day_holiday'] = df['is_holiday_any'].shift(1).fillna(0)
    df['next_day_holiday'] = df['is_holiday_any'].shift(-1).fillna(0)
    df['prev_day_weekend'] = df['is_weekend_israel'].shift(1).fillna(0)
    df['next_day_weekend'] = df['is_weekend_israel'].shift(-1).fillna(0)
    
    # Détection des ponts (jours ouvrables entre weekend et férié)
    df['is_bridge_israel'] = (
        (df['is_workday_israel'] == 1) & 
        (df['is_holiday_any'] == 0) &
        (
            ((df['prev_day_weekend'] == 1) & (df['next_day_holiday'] == 1)) |
            ((df['prev_day_holiday'] == 1) & (df['next_day_weekend'] == 1)) |
            ((df['prev_day_holiday'] == 1) & (df['next_day_holiday'] == 1))
        )
    ).astype(int)
    
    # Jeudi avant week-end long (équivalent du vendredi ailleurs)
    df['thursday_before_long_weekend'] = (
        (df['is_thursday'] == 1) & 
        (df['next_day_weekend'] == 1) &
        (df['Day'].shift(-3).dt.dayofweek.isin([6, 0]))  # Dimanche ou lundi férié
    ).astype(int)
    
    # Dimanche après week-end long
    df['sunday_after_long_weekend'] = (
        (df['is_sunday'] == 1) & 
        (df['prev_day_weekend'] == 1) &
        (df['Day'].shift(-3).dt.dayofweek.isin([3, 4]))  # Jeudi ou vendredi férié précédent
    ).astype(int)
    
    return df

df = detect_bridges_israel(df)

# === 4. STATISTIQUES PAR TYPE DE JOUR ISRAÉLIEN ===
print("\n📊 4. Statistiques de consommation par type de jour israélien...")

def analyze_consumption_israel(df):
    """Analyse la consommation par type de jour israélien"""
    
    # Créer les catégories de jours pour Israël
    categories = []
    for _, row in df.iterrows():
        if row['is_holiday_full'] == 1:
            categories.append('Férié complet')
        elif row['is_holiday_half'] == 1:
            categories.append('Demi-férié')
        elif row['is_bridge_israel'] == 1:
            categories.append('Pont')
        elif row['is_friday'] == 1:
            categories.append('Vendredi (WE)')
        elif row['is_saturday'] == 1:
            categories.append('Samedi (WE)')
        elif row['thursday_before_long_weekend'] == 1:
            categories.append('Jeudi avant WE long')
        elif row['sunday_after_long_weekend'] == 1:
            categories.append('Dimanche après WE long')
        elif row['is_thursday'] == 1:
            categories.append('Jeudi')
        elif row['is_sunday'] == 1:
            categories.append('Dimanche (ouvrable)')
        elif row['is_monday'] == 1:
            categories.append('Lundi')
        else:
            categories.append('Jour ouvrable normal')
    
    df['day_category_israel'] = categories
    
    # Calculer les statistiques
    stats = df.groupby('day_category_israel')['DailyAverage'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(0)
    
    # Référence = moyenne des jours ouvrables (Dimanche-Jeudi, hors fériés)
    workday_mean = df[df['is_workday_israel'] == 1]['DailyAverage'].mean()
    stats['mean_vs_workday'] = ((stats['mean'] / workday_mean) - 1) * 100
    
    return stats.sort_values('mean', ascending=False), df

consumption_stats_israel, df = analyze_consumption_israel(df)

print("\n🏆 CONSOMMATION MOYENNE PAR TYPE DE JOUR (ISRAËL):")
print(consumption_stats_israel[['count', 'mean', 'mean_vs_workday']].to_string())

# === 5. COMPARAISON ANCIEN vs NOUVEAU MODÈLE ===
print("\n⚖️ 5. Comparaison ancien modèle vs modèle israélien...")

# Ancien modèle (samedi-dimanche)
old_weekend_avg = df[df['is_weekend'] == 1]['DailyAverage'].mean()
old_workday_avg = df[df['is_weekend'] == 0]['DailyAverage'].mean()

# Nouveau modèle (vendredi-samedi)
new_weekend_avg = df[df['is_weekend_israel'] == 1]['DailyAverage'].mean()
new_workday_avg = df[df['is_workday_israel'] == 1]['DailyAverage'].mean()

print(f"\n📊 COMPARAISON DES DÉFINITIONS:")
print(f"   ANCIEN (Sam-Dim):  WE={old_weekend_avg:.0f} kWh, Ouvrable={old_workday_avg:.0f} kWh, Diff={old_weekend_avg-old_workday_avg:.0f}")
print(f"   ISRAËL (Ven-Sam):  WE={new_weekend_avg:.0f} kWh, Ouvrable={new_workday_avg:.0f} kWh, Diff={new_weekend_avg-new_workday_avg:.0f}")

# === 6. INTERACTION TEMPÉRATURE-WEEK-END ISRAÉLIEN ===
print("\n🌡️ 6. Interaction température et week-end israélien...")

def analyze_temp_interaction_israel(df):
    """Analyse l'interaction entre température et week-end israélien"""
    
    # Créer des bins de température
    df['temp_category'] = pd.cut(df['TempAvg'], 
                                bins=[-10, 15, 20, 25, 30, 50], 
                                labels=['Froid (<15°)', 'Frais (15-20°)', 'Modéré (20-25°)', 'Chaud (25-30°)', 'Très chaud (>30°)'])
    
    # Analyse pour les week-ends israéliens vs jours ouvrables
    temp_interaction = df.groupby(['temp_category', 'is_weekend_israel'])['DailyAverage'].mean().unstack()
    temp_interaction.columns = ['Jours ouvrables (Dim-Jeu)', 'Week-ends (Ven-Sam)']
    temp_interaction['Différence (%)'] = ((temp_interaction['Week-ends (Ven-Sam)'] / temp_interaction['Jours ouvrables (Dim-Jeu)']) - 1) * 100
    
    return temp_interaction.round(0)

temp_interaction_israel = analyze_temp_interaction_israel(df)
print("\n🌡️ CONSOMMATION PAR TEMPÉRATURE ET WEEK-END ISRAÉLIEN:")
print(temp_interaction_israel.to_string())

# === 7. VISUALISATIONS SPÉCIFIQUES ISRAËL ===
print("\n📊 7. Génération des visualisations pour Israël...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🇮🇱 DIAGNOSTIC WEEK-ENDS ISRAÉLIENS (Vendredi-Samedi)', fontsize=16, fontweight='bold')

# Plot 1: Consommation par jour de la semaine (perspective israélienne)
day_names_israel = ['Dimanche\n(ouvrable)', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi\n(WE)', 'Samedi\n(WE)']
weekday_consumption = df.groupby('weekday')['DailyAverage'].mean()

# Réorganiser pour commencer par dimanche (jour ouvrable en Israël)
weekday_reordered = [weekday_consumption.iloc[6]] + list(weekday_consumption.iloc[0:6])
colors_israel = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'orange', 'orange']

axes[0,0].bar(range(7), weekday_reordered, color=colors_israel)
axes[0,0].set_xticks(range(7))
axes[0,0].set_xticklabels(day_names_israel, rotation=45, fontsize=9)
axes[0,0].set_ylabel('Consommation moyenne (kWh)')
axes[0,0].set_title('Consommation par jour (Semaine israélienne)')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Comparaison ancien vs nouveau modèle
comparison_data = {
    'Week-ends': [old_weekend_avg, new_weekend_avg],
    'Jours ouvrables': [old_workday_avg, new_workday_avg]
}
x = range(2)
width = 0.35

axes[0,1].bar([i - width/2 for i in x], comparison_data['Week-ends'], width, label='Week-ends', alpha=0.8)
axes[0,1].bar([i + width/2 for i in x], comparison_data['Jours ouvrables'], width, label='Jours ouvrables', alpha=0.8)
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(['Ancien\n(Sam-Dim)', 'Israël\n(Ven-Sam)'])
axes[0,1].set_ylabel('Consommation moyenne (kWh)')
axes[0,1].set_title('Comparaison des définitions de week-end')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Évolution temporelle avec marquage correct des jours
sample_period = df[(df['Day'] >= '2022-07-01') & (df['Day'] <= '2022-08-31')]
axes[1,0].plot(sample_period['Day'], sample_period['DailyAverage'], 'b-', alpha=0.7, linewidth=1)

# Marquer les week-ends israéliens
weekend_israel = sample_period[sample_period['is_weekend_israel'] == 1]
holiday_days = sample_period[sample_period['is_holiday_any'] == 1]

axes[1,0].scatter(weekend_israel['Day'], weekend_israel['DailyAverage'], 
                  color='orange', s=50, alpha=0.8, label='Week-ends (Ven-Sam)')
axes[1,0].scatter(holiday_days['Day'], holiday_days['DailyAverage'], 
                  color='red', s=50, alpha=0.8, label='Jours fériés')

axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Consommation (kWh)')
axes[1,0].set_title('Évolution été 2022 (Week-ends israéliens)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].tick_params(axis='x', rotation=45)

# Plot 4: Interaction température-week-end israélien
temp_bins_israel = df.groupby(['temp_category', 'is_weekend_israel'])['DailyAverage'].mean().unstack()
temp_bins_israel.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'orange'])
axes[1,1].set_xlabel('Catégorie de température')
axes[1,1].set_ylabel('Consommation moyenne (kWh)')
axes[1,1].set_title('Interaction Température-Week-end israélien')
axes[1,1].legend(['Jours ouvrables (Dim-Jeu)', 'Week-ends (Ven-Sam)'])
axes[1,1].grid(True, alpha=0.3)
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('diagnostic_weekends_israel.png', dpi=300, bbox_inches='tight')
plt.show()

# === 8. DÉTECTION DES NOUVEAUX PROBLÈMES ===
print("\n🚨 8. Problèmes avec la définition israélienne...")

problems_israel = []

# Vérifier la différence entre vendredi et samedi
friday_avg = df[df['is_friday'] == 1]['DailyAverage'].mean()
saturday_avg = df[df['is_saturday'] == 1]['DailyAverage'].mean()
if abs(friday_avg - saturday_avg) > 5000:
    problems_israel.append(f"PROBLÈME: Vendredi ({friday_avg:.0f}) vs Samedi ({saturday_avg:.0f}) - Différence {abs(friday_avg - saturday_avg):.0f} kWh")

# Vérifier si dimanche est vraiment un jour ouvrable
sunday_avg = df[df['is_sunday'] == 1]['DailyAverage'].mean()
monday_avg = df[df['is_monday'] == 1]['DailyAverage'].mean()
if abs(sunday_avg - monday_avg) > 3000:
    problems_israel.append(f"ATTENTION: Dimanche ({sunday_avg:.0f}) vs Lundi ({monday_avg:.0f}) - Différence {abs(sunday_avg - monday_avg):.0f} kWh")

print(f"\n🚨 {len(problems_israel)} PROBLÈMES AVEC DÉFINITION ISRAÉLIENNE:")
for i, problem in enumerate(problems_israel, 1):
    print(f"   {i}. {problem}")

# === 9. EXPORT DES DONNÉES ISRAÉLIENNES ===
print("\n💾 9. Export des données avec variables israéliennes...")

# Colonnes à exporter pour le modèle israélien
israel_columns = [
    'Day', 'DailyAverage', 'TempAvg', 'TempMin', 'TempMax', 'Precip', 'WindSpeed', 'Pressure',
    'is_holiday_full', 'is_holiday_half', 'is_holiday_any',
    'weekday', 
    'is_sunday', 'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday', 'is_saturday',
    'is_weekend_israel', 'is_workday_israel',
    'is_bridge_israel', 'thursday_before_long_weekend', 'sunday_after_long_weekend',
    'day_category_israel'
]

df_israel = df[israel_columns].copy()
df_israel.to_csv('data_with_israel_temporal_features.csv', index=False)

print(f"✅ Données israéliennes sauvegardées: data_with_israel_temporal_features.csv")
print(f"📊 {len(israel_columns)} colonnes, {len(df_israel)} lignes")

# === 10. RECOMMANDATIONS FINALES ===
print("\n💡 10. Recommandations pour le modèle israélien...")

recommendations_israel = [
    "Utiliser is_weekend_israel (Ven-Sam) au lieu de is_weekend (Sam-Dim)",
    "Distinguer vendredi et samedi avec des variables séparées",
    "Traiter dimanche comme un jour ouvrable normal",
    "Ajouter l'effet 'jeudi avant week-end long'",
    "Créer des interactions température × week-end israélien",
    "Inclure les ponts spécifiques au calendrier israélien"
]

print(f"\n💡 {len(recommendations_israel)} RECOMMANDATIONS POUR ISRAËL:")
for i, rec in enumerate(recommendations_israel, 1):
    print(f"   {i}. {rec}")

print("\n" + "="*70)
print("🇮🇱 DIAGNOSTIC WEEK-ENDS ISRAÉLIENS TERMINÉ !")
print("="*70)
print(f"📊 Graphiques: diagnostic_weekends_israel.png")
print(f"💾 Données israéliennes: data_with_israel_temporal_features.csv") 
print(f"🎯 Différence week-end: {new_weekend_avg-new_workday_avg:.0f} kWh (Fen-Sam vs Dim-Jeu)")
print("="*70) 