import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

print("📊 VISUALISATION DES PRÉDICTIONS HYBRIDES")
print("=" * 55)

# === 1. CHARGER DONNÉES ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df['day_of_week'] = df['Day'].dt.dayofweek

# Patterns historiques
daily_patterns = df.groupby('day_of_week')['DailyAverage'].agg(['mean', 'std']).round(0)
seasonal_patterns = df.groupby(df['Day'].dt.month)['DailyAverage'].mean()

print("✅ Données chargées")

# === 2. FONCTION PRÉDICTION HYBRIDE ===
def predict_hybrid_detailed(start_date, num_days=30):
    """Prédiction hybride avec détails des composants"""
    predictions = []
    current_date = start_date
    
    for i in range(num_days):
        day_of_week = current_date.weekday()
        month = current_date.month
        
        # Composants
        base_pattern = daily_patterns.loc[day_of_week, 'mean']
        monthly_avg = seasonal_patterns.get(month, seasonal_patterns.mean())
        yearly_avg = df['DailyAverage'].mean()
        seasonal_factor = monthly_avg / yearly_avg
        
        temp_by_month = {1: 13, 2: 14, 3: 16, 4: 20, 5: 23, 6: 26, 
                        7: 28, 8: 29, 9: 27, 10: 24, 11: 19, 12: 15}
        expected_temp = temp_by_month.get(month, 20)
        temp_impact = max(0, (expected_temp - 22) * 400)
        
        if day_of_week == 4:  # Vendredi
            special_factor = 0.88
        elif day_of_week == 5:  # Samedi
            special_factor = 0.85
        else:
            special_factor = 1.0
        
        # Prédiction finale
        prediction = (base_pattern * seasonal_factor + temp_impact) * special_factor
        prediction = np.clip(prediction, 35000, 120000)
        
        predictions.append({
            'date': current_date,
            'day_of_week': day_of_week,
            'prediction': prediction,
            'base_pattern': base_pattern,
            'seasonal_factor': seasonal_factor,
            'temp_impact': temp_impact,
            'special_factor': special_factor,
            'month': month
        })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(predictions)

# === 3. GÉNÉRER PRÉDICTIONS ===
last_date = df['Day'].max()
future_start = last_date + timedelta(days=1)
pred_df = predict_hybrid_detailed(future_start, 30)

print(f"✅ 30 jours de prédictions générées")

# === 4. GRAPHIQUES ===
fig = plt.figure(figsize=(20, 16))

# GRAPHIQUE 1: Série temporelle complète
ax1 = plt.subplot(3, 2, 1)
# Historique récent (2 derniers mois)
recent_data = df[df['Day'] >= (last_date - timedelta(days=60))]
plt.plot(recent_data['Day'], recent_data['DailyAverage'], 
         color='steelblue', linewidth=2, label='Historique réel', alpha=0.8)

# Prédictions futures
plt.plot(pred_df['date'], pred_df['prediction'], 
         color='red', linewidth=3, label='Prédictions hybrides', marker='o', markersize=4)

plt.axvline(x=last_date, color='orange', linestyle='--', linewidth=2, label='Aujourd\'hui')
plt.title('📈 Évolution Temporelle: Historique vs Prédictions', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Consommation (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# GRAPHIQUE 2: Patterns par jour de semaine
ax2 = plt.subplot(3, 2, 2)
day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']

# Historique
hist_by_day = [daily_patterns.loc[i, 'mean'] for i in range(7)]
pred_by_day = pred_df.groupby('day_of_week')['prediction'].mean()

x = np.arange(7)
width = 0.35

bars1 = plt.bar(x - width/2, hist_by_day, width, label='Historique', color='lightblue', alpha=0.8)
bars2 = plt.bar(x + width/2, pred_by_day, width, label='Prédictions', color='coral', alpha=0.8)

plt.title('📊 Comparaison par Jour de Semaine', fontsize=14, fontweight='bold')
plt.xlabel('Jour')
plt.ylabel('Consommation moyenne (kWh)')
plt.xticks(x, day_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Ajouter valeurs sur barres
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 500,
             f'{height:.0f}', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 500,
             f'{height:.0f}', ha='center', va='bottom', fontsize=10)

# GRAPHIQUE 3: Composants de prédiction
ax3 = plt.subplot(3, 2, 3)
components_df = pred_df[['date', 'base_pattern', 'temp_impact']].copy()
components_df['seasonal_adjusted'] = pred_df['base_pattern'] * pred_df['seasonal_factor']
components_df['final_prediction'] = pred_df['prediction']

plt.plot(components_df['date'], components_df['base_pattern'], 
         label='Pattern de base', linewidth=2, alpha=0.7)
plt.plot(components_df['date'], components_df['seasonal_adjusted'], 
         label='+ Ajust. saisonnier', linewidth=2, alpha=0.7)
plt.plot(components_df['date'], components_df['final_prediction'], 
         label='Prédiction finale', linewidth=3, color='red')

plt.title('🔧 Décomposition des Composants', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Consommation (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# GRAPHIQUE 4: Distribution des prédictions
ax4 = plt.subplot(3, 2, 4)
plt.hist(df['DailyAverage'], bins=30, alpha=0.6, label='Historique', color='lightblue', density=True)
plt.hist(pred_df['prediction'], bins=20, alpha=0.6, label='Prédictions', color='coral', density=True)

plt.title('📏 Distribution des Consommations', fontsize=14, fontweight='bold')
plt.xlabel('Consommation (kWh)')
plt.ylabel('Densité')
plt.legend()
plt.grid(True, alpha=0.3)

# GRAPHIQUE 5: Erreurs par rapport aux patterns historiques
ax5 = plt.subplot(3, 2, 5)
errors = []
for _, row in pred_df.iterrows():
    historical_avg = daily_patterns.loc[row['day_of_week'], 'mean']
    error_pct = ((row['prediction'] - historical_avg) / historical_avg) * 100
    errors.append(error_pct)

pred_df['error_pct'] = errors

colors = ['green' if abs(e) < 10 else 'orange' if abs(e) < 20 else 'red' for e in errors]
plt.bar(range(len(errors)), errors, color=colors, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='±10% seuil')
plt.axhline(y=-10, color='orange', linestyle='--', alpha=0.7)

plt.title('🎯 Erreurs vs Patterns Historiques', fontsize=14, fontweight='bold')
plt.xlabel('Jours (0-29)')
plt.ylabel('Erreur (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# GRAPHIQUE 6: Calendrier des prédictions
ax6 = plt.subplot(3, 2, 6)
pred_df['week'] = pred_df['date'].dt.isocalendar().week
pred_df['day_name'] = pred_df['date'].dt.day_name()

# Créer matrice pour heatmap
weeks = sorted(pred_df['week'].unique())
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

heatmap_data = np.full((len(weeks), 7), np.nan)
for i, week in enumerate(weeks):
    week_data = pred_df[pred_df['week'] == week]
    for j, day in enumerate(days_order):
        day_data = week_data[week_data['day_name'] == day]
        if not day_data.empty:
            heatmap_data[i, j] = day_data['prediction'].iloc[0]

im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(im, label='Consommation (kWh)')
plt.yticks(range(len(weeks)), [f'Sem {w}' for w in weeks])
plt.xticks(range(7), ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
plt.title('🗓️ Calendrier des Prédictions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('hybrid_predictions_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# === 5. STATISTIQUES RÉSUMÉES ===
print(f"\n📊 STATISTIQUES DES PRÉDICTIONS :")
print("-" * 50)
print(f"🔸 Période        : {pred_df['date'].min().date()} à {pred_df['date'].max().date()}")
print(f"🔸 Moyenne        : {pred_df['prediction'].mean():6.0f} kWh/jour")
print(f"🔸 Médiane        : {pred_df['prediction'].median():6.0f} kWh/jour")
print(f"🔸 Min/Max        : {pred_df['prediction'].min():6.0f} / {pred_df['prediction'].max():6.0f} kWh")
print(f"🔸 Écart-type     : {pred_df['prediction'].std():6.0f} kWh")

print(f"\n🎯 PRÉCISION vs HISTORIQUE :")
print("-" * 50)
print(f"🔸 Erreur moyenne : {abs(np.array(errors)).mean():4.1f}%")
print(f"�� Erreur médiane : {np.median(np.abs(errors)):4.1f}%")
print(f"🔸 Prédictions <10% erreur : {sum(1 for e in errors if abs(e) < 10)}/{len(errors)} ({sum(1 for e in errors if abs(e) < 10)/len(errors)*100:.1f}%)")

print(f"\n💾 Graphique sauvé: hybrid_predictions_analysis.png")
