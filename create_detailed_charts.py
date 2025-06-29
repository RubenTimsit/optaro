import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

print("📈 CRÉATION DES GRAPHIQUES DÉTAILLÉS")
print("=" * 45)

# === 1. CHARGER DONNÉES ===
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df['day_of_week'] = df['Day'].dt.dayofweek

# Patterns historiques
daily_patterns = df.groupby('day_of_week')['DailyAverage'].agg(['mean', 'std']).round(0)
seasonal_patterns = df.groupby(df['Day'].dt.month)['DailyAverage'].mean()

# === 2. PRÉDICTIONS HYBRIDES ===
def predict_hybrid_detailed(start_date, num_days=30):
    predictions = []
    current_date = start_date
    
    for i in range(num_days):
        day_of_week = current_date.weekday()
        month = current_date.month
        
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
            'month': month,
            'is_weekend': day_of_week >= 5
        })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(predictions)

last_date = df['Day'].max()
future_start = last_date + timedelta(days=1)
pred_df = predict_hybrid_detailed(future_start, 30)

# === 3. GRAPHIQUE 1: TIMELINE AVEC ZONES ===
plt.figure(figsize=(16, 8))

# Données récentes
recent_data = df[df['Day'] >= (last_date - timedelta(days=30))]

# Plot principal
plt.plot(recent_data['Day'], recent_data['DailyAverage'], 
         color='steelblue', linewidth=3, label='Historique réel', marker='o', markersize=4)
plt.plot(pred_df['date'], pred_df['prediction'], 
         color='red', linewidth=3, label='Prédictions hybrides', marker='s', markersize=5)

# Zones week-end futures
for _, row in pred_df.iterrows():
    if row['is_weekend']:
        plt.axvspan(row['date'] - timedelta(hours=12), row['date'] + timedelta(hours=12), 
                   alpha=0.2, color='lightcoral', label='Week-end' if row['date'] == pred_df[pred_df['is_weekend']]['date'].iloc[0] else "")

plt.axvline(x=last_date, color='orange', linestyle='--', linewidth=2, label='Transition')
plt.title('🔮 Prédictions Hybrides - Vue Temporelle (30 jours)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Consommation énergétique (kWh)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('timeline_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# === 4. GRAPHIQUE 2: DÉCOMPOSITION PAR COMPOSANTS ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Composants empilés
ax1.fill_between(pred_df['date'], 0, pred_df['base_pattern'], 
                alpha=0.7, label='Pattern de base', color='lightblue')
ax1.fill_between(pred_df['date'], pred_df['base_pattern'], 
                pred_df['base_pattern'] * pred_df['seasonal_factor'], 
                alpha=0.7, label='Ajustement saisonnier', color='lightgreen')
ax1.fill_between(pred_df['date'], pred_df['base_pattern'] * pred_df['seasonal_factor'], 
                pred_df['base_pattern'] * pred_df['seasonal_factor'] + pred_df['temp_impact'], 
                alpha=0.7, label='Impact température', color='lightyellow')

ax1.plot(pred_df['date'], pred_df['prediction'], 
         color='red', linewidth=3, label='Prédiction finale', marker='o', markersize=4)

ax1.set_title('🔧 Décomposition des Composants de Prédiction', fontsize=14, fontweight='bold')
ax1.set_ylabel('Consommation (kWh)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Facteurs d'ajustement
ax2.plot(pred_df['date'], pred_df['seasonal_factor'], 
         color='green', linewidth=2, label='Facteur saisonnier', marker='o')
ax2.plot(pred_df['date'], pred_df['special_factor'], 
         color='purple', linewidth=2, label='Facteur spécial (week-end)', marker='s')

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Référence (1.0)')
ax2.set_title('📊 Facteurs d\'Ajustement', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Facteur multiplicateur')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('components_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()

# === 5. GRAPHIQUE 3: COMPARAISON PAR JOUR ===
plt.figure(figsize=(14, 8))

day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

# Historique avec barres d'erreur
hist_means = [daily_patterns.loc[i, 'mean'] for i in range(7)]
hist_stds = [daily_patterns.loc[i, 'std'] for i in range(7)]

# Prédictions moyennes
pred_by_day = pred_df.groupby('day_of_week')['prediction'].agg(['mean', 'std']).fillna(0)
pred_means = [pred_by_day.loc[i, 'mean'] if i in pred_by_day.index else 0 for i in range(7)]
pred_stds = [pred_by_day.loc[i, 'std'] if i in pred_by_day.index else 0 for i in range(7)]

x = np.arange(7)
width = 0.35

# Barres avec erreurs
bars1 = plt.bar(x - width/2, hist_means, width, yerr=hist_stds, 
               label='Historique (±1σ)', color='lightblue', alpha=0.8, capsize=5)
bars2 = plt.bar(x + width/2, pred_means, width, yerr=pred_stds,
               label='Prédictions (±1σ)', color='coral', alpha=0.8, capsize=5)

# Annotations
for i, (hist, pred) in enumerate(zip(hist_means, pred_means)):
    if pred > 0:  # Only if we have predictions for this day
        diff_pct = ((pred - hist) / hist) * 100
        color = 'green' if abs(diff_pct) < 10 else 'orange' if abs(diff_pct) < 20 else 'red'
        plt.text(i, max(hist, pred) + 2000, f'{diff_pct:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', color=color)

plt.title('📊 Comparaison Historique vs Prédictions par Jour', fontsize=16, fontweight='bold')
plt.xlabel('Jour de la semaine', fontsize=12)
plt.ylabel('Consommation moyenne (kWh)', fontsize=12)
plt.xticks(x, day_names)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('daily_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# === 6. GRAPHIQUE 4: HEATMAP MENSUELLE ===
plt.figure(figsize=(12, 8))

# Créer calendrier
pred_df['day'] = pred_df['date'].dt.day
pred_df['month_name'] = pred_df['date'].dt.strftime('%B')

# Matrice pour heatmap
calendar_data = pred_df.pivot_table(values='prediction', 
                                   index='month_name', 
                                   columns='day', 
                                   fill_value=np.nan)

# Heatmap
sns.heatmap(calendar_data, annot=True, fmt='.0f', cmap='RdYlBu_r', 
           cbar_kws={'label': 'Consommation (kWh)'}, 
           linewidths=0.5, linecolor='white')

plt.title('🗓️ Calendrier des Prédictions - Vue Mensuelle', fontsize=16, fontweight='bold')
plt.xlabel('Jour du mois', fontsize=12)
plt.ylabel('Mois', fontsize=12)
plt.tight_layout()
plt.savefig('monthly_calendar.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 4 graphiques créés:")
print("�� timeline_predictions.png")
print("📁 components_breakdown.png") 
print("📁 daily_comparison.png")
print("📁 monthly_calendar.png")
print("📁 hybrid_predictions_analysis.png (graphique principal)")
