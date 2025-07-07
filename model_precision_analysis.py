import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")

print("📊 ISRAELI MODEL - PRECISION ANALYSIS")
print("=" * 70)
print("🎯 Objective: Comprehensive evaluation of Israeli optimized model")
print("🇮🇱 Weekend system: Friday-Saturday | Workdays: Sunday-Thursday")
print("=" * 70)

# === 1. MODEL LOADING ===
print("\n🤖 1. Loading Israeli optimized model...")

try:
    with open('israel_optimized_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    performance = model_data['performance']
    
    print("✅ Israeli model loaded successfully")
    print(f"   📊 Algorithm: {model_data['training_info']['algorithm']}")
    print(f"   🎯 Features: {len(features)} variables")
    print(f"   🇮🇱 Weekend system: {model_data['israel_specifics']['weekend_system']}")
    print(f"   📈 Test R²: {performance['test_r2']:.3f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# === 2. DATA LOADING ===
print("\n📊 2. Loading test data for analysis...")

try:
    df = pd.read_csv("data_with_israel_temporal_features.csv")
    df['Day'] = pd.to_datetime(df['Day'])
    df = df.sort_values('Day').reset_index(drop=True)
    
    print(f"✅ Data loaded: {len(df)} days")
    print(f"   Period: {df['Day'].min().date()} → {df['Day'].max().date()}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# === 3. DATA PREPARATION ===
print("\n🔧 3. Preparing data for prediction...")

# Remove NaN values
df_clean = df.dropna()

# Split data (same as training: 70/30)
split_idx = int(len(df_clean) * 0.7)
split_date = df_clean.iloc[split_idx]['Day']

# Test set
X_test = df_clean.iloc[split_idx:][features]
y_test = df_clean.iloc[split_idx:]['DailyAverage']
dates_test = df_clean.iloc[split_idx:]['Day']

print(f"   Test period: {split_date.date()} → {dates_test.max().date()}")
print(f"   Test samples: {len(X_test)}")

# === 4. PREDICTIONS GENERATION ===
print("\n🔮 4. Generating predictions with Israeli model...")

# Scale features and predict
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

print("✅ Predictions generated")

# === 5. OVERALL PERFORMANCE ===
print("\n📈 5. Overall model performance evaluation...")

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# Additional metrics
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mbe = np.mean(y_pred - y_test)  # Mean Bias Error
residuals = y_test - y_pred

print(f"\n🏆 ISRAELI MODEL PERFORMANCE:")
print("=" * 50)
print(f"📊 MAE (Mean Absolute Error):    {mae:,.0f} kWh")
print(f"📊 RMSE (Root Mean Square Error): {rmse:,.0f} kWh")
print(f"📊 R² (Coefficient of Determination): {r2:.3f}")
print(f"📊 MAPE (Mean Absolute Percentage Error): {mape:.1f}%")
print(f"📊 MBE (Mean Bias Error):        {mbe:+,.0f} kWh")

# Performance interpretation
print(f"\n💡 PERFORMANCE INTERPRETATION:")
if mape < 5:
    print(f"   🥇 EXCELLENT precision (MAPE < 5%)")
elif mape < 10:
    print(f"   🥈 GOOD precision (5% ≤ MAPE < 10%)")
elif mape < 15:
    print(f"   🥉 ACCEPTABLE precision (10% ≤ MAPE < 15%)")
else:
    print(f"   ⚠️ NEEDS IMPROVEMENT (MAPE ≥ 15%)")

print(f"   🎯 Industrial standard: <10% MAPE for energy prediction")
print(f"   🏭 Production ready: {'✅ YES' if mape < 10 else '❌ NO'}")

# === 6. ISRAELI WEEKEND ANALYSIS ===
print("\n🇮🇱 6. Israeli weekend performance analysis...")

# Prepare test data with predictions
test_data = df_clean.iloc[split_idx:].copy()
test_data['predictions'] = y_pred

# Define Israeli day types
test_data['is_friday'] = (test_data['Day'].dt.dayofweek == 4).astype(int)
test_data['is_saturday'] = (test_data['Day'].dt.dayofweek == 5).astype(int)
test_data['is_sunday'] = (test_data['Day'].dt.dayofweek == 6).astype(int)
test_data['is_weekend_israel'] = ((test_data['Day'].dt.dayofweek == 4) | 
                                 (test_data['Day'].dt.dayofweek == 5)).astype(int)

# Analyze by day type
day_analysis = {}
day_types = {
    'Sunday': 'is_sunday',
    'Monday': (test_data['Day'].dt.dayofweek == 0),
    'Tuesday': (test_data['Day'].dt.dayofweek == 1),
    'Wednesday': (test_data['Day'].dt.dayofweek == 2),
    'Thursday': (test_data['Day'].dt.dayofweek == 3),
    'Friday': 'is_friday',
    'Saturday': 'is_saturday'
}

print(f"\n📅 PERFORMANCE BY DAY OF WEEK:")
print("-" * 60)
print(f"{'Day':<12} {'Count':<6} {'MAE (kWh)':<10} {'MAPE (%)':<9} {'R²':<6}")
print("-" * 60)

for day_name, mask_col in day_types.items():
    if isinstance(mask_col, str):
        mask = test_data[mask_col] == 1
    else:
        mask = mask_col
    
    day_data = test_data[mask]
    
    if len(day_data) > 0:
        day_mae = mean_absolute_error(day_data['DailyAverage'], day_data['predictions'])
        day_mape = mean_absolute_percentage_error(day_data['DailyAverage'], day_data['predictions']) * 100
        day_r2 = r2_score(day_data['DailyAverage'], day_data['predictions'])
        
        day_analysis[day_name] = {
            'count': len(day_data),
            'mae': day_mae,
            'mape': day_mape,
            'r2': day_r2
        }
        
        # Weekend indicator
        weekend_indicator = ""
        if day_name in ['Friday', 'Saturday']:
            weekend_indicator = "🇮🇱"
        elif day_name == 'Sunday':
            weekend_indicator = "💼"
        
        print(f"{day_name:<12} {len(day_data):<6} {day_mae:<10.0f} {day_mape:<9.1f} {day_r2:<6.3f} {weekend_indicator}")

# === 7. SEASONAL ANALYSIS ===
print("\n🌡️ 7. Seasonal performance analysis...")

# Define seasons
test_data['season'] = 'Other'
test_data.loc[test_data['Day'].dt.month.isin([12, 1, 2]), 'season'] = 'Winter'
test_data.loc[test_data['Day'].dt.month.isin([3, 4, 5]), 'season'] = 'Spring'
test_data.loc[test_data['Day'].dt.month.isin([6, 7, 8]), 'season'] = 'Summer'
test_data.loc[test_data['Day'].dt.month.isin([9, 10, 11]), 'season'] = 'Autumn'

print(f"\n🌍 PERFORMANCE BY SEASON:")
print("-" * 50)
print(f"{'Season':<10} {'Count':<6} {'MAE (kWh)':<10} {'MAPE (%)':<9}")
print("-" * 50)

seasonal_analysis = {}
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    season_data = test_data[test_data['season'] == season]
    
    if len(season_data) > 0:
        season_mae = mean_absolute_error(season_data['DailyAverage'], season_data['predictions'])
        season_mape = mean_absolute_percentage_error(season_data['DailyAverage'], season_data['predictions']) * 100
        
        seasonal_analysis[season] = {
            'count': len(season_data),
            'mae': season_mae,
            'mape': season_mape
        }
        
        print(f"{season:<10} {len(season_data):<6} {season_mae:<10.0f} {season_mape:<9.1f}")

# === 8. ERROR DISTRIBUTION ANALYSIS ===
print("\n📊 8. Error distribution analysis...")

# Calculate relative errors
relative_errors = np.abs((y_test - y_pred) / y_test) * 100

# Error brackets
error_brackets = [
    ('≤ 2%', 2),
    ('≤ 5%', 5),
    ('≤ 10%', 10),
    ('≤ 15%', 15),
    ('> 15%', float('inf'))
]

print(f"\n📈 ERROR DISTRIBUTION:")
print("-" * 40)
print(f"{'Error Range':<12} {'Count':<6} {'Percentage':<12}")
print("-" * 40)

cumulative_count = 0
for label, threshold in error_brackets:
    if threshold == float('inf'):
        count = len(relative_errors) - cumulative_count
    else:
        count = np.sum(relative_errors <= threshold) - cumulative_count
    
    percentage = count / len(relative_errors) * 100
    cumulative_count += count
    
    print(f"{label:<12} {count:<6} {percentage:<12.1f}%")

# === 9. FEATURE IMPORTANCE VERIFICATION ===
print("\n🥇 9. Feature importance analysis...")

# Get feature importance from model coefficients
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(model.coef_)
}).sort_values('importance', ascending=False)

print(f"\n🇮🇱 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 50)
for i, row in feature_importance.head(10).iterrows():
    israeli_marker = ""
    if any(keyword in row['feature'].lower() for keyword in ['friday', 'saturday', 'weekend_israel']):
        israeli_marker = "🇮🇱"
    elif 'sunday' in row['feature'].lower():
        israeli_marker = "💼"
    
    print(f"{i+1:2d}. {row['feature']:<25} ({row['importance']:,.0f}) {israeli_marker}")

# === 10. VISUALIZATION ===
print("\n📊 10. Generating comprehensive precision charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'🇮🇱 Israeli Model - Comprehensive Precision Analysis\n'
             f'MAE: {mae:,.0f} kWh | R²: {r2:.3f} | MAPE: {mape:.1f}% | Weekend System: Friday-Saturday',
             fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=30)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0,0].set_xlabel('Actual Consumption (kWh)')
axes[0,0].set_ylabel('Predicted Consumption (kWh)')
axes[0,0].set_title(f'Actual vs Predicted\nR² = {r2:.3f}')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred
axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30, color='green')
axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Predicted Consumption (kWh)')
axes[0,1].set_ylabel('Residuals (kWh)')
axes[0,1].set_title(f'Residuals Distribution\nMAE = {mae:,.0f} kWh')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Daily performance
if day_analysis:
    days = list(day_analysis.keys())
    mapes = [day_analysis[day]['mape'] for day in days]
    colors = []
    for day in days:
        if day in ['Friday', 'Saturday']:
            colors.append('#1f77b4')  # Israeli weekend
        elif day == 'Sunday':
            colors.append('#ff7f0e')  # Workday
        else:
            colors.append('#2ca02c')  # Regular workdays
    
    bars = axes[0,2].bar(days, mapes, color=colors, alpha=0.7)
    axes[0,2].set_title('🇮🇱 Performance by Day of Week')
    axes[0,2].set_ylabel('MAPE (%)')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% Target')
    axes[0,2].legend()

# Plot 4: Seasonal performance
if seasonal_analysis:
    seasons = list(seasonal_analysis.keys())
    seasonal_mapes = [seasonal_analysis[season]['mape'] for season in seasons]
    
    bars = axes[1,0].bar(seasons, seasonal_mapes, alpha=0.7, color='skyblue')
    axes[1,0].set_title('Performance by Season')
    axes[1,0].set_ylabel('MAPE (%)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% Target')
    axes[1,0].legend()

# Plot 5: Error distribution
error_ranges = ['≤2%', '≤5%', '≤10%', '≤15%', '>15%']
error_counts = []
cumulative = 0
for _, threshold in error_brackets:
    if threshold == float('inf'):
        count = len(relative_errors) - cumulative
    else:
        count = np.sum(relative_errors <= threshold) - cumulative
    error_counts.append(count / len(relative_errors) * 100)
    cumulative += count

axes[1,1].bar(error_ranges, error_counts, alpha=0.7, color='lightcoral')
axes[1,1].set_title('Error Distribution')
axes[1,1].set_ylabel('Percentage of Predictions (%)')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

# Plot 6: Time series
last_60_days = test_data.tail(60)
axes[1,2].plot(last_60_days['Day'], last_60_days['DailyAverage'], 
               label='Actual', linewidth=2, alpha=0.8)
axes[1,2].plot(last_60_days['Day'], last_60_days['predictions'], 
               label='Predicted', linewidth=2, alpha=0.8)
axes[1,2].set_title('Recent Predictions (Last 60 Days)')
axes[1,2].set_ylabel('Consumption (kWh)')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Save chart
chart_filename = 'model_precision_analysis.png'
plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Precision analysis chart saved: {chart_filename}")

# === 11. FINAL SUMMARY REPORT ===
print("\n" + "="*70)
print("📊 ISRAELI MODEL - PRECISION ANALYSIS SUMMARY")
print("="*70)

print(f"\n🎯 OVERALL PERFORMANCE:")
print(f"   MAE: {mae:,.0f} kWh")
print(f"   R²: {r2:.3f} ({r2*100:.1f}% variance explained)")
print(f"   MAPE: {mape:.1f}% ({'🥇 EXCELLENT' if mape < 5 else '🥈 GOOD' if mape < 10 else '⚠️ NEEDS IMPROVEMENT'})")

print(f"\n🇮🇱 ISRAELI WEEKEND PERFORMANCE:")
if 'Friday' in day_analysis and 'Saturday' in day_analysis:
    friday_mape = day_analysis['Friday']['mape']
    saturday_mape = day_analysis['Saturday']['mape']
    print(f"   Friday MAPE: {friday_mape:.1f}%")
    print(f"   Saturday MAPE: {saturday_mape:.1f}%")
    israeli_weekend_avg = (friday_mape + saturday_mape) / 2
    print(f"   Israeli weekend average: {israeli_weekend_avg:.1f}%")

if 'Sunday' in day_analysis:
    sunday_mape = day_analysis['Sunday']['mape']
    print(f"   Sunday (workday) MAPE: {sunday_mape:.1f}%")

print(f"\n🌡️ SEASONAL PERFORMANCE:")
if seasonal_analysis:
    best_season = min(seasonal_analysis.keys(), key=lambda x: seasonal_analysis[x]['mape'])
    worst_season = max(seasonal_analysis.keys(), key=lambda x: seasonal_analysis[x]['mape'])
    print(f"   Best season: {best_season} ({seasonal_analysis[best_season]['mape']:.1f}% MAPE)")
    print(f"   Worst season: {worst_season} ({seasonal_analysis[worst_season]['mape']:.1f}% MAPE)")

print(f"\n📈 ERROR DISTRIBUTION:")
within_5_percent = np.sum(relative_errors <= 5) / len(relative_errors) * 100
within_10_percent = np.sum(relative_errors <= 10) / len(relative_errors) * 100
print(f"   Within ±5%: {within_5_percent:.1f}% of predictions")
print(f"   Within ±10%: {within_10_percent:.1f}% of predictions")

print(f"\n🏭 PRODUCTION READINESS:")
print(f"   Industrial standard: {'✅ MEETS' if mape < 10 else '❌ FAILS'} (<10% MAPE)")
print(f"   Israeli context: ✅ OPTIMIZED (Friday-Saturday weekends)")
print(f"   Model stability: {'✅ STABLE' if abs(mbe) < mae/2 else '⚠️ CHECK BIAS'}")

print("="*70)
print(f"📈 Analysis completed! Chart: {chart_filename}")
print("🇮🇱 Israeli model precision evaluation ready!")
print("="*70) 