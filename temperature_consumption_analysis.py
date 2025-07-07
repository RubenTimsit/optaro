import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Chart configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

print("🌡️ TEMPERATURE-CONSUMPTION INFLUENCE ANALYSIS")
print("=" * 70)
print("🎯 Objective: Understand temperature impact on energy consumption")
print("🇮🇱 Context: Israeli industry with Friday-Saturday weekends")
print("=" * 70)

# === 1. DATA LOADING ===
print("\n📊 1. Loading data...")

def create_israeli_features(df):
    """Create Israeli specialized features"""
    df = df.copy()
    
    # Temperature features
    df['temp_squared'] = df['TempAvg'] ** 2
    df['temp_range'] = df['TempMax'] - df['TempMin']
    
    # Heating/cooling needs
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - 25.0)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - 30.0)
    df['heating_needs'] = np.maximum(0, 25.0 - df['TempAvg'])
    
    # Temperature thresholds
    df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
    df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
    df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
    df['temp_below_20'] = (df['TempAvg'] < 20).astype(int)
    
    # Seasons
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_spring'] = ((df['Day'].dt.month >= 3) & (df['Day'].dt.month <= 5)).astype(int)
    df['is_autumn'] = ((df['Day'].dt.month >= 9) & (df['Day'].dt.month <= 11)).astype(int)
    
    # Israeli system
    df['is_friday'] = (df['Day'].dt.dayofweek == 4).astype(int)
    df['is_saturday'] = (df['Day'].dt.dayofweek == 5).astype(int)
    df['is_weekend_israel'] = ((df['Day'].dt.dayofweek == 4) | (df['Day'].dt.dayofweek == 5)).astype(int)
    df['is_workday_israel'] = (df['is_weekend_israel'] == 0).astype(int)
    
    return df

# Loading
try:
    df = pd.read_csv("data_with_context_fixed.csv")
    df['Day'] = pd.to_datetime(df['Day'])
    df = df.sort_values('Day').reset_index(drop=True)
    
    print(f"✅ Data loaded: {len(df)} days")
    print(f"   Period: {df['Day'].min().date()} → {df['Day'].max().date()}")
    print(f"   Temperature: {df['TempAvg'].min():.1f}°C → {df['TempAvg'].max():.1f}°C")
    print(f"   Consumption: {df['DailyAverage'].min():,.0f} → {df['DailyAverage'].max():,.0f} kWh")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Feature creation
df_analysis = create_israeli_features(df)

# === 2. GLOBAL STATISTICAL ANALYSIS ===
print("\n📈 2. Temperature-consumption statistical analysis...")

# Correlations
corr_temp_linear = df_analysis['TempAvg'].corr(df_analysis['DailyAverage'])
corr_temp_squared = df_analysis['temp_squared'].corr(df_analysis['DailyAverage'])

print(f"\n🔗 CORRELATIONS:")
print(f"   Linear temperature:      {corr_temp_linear:+.3f}")
print(f"   Quadratic temperature:   {corr_temp_squared:+.3f}")

# Polynomial regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df_analysis['TempAvg'].values.reshape(-1, 1))
poly_reg = LinearRegression()
poly_reg.fit(X_poly, df_analysis['DailyAverage'])
y_poly_pred = poly_reg.predict(X_poly)
r2_poly = r2_score(df_analysis['DailyAverage'], y_poly_pred)

print(f"   Polynomial regression R²: {r2_poly:.3f}")

# === 3. TEMPERATURE RANGE ANALYSIS ===
print("\n🌡️ 3. Temperature range analysis...")

# Range definition
temp_bins = [df_analysis['TempAvg'].min()-1, 15, 20, 25, 28, 30, df_analysis['TempAvg'].max()+1]
temp_labels = ['<15°C', '15-20°C', '20-25°C', '25-28°C', '28-30°C', '>30°C']
df_analysis['temp_range_cat'] = pd.cut(df_analysis['TempAvg'], bins=temp_bins, labels=temp_labels)

print(f"\n📊 CONSUMPTION BY TEMPERATURE RANGE:")
print("-" * 60)
print(f"{'Range':<10} {'Days':<6} {'Average (kWh)':<15} {'Std Dev':<12}")
print("-" * 60)

temp_stats = {}
for cat in temp_labels:
    data = df_analysis[df_analysis['temp_range_cat'] == cat]['DailyAverage']
    if len(data) > 0:
        temp_stats[cat] = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std()
        }
        print(f"{cat:<10} {len(data):<6} {data.mean():<15,.0f} {data.std():<12,.0f}")

# === 4. SEASONAL ANALYSIS ===
print("\n🌍 4. Seasonal analysis...")

seasons = ['is_winter', 'is_spring', 'is_summer', 'is_autumn']
season_names = ['Winter', 'Spring', 'Summer', 'Autumn']

print(f"\n🌡️ TEMPERATURE-CONSUMPTION RELATIONSHIP BY SEASON:")
print("-" * 70)
print(f"{'Season':<12} {'Days':<6} {'Avg Temp':<10} {'Avg Cons':<12} {'Correlation':<12}")
print("-" * 70)

seasonal_analysis = {}
for season, name in zip(seasons, season_names):
    season_data = df_analysis[df_analysis[season] == 1]
    if len(season_data) > 0:
        temp_mean = season_data['TempAvg'].mean()
        cons_mean = season_data['DailyAverage'].mean()
        correlation = season_data['TempAvg'].corr(season_data['DailyAverage'])
        
        seasonal_analysis[name] = {
            'count': len(season_data),
            'temp_mean': temp_mean,
            'cons_mean': cons_mean,
            'correlation': correlation
        }
        
        print(f"{name:<12} {len(season_data):<6} {temp_mean:<10.1f} {cons_mean:<12,.0f} {correlation:<12.3f}")

# === 5. ISRAELI WEEKENDS ANALYSIS ===
print("\n🇮🇱 5. Israeli weekends influence analysis...")

print(f"\n📅 TEMPERATURE VS CONSUMPTION - ISRAELI WEEKENDS:")
print("-" * 60)

for day_type in ['Workdays', 'Israeli weekends']:
    if day_type == 'Workdays':
        data = df_analysis[df_analysis['is_workday_israel'] == 1]
    else:
        data = df_analysis[df_analysis['is_weekend_israel'] == 1]
    
    if len(data) > 0:
        correlation = data['TempAvg'].corr(data['DailyAverage'])
        temp_mean = data['TempAvg'].mean()
        cons_mean = data['DailyAverage'].mean()
        
        print(f"{day_type}:")
        print(f"   Days: {len(data)}")
        print(f"   Average temperature: {temp_mean:.1f}°C")
        print(f"   Average consumption: {cons_mean:,.0f} kWh")
        print(f"   Correlation: {correlation:+.3f}")
        print()

# === 6. CHART GENERATION ===
print("📊 6. Generating explanatory charts...")

# Main figure configuration
fig = plt.figure(figsize=(20, 24))
fig.suptitle('🌡️ TEMPERATURE-ENERGY CONSUMPTION INFLUENCE ANALYSIS\n🇮🇱 Israeli Industry', 
             fontsize=20, fontweight='bold', y=0.98)

# Chart 1: Main scatter plot
ax1 = plt.subplot(4, 2, 1)
scatter = ax1.scatter(df_analysis['TempAvg'], df_analysis['DailyAverage'], 
                     c=df_analysis['is_weekend_israel'], cmap='viridis', alpha=0.6, s=30)
ax1.plot(np.sort(df_analysis['TempAvg']), 
         y_poly_pred[np.argsort(df_analysis['TempAvg'])], 
         'r-', linewidth=2, label=f'Polynomial regression (R²={r2_poly:.3f})')
ax1.set_xlabel('Average Temperature (°C)')
ax1.set_ylabel('Consumption (kWh)')
ax1.set_title('Temperature-Consumption Relationship\n🟡 Workdays | 🟣 Israeli weekends')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Israeli weekend')

# Chart 2: Box plot by temperature ranges
ax2 = plt.subplot(4, 2, 2)
box_data = [df_analysis[df_analysis['temp_range_cat'] == cat]['DailyAverage'].dropna() 
            for cat in temp_labels]
bp = ax2.boxplot(box_data, labels=temp_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], sns.color_palette("coolwarm", len(temp_labels))):
    patch.set_facecolor(color)
ax2.set_xlabel('Temperature Ranges')
ax2.set_ylabel('Consumption (kWh)')
ax2.set_title('Consumption Distribution by Temperature Range')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Chart 3: Seasonal analysis
ax3 = plt.subplot(4, 2, 3)
for i, (season, name) in enumerate(zip(seasons, season_names)):
    season_data = df_analysis[df_analysis[season] == 1]
    if len(season_data) > 0:
        ax3.scatter(season_data['TempAvg'], season_data['DailyAverage'], 
                   label=f'{name} (r={season_data["TempAvg"].corr(season_data["DailyAverage"]):.3f})',
                   alpha=0.7, s=25)
ax3.set_xlabel('Average Temperature (°C)')
ax3.set_ylabel('Consumption (kWh)')
ax3.set_title('Temperature-Consumption Relationship by Season')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Chart 4: Israeli weekends comparison
ax4 = plt.subplot(4, 2, 4)
workdays = df_analysis[df_analysis['is_workday_israel'] == 1]
weekends = df_analysis[df_analysis['is_weekend_israel'] == 1]
ax4.scatter(workdays['TempAvg'], workdays['DailyAverage'], 
           alpha=0.6, label='Workdays', color='blue', s=25)
ax4.scatter(weekends['TempAvg'], weekends['DailyAverage'], 
           alpha=0.6, label='Israeli weekends', color='red', s=25)
ax4.set_xlabel('Average Temperature (°C)')
ax4.set_ylabel('Consumption (kWh)')
ax4.set_title('🇮🇱 Israeli Weekends Effect on Temperature-Consumption')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Chart 5: Average consumption curve by temperature
ax5 = plt.subplot(4, 2, 5)
temp_rounded = np.round(df_analysis['TempAvg'])
temp_consumption_mean = df_analysis.groupby(temp_rounded)['DailyAverage'].agg(['mean', 'std', 'count'])
temp_valid = temp_consumption_mean[temp_consumption_mean['count'] >= 3]

ax5.plot(temp_valid.index, temp_valid['mean'], 'o-', linewidth=2, markersize=6, color='darkblue')
ax5.fill_between(temp_valid.index, 
                temp_valid['mean'] - temp_valid['std'], 
                temp_valid['mean'] + temp_valid['std'], 
                alpha=0.3, color='lightblue')
ax5.axhline(y=df_analysis['DailyAverage'].mean(), color='red', linestyle='--', 
           label=f'Overall average: {df_analysis["DailyAverage"].mean():,.0f} kWh')
ax5.set_xlabel('Temperature (°C)')
ax5.set_ylabel('Average Consumption (kWh)')
ax5.set_title('Consumption Profile by Temperature\n(±1 std dev)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Chart 6: Temperature thresholds analysis
ax6 = plt.subplot(4, 2, 6)
thresholds = ['temp_below_20', 'temp_above_25', 'temp_above_28', 'temp_above_30']
threshold_names = ['<20°C', '>25°C', '>28°C', '>30°C']
threshold_consumption = []

for threshold in thresholds:
    mask = df_analysis[threshold] == 1
    if mask.sum() > 0:
        threshold_consumption.append(df_analysis[mask]['DailyAverage'].mean())
    else:
        threshold_consumption.append(0)

bars = ax6.bar(threshold_names, threshold_consumption, color=['lightblue', 'orange', 'red', 'darkred'])
ax6.axhline(y=df_analysis['DailyAverage'].mean(), color='black', linestyle='--', 
           label=f'Average: {df_analysis["DailyAverage"].mean():,.0f} kWh')
ax6.set_ylabel('Average Consumption (kWh)')
ax6.set_title('Consumption by Temperature Thresholds')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Add values on bars
for bar, value in zip(bars, threshold_consumption):
    if value > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')

# Chart 7: Consumption variance analysis by temperature
ax7 = plt.subplot(4, 2, 7)
temp_variance = df_analysis.groupby(temp_rounded)['DailyAverage'].var()
temp_variance_valid = temp_variance[temp_consumption_mean['count'] >= 3]

ax7.plot(temp_variance_valid.index, temp_variance_valid.values, 'o-', 
         linewidth=2, markersize=6, color='purple')
ax7.set_xlabel('Temperature (°C)')
ax7.set_ylabel('Consumption Variance')
ax7.set_title('Consumption Variability by Temperature')
ax7.grid(True, alpha=0.3)

# Chart 8: Temperature-month heatmap
ax8 = plt.subplot(4, 2, 8)
df_analysis['month'] = df_analysis['Day'].dt.month
temp_month_pivot = df_analysis.pivot_table(values='DailyAverage', 
                                         index='month', 
                                         columns=pd.cut(df_analysis['TempAvg'], bins=8),
                                         aggfunc='mean')
sns.heatmap(temp_month_pivot, cmap='RdYlBu_r', annot=False, fmt='.0f', 
           cbar_kws={'label': 'Consumption (kWh)'}, ax=ax8)
ax8.set_xlabel('Temperature Ranges')
ax8.set_ylabel('Month')
ax8.set_title('Heatmap Consumption: Month × Temperature')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('temperature_consumption_analysis_en.png', dpi=300, bbox_inches='tight')
print("✅ Main chart saved: temperature_consumption_analysis_en.png")

# === 7. DETAILED THRESHOLD ANALYSIS ===
print("\n🎯 7. Detailed critical thresholds analysis...")

print(f"\n🌡️ TEMPERATURE THRESHOLDS ANALYSIS:")
print("=" * 60)

# Cooling/heating needs
cooling_light_days = df_analysis[df_analysis['cooling_needs_light'] > 0]
cooling_heavy_days = df_analysis[df_analysis['cooling_needs_heavy'] > 0]
heating_days = df_analysis[df_analysis['heating_needs'] > 0]

print(f"📊 ENERGY NEEDS:")
print(f"   Days with light cooling (>25°C): {len(cooling_light_days)} days")
print(f"      → Average consumption: {cooling_light_days['DailyAverage'].mean():,.0f} kWh")
print(f"   Days with heavy cooling (>30°C): {len(cooling_heavy_days)} days")
print(f"      → Average consumption: {cooling_heavy_days['DailyAverage'].mean():,.0f} kWh")
print(f"   Days with heating (<25°C): {len(heating_days)} days")
print(f"      → Average consumption: {heating_days['DailyAverage'].mean():,.0f} kWh")

# Thermal equilibrium point
optimal_temp_range = df_analysis[(df_analysis['TempAvg'] >= 22) & (df_analysis['TempAvg'] <= 26)]
print(f"\n🎯 COMFORT ZONE (22-26°C):")
print(f"   Days: {len(optimal_temp_range)}")
print(f"   Average consumption: {optimal_temp_range['DailyAverage'].mean():,.0f} kWh")
print(f"   Standard deviation: {optimal_temp_range['DailyAverage'].std():,.0f} kWh")

# === 8. EXECUTIVE SUMMARY ===
print("\n" + "=" * 70)
print("📋 EXECUTIVE SUMMARY - TEMPERATURE INFLUENCE")
print("=" * 70)

print(f"\n🔗 KEY CORRELATIONS:")
print(f"   • Linear temperature: {corr_temp_linear:+.3f}")
print(f"   • Quadratic temperature: {corr_temp_squared:+.3f} (better)")
print(f"   • Polynomial regression R²: {r2_poly:.3f}")

print(f"\n🌡️ CRITICAL THRESHOLDS:")
if len(cooling_heavy_days) > 0:
    temp_max_consumption = cooling_heavy_days['DailyAverage'].mean()
    print(f"   • >30°C: Maximum consumption ({temp_max_consumption:,.0f} kWh)")
if len(optimal_temp_range) > 0:
    temp_optimal_consumption = optimal_temp_range['DailyAverage'].mean()
    print(f"   • 22-26°C: Optimal zone ({temp_optimal_consumption:,.0f} kWh)")

print(f"\n🇮🇱 ISRAELI WEEKENDS EFFECT:")
workday_corr = workdays['TempAvg'].corr(workdays['DailyAverage'])
weekend_corr = weekends['TempAvg'].corr(weekends['DailyAverage'])
print(f"   • Workdays: r = {workday_corr:+.3f}")
print(f"   • Israeli weekends: r = {weekend_corr:+.3f}")

print(f"\n🌍 SEASONAL VARIATIONS:")
for name in season_names:
    if name in seasonal_analysis:
        corr = seasonal_analysis[name]['correlation']
        print(f"   • {name}: r = {corr:+.3f}")

print(f"\n📈 RECOMMENDATIONS:")
print(f"   1. Enhanced monitoring for T>28°C (high cooling demand)")
print(f"   2. Energy optimization in 22-26°C zone (stable consumption)")
print(f"   3. Israeli weekends adaptation (different pattern)")
print(f"   4. Seasonal planning (summer = strong correlation)")

print("\n" + "=" * 70)
print("🎯 Complete analysis finished!")
print("📊 Charts: temperature_consumption_analysis_en.png")
print("🇮🇱 Model adapted to Israeli context")
print("=" * 70) 