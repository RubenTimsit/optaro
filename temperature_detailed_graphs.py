import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12

print("📊 DETAILED TEMPERATURE-CONSUMPTION CHARTS GENERATION")
print("=" * 70)

# === DATA LOADING AND PREPARATION ===
def create_israeli_features(df):
    df = df.copy()
    df['temp_squared'] = df['TempAvg'] ** 2
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - 25.0)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - 30.0)
    df['heating_needs'] = np.maximum(0, 25.0 - df['TempAvg'])
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_spring'] = ((df['Day'].dt.month >= 3) & (df['Day'].dt.month <= 5)).astype(int)
    df['is_autumn'] = ((df['Day'].dt.month >= 9) & (df['Day'].dt.month <= 11)).astype(int)
    df['is_weekend_israel'] = ((df['Day'].dt.dayofweek == 4) | (df['Day'].dt.dayofweek == 5)).astype(int)
    return df

# Loading
df = pd.read_csv("data_with_context_fixed.csv")
df['Day'] = pd.to_datetime(df['Day'])
df = df.sort_values('Day').reset_index(drop=True)
df_analysis = create_israeli_features(df)

print(f"✅ Data prepared: {len(df_analysis)} days")

# === CHART 1: MAIN TEMPERATURE-CONSUMPTION RELATIONSHIP ===
print("📊 1. Main relationship chart...")

plt.figure(figsize=(14, 10))

# Polynomial regression
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(df_analysis['TempAvg'].values.reshape(-1, 1))
poly_reg = LinearRegression()
poly_reg.fit(X_poly, df_analysis['DailyAverage'])
temp_range = np.linspace(df_analysis['TempAvg'].min(), df_analysis['TempAvg'].max(), 100)
X_poly_pred = poly_features.transform(temp_range.reshape(-1, 1))
y_poly_pred = poly_reg.predict(X_poly_pred)

# Scatter plot with Israeli weekends
weekends = df_analysis[df_analysis['is_weekend_israel'] == 1]
workdays = df_analysis[df_analysis['is_weekend_israel'] == 0]

plt.scatter(workdays['TempAvg'], workdays['DailyAverage'], 
           alpha=0.6, s=40, color='steelblue', label='Workdays', edgecolor='white', linewidth=0.5)
plt.scatter(weekends['TempAvg'], weekends['DailyAverage'], 
           alpha=0.8, s=40, color='orangered', label='🇮🇱 Israeli weekends', edgecolor='white', linewidth=0.5)

# Regression curve
plt.plot(temp_range, y_poly_pred, 'red', linewidth=3, label='Polynomial regression', alpha=0.9)

# Temperature zones
plt.axvspan(7, 20, alpha=0.15, color='lightblue', label='Heating zone')
plt.axvspan(22, 26, alpha=0.15, color='lightgreen', label='Comfort zone')
plt.axvspan(28, 31, alpha=0.15, color='lightcoral', label='Intensive cooling zone')

plt.xlabel('Average Temperature (°C)', fontsize=14, fontweight='bold')
plt.ylabel('Energy Consumption (kWh)', fontsize=14, fontweight='bold')
plt.title('🌡️ TEMPERATURE-ENERGY CONSUMPTION RELATIONSHIP\n🇮🇱 Israeli Industry with Friday-Saturday Weekends', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

# Statistics on chart
corr = df_analysis['TempAvg'].corr(df_analysis['DailyAverage'])
plt.text(0.02, 0.98, f'Correlation: {corr:+.3f}\nPolynomial R²: {poly_reg.score(X_poly, df_analysis["DailyAverage"]):.3f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('temperature_main_relation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: temperature_main_relation.png")

# === CHART 2: DETAILED SEASONAL ANALYSIS ===
print("📊 2. Seasonal analysis chart...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🌍 SEASONAL TEMPERATURE-CONSUMPTION ANALYSIS', fontsize=16, fontweight='bold')

seasons = [
    ('is_winter', 'Winter', 'lightblue', ax1),
    ('is_spring', 'Spring', 'lightgreen', ax2),
    ('is_summer', 'Summer', 'orange', ax3),
    ('is_autumn', 'Autumn', 'brown', ax4)
]

for season_col, season_name, color, ax in seasons:
    season_data = df_analysis[df_analysis[season_col] == 1]
    
    if len(season_data) > 0:
        # Scatter plot
        ax.scatter(season_data['TempAvg'], season_data['DailyAverage'], 
                  alpha=0.7, s=30, color=color, edgecolor='white', linewidth=0.5)
        
        # Linear regression
        z = np.polyfit(season_data['TempAvg'], season_data['DailyAverage'], 1)
        p = np.poly1d(z)
        ax.plot(season_data['TempAvg'].sort_values(), p(season_data['TempAvg'].sort_values()), 
                'red', linewidth=2, alpha=0.8)
        
        # Statistics
        corr = season_data['TempAvg'].corr(season_data['DailyAverage'])
        temp_mean = season_data['TempAvg'].mean()
        cons_mean = season_data['DailyAverage'].mean()
        
        ax.set_title(f'{season_name}\nCorr: {corr:+.3f} | Avg Temp: {temp_mean:.1f}°C | Avg Cons: {cons_mean:,.0f} kWh')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Consumption (kWh)')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_seasonal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: temperature_seasonal_analysis.png")

# === CHART 3: TEMPERATURE THRESHOLDS AND ENERGY NEEDS ===
print("📊 3. Critical thresholds chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('🌡️ CRITICAL TEMPERATURE THRESHOLDS', fontsize=16, fontweight='bold')

# Average chart by thresholds
temp_bins = [df_analysis['TempAvg'].min()-1, 15, 20, 25, 28, 30, df_analysis['TempAvg'].max()+1]
temp_labels = ['<15°C\n(Cold)', '15-20°C\n(Cool)', '20-25°C\n(Moderate)', '25-28°C\n(Warm)', '28-30°C\n(Hot)', '>30°C\n(Extreme)']
df_analysis['temp_category'] = pd.cut(df_analysis['TempAvg'], bins=temp_bins, labels=temp_labels)

temp_stats = []
counts = []
for cat in temp_labels:
    data = df_analysis[df_analysis['temp_category'] == cat]['DailyAverage']
    if len(data) > 0:
        temp_stats.append(data.mean())
        counts.append(len(data))
    else:
        temp_stats.append(0)
        counts.append(0)

colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
bars = ax1.bar(temp_labels, temp_stats, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.axhline(y=df_analysis['DailyAverage'].mean(), color='black', linestyle='--', linewidth=2,
           label=f'Overall average: {df_analysis["DailyAverage"].mean():,.0f} kWh')
ax1.set_ylabel('Average Consumption (kWh)', fontweight='bold')
ax1.set_title('Consumption by Temperature Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add values and counters
for bar, value, count in zip(bars, temp_stats, counts):
    if value > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f'{value:,.0f} kWh\n({count} days)', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)

# Energy needs chart
categories = ['Heating\n(<25°C)', 'Comfort Zone\n(22-26°C)', 'Cooling\n(>25°C)', 'Heavy Cooling\n(>30°C)']
heating_days = df_analysis[df_analysis['heating_needs'] > 0]
comfort_days = df_analysis[(df_analysis['TempAvg'] >= 22) & (df_analysis['TempAvg'] <= 26)]
cooling_days = df_analysis[df_analysis['cooling_needs_light'] > 0]
cooling_heavy_days = df_analysis[df_analysis['cooling_needs_heavy'] > 0]

needs_cons = [
    heating_days['DailyAverage'].mean(),
    comfort_days['DailyAverage'].mean(),
    cooling_days['DailyAverage'].mean(),
    cooling_heavy_days['DailyAverage'].mean()
]

needs_counts = [len(heating_days), len(comfort_days), len(cooling_days), len(cooling_heavy_days)]

colors2 = ['lightblue', 'lightgreen', 'orange', 'red']
bars2 = ax2.bar(categories, needs_cons, color=colors2, alpha=0.8, edgecolor='black', linewidth=1)
ax2.axhline(y=df_analysis['DailyAverage'].mean(), color='black', linestyle='--', linewidth=2)
ax2.set_ylabel('Average Consumption (kWh)', fontweight='bold')
ax2.set_title('Energy Needs by Type')
ax2.grid(True, alpha=0.3)

for bar, value, count in zip(bars2, needs_cons, needs_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f'{value:,.0f} kWh\n({count} days)', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('temperature_thresholds_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: temperature_thresholds_analysis.png")

# === CHART 4: ISRAELI WEEKENDS COMPARISON ===
print("📊 4. Israeli weekends chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('🇮🇱 ISRAELI WEEKENDS INFLUENCE ON TEMPERATURE-CONSUMPTION', fontsize=16, fontweight='bold')

# Direct comparison
workdays = df_analysis[df_analysis['is_weekend_israel'] == 0]
weekends = df_analysis[df_analysis['is_weekend_israel'] == 1]

ax1.scatter(workdays['TempAvg'], workdays['DailyAverage'], 
           alpha=0.6, s=35, color='steelblue', label='Workdays', edgecolor='white', linewidth=0.5)
ax1.scatter(weekends['TempAvg'], weekends['DailyAverage'], 
           alpha=0.8, s=35, color='orangered', label='Israeli weekends', edgecolor='white', linewidth=0.5)

# Separate regressions
z1 = np.polyfit(workdays['TempAvg'], workdays['DailyAverage'], 1)
z2 = np.polyfit(weekends['TempAvg'], weekends['DailyAverage'], 1)
p1 = np.poly1d(z1)
p2 = np.poly1d(z2)

ax1.plot(workdays['TempAvg'].sort_values(), p1(workdays['TempAvg'].sort_values()), 
         'blue', linewidth=2, alpha=0.8, label='Workdays trend')
ax1.plot(weekends['TempAvg'].sort_values(), p2(weekends['TempAvg'].sort_values()), 
         'red', linewidth=2, alpha=0.8, label='Weekends trend')

corr_work = workdays['TempAvg'].corr(workdays['DailyAverage'])
corr_weekend = weekends['TempAvg'].corr(weekends['DailyAverage'])

ax1.set_xlabel('Temperature (°C)', fontweight='bold')
ax1.set_ylabel('Consumption (kWh)', fontweight='bold')
ax1.set_title('Direct Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax1.text(0.02, 0.98, f'Correlations:\nWorkdays: {corr_work:+.3f}\nWeekends: {corr_weekend:+.3f}', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Consumption differences by temperature
temp_rounded = np.round(df_analysis['TempAvg'])
work_by_temp = workdays.groupby(np.round(workdays['TempAvg']))['DailyAverage'].mean()
weekend_by_temp = weekends.groupby(np.round(weekends['TempAvg']))['DailyAverage'].mean()

common_temps = set(work_by_temp.index) & set(weekend_by_temp.index)
common_temps = sorted([t for t in common_temps if work_by_temp[t] and weekend_by_temp[t]])

if len(common_temps) > 5:
    work_values = [work_by_temp[t] for t in common_temps]
    weekend_values = [weekend_by_temp[t] for t in common_temps]
    
    ax2.plot(common_temps, work_values, 'o-', color='steelblue', linewidth=2, 
             markersize=8, label='Workdays')
    ax2.plot(common_temps, weekend_values, 'o-', color='orangered', linewidth=2, 
             markersize=8, label='Israeli weekends')
    
    ax2.fill_between(common_temps, work_values, weekend_values, alpha=0.2, color='gray')
    
    ax2.set_xlabel('Temperature (°C)', fontweight='bold')
    ax2.set_ylabel('Average Consumption (kWh)', fontweight='bold')
    ax2.set_title('Consumption Profiles by Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Average difference
    diff_mean = np.mean([work_values[i] - weekend_values[i] for i in range(len(work_values))])
    ax2.text(0.02, 0.98, f'Average difference:\n{diff_mean:+.0f} kWh\n(Workdays - Weekends)', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('temperature_israeli_weekends.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: temperature_israeli_weekends.png")

# === CHART 5: MONTHLY TEMPERATURE PROFILE ===
print("📊 5. Monthly profile chart...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('📅 MONTHLY TEMPERATURE-CONSUMPTION PROFILES', fontsize=16, fontweight='bold')

# Monthly averages
df_analysis['month'] = df_analysis['Day'].dt.month
monthly_stats = df_analysis.groupby('month').agg({
    'TempAvg': ['mean', 'std'],
    'DailyAverage': ['mean', 'std'],
    'Day': 'count'
}).round(1)

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Temperature chart
temp_means = [monthly_stats.loc[i, ('TempAvg', 'mean')] if i in monthly_stats.index else 0 for i in range(1, 13)]
temp_stds = [monthly_stats.loc[i, ('TempAvg', 'std')] if i in monthly_stats.index else 0 for i in range(1, 13)]

ax1.plot(months, temp_means, 'o-', color='red', linewidth=3, markersize=8, label='Average temperature')
ax1.fill_between(months, [temp_means[i] - temp_stds[i] for i in range(12)], 
                 [temp_means[i] + temp_stds[i] for i in range(12)], alpha=0.3, color='red')
ax1.set_ylabel('Temperature (°C)', fontweight='bold', color='red')
ax1.set_title('Annual Temperature Profile')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='red')

# Consumption chart
cons_means = [monthly_stats.loc[i, ('DailyAverage', 'mean')] if i in monthly_stats.index else 0 for i in range(1, 13)]
cons_stds = [monthly_stats.loc[i, ('DailyAverage', 'std')] if i in monthly_stats.index else 0 for i in range(1, 13)]

ax2.plot(months, cons_means, 'o-', color='blue', linewidth=3, markersize=8, label='Average consumption')
ax2.fill_between(months, [cons_means[i] - cons_stds[i] for i in range(12)], 
                 [cons_means[i] + cons_stds[i] for i in range(12)], alpha=0.3, color='blue')
ax2.set_ylabel('Consumption (kWh)', fontweight='bold', color='blue')
ax2.set_xlabel('Month', fontweight='bold')
ax2.set_title('Annual Consumption Profile')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', labelcolor='blue')

plt.tight_layout()
plt.savefig('temperature_monthly_profiles.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: temperature_monthly_profiles.png")

print("\n" + "=" * 70)
print("🎯 DETAILED CHARTS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("📊 Files created:")
print("   1. temperature_main_relation.png - Main relationship")
print("   2. temperature_seasonal_analysis.png - Seasonal analysis")
print("   3. temperature_thresholds_analysis.png - Critical thresholds")
print("   4. temperature_israeli_weekends.png - Israeli weekends")
print("   5. temperature_monthly_profiles.png - Monthly profiles")
print("🇮🇱 All adapted to Israeli context!")
print("=" * 70) 