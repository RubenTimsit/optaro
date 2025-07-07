import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("🇮🇱 ISRAELI WEEKEND PATTERNS - DIAGNOSTIC ANALYSIS")
print("=" * 70)
print("🎯 Objective: Analyze consumption differences between global vs Israeli weekends")
print("🌍 Global weekends: Saturday-Sunday | 🇮🇱 Israeli weekends: Friday-Saturday")
print("=" * 70)

# Load data
try:
    df = pd.read_csv("data_with_context_fixed.csv")
    df['Day'] = pd.to_datetime(df['Day'])
    df = df.sort_values('Day').reset_index(drop=True)
    print(f"✅ Data loaded: {len(df)} days")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Define weekend systems
df['is_friday'] = (df['Day'].dt.dayofweek == 4).astype(int)
df['is_saturday'] = (df['Day'].dt.dayofweek == 5).astype(int)
df['is_sunday'] = (df['Day'].dt.dayofweek == 6).astype(int)

df['is_weekend_global'] = ((df['Day'].dt.dayofweek == 5) | (df['Day'].dt.dayofweek == 6)).astype(int)
df['is_weekend_israel'] = ((df['Day'].dt.dayofweek == 4) | (df['Day'].dt.dayofweek == 5)).astype(int)

# Analyze consumption
friday_avg = df[df['is_friday'] == 1]['DailyAverage'].mean()
saturday_avg = df[df['is_saturday'] == 1]['DailyAverage'].mean()
sunday_avg = df[df['is_sunday'] == 1]['DailyAverage'].mean()

print(f"\n📊 CONSUMPTION ANALYSIS:")
print(f"Friday:   {friday_avg:,.0f} kWh (🇮🇱 Weekend start)")
print(f"Saturday: {saturday_avg:,.0f} kWh (🇮🇱🌍 Both systems)")
print(f"Sunday:   {sunday_avg:,.0f} kWh (🌍 Global weekend)")

difference = saturday_avg - sunday_avg
print(f"\n🔍 Saturday-Sunday difference: {difference:+,.0f} kWh")

if abs(difference) > 5000:
    print("⚠️ SIGNIFICANT DIFFERENCE - Israeli weekend pattern detected!")
    print("🇮🇱 Recommendation: Use Israeli weekend model (Friday-Saturday)")
else:
    print("ℹ️  Minor difference - further analysis recommended")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🇮🇱 Israeli vs Global Weekend Analysis', fontsize=14)

# Daily patterns
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_avgs = []
for i in range(7):
    day_data = df[df['Day'].dt.dayofweek == i]['DailyAverage']
    day_avgs.append(day_data.mean() if len(day_data) > 0 else 0)

colors = ['green', 'green', 'green', 'green', 'blue', 'purple', 'orange']
axes[0,0].bar(days, day_avgs, color=colors, alpha=0.7)
axes[0,0].set_title('Daily Consumption Patterns')
axes[0,0].set_ylabel('Average Consumption (kWh)')

# Weekend comparison
global_weekend_avg = df[df['is_weekend_global'] == 1]['DailyAverage'].mean()
israel_weekend_avg = df[df['is_weekend_israel'] == 1]['DailyAverage'].mean()

axes[0,1].bar(['Global\n(Sat-Sun)', 'Israeli\n(Fri-Sat)'], 
              [global_weekend_avg, israel_weekend_avg], 
              color=['orange', 'blue'], alpha=0.7)
axes[0,1].set_title('Weekend Systems Comparison')
axes[0,1].set_ylabel('Average Consumption (kWh)')

# Critical analysis
critical_days = ['Friday', 'Saturday', 'Sunday']
critical_avgs = [friday_avg, saturday_avg, sunday_avg]
critical_colors = ['blue', 'purple', 'orange']

axes[1,0].bar(critical_days, critical_avgs, color=critical_colors, alpha=0.7)
axes[1,0].set_title('Critical Analysis: Fri-Sat-Sun')
axes[1,0].set_ylabel('Average Consumption (kWh)')
axes[1,0].text(1, max(critical_avgs) * 0.9, f'Sat-Sun gap:\n{difference:+,.0f} kWh', 
               ha='center', fontsize=10, bbox=dict(boxstyle="round", facecolor="yellow"))

# Summary
summary_text = f"""DIAGNOSTIC RESULTS:
• Saturday avg: {saturday_avg:,.0f} kWh
• Sunday avg: {sunday_avg:,.0f} kWh  
• Difference: {difference:+,.0f} kWh

RECOMMENDATION:
{"Israeli weekend model (Fri-Sat)" if abs(difference) > 5000 else "Further analysis needed"}"""

axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes, 
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
axes[1,1].set_title('Diagnostic Summary')
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig('israel_weekend_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Analysis complete! Chart saved.")
print(f"🇮🇱 Recommendation: {'Use Israeli weekend model' if abs(difference) > 5000 else 'Further analysis needed'}")
