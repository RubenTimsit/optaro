import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🇮🇱 ISRAELI ENERGY CONSUMPTION - SIMPLE PERIOD COMPARATOR")
print("=" * 70)
print("🎯 Quick comparison between two periods using Israeli optimized model")
print("🗓️  Israeli calendar: Friday-Saturday weekends | Sunday-Thursday workdays")
print("=" * 70)

# === CONFIGURATION ===
print("\n⚙️  CONFIGURATION:")

# Define comparison periods
PERIOD_1_START = "2024-06-01"  # Summer period
PERIOD_1_END = "2024-08-31"
PERIOD_1_NAME = "Summer 2024"

PERIOD_2_START = "2024-12-01"  # Winter period
PERIOD_2_END = "2025-02-28"
PERIOD_2_NAME = "Winter 2024-2025"

print(f"   📅 Period 1: {PERIOD_1_NAME} ({PERIOD_1_START} → {PERIOD_1_END})")
print(f"   📅 Period 2: {PERIOD_2_NAME} ({PERIOD_2_START} → {PERIOD_2_END})")

# === MODEL LOADING ===
print("\n🤖 Loading Israeli optimized model...")

try:
    with open('israel_optimized_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    print("✅ Israeli model loaded successfully")
    print(f"   🇮🇱 Weekend system: {model_data['israel_specifics']['weekend_system']}")
    print(f"   📊 Features: {len(features)} variables")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# === DATA LOADING ===
print("\n📊 Loading historical data...")

try:
    df = pd.read_csv("data_with_israel_temporal_features.csv")
    df['Day'] = pd.to_datetime(df['Day'])
    df = df.sort_values('Day').reset_index(drop=True)
    
    print(f"✅ Data loaded: {len(df)} days")
    print(f"   Period: {df['Day'].min().date()} → {df['Day'].max().date()}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# === PERIOD EXTRACTION ===
print("\n📅 Extracting comparison periods...")

# Period 1
period1_mask = (df['Day'] >= PERIOD_1_START) & (df['Day'] <= PERIOD_1_END)
period1_data = df[period1_mask].copy()

# Period 2
period2_mask = (df['Day'] >= PERIOD_2_START) & (df['Day'] <= PERIOD_2_END)
period2_data = df[period2_mask].copy()

print(f"   Period 1 ({PERIOD_1_NAME}): {len(period1_data)} days")
print(f"   Period 2 ({PERIOD_2_NAME}): {len(period2_data)} days")

if len(period1_data) == 0 or len(period2_data) == 0:
    print("❌ One or both periods have no data")
    exit(1)

# === ISRAELI CALENDAR ANALYSIS ===
print("\n🇮🇱 Israeli calendar analysis...")

def analyze_israeli_calendar(data, period_name):
    """Analyze consumption patterns using Israeli calendar"""
    
    # Israeli day classification
    data['is_weekend_israel'] = ((data['Day'].dt.dayofweek == 4) | 
                                 (data['Day'].dt.dayofweek == 5)).astype(int)  # Fri-Sat
    
    # Statistics
    total_days = len(data)
    israeli_weekends = data['is_weekend_israel'].sum()
    workdays = total_days - israeli_weekends
    
    # Consumption analysis
    avg_consumption = data['DailyAverage'].mean()
    weekend_consumption = data[data['is_weekend_israel'] == 1]['DailyAverage'].mean()
    workday_consumption = data[data['is_weekend_israel'] == 0]['DailyAverage'].mean()
    
    # Weather analysis
    avg_temp = data['TempAvg'].mean()
    min_temp = data['TempAvg'].min()
    max_temp = data['TempAvg'].max()
    
    return {
        'period_name': period_name,
        'total_days': total_days,
        'israeli_weekends': israeli_weekends,
        'workdays': workdays,
        'avg_consumption': avg_consumption,
        'weekend_consumption': weekend_consumption,
        'workday_consumption': workday_consumption,
        'avg_temp': avg_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'weekend_reduction': ((workday_consumption - weekend_consumption) / workday_consumption * 100) if workday_consumption > 0 else 0
    }

# Analyze both periods
analysis1 = analyze_israeli_calendar(period1_data, PERIOD_1_NAME)
analysis2 = analyze_israeli_calendar(period2_data, PERIOD_2_NAME)

# === COMPARISON RESULTS ===
print(f"\n📊 ISRAELI CALENDAR COMPARISON RESULTS:")
print("=" * 80)

print(f"\n🇮🇱 {PERIOD_1_NAME.upper()}:")
print("-" * 40)
print(f"📅 Total days: {analysis1['total_days']}")
print(f"🇮🇱 Israeli weekends (Fri-Sat): {analysis1['israeli_weekends']} days")
print(f"💼 Workdays (Sun-Thu): {analysis1['workdays']} days")
print(f"⚡ Average consumption: {analysis1['avg_consumption']:,.0f} kWh/day")
print(f"🇮🇱 Weekend consumption: {analysis1['weekend_consumption']:,.0f} kWh/day")
print(f"💼 Workday consumption: {analysis1['workday_consumption']:,.0f} kWh/day")
print(f"📉 Weekend reduction: {analysis1['weekend_reduction']:.1f}%")
print(f"🌡️  Average temperature: {analysis1['avg_temp']:.1f}°C ({analysis1['min_temp']:.1f}°C - {analysis1['max_temp']:.1f}°C)")

print(f"\n🇮🇱 {PERIOD_2_NAME.upper()}:")
print("-" * 40)
print(f"📅 Total days: {analysis2['total_days']}")
print(f"🇮🇱 Israeli weekends (Fri-Sat): {analysis2['israeli_weekends']} days")
print(f"💼 Workdays (Sun-Thu): {analysis2['workdays']} days")
print(f"⚡ Average consumption: {analysis2['avg_consumption']:,.0f} kWh/day")
print(f"🇮🇱 Weekend consumption: {analysis2['weekend_consumption']:,.0f} kWh/day")
print(f"💼 Workday consumption: {analysis2['workday_consumption']:,.0f} kWh/day")
print(f"📉 Weekend reduction: {analysis2['weekend_reduction']:.1f}%")
print(f"🌡️  Average temperature: {analysis2['avg_temp']:.1f}°C ({analysis2['min_temp']:.1f}°C - {analysis2['max_temp']:.1f}°C)")

# === COMPARATIVE ANALYSIS ===
print(f"\n🆚 COMPARATIVE ANALYSIS:")
print("=" * 50)

# Consumption differences
consumption_diff = analysis2['avg_consumption'] - analysis1['avg_consumption']
consumption_diff_pct = (consumption_diff / analysis1['avg_consumption']) * 100

weekend_diff = analysis2['weekend_consumption'] - analysis1['weekend_consumption']
weekend_diff_pct = (weekend_diff / analysis1['weekend_consumption']) * 100

workday_diff = analysis2['workday_consumption'] - analysis1['workday_consumption']
workday_diff_pct = (workday_diff / analysis1['workday_consumption']) * 100

# Temperature differences
temp_diff = analysis2['avg_temp'] - analysis1['avg_temp']

print(f"⚡ CONSUMPTION DIFFERENCES:")
print(f"   Overall: {consumption_diff:+,.0f} kWh/day ({consumption_diff_pct:+.1f}%)")
print(f"   🇮🇱 Israeli weekends: {weekend_diff:+,.0f} kWh/day ({weekend_diff_pct:+.1f}%)")
print(f"   💼 Workdays: {workday_diff:+,.0f} kWh/day ({workday_diff_pct:+.1f}%)")

print(f"\n🌡️  TEMPERATURE DIFFERENCE:")
print(f"   Average: {temp_diff:+.1f}°C")

print(f"\n🇮🇱 ISRAELI WEEKEND BEHAVIOR:")
print(f"   {PERIOD_1_NAME} weekend reduction: {analysis1['weekend_reduction']:.1f}%")
print(f"   {PERIOD_2_NAME} weekend reduction: {analysis2['weekend_reduction']:.1f}%")
print(f"   Difference: {analysis2['weekend_reduction'] - analysis1['weekend_reduction']:+.1f} percentage points")

# === BUSINESS INSIGHTS ===
print(f"\n💼 BUSINESS INSIGHTS:")
print("=" * 50)

if abs(consumption_diff_pct) > 10:
    print(f"🚨 SIGNIFICANT consumption difference ({consumption_diff_pct:+.1f}%)")
    if consumption_diff_pct > 0:
        print(f"   {PERIOD_2_NAME} consumes {consumption_diff_pct:.1f}% MORE than {PERIOD_1_NAME}")
    else:
        print(f"   {PERIOD_2_NAME} consumes {abs(consumption_diff_pct):.1f}% LESS than {PERIOD_1_NAME}")
else:
    print(f"✅ Consumption difference is moderate ({consumption_diff_pct:+.1f}%)")

if abs(temp_diff) > 10:
    print(f"🌡️  SIGNIFICANT temperature difference ({temp_diff:+.1f}°C)")
    print(f"   Temperature likely explains consumption variation")
else:
    print(f"🌡️  Moderate temperature difference ({temp_diff:+.1f}°C)")

# Weekend behavior analysis
if abs(analysis2['weekend_reduction'] - analysis1['weekend_reduction']) > 2:
    print(f"🇮🇱 DIFFERENT Israeli weekend behavior between periods")
    if analysis2['weekend_reduction'] > analysis1['weekend_reduction']:
        print(f"   {PERIOD_2_NAME} has STRONGER weekend reduction pattern")
    else:
        print(f"   {PERIOD_1_NAME} has STRONGER weekend reduction pattern")
else:
    print(f"✅ Consistent Israeli weekend behavior across periods")

# === MODEL PREDICTIONS (if we have future dates) ===
print(f"\n🔮 MODEL PREDICTION CAPABILITY:")
print("-" * 40)

# Check if we can make predictions for these periods
available_features = set(features)
period1_features = set(period1_data.columns)
period2_features = set(period2_data.columns)

period1_can_predict = available_features.issubset(period1_features)
period2_can_predict = available_features.issubset(period2_features)

if period1_can_predict and period2_can_predict:
    print(f"✅ Model can predict both periods")
    print(f"   All {len(features)} required features available")
else:
    print(f"⚠️  Model predictions limited:")
    if not period1_can_predict:
        missing1 = available_features - period1_features
        print(f"   {PERIOD_1_NAME}: Missing {len(missing1)} features")
    if not period2_can_predict:
        missing2 = available_features - period2_features
        print(f"   {PERIOD_2_NAME}: Missing {len(missing2)} features")

# === SUMMARY ===
print(f"\n" + "="*70)
print(f"🇮🇱 ISRAELI ENERGY COMPARISON - SUMMARY")
print("="*70)

print(f"📅 Compared periods:")
print(f"   🔵 {PERIOD_1_NAME}: {len(period1_data)} days, avg {analysis1['avg_consumption']:,.0f} kWh/day")
print(f"   🔴 {PERIOD_2_NAME}: {len(period2_data)} days, avg {analysis2['avg_consumption']:,.0f} kWh/day")

print(f"\n🇮🇱 Israeli calendar insights:")
print(f"   Weekend system: Friday-Saturday consistently applied")
print(f"   Weekend reduction: {analysis1['weekend_reduction']:.1f}% vs {analysis2['weekend_reduction']:.1f}%")

print(f"\n📊 Key difference: {consumption_diff:+,.0f} kWh/day ({consumption_diff_pct:+.1f}%)")

if abs(consumption_diff_pct) > 15:
    print(f"   🚨 MAJOR difference - requires investigation")
elif abs(consumption_diff_pct) > 5:
    print(f"   ⚠️  MODERATE difference - monitor closely")
else:
    print(f"   ✅ MINOR difference - normal variation")

print("="*70)
print("🇮🇱 Israeli period comparison completed!")
print("💡 Use 'interactive_period_comparator.py' for custom period analysis")
print("="*70) 