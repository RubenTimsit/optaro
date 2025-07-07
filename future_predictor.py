import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🔮 ISRAELI ENERGY CONSUMPTION - FUTURE PREDICTOR")
print("=" * 70)
print("🎯 Generate future consumption predictions using Israeli optimized model")
print("🇮🇱 Israeli calendar: Friday-Saturday weekends | Sunday-Thursday workdays")
print("=" * 70)

# Configuration
PREDICTION_START = "2025-07-12"
PREDICTION_END = "2025-08-15"
PREDICTION_NAME = "Summer 2025 Forecast"

print(f"\n📅 Prediction period: {PREDICTION_NAME}")
print(f"   From: {PREDICTION_START} → To: {PREDICTION_END}")

# Weather assumptions
WEATHER_ASSUMPTIONS = {
    'avg_temp': 28.5,
    'temp_variation': 5.0,
    'min_temp_base': 22.0,
    'max_temp_base': 35.0,
    'precipitation': 0.1,
    'wind_speed': 12.0,
    'pressure': 1013.0
}

print(f"   🌡️ Weather: {WEATHER_ASSUMPTIONS['avg_temp']}°C average")

# Load model
print("\n🤖 Loading Israeli optimized model...")
try:
    with open('israel_optimized_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    performance = model_data['performance']
    
    print("✅ Israeli model loaded successfully")
    print(f"   🇮🇱 Weekend system: {model_data['israel_specifics']['weekend_system']}")
    print(f"   📊 Features: {len(features)} variables")
    print(f"   📈 Model accuracy: R² = {performance['test_r2']:.3f}, MAPE = {performance['test_mape']:.1f}%")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load historical data for context
print("\n📊 Loading historical data...")
try:
    df_historical = pd.read_csv("data_with_israel_temporal_features.csv")
    df_historical['Day'] = pd.to_datetime(df_historical['Day'])
    df_historical = df_historical.sort_values('Day').reset_index(drop=True)
    
    last_consumption = df_historical['DailyAverage'].iloc[-1]
    last_week_avg = df_historical['DailyAverage'].tail(7).mean()
    
    print(f"✅ Historical data: {len(df_historical)} days")
    print(f"   Last consumption: {last_consumption:,.0f} kWh")
    print(f"   Last week average: {last_week_avg:,.0f} kWh")
    
except Exception as e:
    print(f"❌ Error loading historical data: {e}")
    exit(1)

# Create future dataset
print("\n🔧 Creating future dataset...")

def create_future_dataset(start_date, end_date, weather_assumptions, last_consumption):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_future = pd.DataFrame({'Day': date_range})
    
    print(f"   📅 Created {len(df_future)} future days")
    
    # Simulate weather
    np.random.seed(42)
    base_temp = weather_assumptions['avg_temp']
    temp_variation = weather_assumptions['temp_variation']
    
    day_of_year = df_future['Day'].dt.dayofyear
    seasonal_factor = 0.8 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
    
    df_future['TempAvg'] = (base_temp + seasonal_factor + 
                           np.random.normal(0, temp_variation/3, len(df_future)))
    df_future['TempMin'] = df_future['TempAvg'] - 3 - np.random.uniform(0, 2, len(df_future))
    df_future['TempMax'] = df_future['TempAvg'] + 5 + np.random.uniform(0, 3, len(df_future))
    
    df_future['Precip'] = np.random.exponential(weather_assumptions['precipitation'], len(df_future))
    df_future['WindSpeed'] = np.random.normal(weather_assumptions['wind_speed'], 2, len(df_future))
    df_future['Pressure'] = np.random.normal(weather_assumptions['pressure'], 5, len(df_future))
    
    # Weather features
    df_future['temp_range'] = df_future['TempMax'] - df_future['TempMin']
    df_future['temp_ma_7'] = df_future['TempAvg'].rolling(window=7, min_periods=1).mean()
    df_future['temp_ma_30'] = df_future['TempAvg'].rolling(window=30, min_periods=1).mean()
    df_future['temp_squared'] = df_future['TempAvg'] ** 2
    
    df_future['precip_ma_7'] = df_future['Precip'].rolling(window=7, min_periods=1).mean()
    df_future['has_rain'] = (df_future['Precip'] > 0).astype(int)
    df_future['wind_ma_7'] = df_future['WindSpeed'].rolling(window=7, min_periods=1).mean()
    df_future['pressure_ma_7'] = df_future['Pressure'].rolling(window=30, min_periods=1).mean()
    
    # Temperature thresholds
    df_future['cooling_needs_light'] = np.maximum(0, df_future['TempAvg'] - 25.0)
    df_future['cooling_needs_heavy'] = np.maximum(0, df_future['TempAvg'] - 30.0)
    df_future['heating_needs'] = np.maximum(0, 25.0 - df_future['TempAvg'])
    
    df_future['temp_above_25'] = (df_future['TempAvg'] > 25).astype(int)
    df_future['temp_above_28'] = (df_future['TempAvg'] > 28).astype(int)
    df_future['temp_above_30'] = (df_future['TempAvg'] > 30).astype(int)
    
    # Seasons
    df_future['is_summer'] = ((df_future['Day'].dt.month >= 6) & (df_future['Day'].dt.month <= 8)).astype(int)
    df_future['is_winter'] = ((df_future['Day'].dt.month == 12) | (df_future['Day'].dt.month <= 2)).astype(int)
    df_future['is_mid_summer'] = (df_future['Day'].dt.month == 7).astype(int)
    
    # Israeli calendar
    df_future['is_sunday'] = (df_future['Day'].dt.dayofweek == 6).astype(int)
    df_future['is_monday'] = (df_future['Day'].dt.dayofweek == 0).astype(int)
    df_future['is_tuesday'] = (df_future['Day'].dt.dayofweek == 1).astype(int)
    df_future['is_wednesday'] = (df_future['Day'].dt.dayofweek == 2).astype(int)
    df_future['is_thursday'] = (df_future['Day'].dt.dayofweek == 3).astype(int)
    df_future['is_friday'] = (df_future['Day'].dt.dayofweek == 4).astype(int)
    df_future['is_saturday'] = (df_future['Day'].dt.dayofweek == 5).astype(int)
    
    df_future['is_weekend_israel'] = ((df_future['Day'].dt.dayofweek == 4) | 
                                     (df_future['Day'].dt.dayofweek == 5)).astype(int)
    
    df_future['is_holiday'] = 0
    
    # Cyclical features
    df_future['month_sin'] = np.sin(2 * np.pi * df_future['Day'].dt.month / 12)
    df_future['month_cos'] = np.cos(2 * np.pi * df_future['Day'].dt.month / 12)
    df_future['day_of_year_sin'] = np.sin(2 * np.pi * df_future['Day'].dt.dayofyear / 365)
    df_future['day_of_year_cos'] = np.cos(2 * np.pi * df_future['Day'].dt.dayofyear / 365)
    
    # Israeli interactions
    df_future['temp_x_weekend_israel'] = df_future['TempAvg'] * df_future['is_weekend_israel']
    df_future['temp_x_friday'] = df_future['TempAvg'] * df_future['is_friday']
    df_future['temp_x_saturday'] = df_future['TempAvg'] * df_future['is_saturday']
    df_future['temp_x_sunday'] = df_future['TempAvg'] * df_future['is_sunday']
    
    # Other interactions
    df_future['temp_x_summer'] = df_future['TempAvg'] * df_future['is_summer']
    df_future['temp_x_mid_summer'] = df_future['TempAvg'] * df_future['is_mid_summer']
    df_future['temp_squared_x_summer'] = df_future['temp_squared'] * df_future['is_summer']
    df_future['temp_x_wind'] = df_future['TempAvg'] * df_future['WindSpeed']
    df_future['pressure_x_temp'] = df_future['Pressure'] * df_future['TempAvg']
    
    # Temporal features
    reference_date = pd.to_datetime('2022-01-01')
    df_future['time_trend'] = (df_future['Day'] - reference_date).dt.days / 365.25
    
    # Lag features (will be filled during prediction)
    df_future['consumption_lag_1'] = np.nan
    df_future['consumption_lag_7'] = np.nan
    
    # End-of-year features
    df_future['is_december'] = (df_future['Day'].dt.month == 12).astype(int)
    df_future['days_to_new_year'] = 32 - df_future['Day'].dt.day
    df_future['is_end_of_year'] = ((df_future['Day'].dt.month == 12) & 
                                  (df_future['Day'].dt.day >= 15)).astype(int)
    
    print(f"   🇮🇱 Israeli features created: {len([col for col in df_future.columns if col != 'Day'])} variables")
    
    return df_future

df_future = create_future_dataset(PREDICTION_START, PREDICTION_END, WEATHER_ASSUMPTIONS, last_consumption)

# Generate predictions iteratively
print("\n🔮 Generating iterative predictions...")

predictions = []
prediction_dates = []

current_lag_1 = last_consumption
last_7_values = [last_consumption] * 7

for i, row in df_future.iterrows():
    current_row = row.copy()
    
    # Set lag features
    current_row['consumption_lag_1'] = current_lag_1
    current_row['consumption_lag_7'] = np.mean(last_7_values)
    
    try:
        feature_values = [current_row[feature] for feature in features]
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        prediction = model.predict(feature_scaled)[0]
        
        predictions.append(prediction)
        prediction_dates.append(row['Day'])
        
        # Update lags
        current_lag_1 = prediction
        last_7_values = last_7_values[1:] + [prediction]
        
    except Exception as e:
        print(f"⚠️ Error predicting day {row['Day'].date()}: {e}")
        prediction = last_consumption
        predictions.append(prediction)
        prediction_dates.append(row['Day'])

# Create results
df_predictions = pd.DataFrame({
    'Day': prediction_dates,
    'PredictedConsumption': predictions
})

df_predictions['is_weekend_israel'] = ((df_predictions['Day'].dt.dayofweek == 4) | 
                                      (df_predictions['Day'].dt.dayofweek == 5)).astype(int)
df_predictions['day_name'] = df_predictions['Day'].dt.day_name()

print(f"✅ Predictions generated for {len(predictions)} days")

# Analysis
avg_prediction = np.mean(predictions)
min_prediction = np.min(predictions)
max_prediction = np.max(predictions)

print(f"\n🇮🇱 FUTURE CONSUMPTION FORECAST:")
print("=" * 50)
print(f"📅 Period: {PREDICTION_START} → {PREDICTION_END}")
print(f"⚡ Average consumption: {avg_prediction:,.0f} kWh/day")
print(f"📈 Range: {min_prediction:,.0f} - {max_prediction:,.0f} kWh/day")

# Israeli weekend analysis
israeli_weekends = df_predictions[df_predictions['is_weekend_israel'] == 1]
workdays = df_predictions[df_predictions['is_weekend_israel'] == 0]

if len(israeli_weekends) > 0 and len(workdays) > 0:
    weekend_avg = israeli_weekends['PredictedConsumption'].mean()
    workday_avg = workdays['PredictedConsumption'].mean()
    weekend_reduction = ((workday_avg - weekend_avg) / workday_avg) * 100
    
    print(f"\n🇮🇱 ISRAELI WEEKEND PATTERN:")
    print(f"   🇮🇱 Weekends (Fri-Sat): {len(israeli_weekends)} days, avg {weekend_avg:,.0f} kWh")
    print(f"   💼 Workdays (Sun-Thu): {len(workdays)} days, avg {workday_avg:,.0f} kWh")
    print(f"   📉 Weekend reduction: {weekend_reduction:.1f}%")

# Daily breakdown
print(f"\n📅 DAILY BREAKDOWN:")
print("-" * 50)
for _, row in df_predictions.iterrows():
    day_type = "🇮🇱 Weekend" if row['is_weekend_israel'] == 1 else "💼 Workday"
    print(f"{row['Day'].strftime('%Y-%m-%d')} {row['day_name']:<10} "
          f"{row['PredictedConsumption']:,.0f} kWh {day_type}")

# Visualization
print("\n📊 Creating prediction visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'🇮🇱 Israeli Energy Forecast - {PREDICTION_NAME}\n'
             f'Average: {avg_prediction:,.0f} kWh/day | Israeli Model (Fri-Sat weekends)',
             fontsize=14, fontweight='bold')

# Time series
colors = ['blue' if weekend == 0 else 'red' for weekend in df_predictions['is_weekend_israel']]
axes[0,0].scatter(df_predictions['Day'], df_predictions['PredictedConsumption'], 
                  c=colors, alpha=0.7, s=50)
axes[0,0].plot(df_predictions['Day'], df_predictions['PredictedConsumption'], 
               color='gray', alpha=0.5, linewidth=1)
axes[0,0].set_title('Daily Consumption Forecast')
axes[0,0].set_ylabel('Consumption (kWh)')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# Weekend comparison
if len(israeli_weekends) > 0 and len(workdays) > 0:
    comparison_data = ['Workdays\n(Sun-Thu)', 'Israeli Weekends\n(Fri-Sat)']
    comparison_values = [workday_avg, weekend_avg]
    
    bars = axes[0,1].bar(comparison_data, comparison_values, 
                        color=['blue', 'red'], alpha=0.7)
    axes[0,1].set_title('🇮🇱 Israeli Calendar Comparison')
    axes[0,1].set_ylabel('Average Consumption (kWh)')
    axes[0,1].grid(True, alpha=0.3)

# Distribution
axes[1,0].hist(predictions, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,0].axvline(avg_prediction, color='red', linestyle='--', linewidth=2, 
                  label=f'Average: {avg_prediction:,.0f} kWh')
axes[1,0].set_title('Consumption Distribution')
axes[1,0].set_xlabel('Consumption (kWh)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Weekly pattern
df_predictions['weekday'] = df_predictions['Day'].dt.day_name()
weekday_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
weekday_avgs = []
weekday_colors = []

for day in weekday_order:
    day_data = df_predictions[df_predictions['weekday'] == day]['PredictedConsumption']
    if len(day_data) > 0:
        weekday_avgs.append(day_data.mean())
        weekday_colors.append('red' if day in ['Friday', 'Saturday'] else 'blue')
    else:
        weekday_avgs.append(0)
        weekday_colors.append('gray')

bars = axes[1,1].bar(weekday_order, weekday_avgs, color=weekday_colors, alpha=0.7)
axes[1,1].set_title('🇮🇱 Weekly Pattern (Israeli Calendar)')
axes[1,1].set_ylabel('Average Consumption (kWh)')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()

chart_filename = f'future_predictions_{PREDICTION_START.replace("-", "")}_{PREDICTION_END.replace("-", "")}.png'
plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Prediction chart saved: {chart_filename}")

# Export results
csv_filename = f'predictions_{PREDICTION_START.replace("-", "")}_{PREDICTION_END.replace("-", "")}.csv'
df_predictions.to_csv(csv_filename, index=False)
print(f"✅ Predictions exported to: {csv_filename}")

# Summary
print(f"\n" + "="*70)
print(f"🔮 ISRAELI FUTURE PREDICTION - SUMMARY")
print("="*70)
print(f"📅 Forecast: {PREDICTION_START} → {PREDICTION_END}")
print(f"🇮🇱 Israeli calendar: Friday-Saturday weekends applied")
print(f"⚡ Average: {avg_prediction:,.0f} kWh/day")
print(f"📊 Total: {sum(predictions):,.0f} kWh")
if len(israeli_weekends) > 0 and len(workdays) > 0:
    print(f"🇮🇱 Weekend reduction: {weekend_reduction:.1f}%")
print(f"🎯 Model confidence: {performance['test_mape']:.1f}% MAPE")
print("="*70)
print("🔮 Future prediction completed!")
print("🇮🇱 Ready for Israeli industrial energy planning!")
print("="*70)
