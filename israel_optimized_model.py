import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")

print("🇮🇱 ISRAELI OPTIMIZED ENERGY PREDICTION MODEL")
print("=" * 70)

# === 1. DATA LOADING ===
print("\n📊 1. Loading historical data...")

try:
    df = pd.read_csv("data_with_context_fixed.csv")
    df['Day'] = pd.to_datetime(df['Day'])
    df = df.sort_values('Day').reset_index(drop=True)
    
    print(f"✅ Data loaded successfully: {len(df)} days")
    print(f"   Period: {df['Day'].min().date()} → {df['Day'].max().date()}")
    print(f"   Average consumption: {df['DailyAverage'].mean():,.0f} kWh/day")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# === 2. ISRAELI FEATURE ENGINEERING ===
print("\n🇮🇱 2. Creating Israeli specialized features...")

def create_israeli_features(df):
    """Create features optimized for Israeli operational context"""
    df = df.copy()
    
    # === WEATHER FEATURES ===
    df['temp_range'] = df['TempMax'] - df['TempMin']
    df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
    df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
    df['temp_squared'] = df['TempAvg'] ** 2
    
    df['precip_ma_7'] = df['Precip'].rolling(window=7, min_periods=1).mean()
    df['has_rain'] = (df['Precip'] > 0).astype(int)
    df['wind_ma_7'] = df['WindSpeed'].rolling(window=7, min_periods=1).mean()
    df['pressure_ma_7'] = df['Pressure'].rolling(window=30, min_periods=1).mean()
    
    # === TEMPERATURE THRESHOLDS ===
    df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - 25.0)
    df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - 30.0)
    df['heating_needs'] = np.maximum(0, 25.0 - df['TempAvg'])
    
    df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
    df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
    df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
    
    # === SEASONS ===
    df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
    df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
    df['is_mid_summer'] = (df['Day'].dt.month == 7).astype(int)
    
    # === ISRAELI WEEKDAY SYSTEM ===
    df['is_sunday'] = (df['Day'].dt.dayofweek == 6).astype(int)      # Sunday = workday in Israel
    df['is_monday'] = (df['Day'].dt.dayofweek == 0).astype(int)
    df['is_tuesday'] = (df['Day'].dt.dayofweek == 1).astype(int)
    df['is_wednesday'] = (df['Day'].dt.dayofweek == 2).astype(int)
    df['is_thursday'] = (df['Day'].dt.dayofweek == 3).astype(int)
    df['is_friday'] = (df['Day'].dt.dayofweek == 4).astype(int)      # Friday = weekend in Israel
    df['is_saturday'] = (df['Day'].dt.dayofweek == 5).astype(int)    # Saturday = weekend in Israel
    
    # === ISRAELI WEEKENDS (FRIDAY-SATURDAY) ===
    df['is_weekend_israel'] = ((df['Day'].dt.dayofweek == 4) | (df['Day'].dt.dayofweek == 5)).astype(int)
    
    # === ISRAELI HOLIDAYS ===
    df['is_holiday'] = 0  # Simplified for this version
    
    # === CYCLICAL FEATURES ===
    df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
    
    # === ISRAELI CULTURAL INTERACTIONS ===
    df['temp_x_weekend_israel'] = df['TempAvg'] * df['is_weekend_israel']
    df['temp_x_friday'] = df['TempAvg'] * df['is_friday']
    df['temp_x_saturday'] = df['TempAvg'] * df['is_saturday']
    df['temp_x_sunday'] = df['TempAvg'] * df['is_sunday']
    
    # === OTHER INTERACTIONS ===
    df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
    df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
    df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
    df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
    df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
    
    # === TEMPORAL FEATURES ===
    reference_date = pd.to_datetime('2022-01-01')
    df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
    
    # === LAG FEATURES ===
    df['consumption_lag_1'] = df['DailyAverage'].shift(1)
    df['consumption_lag_7'] = df['DailyAverage'].shift(7)
    
    # === END-OF-YEAR FEATURES ===
    df['is_december'] = (df['Day'].dt.month == 12).astype(int)
    df['days_to_new_year'] = 32 - df['Day'].dt.day
    df['is_end_of_year'] = ((df['Day'].dt.month == 12) & (df['Day'].dt.day >= 15)).astype(int)
    
    return df

# Create Israeli features
df_features = create_israeli_features(df)

# Select feature columns (exclude date and target)
feature_columns = [col for col in df_features.columns 
                  if col not in ['Day', 'DailyAverage'] and not col.startswith('Temp')]

print(f"✅ Israeli features created: {len(feature_columns)} variables")
print(f"   🇮🇱 Key Israeli features: is_weekend_israel, is_friday, is_saturday, is_sunday")
print(f"   📊 Cultural interactions: temp_x_weekend_israel, temp_x_friday, temp_x_saturday")

# === 3. DATA PREPARATION ===
print("\n🔧 3. Data preparation and cleaning...")

# Remove rows with NaN (from lag features)
df_clean = df_features.dropna()
print(f"   Data after cleaning: {len(df_clean)} days (lost {len(df_features) - len(df_clean)} days from lags)")

# Prepare features and target
X = df_clean[feature_columns]
y = df_clean['DailyAverage']

print(f"   Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

# === 4. TRAIN/TEST SPLIT ===
print("\n🎯 4. Train/test split with temporal validation...")

# Temporal split (70/30)
split_idx = int(len(df_clean) * 0.7)
split_date = df_clean.iloc[split_idx]['Day']

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"   Training set: {len(X_train)} days (until {split_date.date()})")
print(f"   Test set: {len(X_test)} days (from {split_date.date()})")

# === 5. FEATURE SCALING ===
print("\n⚖️ 5. Feature normalization...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features normalized with StandardScaler")

# === 6. MODEL TRAINING ===
print("\n🤖 6. Training Israeli optimized Ridge Regression model...")

# Train Ridge model with optimal alpha
model = Ridge(alpha=10.0, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Israeli model trained successfully")

# === 7. PREDICTIONS ===
print("\n🔮 7. Generating predictions...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("✅ Predictions generated")

# === 8. PERFORMANCE EVALUATION ===
print("\n📊 8. Israeli model performance evaluation...")

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

print(f"\n🏆 ISRAELI MODEL PERFORMANCE RESULTS:")
print("=" * 60)
print(f"🎯 Training MAE:    {train_mae:,.0f} kWh")
print(f"🎯 Test MAE:        {test_mae:,.0f} kWh    (🇮🇱 Production metric)")
print(f"📈 Training R²:     {train_r2:.3f}")
print(f"📈 Test R²:         {test_r2:.3f}        (🇮🇱 {test_r2*100:.1f}% variance explained)")
print(f"📊 Test MAPE:       {test_mape:.1f}%           (🇮🇱 Industrial precision)")
print(f"⚖️ Overfitting:     {train_r2 - test_r2:+.3f}       (🇮🇱 Generalization control)")

# === 9. FEATURE IMPORTANCE ANALYSIS ===
print("\n🥇 9. Feature importance analysis...")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': np.abs(model.coef_)
}).sort_values('importance', ascending=False)

print(f"\n🇮🇱 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 50)
for i, row in feature_importance.head(10).iterrows():
    print(f"{row.name+1:2d}. {row['feature']:<25} ({row['importance']:,.0f})")

# === 10. ISRAELI WEEKEND ANALYSIS ===
print("\n🇮🇱 10. Israeli weekend pattern analysis...")

# Analyze predictions by Israeli day type
test_data = df_clean.iloc[split_idx:].copy()
test_data['predictions'] = y_test_pred

# Weekend analysis
israeli_weekend_mask = test_data['is_weekend_israel'] == 1
friday_mask = test_data['is_friday'] == 1
saturday_mask = test_data['is_saturday'] == 1
sunday_mask = test_data['is_sunday'] == 1

print(f"\n📅 ISRAELI WEEKEND PERFORMANCE:")
print("-" * 40)

if israeli_weekend_mask.sum() > 0:
    weekend_mae = mean_absolute_error(
        test_data[israeli_weekend_mask]['DailyAverage'],
        test_data[israeli_weekend_mask]['predictions']
    )
    weekend_mape = mean_absolute_percentage_error(
        test_data[israeli_weekend_mask]['DailyAverage'],
        test_data[israeli_weekend_mask]['predictions']
    ) * 100
    print(f"🇮🇱 Israeli weekends (Fri-Sat): {weekend_mape:.1f}% MAPE ({weekend_mae:.0f} kWh MAE)")

if friday_mask.sum() > 0:
    friday_mape = mean_absolute_percentage_error(
        test_data[friday_mask]['DailyAverage'],
        test_data[friday_mask]['predictions']
    ) * 100
    print(f"📅 Fridays:                  {friday_mape:.1f}% MAPE")

if saturday_mask.sum() > 0:
    saturday_mape = mean_absolute_percentage_error(
        test_data[saturday_mask]['DailyAverage'],
        test_data[saturday_mask]['predictions']
    ) * 100
    print(f"📅 Saturdays:                {saturday_mape:.1f}% MAPE")

if sunday_mask.sum() > 0:
    sunday_mape = mean_absolute_percentage_error(
        test_data[sunday_mask]['DailyAverage'],
        test_data[sunday_mask]['predictions']
    ) * 100
    print(f"💼 Sundays (workday):        {sunday_mape:.1f}% MAPE")

# === 11. VISUALIZATION ===
print("\n📊 11. Generating validation charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🇮🇱 Israeli Optimized Model - Validation Results\n'
             f'MAE: {test_mae:,.0f} kWh | R²: {test_r2:.3f} | MAPE: {test_mape:.1f}% | Israeli Weekends: Fri-Sat',
             fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0,0].scatter(y_test, y_test_pred, alpha=0.6, color='blue', s=30)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0,0].set_xlabel('Actual Consumption (kWh)')
axes[0,0].set_ylabel('Predicted Consumption (kWh)')
axes[0,0].set_title(f'Actual vs Predicted\nR² = {test_r2:.3f}')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_test_pred
axes[0,1].scatter(y_test_pred, residuals, alpha=0.6, color='green', s=30)
axes[0,1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Predicted Consumption (kWh)')
axes[0,1].set_ylabel('Residuals (kWh)')
axes[0,1].set_title(f'Residuals Analysis\nMAE = {test_mae:,.0f} kWh')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
top_features = feature_importance.head(15)
axes[1,0].barh(range(len(top_features)), top_features['importance'], alpha=0.7)
axes[1,0].set_yticks(range(len(top_features)))
axes[1,0].set_yticklabels(top_features['feature'])
axes[1,0].set_xlabel('Feature Importance (|Coefficient|)')
axes[1,0].set_title('🇮🇱 Top 15 Israeli Features')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Time Series Validation
test_dates = df_clean.iloc[split_idx:]['Day']
axes[1,1].plot(test_dates, y_test.values, label='Actual', linewidth=2, alpha=0.8)
axes[1,1].plot(test_dates, y_test_pred, label='Predicted', linewidth=2, alpha=0.8)
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Consumption (kWh)')
axes[1,1].set_title('Time Series Validation')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Save chart
chart_filename = 'israel_optimized_model_validation.png'
plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Validation chart saved: {chart_filename}")

# === 12. MODEL EXPORT ===
print("\n💾 12. Exporting Israeli optimized model...")

# Prepare model data for export
model_data = {
    'model': model,
    'scaler': scaler,
    'features': feature_columns,
    'performance': {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'overfitting': train_r2 - test_r2
    },
    'israel_specifics': {
        'weekend_system': 'Friday-Saturday',
        'workdays': 'Sunday-Thursday',
        'total_features': len(feature_columns),
        'cultural_features': ['is_weekend_israel', 'is_friday', 'is_saturday', 'is_sunday',
                             'temp_x_weekend_israel', 'temp_x_friday', 'temp_x_saturday']
    },
    'training_info': {
        'algorithm': 'Ridge Regression',
        'alpha': 10.0,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'split_date': split_date,
        'features_count': len(feature_columns)
    }
}

# Save model
model_filename = 'israel_optimized_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✅ Israeli model exported: {model_filename}")

# === 13. FINAL SUMMARY ===
print("\n" + "="*70)
print("🇮🇱 ISRAELI OPTIMIZED MODEL - TRAINING COMPLETED!")
print("="*70)
print(f"🎯 Performance: MAE {test_mae:,.0f} kWh | R² {test_r2:.3f} | MAPE {test_mape:.1f}%")
print(f"🇮🇱 Israeli Context: Friday-Saturday weekends perfectly integrated")
print(f"📊 Features: {len(feature_columns)} specialized variables")
print(f"🏭 Production Ready: {model_filename} exported")
print(f"📈 Chart: {chart_filename} generated")
print("="*70)
print("💡 Ready for Israeli industrial energy prediction!")
print("🇮🇱 Weekends: Friday-Saturday | Workdays: Sunday-Thursday")
print("="*70) 