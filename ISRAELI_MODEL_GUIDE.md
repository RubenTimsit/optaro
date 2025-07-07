# 🇮🇱 OPTARO - Israeli Optimized Energy Prediction Model

## Overview

OPTARO is an advanced energy consumption prediction system specifically optimized for Israeli operational context. The model leverages machine learning to predict daily industrial energy consumption while accounting for Israel's unique cultural and operational patterns.

## 🏭 Industrial Context

**Target Application:** Industrial energy consumption prediction  
**Temporal Resolution:** Daily predictions  
**Operational Environment:** Israeli industrial facility  
**Data Period:** 1,114 days (2022-2025)  
**Average Consumption:** ~80,000 kWh/day  

## 🇮🇱 Israeli Calendar Integration

### Weekend System
- **Israeli Weekends:** Friday-Saturday (🇮🇱)
- **Workdays:** Sunday-Thursday (💼)
- **Cultural Adaptation:** Model recognizes Israeli operational calendar

### Operational Patterns
- **Friday:** Weekend start, reduced industrial activity
- **Saturday:** Traditional Sabbath, minimal operations  
- **Sunday:** Primary workday start (not weekend)
- **Monday-Thursday:** Full industrial operations

## 🤖 Model Architecture

### Algorithm
- **Base Algorithm:** Ridge Regression
- **Regularization:** L2 with α = 10.0
- **Feature Engineering:** 57 specialized variables
- **Preprocessing:** StandardScaler normalization

### Key Features Categories

#### 1. Weather Features (12 variables)
```
- TempAvg, TempMin, TempMax
- Temperature moving averages (7-day, 30-day)
- Temperature squared (quadratic effects)
- Precipitation patterns
- Wind speed and atmospheric pressure
- Temperature thresholds (25°C, 28°C, 30°C)
```

#### 2. Israeli Calendar Features (7 variables)
```
- is_sunday, is_monday, is_tuesday, is_wednesday, is_thursday
- is_friday (🇮🇱 weekend start)
- is_saturday (🇮🇱 weekend)
- is_weekend_israel (Friday-Saturday combined)
```

#### 3. Seasonal Features (6 variables)
```
- is_summer, is_winter, is_mid_summer
- Cyclical encoding: month_sin, month_cos
- Day of year: day_of_year_sin, day_of_year_cos
```

#### 4. Israeli Cultural Interactions (4 variables)
```
- temp_x_weekend_israel: Temperature × Israeli weekend
- temp_x_friday: Temperature × Friday
- temp_x_saturday: Temperature × Saturday  
- temp_x_sunday: Temperature × Sunday (workday)
```

#### 5. Thermal Comfort Features (5 variables)
```
- cooling_needs_light (>25°C)
- cooling_needs_heavy (>30°C)
- heating_needs (<25°C)
- Temperature range (daily variation)
- Summer temperature interactions
```

#### 6. Temporal Features (3 variables)
```
- time_trend: Long-term temporal evolution
- consumption_lag_1: Previous day consumption
- consumption_lag_7: 7-day average consumption
```

#### 7. End-of-Year Features (3 variables)
```
- is_december: December month indicator
- days_to_new_year: Countdown to new year
- is_end_of_year: December 15-31 period
```

## 📊 Model Performance

### Accuracy Metrics
- **R² Score:** 0.962 (96.2% variance explained)
- **MAE (Mean Absolute Error):** 3,150 kWh
- **MAPE (Mean Absolute Percentage Error):** 4.3%
- **Industrial Standard:** ✅ Meets <10% MAPE requirement

### Israeli Weekend Performance
- **Friday MAPE:** 3.8% 
- **Saturday MAPE:** 5.5%
- **Sunday MAPE:** 4.4% (workday)
- **Weekend Reduction:** ~15% consumption decrease Fri-Sat

### Seasonal Performance
- **Summer:** 2.7% MAPE (best season)
- **Winter:** 5.5% MAPE
- **Spring/Autumn:** 4.0-4.5% MAPE

### Prediction Accuracy Distribution
- **Within ±5% error:** 64.3% of predictions
- **Within ±10% error:** 94.6% of predictions
- **Excellent precision:** <5% MAPE (industrial excellence)

## 🥇 Feature Importance Ranking

**Top 10 Most Important Features:**

1. **consumption_lag_1** (6,792) - Previous day consumption
2. **temp_squared** (2,590) - Quadratic temperature effect
3. **is_friday** (1,685) - Friday (Israeli weekend start) 🇮🇱
4. **heating_needs** (1,635) - Heating requirements (<25°C)
5. **temp_x_weekend_israel** (1,502) - Temperature × Israeli weekend 🇮🇱
6. **temp_ma_7** (1,445) - 7-day temperature average
7. **is_saturday** (1,398) - Saturday (Israeli weekend) 🇮🇱
8. **cooling_needs_heavy** (1,287) - Heavy cooling needs (>30°C)
9. **temp_x_friday** (1,201) - Temperature × Friday interaction 🇮🇱
10. **is_summer** (1,156) - Summer season indicator

## 🔧 Usage Guide

### Model Files
```
israel_optimized_model.pkl     # Trained model + metadata
israel_optimized_model.py      # Training script
model_precision_analysis.py    # Performance evaluation
```

### Analysis Tools
```
israel_weekend_diagnostics.py  # Weekend pattern analysis
simple_period_comparator.py    # Quick period comparison
future_predictor.py            # Future consumption forecasting
```

### Data Requirements
```
- Daily consumption data (DailyAverage)
- Weather data: Temperature, precipitation, wind, pressure
- Israeli calendar integration
- Minimum 1+ years historical data for training
```

### Prediction Workflow

1. **Data Preparation**
   ```python
   # Load historical data with Israeli features
   df = pd.read_csv("data_with_israel_temporal_features.csv")
   ```

2. **Model Loading**
   ```python
   import pickle
   with open('israel_optimized_model.pkl', 'rb') as f:
       model_data = pickle.load(f)
   ```

3. **Feature Engineering**
   ```python
   # Create 57 Israeli-specific features
   # Include weekend system: Friday-Saturday
   # Add cultural interactions
   ```

4. **Prediction**
   ```python
   # Scale features
   features_scaled = scaler.transform(features)
   
   # Predict
   prediction = model.predict(features_scaled)
   ```

## 🇮🇱 Israeli Operational Insights

### Weekly Pattern
- **Peak Consumption:** Tuesday-Thursday (full operations)
- **Medium Consumption:** Sunday-Monday (ramp-up period)  
- **Low Consumption:** Friday-Saturday (Israeli weekends)

### Cultural Adaptations
- **Friday Pattern:** Early weekend start, production wind-down
- **Saturday Pattern:** Sabbath observance, minimal operations
- **Sunday Pattern:** Week start (not weekend like global standard)
- **Holiday Integration:** Framework for Israeli holidays

### Temperature Sensitivity
- **Cooling Season:** Strong AC demand >25°C
- **Heating Season:** Moderate heating <25°C
- **Peak Efficiency:** 20-25°C temperature range
- **Weekend Interaction:** Lower temperature sensitivity on Fri-Sat

## 📈 Business Applications

### Energy Planning
- **Daily Forecasting:** Next 1-35 days consumption
- **Seasonal Planning:** Summer/winter consumption patterns
- **Weekend Scheduling:** Friday-Saturday maintenance windows
- **Peak Management:** Anticipate high consumption periods

### Operational Optimization
- **Maintenance Scheduling:** Israeli weekend opportunities
- **Energy Procurement:** Demand-based purchasing
- **Cost Management:** Peak avoidance strategies
- **Efficiency Monitoring:** Deviation detection

### Risk Management
- **Demand Spikes:** Early warning system
- **Weather Impact:** Temperature-based adjustments
- **Calendar Events:** Israeli holiday preparations
- **Supply Planning:** Consumption forecasting

## 🔬 Technical Specifications

### Model Hyperparameters
```python
Ridge(alpha=10.0, random_state=42)
StandardScaler()
```

### Training Configuration
```python
- Train/Test Split: 70/30 temporal
- Cross-validation: Time series aware
- Feature Selection: Domain expertise + correlation analysis
- Regularization: L2 penalty for generalization
```

### Performance Validation
```python
- MAE: Production metric
- R²: Variance explanation
- MAPE: Industrial standard compliance
- Residual Analysis: Bias detection
- Weekend Analysis: Cultural pattern validation
```

## 📋 Maintenance & Updates

### Model Retraining
- **Frequency:** Quarterly or when MAPE >7%
- **Data Requirements:** Minimum 6 months new data
- **Validation:** Israeli weekend pattern preservation
- **Performance Threshold:** Maintain <5% MAPE target

### Feature Updates
- **Israeli Holidays:** Annual calendar integration
- **Weather Features:** Climate change adaptations
- **Operational Changes:** Industrial schedule modifications
- **Cultural Evolution:** Israeli business pattern updates

### Monitoring Metrics
- **Daily:** Prediction accuracy tracking
- **Weekly:** Israeli weekend pattern validation
- **Monthly:** Seasonal performance review
- **Quarterly:** Full model performance assessment

---

## 📞 Support & Documentation

**Model Version:** Israeli Optimized v1.0  
**Last Updated:** 2025  
**Compatibility:** Python 3.8+, scikit-learn 1.0+  
**Israeli Calendar:** Integrated Friday-Saturday weekends  

**Performance Standard:** Industrial excellence (<5% MAPE)  
**Cultural Compliance:** ✅ Israeli operational calendar  
**Production Ready:** ✅ Validated for industrial deployment 