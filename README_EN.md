# OPTARO - Optimized Industrial Energy Prediction System for Israel 🇮🇱

## 🎯 Project Overview

**OPTARO** is an intelligent industrial energy consumption prediction system using an **Israeli optimized model** achieving exceptional performance. The project includes advanced diagnostic tools and period comparison features, specially adapted to the **Israeli cultural and operational context**.

### 🏭 Israeli Industrial Context
- **Data**: 1,114 days of energy consumption (2022-2025)
- **Weather variables**: Temperature (Min/Max/Avg), Precipitation, Wind, Pressure
- **Israeli calendar**: Friday-Saturday weekends, Sunday workdays
- **Final performance**: MAE 3,150 kWh, R² 0.962
- **Improvement vs classic model**: +19.0% precision

---

## 🇮🇱 Israeli Model Specificities

### 📅 Israeli Weekend System
- **Weekends**: Friday-Saturday (instead of Saturday-Sunday)
- **Workdays**: Sunday-Thursday
- **Major impact**: 11,000 kWh difference between Saturday (71,925 kWh) vs Sunday (82,889 kWh)
- **Weekend precision**: Friday 3.8%, Saturday 5.5%, Sunday 4.4% relative error

### 🎯 Israeli Specialized Features
- **Individual days**: `is_friday`, `is_saturday`, `is_sunday`, etc.
- **Israeli weekend**: `is_weekend_israel` (Friday-Saturday)
- **Temperature interactions**: `temp_x_weekend_israel`, `temp_x_friday`, `temp_x_saturday`
- **Cultural system**: Complete adaptation to Israeli operational patterns

---

## 🏆 Final Results - Israeli Optimized Model

### 📊 Exceptional Performance
```
🇮🇱 ISRAELI FINAL MODEL (modele_optimise_israel.pkl)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Test MAE       : 3,150 kWh    (🔥 19.0% improvement vs classic model)
📈 Test R²        : 0.962        (🔥 Excellent - 96.2% variance explained)
📊 MAPE           : 4.3%         (🔥 Industrial precision < 5%)
⚖️ Overfitting    : Controlled   (🔥 Robust generalization)
📊 Features       : 57 variables (Israel specialized)
🇮🇱 Calendar      : Friday-Saturday weekends
```

### 🎯 Precision Analysis by Context

#### 📊 Precision by Consumption Quartiles
- **Q1 (low loads)**: 5.8% MAPE (good)
- **Q2 (low-medium)**: 4.6% MAPE (excellent) 
- **Q3 (medium-high)**: 3.9% MAPE (excellent)
- **Q4 (high loads)**: 2.8% MAPE (optimal)

#### 🗓️ Precision by Seasons
- **Spring**: 4.7% MAPE
- **Summer**: 2.7% MAPE (best)
- **Autumn**: 4.5% MAPE
- **Winter**: 5.5% MAPE

#### 🇮🇱 Israeli Weekend Precision (Problem Solved!)
- **Friday**: 3.8% relative error (excellent)
- **Saturday**: 5.5% relative error (good)
- **Sunday**: 4.4% relative error (excellent - workday)

### 🥇 Top Features - Israeli Model
1. **`consumption_lag_1`** (6,792) - D-1 consumption (critical)
2. **`temp_squared`** (2,590) - Quadratic temperature effect
3. **`is_friday`** (1,685) - Friday (Israeli weekend start)
4. **`heating_needs`** (1,635) - Heating requirements
5. **`temp_x_weekend_israel`** (1,502) - Temperature × Israeli weekend interaction
6. **`is_saturday`** (1,445) - Saturday (Israeli weekend)
7. **`consumption_lag_7`** (1,318) - D-7 consumption
8. **`temp_x_friday`** (1,285) - Temperature × Friday interaction

### 🎯 Weekend Problem Diagnosis (RESOLVED)
**Initial problem identified**: Classic model didn't represent weekends and holidays well in Israel

**Diagnostic analysis**:
- 11,000 kWh difference Saturday vs Sunday
- 19,500 kWh difference "bridge days" vs normal days
- Temperature-weekend interaction varying from -7% to -2%

**Israeli solution**:
- ✅ Model adapted to Friday-Saturday calendar
- ✅ Specialized variables per weekday
- ✅ Weather × Israeli context interactions
- ✅ Weekend precision: 3.8% to 5.5% (vs >10% before)

---

## 🛠️ Available Tools (Israeli Updated)

### 🇮🇱 1. Main Israeli Optimized Model
```bash
python modele_optimise_israel.py
```
**The reference model** adapted to Israeli context
- **57 specialized features** including Israeli system
- **Friday-Saturday weekends** perfectly managed
- **Temporal validation** with Israeli patterns
- **Automatic diagnostic** weekends and holidays
- **Export**: Production-ready `modele_optimise_israel.pkl`

**Outputs**:
- Saved model with Israeli metadata
- Weekend validation graphics
- Precision analysis per weekday
- Culturally adapted performance metrics

### 📊 2. Period Comparator (Israeli Version)
```bash
python comparateur_periodes.py
```
**Interactive interface** to compare two periods with Israeli calendar
- **Flexible input**: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY
- **Israeli calendar**: Friday-Saturday weekends automatically detected
- **4 automatic charts**: evolution, distributions, temperature, Israeli weekday
- **Specialized analysis**: Israeli weekend impact, cultural patterns
- **Optional export**: High-resolution CSV and PNG

**Israeli usage example**:
```
📅 Period 1: 01/07/2024 → 31/07/2024 (Summer 2024)
📅 Period 2: 01/12/2024 → 31/12/2024 (Winter 2024)
🇮🇱 Weekends detected: 8 Friday-Saturdays vs 9 in winter
🎯 Result: Summer +79.8% vs Winter (+€179,944)
🌡️ Factor: +14.7°C average temperature
📊 Charts: 4 Israeli analyses generated
```

### ⚡ 3. Simple Israeli Comparator
```bash
python comparateur_simple.py
```
**Fast version** with examples adapted to Israeli context
- **4 ready comparisons**: Summer vs Summer, Winter vs Summer, monthly, etc.
- **Israeli calendar**: Automatic Friday-Saturday weekend detection
- **Cultural statistics**: Fridays, Saturdays, Sundays separated
- **Text mode**: Clear statistics with Israeli context
- **Simple export**: CSV with Israeli breakdown

**Available comparisons**:
1. 🌞 Summer 2024 vs Summer 2023
2. ❄️ Winter vs Summer 2024  
3. 🗓️ June 2024 vs June 2023
4. 🔥 August vs September 2024
5. ✏️ Manual input

**New Israeli statistics**:
```
Weekends (Fri-Sat)      |            8 |           9 |        -1
Fridays                 |            4 |           4 |         0
Saturdays               |            4 |           5 |        -1
Sundays (workday)       |            4 |           4 |         0
```

### 🔮 4. Future Consumption Predictor (Israeli)
```bash
python predicteur_futur.py
```
**Future predictions** with weather simulation and Israeli calendar
- **Smart weather**: Based on real Israeli historical data
- **Israeli calendar**: Friday-Saturday weekends in simulations
- **Adapted lags**: Israeli pattern continuation
- **Historical comparison**: vs same periods with correct calendar
- **Visualizations**: 4 charts adapted to Israeli context

**July 2025 Example - Israeli Version**:
```
🇮🇱 Period: 2025-07-01 → 2025-07-31
🌡️ Simulated temperature: 28.6°C (realistic)
⚡ Expected consumption: 2,640,665 kWh
💰 Estimated cost: €396,100
📅 Weekends (Fri-Sat): 8 days detected
💼 Sunday workdays: 4 days
📉 vs July 2024: -22.2% (-752,946 kWh)
```

### 🔍 5. Israeli Weekend Diagnostics
```bash
python diagnostic_weekends_feries_israel.py
```
**Specialized analysis** of Israeli weekend patterns
- **Saturday vs Sunday comparison**: Quantified differences
- **Temperature impact**: Israeli weekend interactions
- **"Bridge days"**: Cultural pattern analysis
- **Calendar validation**: Israeli adaptation verification

---

## 📁 Final Project Structure (Israel)

```
optaro-main/
├── 📊 DATA
│   ├── data_with_context_fixed.csv              # Main dataset (1,114 days)
│   └── data_with_israel_temporal_features.csv   # Dataset with Israeli features
│
├── 🇮🇱 ISRAELI OPTIMIZED MODEL (PRODUCTION)
│   ├── modele_optimise_israel.py                # Israeli final model script
│   ├── modele_optimise_israel.pkl               # Model + scaler + metadata
│   └── modele_optimise_israel_validation.png    # Israeli validation charts
│
├── 📊 COMPARISON TOOLS (ISRAEL)
│   ├── comparateur_periodes.py                  # Israeli interactive comparator
│   ├── comparateur_simple.py                    # Israeli fast comparator
│   └── predicteur_futur.py                      # Israeli future predictions
│
├── 🔍 ISRAELI DIAGNOSTICS
│   ├── diagnostic_weekends_feries_israel.py     # Israeli weekend diagnostics
│   ├── diagnostic_weekends_israel.png           # Weekend pattern analysis
│   └── analyse_precision_modele_israel.png      # Complete precision analysis
│
├── 📊 PRECISION ANALYSIS
│   ├── analyse_precision_modele.py              # Complete precision analysis
│   └── predictions_futures_20250712_20250815.png # Prediction examples
│
└── 📖 DOCUMENTATION
    ├── README.md                                # French version (Israel)
    ├── README_EN.md                             # This file (English - Israel)
    └── .gitignore                               # Git configuration
```

---

## 🚀 Installation and Quick Start (Israel)

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pickle
```

### Israeli Express Setup
1. **Verify** presence of `data_with_context_fixed.csv`
2. **Test Israeli model**: `python modele_optimise_israel.py`
3. **Diagnose weekends**: `python diagnostic_weekends_feries_israel.py`
4. **Compare periods**: `python comparateur_simple.py`
5. **Predict future**: `python predicteur_futur.py`

---

## 📊 Production Usage (Israel)

### 🇮🇱 Loading Israeli Model
```python
import pickle
import pandas as pd

# Load complete Israeli model
with open('modele_optimise_israel.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler'] 
features = model_data['features']  # 57 Israeli features
performance = model_data['performance']

print(f"Performance: MAE {performance['test_mae']:.0f} kWh, R² {performance['test_r2']:.3f}")
# Output: Performance: MAE 3150 kWh, R² 0.962
print("🇮🇱 Weekends: Friday-Saturday | Workdays: Sunday-Thursday")
```

### 🔮 Prediction with Israeli Calendar
```python
# Prepare data with 57 required Israeli features
new_data = create_features_israel(raw_data)  # Function with Israeli calendar
X_scaled = scaler.transform(new_data[features])
prediction = model.predict(X_scaled)

print(f"Predicted consumption: {prediction[0]:,.0f} kWh")

# Check day type
if new_data['is_weekend_israel'].iloc[0]:
    print("🇮🇱 Israeli weekend (Friday or Saturday)")
elif new_data['is_sunday'].iloc[0]:
    print("💼 Sunday - Workday in Israel")
```

### 📊 Classic vs Israeli Calendar Comparison
```python
# Comparison of both approaches
print("BEFORE (Classic model):")
print("- Weekends: Saturday-Sunday")
print("- MAE: 3,889 kWh")
print("- Weekend errors: >10%")

print("\nAFTER (Israeli model):")
print("- Weekends: Friday-Saturday")  
print("- MAE: 3,150 kWh (-19%)")
print("- Weekend errors: 3.8-5.5%")
print("🇮🇱 Cultural adaptation successful!")
```

---

## 🎯 Technical Key Points (Israel)

### ✅ What Works Exceptionally
- **Israeli calendar**: Friday-Saturday weekends perfectly managed
- **Day-specific variables**: `is_friday`, `is_saturday`, `is_sunday` critical
- **Cultural interactions**: `temp_x_weekend_israel` very performant
- **Weekend precision**: Error divided by 2-3 vs classic model
- **Contextual adaptation**: 57 specialized Israeli features

### 🇮🇱 Specific Israeli Advantages
- **Weekend problem resolution**: Finally accurate predictions!
- **Sunday workday**: Correctly treated as work day
- **Cultural patterns**: Friday vs Saturday consumption differentiated
- **Weather interactions**: Adapted to Israeli operational patterns
- **Cultural validation**: Metrics adapted to local context

### 🔄 Possible Future Improvements
- **Israeli holidays**: Complete religious calendar integration
- **Peak hours**: Israeli work schedule adaptation
- **Regional seasons**: Region-specific climate features
- **Cultural events**: Rosh Hashana, Yom Kippur patterns, etc.
- **Cultural ensembling**: Regional model combination

### ⚠️ Specific Limitations
- **Lag data**: D-1, D-7 still required
- **Calendar changes**: Retraining if cultural modifications
- **Geographic transfer**: Model specific to Israeli context
- **Holidays**: Simplified in current version

---

## 📈 Performance Summary (Classic vs Israeli)

| Metric | Classic Model | Israeli Model | Improvement |
|--------|---------------|---------------|-------------|
| **MAE Test** | 3,889 kWh | **3,150 kWh** | **🇮🇱 +19.0%** |
| **R² Test** | 0.941 | **0.962** | **🇮🇱 +2.2%** |
| **MAPE** | 5.3% | **4.3%** | **🇮🇱 +18.9%** |
| **Weekends** | >10% error | **3.8-5.5%** | **🇮🇱 +50-60%** |
| **Calendar** | Saturday-Sunday | **Friday-Saturday** | **🇮🇱 Cultural** |
| **Features** | 40 | **57** | **🇮🇱 Specialized** |
| **Context** | Generic | **Israeli** | **🇮🇱 Adapted** |

## 🎯 Business Impact (Israel)

### 💰 Financial Impact
- **Improved precision**: 19% error reduction = operational savings
- **Weekend planning**: Accurate Friday-Saturday predictions
- **Calendar optimization**: Respect for local cultural patterns
- **Waste reduction**: Better peak/low anticipation

### 🏭 Operational Impact  
- **Maintenance planning**: Israeli calendar respected
- **Team management**: Sunday (workday) load anticipation
- **Supply management**: Accurate Israeli weekend forecasts
- **Dashboards**: Metrics adapted to local context

**🏆 CONCLUSION: Reliable Israeli industrial model with complete cultural adaptation and exceptional performance for local operational context.**

---

*Last update: Model optimized for Israel with complete weekend diagnostics and culturally adapted tools* 