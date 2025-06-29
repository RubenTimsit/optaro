import pandas as pd

# === Fichiers source ===
FILE_CONSO = "data_cleaned.csv"
FILE_METEO = "weather_haifa_enriched.csv"
FILE_JOURS_FERIES = "jours_feries.csv"
OUTPUT_FILE = "data_with_context.csv"

# === 1. Charger les fichiers ===
df = pd.read_csv(FILE_CONSO)
weather = pd.read_csv(FILE_METEO)
feries = pd.read_csv(FILE_JOURS_FERIES)

# === 2. Conversion des dates ===
df["Day"] = pd.to_datetime(df["Day"])
weather["Day"] = pd.to_datetime(weather["Day"])
feries["Day"] = pd.to_datetime(feries["Day"])

# === 3. Fusion consommation + météo (left join)
df = df.merge(weather, on="Day", how="left")

# === 4. Ajouter is_holiday_full et is_holiday_half
# Comparaison sur la partie date uniquement
feries_days_full = set(feries[feries["type"] == "full"]["Day"].dt.date)
feries_days_half = set(feries[feries["type"] == "half"]["Day"].dt.date)

df["is_holiday_full"] = df["Day"].dt.date.isin(feries_days_full).astype(int)
df["is_holiday_half"] = df["Day"].dt.date.isin(feries_days_half).astype(int)

# === 5. Sauvegarde
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Données enrichies sauvegardées dans : {OUTPUT_FILE}")
print(df.head())
