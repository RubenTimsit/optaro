from meteostat import Daily, Stations
from datetime import datetime
import pandas as pd

# === Étape 1 – Définir la période d'intérêt ===
start = datetime(2022, 5, 1)
end = datetime(2025, 12, 31)

# === Étape 2 – Trouver les stations ===

# Haïfa
stations_haifa = Stations()
station_haifa = stations_haifa.nearby(32.794, 34.989).fetch(1).index[0]

# Tel Aviv (fallback)
stations_tlv = Stations()
station_tlv = stations_tlv.nearby(32.08, 34.78).fetch(1).index[0]

# === Étape 3 – Récupérer les données ===

# Haïfa
df_haifa = Daily(station_haifa, start, end).fetch().reset_index()
df_haifa = df_haifa[["time", "tavg", "tmin", "tmax", "prcp", "wspd", "pres"]]
df_haifa.columns = ["Day", "TempAvg", "TempMin", "TempMax", "Precip", "WindSpeed", "Pressure"]

# Tel Aviv
df_tlv = Daily(station_tlv, start, end).fetch().reset_index()
df_tlv = df_tlv[["time", "prcp"]]
df_tlv.columns = ["Day", "Precip_TelAviv"]

# === Étape 4 – Fusionner les deux DataFrames sur la date ===
df_combined = pd.merge(df_haifa, df_tlv, on="Day", how="left")

# === Étape 5 – Remplacer les NaN dans Precip par Tel Aviv ===
df_combined["Precip"] = df_combined["Precip"].fillna(df_combined["Precip_TelAviv"])

# (optionnel) mettre les jours sans info Tel Aviv à 0
df_combined["Precip"] = df_combined["Precip"].fillna(0)

# === Étape 6 – Nettoyage et sauvegarde ===
df_combined.drop(columns=["Precip_TelAviv"], inplace=True)
df_combined.to_csv("weather_haifa_enriched.csv", index=False)

print("✅ Météo enrichie de Haïfa sauvegardée dans : weather_haifa_enriched.csv")
print(df_combined.head())
