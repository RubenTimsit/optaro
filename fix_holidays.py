import pandas as pd

# === Configuration ===
FILE_CONSO = "data_cleaned.csv"
FILE_METEO = "weather_haifa_enriched.csv" 
FILE_JOURS_FERIES = "jours_feries.csv"
OUTPUT_FILE = "data_with_context_fixed.csv"

print("🔧 Correction de l'intégration des jours fériés...")

# === 1. Charger et nettoyer le fichier jours fériés ===
print("📅 Nettoyage du fichier jours fériés...")

# Lire le fichier ligne par ligne pour gérer les commentaires
holidays_data = []
with open(FILE_JOURS_FERIES, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
for i, line in enumerate(lines):
    if i == 0:  # Header
        continue
    line = line.strip()
    if line and not line.startswith('#'):
        # Extraire seulement les données avant le commentaire #
        data_part = line.split('#')[0].strip()
        if ',' in data_part:
            parts = data_part.split(',')
            if len(parts) >= 2:
                date_str = parts[0].strip()
                type_str = parts[1].strip()
                holidays_data.append({'Day': date_str, 'type': type_str})

# Créer DataFrame des jours fériés nettoyé
feries = pd.DataFrame(holidays_data)
print(f"Jours fériés trouvés : {len(feries)}")
print("Échantillon :")
print(feries.head(10))

# === 2. Charger les autres fichiers ===
df = pd.read_csv(FILE_CONSO)
weather = pd.read_csv(FILE_METEO)

# === 3. Conversion des dates ===
df["Day"] = pd.to_datetime(df["Day"])
weather["Day"] = pd.to_datetime(weather["Day"])
feries["Day"] = pd.to_datetime(feries["Day"])

print(f"\n📊 Plages de dates :")
print(f"Consommation : {df['Day'].min().date()} → {df['Day'].max().date()}")
print(f"Météo : {weather['Day'].min().date()} → {weather['Day'].max().date()}")
print(f"Jours fériés : {feries['Day'].min().date()} → {feries['Day'].max().date()}")

# === 4. Fusion consommation + météo ===
df = df.merge(weather, on="Day", how="left")

# === 5. Ajouter les jours fériés correctement ===
# Créer les ensembles de dates pour chaque type de jour férié
feries_full = set(feries[feries["type"] == "full"]["Day"].dt.date)
feries_half = set(feries[feries["type"] == "half"]["Day"].dt.date)

print(f"\n🎯 Nombre de jours fériés par type :")
print(f"Jours fériés complets : {len(feries_full)}")
print(f"Jours fériés partiels : {len(feries_half)}")

# Appliquer les marqueurs
df["is_holiday_full"] = df["Day"].dt.date.isin(feries_full).astype(int)
df["is_holiday_half"] = df["Day"].dt.date.isin(feries_half).astype(int)

# === 6. Vérification ===
full_holidays_count = df["is_holiday_full"].sum()
half_holidays_count = df["is_holiday_half"].sum()

print(f"\n✅ Résultats de l'intégration :")
print(f"Jours fériés complets marqués : {full_holidays_count}")
print(f"Jours fériés partiels marqués : {half_holidays_count}")

# Afficher quelques exemples de jours fériés détectés
print(f"\n📝 Exemples de jours fériés détectés :")
holiday_examples = df[(df["is_holiday_full"] == 1) | (df["is_holiday_half"] == 1)][["Day", "DailyAverage", "is_holiday_full", "is_holiday_half"]].head(10)
print(holiday_examples)

# === 7. Sauvegarde ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Données corrigées sauvegardées dans : {OUTPUT_FILE}")
print(f"Nombre total de lignes : {len(df)}")

# === 8. Statistiques finales ===
print(f"\n📈 Statistiques finales :")
print(f"Période couverte : {df['Day'].min().date()} → {df['Day'].max().date()}")
print(f"Consommation moyenne jours normaux : {df[df['is_holiday_full'] == 0]['DailyAverage'].mean():.0f}")
if full_holidays_count > 0:
    print(f"Consommation moyenne jours fériés complets : {df[df['is_holiday_full'] == 1]['DailyAverage'].mean():.0f}")
if half_holidays_count > 0:
    print(f"Consommation moyenne jours fériés partiels : {df[df['is_holiday_half'] == 1]['DailyAverage'].mean():.0f}") 