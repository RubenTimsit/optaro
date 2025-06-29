import pandas as pd

# === Configuration ===
INPUT_FILE = "data_7_years.csv"
OUTPUT_FILE = "data_cleaned.csv"

# === Étape 1 – Charger les données ===
df = pd.read_csv(INPUT_FILE)
print(f"📥 Données chargées depuis : {INPUT_FILE}")
print(f"Nombre de lignes brutes : {len(df)}")

# === Étape 2 – Conversion de la date ===
df["Day"] = pd.to_datetime(df["Day"], errors="coerce")

# === Étape 3 – Tri par SourceID et date ===
df.sort_values(["SourceID", "Day"], inplace=True)

# === Étape 4 – Suppression des doublons ===
df.drop_duplicates(inplace=True)

# === Étape 5 – Valeurs manquantes ===
missing = df.isnull().sum()
print("\n🔎 Valeurs manquantes par colonne :\n", missing)

# On supprime les lignes où la date ou la consommation est manquante
df.dropna(subset=["Day", "DailyAverage", "SourceID"], inplace=True)

# === Étape 6 – Nettoyage des valeurs aberrantes ===
# On élimine les valeurs négatives et très extrêmes (> 10 000 000)
df = df[df["DailyAverage"] >= 0]
df = df[df["DailyAverage"] < 1e7]  # seuil à ajuster si besoin

# === Étape 7 – Statistiques utiles ===
print("\n📊 Statistiques après nettoyage :")
print(f"Période couverte : {df['Day'].min().date()} → {df['Day'].max().date()}")
print(f"Nombre de SourceID uniques : {df['SourceID'].nunique()}")
print(f"Nombre de lignes finales : {len(df)}")

# === Étape 8 – Sauvegarde ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Données nettoyées sauvegardées dans : {OUTPUT_FILE}")
print(df.head())
