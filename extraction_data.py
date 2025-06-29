import requests
import pandas as pd

API_URL = "http://10.10.32.56:8000/query"
OUTPUT_FILE = "data_7_years.csv"

# === Requête SQL pour les 7 dernières années ===
sql_query = """
WITH TimeRange AS (
    SELECT CAST(DATEADD(YEAR, -7, MAX(TimestampUTC)) AS DATE) AS FromDate,
           CAST(MAX(TimestampUTC) AS DATE) AS ToDate
    FROM DataLog2
)
SELECT 
    CAST(D.TimestampUTC AS DATE) AS Day,
    D.SourceID,
    D.QuantityID,
    ST.Name AS SourceTypeName,
    MAX(D.Value) - MIN(D.Value) AS DailyAverage
FROM DataLog2 D
JOIN Source S ON D.SourceID = S.ID
JOIN SourceType ST ON S.SourceTypeID = ST.ID,
     TimeRange
WHERE 
    D.QuantityID = 129
    AND S.SourceTypeID = 5
    AND CAST(D.TimestampUTC AS DATE) BETWEEN TimeRange.FromDate AND TimeRange.ToDate
GROUP BY 
    CAST(D.TimestampUTC AS DATE), D.SourceID, D.QuantityID, ST.Name
ORDER BY 
    Day
"""

# === Requête HTTP ===
payload = {
    "query": sql_query,
    "limit": 100000  # augmente si besoin
}

print("🔄 Envoi de la requête à l'API...")
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    try:
        response_json = response.json()
        data = response_json.get("data", [])

        if not data:
            print("⚠️ Aucune donnée reçue.")
        else:
            df = pd.DataFrame(data)
            df["Day"] = pd.to_datetime(df["Day"])
            df.sort_values(["SourceID", "Day"], inplace=True)
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"✅ Données des 7 dernières années sauvegardées dans : {OUTPUT_FILE}")
            print(df.head())
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des données : {e}")
else:
    print(f"❌ Erreur HTTP {response.status_code} : {response.text}")
