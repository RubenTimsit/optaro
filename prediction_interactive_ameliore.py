import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# === FONCTION DE DÉTECTION DES JOURS FÉRIÉS BASÉE SUR LES VRAIES DONNÉES ===
def creer_detecteur_jours_feries(csv_path="data_with_context_fixed.csv"):
    """
    Crée un détecteur de jours fériés basé sur les vraies données historiques
    """
    df = pd.read_csv(csv_path)
    df['Day'] = pd.to_datetime(df['Day'])
    
    # Extraire les jours fériés réels
    jours_feries = df[(df['is_holiday_full'] == 1) | (df['is_holiday_half'] == 1)].copy()
    
    # Créer des patterns récurrents
    patterns_feries = set()
    
    for _, row in jours_feries.iterrows():
        date = row['Day']
        # Ajouter (mois, jour) pour les fêtes fixes
        patterns_feries.add((date.month, date.day))
    
    print(f"🎉 {len(patterns_feries)} patterns de jours fériés détectés")
    
    def detecter_jour_ferie(date):
        """Détecte si une date est un jour férié"""
        return 1 if (date.month, date.day) in patterns_feries else 0
    
    return detecter_jour_ferie, patterns_feries

# Créer le détecteur global
detecteur_feries, patterns_feries_globaux = creer_detecteur_jours_feries()

class PredicteurConsommationComplet:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.mae = None
        self.std_residuals = None
        self.temp_median = None
        self.patterns_feries = patterns_feries_globaux
        
        # Features complètes avec TOUTES les variables météo
        self.features = [
            # === TEMPÉRATURE ===
            'TempAvg', 'TempMin', 'TempMax',
            'temp_range', 'temp_ma_7', 'temp_ma_30', 'temp_squared',
            
            # === PRÉCIPITATIONS ===
            'Precip', 'precip_ma_7', 'has_rain',
            
            # === VENT ET PRESSION ===
            'WindSpeed', 'wind_ma_7', 'Pressure', 'pressure_ma_7',
            
            # === SEUILS TEMPÉRATURE ===
            'cooling_needs_light', 'cooling_needs_heavy', 'heating_needs',
            'temp_above_25', 'temp_above_28', 'temp_above_30',
            
            # === SAISONS ===
            'is_summer', 'is_winter', 'is_mid_summer',
            'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
            
            # === INTERACTIONS AVANCÉES ===
            'temp_x_summer', 'temp_x_mid_summer', 'temp_squared_x_summer',
            'temp_x_wind', 'pressure_x_temp',
            
            # === TEMPOREL ===
            'time_trend', 'is_weekend', 'is_holiday'
        ]
        
    def create_features_completes(self, df, temp_median=None):
        """Créé des features complètes avec TOUTES les variables météo"""
        df = df.copy()
        
        if temp_median is None:
            temp_median = df['TempAvg'].median()
        
        # === FEATURES TEMPÉRATURE COMPLÈTES ===
        df['temp_range'] = df['TempMax'] - df['TempMin']  # Amplitude thermique
        df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
        df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
        df['temp_squared'] = df['TempAvg'] ** 2
        
        # === FEATURES PRÉCIPITATIONS ===
        df['precip_ma_7'] = df['Precip'].rolling(window=7, min_periods=1).mean()
        df['has_rain'] = (df['Precip'] > 0).astype(int)
        
        # === FEATURES VENT ET PRESSION ===
        df['wind_ma_7'] = df['WindSpeed'].rolling(window=7, min_periods=1).mean()
        df['pressure_ma_7'] = df['Pressure'].rolling(window=30, min_periods=1).mean()
        
        # === SEUILS TEMPÉRATURE OPTIMISÉS ===
        temp_25, temp_30 = 25.0, 30.0
        df['cooling_needs_light'] = np.maximum(0, df['TempAvg'] - temp_25)
        df['cooling_needs_heavy'] = np.maximum(0, df['TempAvg'] - temp_30)
        df['heating_needs'] = np.maximum(0, temp_25 - df['TempAvg'])
        
        # Seuils binaires
        df['temp_above_25'] = (df['TempAvg'] > 25).astype(int)
        df['temp_above_28'] = (df['TempAvg'] > 28).astype(int)
        df['temp_above_30'] = (df['TempAvg'] > 30).astype(int)
        
        # === SAISONS EXPLICITES ===
        df['is_summer'] = ((df['Day'].dt.month >= 6) & (df['Day'].dt.month <= 8)).astype(int)
        df['is_winter'] = ((df['Day'].dt.month == 12) | (df['Day'].dt.month <= 2)).astype(int)
        df['is_mid_summer'] = (df['Day'].dt.month == 7).astype(int)
        
        # === FEATURES CYCLIQUES ===
        df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
        
        # === INTERACTIONS MÉTÉO AVANCÉES ===
        df['temp_x_summer'] = df['TempAvg'] * df['is_summer']
        df['temp_x_mid_summer'] = df['TempAvg'] * df['is_mid_summer']
        df['temp_squared_x_summer'] = df['temp_squared'] * df['is_summer']
        df['temp_x_wind'] = df['TempAvg'] * df['WindSpeed']
        df['pressure_x_temp'] = df['Pressure'] * df['TempAvg']
        
        # === TEMPOREL ===
        reference_date = pd.to_datetime('2022-01-01')
        df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
        
        # === JOURS SPÉCIAUX ===
        df['is_weekend'] = (df['Day'].dt.dayofweek >= 5).astype(int)
        
        # 🔥 UTILISATION DU VRAI DÉTECTEUR DE JOURS FÉRIÉS
        df['is_holiday'] = df['Day'].apply(detecteur_feries)
        
        return df, temp_median
    
    def entrainer_modele_complet(self):
        """Entraîne le modèle avec TOUTES les variables météo"""
        print("🚀 MODÈLE COMPLET - TOUTES VARIABLES MÉTÉO")
        print("=" * 60)
        
        # Chargement des données
        print("📊 Chargement des données complètes...")
        df = pd.read_csv("data_with_context_fixed.csv")
        df['Day'] = pd.to_datetime(df['Day'])
        df = df.sort_values('Day').reset_index(drop=True)
        
        print(f"📊 Données chargées: {len(df)} jours")
        print(f"📊 Variables météo: TempAvg, TempMin, TempMax, Precip, WindSpeed, Pressure")
        
        # Vérifier les données manquantes
        print("🔍 Vérification données manquantes:")
        for col in ['TempAvg', 'TempMin', 'TempMax', 'Precip', 'WindSpeed', 'Pressure']:
            missing = df[col].isnull().sum()
            print(f"   • {col}: {missing} valeurs manquantes")
        
        # Créer les features complètes
        print("🔧 Création des features complètes...")
        df_features, self.temp_median = self.create_features_completes(df)
        
        # Utiliser les vraies données de jours fériés pour l'entraînement
        vraie_holiday = (df['is_holiday_full'] + df['is_holiday_half'] > 0).astype(int)
        df_features['is_holiday'] = vraie_holiday
        
        print(f"✅ {len(self.features)} features créées")
        print(f"🎉 {vraie_holiday.sum()} jours fériés dans les données")
        
        # Split temporel
        split_idx = int(len(df_features) * 0.7)
        train_data = df_features.iloc[:split_idx].copy()
        test_data = df_features.iloc[split_idx:].copy()
        
        print(f"📊 Split: Train={len(train_data)} jours, Test={len(test_data)} jours")
        
        # Normalisation
        print("📐 Normalisation des features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(train_data[self.features])
        X_test_scaled = self.scaler.transform(test_data[self.features])
        
        y_train = train_data['DailyAverage'].values
        y_test = test_data['DailyAverage'].values
        
        # Entraînement avec validation croisée
        print("🤖 Entraînement avec validation croisée...")
        
        models = {
            'Lasso': Lasso(alpha=1.0, random_state=42),
            'Ridge': Ridge(alpha=10.0, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        best_name = ""
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_train_scaled):
                X_fold_train = X_train_scaled[train_idx]
                X_fold_val = X_train_scaled[val_idx]
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                val_pred = model.predict(X_fold_val)
                r2 = r2_score(y_fold_val, val_pred)
                scores.append(r2)
            
            mean_score = np.mean(scores)
            print(f"   • {name}: R² CV = {mean_score:.3f} ± {np.std(scores):.3f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Entraînement final
        print(f"\n🏆 Meilleur modèle: {best_name}")
        self.model = best_model
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation finale
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        self.mae = mean_absolute_error(y_test, test_pred)
        
        residuals = y_test - test_pred
        self.std_residuals = np.std(residuals)
        
        print(f"\n✅ MODÈLE COMPLET ENTRAÎNÉ:")
        print(f"   • Modèle sélectionné: {best_name}")
        print(f"   • Train R²: {train_r2:.3f}")
        print(f"   • Test R²: {test_r2:.3f}")
        print(f"   • Test MAE: {self.mae:.0f} kWh")
        print(f"   • Écart-type résidus: {self.std_residuals:.0f} kWh")
        print(f"   • Overfitting: {train_r2 - test_r2:.3f}")
        
        # Analyse des features importantes
        if hasattr(self.model, 'coef_'):
            feature_importance = abs(self.model.coef_)
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 TOP 10 FEATURES LES PLUS IMPORTANTES:")
            for i, row in importance_df.head(10).iterrows():
                print(f"   {i+1:2d}. {row['feature']:25s}: {row['importance']:8.0f}")
        
        # Sauvegarde
        self.sauvegarder_modele_complet()
        
    def sauvegarder_modele_complet(self):
        """Sauvegarde le modèle complet"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'mae': self.mae,
            'std_residuals': self.std_residuals,
            'temp_median': self.temp_median,
            'features': self.features,
            'patterns_feries': self.patterns_feries,
            'version': 'complet_v1_meteo'
        }
        
        with open('modele_prediction_complet.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("💾 Modèle complet sauvegardé dans 'modele_prediction_complet.pkl'")
    
    def charger_modele_complet(self):
        """Charge le modèle complet"""
        if os.path.exists('modele_prediction_complet.pkl'):
            try:
                with open('modele_prediction_complet.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.mae = model_data['mae']
                self.std_residuals = model_data.get('std_residuals', self.mae)
                self.temp_median = model_data['temp_median']
                self.features = model_data['features']
                self.patterns_feries = model_data.get('patterns_feries', patterns_feries_globaux)
                
                print("✅ Modèle complet chargé avec succès!")
                print(f"   Version: {model_data.get('version', 'complet_v1')}")
                print(f"   Features: {len(self.features)} (incluant toutes variables météo)")
                print(f"   MAE: {self.mae:.0f} kWh")
                print(f"   Jours fériés: {len(self.patterns_feries)} patterns")
                return True
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement: {e}")
                return False
        else:
            print("❌ Modèle complet non trouvé. Entraînement nécessaire...")
            return False

    def obtenir_donnees_meteo_completes(self, date):
        """Obtient toutes les données météo pour une date"""
        print(f"\n🌤️ DONNÉES MÉTÉO pour le {date.strftime('%d/%m/%Y')}:")
        print("1. Saisir toutes les données météo manuellement")
        print("2. Utiliser les moyennes saisonnières")
        
        choix = input("Votre choix (1 ou 2): ").strip()
        
        if choix == "1":
            meteo = {}
            print("\n📊 Saisissez les données météo:")
            
            try:
                meteo['TempAvg'] = float(input("Température moyenne (°C): "))
                meteo['TempMin'] = float(input("Température minimale (°C): "))
                meteo['TempMax'] = float(input("Température maximale (°C): "))
                meteo['Precip'] = float(input("Précipitations (mm): "))
                meteo['WindSpeed'] = float(input("Vitesse du vent (km/h): "))
                meteo['Pressure'] = float(input("Pression atmosphérique (hPa): "))
                
                return meteo
            except ValueError:
                print("❌ Erreur de saisie. Utilisation des moyennes saisonnières.")
                choix = "2"
        
        if choix == "2":
            # Moyennes saisonnières complètes
            mois = date.month
            moyennes_saisonnieres = {
                1: {'TempAvg': 15.2, 'TempMin': 11.0, 'TempMax': 19.5, 'Precip': 3.2, 'WindSpeed': 12.5, 'Pressure': 1014.5},
                2: {'TempAvg': 16.1, 'TempMin': 12.0, 'TempMax': 20.3, 'Precip': 2.8, 'WindSpeed': 13.0, 'Pressure': 1013.8},
                3: {'TempAvg': 18.5, 'TempMin': 14.2, 'TempMax': 22.8, 'Precip': 2.1, 'WindSpeed': 12.8, 'Pressure': 1013.2},
                4: {'TempAvg': 21.3, 'TempMin': 16.8, 'TempMax': 25.9, 'Precip': 1.5, 'WindSpeed': 12.2, 'Pressure': 1012.5},
                5: {'TempAvg': 24.8, 'TempMin': 20.1, 'TempMax': 29.5, 'Precip': 0.8, 'WindSpeed': 11.8, 'Pressure': 1011.8},
                6: {'TempAvg': 27.2, 'TempMin': 22.5, 'TempMax': 31.8, 'Precip': 0.3, 'WindSpeed': 11.5, 'Pressure': 1009.2},
                7: {'TempAvg': 28.9, 'TempMin': 24.2, 'TempMax': 33.5, 'Precip': 0.1, 'WindSpeed': 11.2, 'Pressure': 1006.8},
                8: {'TempAvg': 29.1, 'TempMin': 24.8, 'TempMax': 33.2, 'Precip': 0.2, 'WindSpeed': 11.8, 'Pressure': 1007.2},
                9: {'TempAvg': 27.4, 'TempMin': 23.1, 'TempMax': 31.5, 'Precip': 0.8, 'WindSpeed': 12.1, 'Pressure': 1009.8},
                10: {'TempAvg': 24.2, 'TempMin': 19.8, 'TempMax': 28.5, 'Precip': 2.1, 'WindSpeed': 12.5, 'Pressure': 1012.5},
                11: {'TempAvg': 20.1, 'TempMin': 16.2, 'TempMax': 24.8, 'Precip': 3.8, 'WindSpeed': 12.8, 'Pressure': 1014.2},
                12: {'TempAvg': 16.8, 'TempMin': 12.5, 'TempMax': 21.2, 'Precip': 4.2, 'WindSpeed': 13.2, 'Pressure': 1015.1}
            }
            
            meteo = moyennes_saisonnieres[mois]
            print(f"📊 Moyennes saisonnières pour {date.strftime('%B')}:")
            for var, val in meteo.items():
                print(f"   • {var}: {val}")
            
            return meteo
    
    def predire_consommation_complete(self, date, donnees_meteo):
        """Prédiction avec toutes les variables météo"""
        # Créer un DataFrame temporaire
        temp_df = pd.DataFrame({
            'Day': [date],
            **donnees_meteo
        })
        temp_df['Day'] = pd.to_datetime(temp_df['Day'])
        
        # Créer les features complètes
        df_features, _ = self.create_features_completes(temp_df, self.temp_median)
        
        # Normalisation
        X_scaled = self.scaler.transform(df_features[self.features])
        
        # Prédiction
        prediction = self.model.predict(X_scaled)[0]
        
        # Intervalle de confiance
        marge_erreur = 1.5 * self.std_residuals
        fourchette_basse = max(0, prediction - marge_erreur)
        fourchette_haute = prediction + marge_erreur
        
        return prediction, fourchette_basse, fourchette_haute, df_features
    
    def interface_interactive_complete(self):
        """Interface interactive complète avec toutes les variables météo"""
        print("🚀 PRÉDICTEUR COMPLET - TOUTES VARIABLES MÉTÉO")
        print("🌤️ Température, Précipitations, Vent, Pression")
        print("=" * 70)
        
        while True:
            try:
                print("\n📅 Saisissez une date future:")
                date_str = input("Date (format: JJ/MM/AAAA): ").strip()
                
                try:
                    date = datetime.strptime(date_str, "%d/%m/%Y")
                except ValueError:
                    print("❌ Format de date invalide. Utilisez JJ/MM/AAAA")
                    continue
                
                if date.date() <= datetime.now().date():
                    print("⚠️ Veuillez saisir une date future")
                    continue
                
                # Obtenir toutes les données météo
                donnees_meteo = self.obtenir_donnees_meteo_completes(date)
                
                # Prédiction complète
                print("\n🤖 Calcul de la prédiction complète...")
                prediction, fourchette_basse, fourchette_haute, features = self.predire_consommation_complete(date, donnees_meteo)
                
                # Affichage complet des résultats
                print("\n" + "="*80)
                print("🚀 PRÉDICTION COMPLÈTE - TOUTES VARIABLES MÉTÉO")
                print("="*80)
                print(f"📅 Date: {date.strftime('%A %d %B %Y')}")
                print(f"\n🌤️ DONNÉES MÉTÉO UTILISÉES:")
                for var, val in donnees_meteo.items():
                    print(f"   • {var}: {val}")
                
                print(f"\n🎯 RÉSULTATS DE PRÉDICTION:")
                print(f"   • Prédiction centrale: {prediction:,.0f} kWh")
                print(f"   • Fourchette estimée: {fourchette_basse:,.0f} - {fourchette_haute:,.0f} kWh")
                print(f"   • Marge d'erreur: ±{1.5 * self.std_residuals:,.0f} kWh")
                
                # Analyse contextuelle complète
                print(f"\n📋 ANALYSE CONTEXTUELLE COMPLÈTE:")
                is_weekend = date.weekday() >= 5
                print(f"   • Type de jour: {'🏖️ Weekend' if is_weekend else '🏢 Semaine'}")
                
                is_holiday = detecteur_feries(date)
                if is_holiday:
                    print(f"   • Jour férié: 🎉 OUI - Consommation réduite attendue")
                else:
                    print(f"   • Jour férié: 🏢 NON - Consommation normale")
                
                saison = ["🥶 Hiver", "🌸 Printemps", "☀️ Été", "🍂 Automne"][(date.month-1)//3]
                print(f"   • Saison: {saison}")
                
                temp_avg = donnees_meteo['TempAvg']
                if temp_avg > 30:
                    print(f"   • Climat: 🔥 TRÈS CHAUD - Forte climatisation")
                elif temp_avg > 25:
                    print(f"   • Climat: ☀️ CHAUD - Climatisation active")
                else:
                    print(f"   • Climat: 🌤️ TEMPÉRÉ - Climatisation minimale")
                
                if donnees_meteo['Precip'] > 5:
                    print(f"   • Précipitations: 🌧️ FORTES - Impact possible")
                elif donnees_meteo['Precip'] > 0:
                    print(f"   • Précipitations: 🌦️ Légères")
                else:
                    print(f"   • Précipitations: ☀️ Aucune")
                
                print("="*80)
                
                # Continuer ?
                print("\n🔄 Voulez-vous faire une autre prédiction?")
                continuer = input("Tapez 'oui' pour continuer ou 'non' pour quitter: ").strip().lower()
                
                if continuer not in ['oui', 'o', 'yes', 'y']:
                    print("👋 Au revoir!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                print("Veuillez réessayer...")

def main():
    """Fonction principale"""
    predicteur = PredicteurConsommationComplet()
    
    # Charger ou entraîner le modèle complet
    if not predicteur.charger_modele_complet():
        print("🚀 Entraînement du modèle complet nécessaire...")
        predicteur.entrainer_modele_complet()
    
    # Interface interactive complète
    predicteur.interface_interactive_complete()

if __name__ == "__main__":
    main() 