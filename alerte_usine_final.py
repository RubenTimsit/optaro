import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SystemeAlerteUsine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.mae = None
        self.std_residuals = None
        self.temp_median = None
        self.features = []
        self.model_loaded = False
        
    def charger_modele_robuste(self):
        """Charge le modèle robuste et recalcule les métriques précises"""
        print("🔧 CHARGEMENT DU MODÈLE ROBUSTE...")
        
        # Essayer de charger le modèle pré-entraîné
        if os.path.exists('modele_prediction.pkl'):
            try:
                with open('modele_prediction.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.mae = model_data['mae']
                self.temp_median = model_data['temp_median']
                self.features = model_data['features']
                
                print("✅ Modèle robuste chargé avec succès!")
                
                # Recalculer les métriques précises sur les données
                self.recalculer_metriques()
                self.model_loaded = True
                return True
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement: {e}")
                return False
        else:
            print("❌ Modèle non trouvé. Veuillez d'abord exécuter prediction_interactive.py")
            return False
    
    def recalculer_metriques(self):
        """Recalcule les métriques précises du modèle sur les données actuelles"""
        print("📊 Recalcul des métriques précises...")
        
        try:
            # Charger les données
            df = pd.read_csv("data_with_context_fixed.csv")
            df['Day'] = pd.to_datetime(df['Day'])
            df = df.sort_values('Day').reset_index(drop=True)
            
            # Créer les features (même fonction que dans le modèle)
            df_features = self.create_robust_features(df)
            
            # Split comme dans le modèle original
            split_idx = int(len(df_features) * 0.7)
            test_data = df_features.iloc[split_idx:].copy()
            
            # Prédictions sur test
            X_test_scaled = self.scaler.transform(test_data[self.features])
            y_test = test_data['DailyAverage'].values
            test_pred = self.model.predict(X_test_scaled)
            
            # Calcul des métriques précises
            self.mae = mean_absolute_error(y_test, test_pred)
            residuals = y_test - test_pred
            self.std_residuals = np.std(residuals)
            
            print(f"✅ Métriques recalculées:")
            print(f"   📈 MAE réelle: {self.mae:,.0f} kWh")
            print(f"   📊 Écart-type résidus: {self.std_residuals:,.0f} kWh")
            
        except Exception as e:
            print(f"⚠️ Erreur recalcul métriques: {e}")
            # Valeurs par défaut du modèle
            print("📊 Utilisation des métriques du modèle chargé")
    
    def create_robust_features(self, df):
        """Recrée les features robustes (même logique que le modèle)"""
        df = df.copy()
        
        # Features temporelles cycliques
        df['month_sin'] = np.sin(2 * np.pi * df['Day'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Day'].dt.month / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['Day'].dt.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['Day'].dt.dayofyear / 365)
        
        # Tendance temporelle
        reference_date = pd.to_datetime('2022-01-01')
        df['time_trend'] = (df['Day'] - reference_date).dt.days / 365.25
        
        # Features météo robustes
        df['temp_ma_7'] = df['TempAvg'].rolling(window=7, min_periods=1).mean()
        df['temp_ma_30'] = df['TempAvg'].rolling(window=30, min_periods=1).mean()
        df['temp_deviation_7'] = df['TempAvg'] - df['temp_ma_7']
        df['temp_deviation_30'] = df['TempAvg'] - df['temp_ma_30']
        
        df['temp_above_median'] = (df['TempAvg'] > self.temp_median).astype(int)
        df['cooling_needs'] = np.maximum(0, df['TempAvg'] - self.temp_median)
        df['heating_needs'] = np.maximum(0, self.temp_median - df['TempAvg'])
        
        # Features jours spéciaux
        df['is_weekend'] = (df['Day'].dt.dayofweek >= 5).astype(int)
        df['is_holiday'] = 0  # Par défaut
        
        # Features d'interaction
        df['temp_x_weekend'] = df['TempAvg'] * df['is_weekend']
        df['temp_x_season'] = df['TempAvg'] * df['month_sin']
        
        return df
    
    def predire_consommation(self, date, temperature):
        """Prédit la consommation pour une date et température données"""
        if not self.model_loaded:
            raise Exception("Modèle non chargé. Impossible de faire une prédiction.")
        
        # Créer DataFrame temporaire
        temp_df = pd.DataFrame({
            'Day': [date],
            'TempAvg': [temperature]
        })
        temp_df['Day'] = pd.to_datetime(temp_df['Day'])
        
        # Créer features
        df_features = self.create_robust_features(temp_df)
        
        # Normaliser et prédire
        X_scaled = self.scaler.transform(df_features[self.features])
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction
    
    def detecter_anomalie(self, actual, predicted):
        """
        Détecte si une consommation observée est anormale
        Utilise les métriques RÉELLES du modèle robuste
        """
        
        error = abs(actual - predicted)
        relative_error = error / predicted * 100 if predicted > 0 else 0
        
        # Utiliser l'écart-type RÉEL des résidus
        std_to_use = self.std_residuals if self.std_residuals else self.mae
        z_score = abs((actual - predicted) / std_to_use) if std_to_use > 0 else 0
        
        # Probabilités basées sur la distribution normale
        prob_normal = 2 * (1 - stats.norm.cdf(z_score))
        prob_anomaly = 1 - prob_normal
        
        # Seuils d'alerte ajustés pour le modèle robuste
        if z_score > 2.5:  # Plus strict pour modèle robuste
            alert_level = "🔴 CRITIQUE"
            status = "ANOMALIE MAJEURE"
            urgency = "IMMÉDIATE"
        elif z_score > 2.0:
            alert_level = "🟠 ÉLEVÉ"
            status = "ANOMALIE MODÉRÉE"
            urgency = "ÉLEVÉE"
        elif z_score > 1.5:
            alert_level = "🟡 MOYEN"
            status = "ANOMALIE LÉGÈRE"
            urgency = "MODÉRÉE"
        else:
            alert_level = "🟢 NORMAL"
            status = "FONCTIONNEMENT NORMAL"
            urgency = "AUCUNE"
        
        return {
            'alert_level': alert_level,
            'status': status,
            'urgency': urgency,
            'error_kwh': error,
            'error_percent': relative_error,
            'z_score': z_score,
            'prob_anomaly': prob_anomaly * 100,
            'prob_normal': prob_normal * 100
        }
    
    def generer_rapport_alerte(self, predicted, actual, timestamp=None, context=""):
        """Génère un rapport d'alerte complet avec modèle robuste"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        result = self.detecter_anomalie(actual, predicted)
        
        print(f"\n" + "="*70)
        print(f"🏭 RAPPORT D'ALERTE ROBUSTE - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if context:
            print(f"📝 Contexte: {context}")
        print(f"🤖 Modèle: LASSO Régularisé (MAE={self.mae:,.0f} kWh)")
        print(f"="*70)
        
        print(f"🔮 Consommation prévue:    {predicted:>10,.0f} kWh")
        print(f"📊 Consommation observée:  {actual:>10,.0f} kWh")
        print(f"📉 Écart absolu:           {result['error_kwh']:>10,.0f} kWh")
        print(f"📈 Écart relatif:          {result['error_percent']:>10.1f}%")
        print(f"🎯 Z-Score (sigmas):       {result['z_score']:>10.2f}")
        print(f"")
        print(f"🚨 NIVEAU D'ALERTE: {result['alert_level']}")
        print(f"📋 STATUT: {result['status']}")
        print(f"⚡ URGENCE: {result['urgency']}")
        print(f"")
        print(f"📊 PROBABILITÉ D'ANOMALIE: {result['prob_anomaly']:>8.1f}%")
        print(f"📊 PROBABILITÉ NORMALE:    {result['prob_normal']:>8.1f}%")
        print(f"")
        
        # Actions recommandées selon la probabilité
        if result['prob_anomaly'] > 95:
            print(f"🔴 ACTIONS CRITIQUES IMMÉDIATES:")
            print(f"   1. 🛑 ARRÊT SÉCURITAIRE DE LA PRODUCTION")
            print(f"   2. 📞 ALERTER L'ÉQUIPE D'URGENCE") 
            print(f"   3. 🔍 INSPECTION COMPLÈTE DU SYSTÈME")
            print(f"   4. 📊 VÉRIFIER TOUS LES CAPTEURS")
            print(f"   5. 📝 DOCUMENTER L'INCIDENT MAJEUR")
        elif result['prob_anomaly'] > 85:
            print(f"🟠 ACTIONS URGENTES:")
            print(f"   1. ⚠️  RÉDUIRE LA PRODUCTION TEMPORAIREMENT")
            print(f"   2. 👥 ALERTER L'ÉQUIPE MAINTENANCE")
            print(f"   3. 🔧 INSPECTION DES ÉQUIPEMENTS PRINCIPAUX")
            print(f"   4. 📈 MONITORING CONTINU")
            print(f"   5. 📋 PRÉPARER RAPPORT D'INCIDENT")
        elif result['prob_anomaly'] > 70:
            print(f"🟡 SURVEILLANCE RENFORCÉE:")
            print(f"   1. 👀 MONITORING CONTINU PROCHAINES 4H")
            print(f"   2. 🔧 VÉRIFICATION ÉQUIPEMENTS SENSIBLES")
            print(f"   3. 📊 CONTRÔLE CAPTEURS PRINCIPAUX")
            print(f"   4. 👥 ALERTER SUPERVISEUR DE POSTE")
            print(f"   5. 📝 NOTER DANS RAPPORT DE QUART")
        else:
            print(f"🟢 FONCTIONNEMENT NORMAL:")
            print(f"   • Aucune action particulière requise")
            print(f"   • Système dans les paramètres normaux")
            print(f"   • Continuer surveillance de routine")
        
        print(f"="*70)
        
        return result
    
    def mode_interactif(self):
        """Mode interactif pour tester le système d'alerte"""
        print("🚨 MODE INTERACTIF - SYSTÈME D'ALERTE")
        print("=" * 50)
        
        while True:
            try:
                print("\n📊 NOUVELLE ANALYSE:")
                print("1. Saisir consommation prévue et observée")
                print("2. Prédire puis comparer avec l'observé")
                print("3. Quitter")
                
                choix = input("\nVotre choix (1, 2 ou 3): ").strip()
                
                if choix == "3":
                    print("👋 Au revoir!")
                    break
                    
                elif choix == "1":
                    # Mode simple : saisie directe
                    try:
                        predicted = float(input("Consommation prévue (kWh): "))
                        actual = float(input("Consommation observée (kWh): "))
                        context = input("Contexte (optionnel): ").strip()
                        
                        self.generer_rapport_alerte(predicted, actual, context=context)
                        
                    except ValueError:
                        print("❌ Veuillez saisir des nombres valides")
                        
                elif choix == "2":
                    # Mode avec prédiction automatique
                    try:
                        print("\n📅 Date d'analyse:")
                        date_str = input("Date (JJ/MM/AAAA): ").strip()
                        date = datetime.strptime(date_str, "%d/%m/%Y")
                        
                        temp = float(input("Température (°C): "))
                        actual = float(input("Consommation observée (kWh): "))
                        
                        # Prédiction automatique
                        predicted = self.predire_consommation(date, temp)
                        
                        context = f"Date: {date.strftime('%A %d/%m/%Y')}, Temp: {temp}°C"
                        self.generer_rapport_alerte(predicted, actual, context=context)
                        
                    except ValueError as e:
                        print(f"❌ Erreur de saisie: {e}")
                    except Exception as e:
                        print(f"❌ Erreur: {e}")
                
                else:
                    print("❌ Choix invalide")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break

def main():
    """Fonction principale du système d'alerte"""
    print("🚨 SYSTÈME D'ALERTE USINE - VERSION ROBUSTE")
    print("Utilise le modèle LASSO régularisé avec métriques réelles")
    print("=" * 60)
    
    # Initialiser le système
    systeme = SystemeAlerteUsine()
    
    # Charger le modèle robuste
    if not systeme.charger_modele_robuste():
        print("\n❌ IMPOSSIBLE DE CHARGER LE MODÈLE")
        print("👉 Veuillez d'abord exécuter: python prediction_interactive.py")
        return
    
    print(f"\n📊 SYSTÈME INITIALISÉ AVEC SUCCÈS:")
    print(f"   🤖 Modèle: LASSO Régularisé")
    print(f"   📈 MAE: {systeme.mae:,.0f} kWh")
    print(f"   📊 Écart-type: {systeme.std_residuals:,.0f} kWh" if systeme.std_residuals else "   📊 Écart-type: Calculé sur MAE")
    print(f"   🎯 Prêt pour détection d'anomalies")
    
    # Tests automatiques avec les nouvelles métriques
    print(f"\n🎬 DÉMONSTRATIONS AVEC MÉTRIQUES ROBUSTES")
    
    cas_tests = [
        {
            'nom': "🟢 JOURNÉE NORMALE",
            'predit': 75000,
            'observe': 73500,
            'contexte': "Fonctionnement standard"
        },
        {
            'nom': "🟡 ÉCART MODÉRÉ", 
            'predit': 80000,
            'observe': 85000,
            'contexte': "Production légèrement élevée"
        },
        {
            'nom': "🟠 BAISSE SIGNIFICATIVE",
            'predit': 70000,
            'observe': 55000,
            'contexte': "Possible arrêt d'équipement"
        },
        {
            'nom': "🔴 ANOMALIE MAJEURE",
            'predit': 80000,
            'observe': 40000,
            'contexte': "Chute critique 50% - défaillance système"
        }
    ]
    
    for i, cas in enumerate(cas_tests, 1):
        print(f"\n🔸 CAS {i}/{len(cas_tests)}: {cas['nom']}")
        result = systeme.generer_rapport_alerte(
            predicted=cas['predit'],
            actual=cas['observe'],
            context=cas['contexte']
        )
    
    # Mode interactif
    print(f"\n🔄 PASSER EN MODE INTERACTIF ?")
    choix = input("Tapez 'oui' pour continuer en mode interactif: ").strip().lower()
    
    if choix in ['oui', 'o', 'yes', 'y']:
        systeme.mode_interactif()
    
    print(f"\n✅ SYSTÈME D'ALERTE OPÉRATIONNEL !")

if __name__ == "__main__":
    main() 