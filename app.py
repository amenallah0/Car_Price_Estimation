import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

try:
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('car_prices.csv')
    print("Données chargées avec succès!")

    # Nettoyer les données
    print("\nNettoyage des données...")
    df = df.dropna()
    print(f"Nombre de voitures dans la base : {len(df)}")

    # Sélectionner les features pertinentes
    features = ['year', 'make', 'model', 'trim', 'body', 'transmission', 
               'condition', 'odometer', 'color', 'interior']

    # Prendre un exemple réel de la base de données
    exemple_index = 0  # On prend la première voiture comme exemple
    voiture_exemple = df.iloc[exemple_index]

    # Encoder les variables catégorielles
    print("\nPréparation du modèle...")
    encoders = {}
    for column in features:
        if df[column].dtype == 'object':
            encoders[column] = LabelEncoder()
            df[column] = encoders[column].fit_transform(df[column].astype(str))

    # Préparer X et y
    X = df[features]
    y = df['sellingprice']

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normaliser les features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Créer et entraîner le modèle
    print("Entraînement du modèle d'IA...")
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Modèle entraîné en {training_time:.2f} secondes")

    # Évaluer la performance
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print(f"Précision du modèle : {r2:.2%}")

    # Créer la voiture test avec des valeurs existantes
    voiture_test = {
        'year': int(voiture_exemple['year']),
        'make': str(voiture_exemple['make']),
        'model': str(voiture_exemple['model']),
        'trim': str(voiture_exemple['trim']),
        'body': str(voiture_exemple['body']),
        'transmission': str(voiture_exemple['transmission']),
        'condition': float(voiture_exemple['condition']),
        'odometer': float(voiture_exemple['odometer']),
        'color': str(voiture_exemple['color']),
        'interior': str(voiture_exemple['interior'])
    }

    # Fonction de prédiction
    def predict_price(input_data):
        input_df = pd.DataFrame([input_data])
        for column in features:
            if input_df[column].dtype == 'object':
                input_df[column] = encoders[column].transform(input_df[column].astype(str))
        input_scaled = scaler.transform(input_df)
        return model.predict(input_scaled)[0]

    # Faire la prédiction
    prix_estime = predict_price(voiture_test)
    prix_reel = voiture_exemple['sellingprice']

    # Afficher les résultats
    print("\n" + "="*50)
    print("DÉTAILS DE LA VOITURE TESTÉE:")
    print("="*50)
    for key, value in voiture_test.items():
        print(f"{key.capitalize()}: {value}")

    print("\n" + "="*50)
    print("RÉSULTATS DE L'ESTIMATION:")
    print("="*50)
    print(f"Prix réel      : ${prix_reel:,.2f}")
    print(f"Prix estimé    : ${prix_estime:,.2f}")
    print(f"Différence     : ${abs(prix_reel - prix_estime):,.2f}")
    print(f"Erreur         : {abs(prix_reel - prix_estime)/prix_reel:.1%}")

except Exception as e:
    print(f"\nUne erreur est survenue : {str(e)}")
    print("Détails supplémentaires:")
    import traceback
    print(traceback.format_exc())
