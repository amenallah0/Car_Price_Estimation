import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import time

try:
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    df = pd.read_csv('car_prices.csv')
    
    # Feature Engineering avancé
    df['age'] = 2024 - df['year']
    df['miles_per_year'] = df['odometer'] / df['age']
    df['condition_score'] = df['condition'] * (1 - df['age']/100)
    
    # Nettoyage avancé des données
    df = df.dropna()
    df = df[df['sellingprice'] > 100]  # Éliminer les prix aberrants
    df = df[df['odometer'] < 500000]    # Éliminer les kilométrages aberrants
    
    print(f"Nombre de voitures dans la base : {len(df)}")

    # Features sélectionnées (sans les catégories qui causaient l'erreur)
    features = [
        'year', 'make', 'model', 'trim', 'body', 'transmission',
        'condition', 'odometer', 'color', 'interior',
        'age', 'miles_per_year', 'condition_score'
    ]

    # Préparation des encodeurs
    encoders = {}
    for column in features:
        if df[column].dtype == 'object':
            encoders[column] = LabelEncoder()
            df[column] = encoders[column].fit_transform(df[column].astype(str))

    # Préparation des données
    X = df[features]
    y = df['sellingprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création des modèles
    print("\nCréation et optimisation des modèles...")
    
    # Random Forest optimisé
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    # Gradient Boosting optimisé
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )

    # Entraînement des modèles
    print("Entraînement des modèles...")
    start_time = time.time()
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Modèles entraînés en {training_time:.2f} secondes")

    # Prédictions et évaluation
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    # Moyenne pondérée des prédictions
    final_pred = 0.6 * rf_pred + 0.4 * gb_pred

    # Métriques de performance
    print("\nPERFORMANCE DES MODÈLES:")
    print("="*50)
    print(f"Random Forest R² Score: {r2_score(y_test, rf_pred):.4f}")
    print(f"Gradient Boosting R² Score: {r2_score(y_test, gb_pred):.4f}")
    print(f"Ensemble R² Score: {r2_score(y_test, final_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, final_pred):.2%}")

    # Voiture test
    voiture_test = {
        'year': 2015,
        'make': 'Kia',
        'model': 'Sorento',
        'trim': 'LX',
        'body': 'SUV',
        'transmission': 'automatic',
        'condition': 5.0,
        'odometer': 200000.0,
        'color': 'white',
        'interior': 'black'
    }

    # Ajout des features calculées
    voiture_test['age'] = 2024 - voiture_test['year']
    voiture_test['miles_per_year'] = voiture_test['odometer'] / voiture_test['age']
    voiture_test['condition_score'] = voiture_test['condition'] * (1 - voiture_test['age']/100)
    
    # Prédiction avec l'ensemble des modèles
    def predict_price(input_data):
        input_df = pd.DataFrame([input_data])
        
        for column in features:
            if input_df[column].dtype == 'object':
                input_df[column] = encoders[column].transform(input_df[column].astype(str))
        
        rf_price = rf_model.predict(input_df)[0]
        gb_price = gb_model.predict(input_df)[0]
        
        return 0.6 * rf_price + 0.4 * gb_price

    # Prédiction finale
    prix_estime = predict_price(voiture_test)

    # Affichage des résultats
    print("\n" + "="*50)
    print("IMPORTANCE DES CARACTÉRISTIQUES:")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")

    print("\n" + "="*50)
    print("DÉTAILS DE LA VOITURE TESTÉE:")
    print("="*50)
    for key, value in voiture_test.items():
        print(f"{key.capitalize()}: {value}")

    print("\n" + "="*50)
    print("ESTIMATION FINALE:")
    print("="*50)
    print(f"Prix estimé    : ${prix_estime:,.2f}")
    print(f"Intervalle de confiance: ${prix_estime*0.9:,.2f} - ${prix_estime*1.1:,.2f}")

except Exception as e:
    print(f"\nUne erreur est survenue : {str(e)}")
    print("Détails supplémentaires:")
    import traceback
    print(traceback.format_exc())
