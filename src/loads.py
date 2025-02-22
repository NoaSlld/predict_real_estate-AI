import pandas as pd
import os
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from src.models import regressionRandomForest

MODEL_PATH = "../data/regressionRandomForest.pkl"
ENCODER_PATH = "../data/regressionRandomForest_encoder.pkl"


def load_prepare_data():
    print("---------Chargement---------")

    # Chargement du dataset en ne gardant que les colonnes pertinentes
    df = pd.read_csv("./data/full_cleaned.csv", usecols=['date_mutation', 'valeur_fonciere', 'code_postal', 'nom_commune', 'surface_reelle_bati'], dtype={'code_departement': str})
    print("* Dataset chargé")

    print(df[['valeur_fonciere', 'surface_reelle_bati']].describe())


    # Conversion de la date en année uniquement
    df['date_mutation'] = pd.to_datetime(df['date_mutation']).dt.year
    print("* Conversion de la date effectuée")

    # Encodage des variables catégoriques
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[['nom_commune', 'code_postal']] = encoder.fit_transform(df[['nom_commune', 'code_postal']])
    print("* Encodage terminé")

    # Séparation des variables
    y = df['valeur_fonciere']
    X = df.drop(columns=['valeur_fonciere'])

    # Division en train (70%), test (20%), validation (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, shuffle=True, random_state=42)
    print("* Séparation du dataset effectuée")

    return X_train, X_val, X_test, y_train, y_val, y_test, encoder



def load_or_train_model(train_function, X_train, y_train, X_val, y_val, X_test, y_test, encoder, model_path, encoder_path):

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        print("* Chargement du modèle et de l'encodeur existants...")
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        print("* Modèle et encodeur chargés")
    else:
        print("* Aucun modèle trouvé, entraînement en cours...")
        model = train_function(X_train, y_train, X_val, y_val, X_test, y_test, encoder, model_path, encoder_path)
        
        # Sauvegarde du modèle et de l'encodeur après entraînement
        joblib.dump(model, model_path)
        joblib.dump(encoder, encoder_path)
        print(f"* Modèle sauvegardé sous '{model_path}'")
        print(f"* Encodeur sauvegardé sous '{encoder_path}'")
    
    return model, encoder
