import numpy as np
import pandas as pd
from datetime import datetime

def predict_price(model, area, city_name, postal_code, encoder):
    """
    Fonction de prédiction du prix d'un bien immobilier en fonction de la superficie, la ville et l'année actuelle.
    
    :param model: Modèle de régression entraîné
    :param area: Superficie du bien en m²
    :param city_name: Nom de la ville
    :param postal_code: Code postal du bien
    :param encoder: Encodeur pour transformer les variables catégoriques
    :return: Prix prédit du bien immobilier
    """

    print("---------Prédiction---------")

    # Transformer l'entrée en DataFrame avec les mêmes colonnes que l'entraînement
    input_df = pd.DataFrame([[city_name, postal_code]], columns=['nom_commune', 'code_postal'])

    # Obtenir l'année actuelle
    current_year = datetime.now().year

    # Créer un DataFrame avec toutes les features dans le bon ordre
    input_data = pd.DataFrame([[input_df.loc[0, 'nom_commune'], input_df.loc[0, 'code_postal'], area, current_year]],columns=model.feature_names_in_)

    # Prédire le prix
    predicted_price = model.predict(input_data)

    print(f"Le prix estimé pour {city_name} ({postal_code}) avec une surface de {area} m² est de {predicted_price[0]:,.2f} €")
    return predicted_price[0]
