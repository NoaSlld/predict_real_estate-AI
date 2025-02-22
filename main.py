import pandas as pd
from src import analyse
from src.models import regressionLineare, regressionRandomForest, regressionLightGBM
from src.input import get_user_input
from src.prediction import predict_price
from src.loads import load_prepare_data, load_or_train_model


MODEL_RF_PATH = "./data/random_forest.pkl"
ENCODER_PATH = "./data/encoder.pkl"
MODEL_LGB_PATH = "./data/lightgbm.pkl"

def main():
    # Visualisation des données
    # displayGraphs()

    X_train, X_val, X_test, y_train, y_val, y_test, encoder = load_prepare_data()

    # Entraînement des modèles
    print("--------- Régression Linéaire ---------")
    modeleLineaire = regressionLineare(X_train, y_train, X_val, y_val, X_test, y_test)
    print("* Modèle linéaire entrainé")


    print("----------- Random Forest -----------")
    modeleRandomForest, encoder = load_or_train_model(
        regressionRandomForest, X_train, y_train, X_val, y_val, X_test, y_test, encoder, MODEL_RF_PATH, ENCODER_PATH)

    print("----------- LightGBM -----------")
    modeleLightGBM, encoder = load_or_train_model(
        regressionLightGBM, X_train, y_train, X_val, y_val, X_test, y_test, encoder, MODEL_LGB_PATH, ENCODER_PATH)
    print("Modèle LightGBM prêt")


    # Récupération des entrées utilisateur
    area, city_name, postal_code = get_user_input(encoder)

    # Prédictions avec les modèles
    predict_price(modeleLineaire, area, city_name, postal_code, encoder)
    print("avec la régression linéaire")

    predict_price(modeleRandomForest, area, city_name, postal_code, encoder)
    print("avec Random Forest")

    predict_price(modeleLightGBM, area, city_name, postal_code, encoder)
    print("avec LightGBM")



if __name__ == "__main__":
    main()


def displayGraphs():
    analyse.evolutionPricesOverYears()
    analyse.mostExpensiveSquareMetre()
    analyse.biggestPriceIncreaseOverYears()
    analyse.repartitionMontantVentes()
