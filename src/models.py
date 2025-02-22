from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import lightgbm as lgb
from lightgbm import early_stopping
from src.evaluation import evaluate_model


def regressionLineare(X_train, y_train, X_val, y_val, X_test, y_test):

    model = LinearRegression()
    print("* Modèle créé")

    print("* Début de l'entraînement...")
    model.fit(X_train, y_train)
    print("* Entraînement terminé")

    evaluate_model(model, X_val, y_val, X_test, y_test)

    return model


def regressionRandomForest(X_train, y_train, X_val, y_val, X_test, y_test, encoder, model_path, encoder_path):

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("* Modèle créé")

    print("* Début de l'entraînement...")
    model.fit(X_train, y_train)
    print("* Entraînement terminé")

    # Sauvegarde du modèle et de l'encodeur avec les chemins passés en paramètre
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"* Modèle sauvegardé sous '{model_path}'")
    print(f"* Encodeur sauvegardé sous '{encoder_path}'")

    evaluate_model(model, X_val, y_val, X_test, y_test)

    return model


def regressionLightGBM(X_train, y_train, X_val, y_val, X_test, y_test, encoder, model_path, encoder_path):
    
    # Initialisation du modèle avec les hyperparamètres
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)

    print("* Début de l'entraînement...")
    
    # Entraînement avec early stopping
    model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='l2',
          callbacks=[early_stopping(50)])
    
    print("* Entraînement terminé")
    
    # Sauvegarde du modèle et de l'encodeur
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"* Modèle sauvegardé sous '{model_path}'")
    print(f"* Encodeur sauvegardé sous '{encoder_path}'")
    
    # Évaluation du modèle
    evaluate_model(model, X_val, y_val, X_test, y_test)
    
    return model
