from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_val, y_val, X_test, y_test):
    # Prédictions sur validation et test
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = mse ** 0.5  # Racine carrée de MSE

    print(f"R² : {r2:.3f}")
    print(f"MAE : {mae:.2f} € (moyenne des erreurs absolues)")
    print(f"MSE : {mse:.2f} (moyenne des erreurs quadratiques)")
    print(f"RMSE : {rmse:.2f} (erreur quadratique moyenne)")
