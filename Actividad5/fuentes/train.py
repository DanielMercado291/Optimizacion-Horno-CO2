import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
mlflow.set_tracking_uri("sqlite:///../../mlflow_horno.db")
mlflow.set_experiment("Optimizacion_Horno_Actividad5")
df = pd.read_csv("../datos/datos_limp/horno_limpio.csv").dropna()
X = df.drop(columns=["Fecha", "%Gases NO".replace("%%%%", "%%")])
y = df["%Gases NO".replace("%%%%", "%%")]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for max_iteracion in [50, 100]:
    with mlflow.start_run(run_name=f"HistGradient_iter_{max_iteracion}"):
        model = HistGradientBoostingRegressor(max_iter=max_iteracion, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mlflow.log_param("max_iter", max_iteracion)
        mlflow.log_metric("R2_Test", r2_score(y_test, preds))
        mlflow.sklearn.log_model(model, "modelo", serialization_format="pickle")
        print(f"Corrida con max_iter={max_iteracion} registrada exitosamente.")