import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
	RandomForestRegressor,
	ExtraTreesRegressor,
	GradientBoostingRegressor,
	HistGradientBoostingRegressor,
)
from sklearn.metrics import r2_score

# Conectamos a tu base local SQLite del horno
mlflow.set_tracking_uri("sqlite:///../../mlflow_horno.db")
mlflow.set_experiment("Optimizacion_Horno_Actividad5")

df = pd.read_csv("../datos/datos_limp/horno_limpio.csv").dropna()
X = df.drop(columns=["Fecha", "%Gases NO"])
y = df["%Gases NO"]
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Declaramos los 4 modelos oficiales del proyecto del horno
modelos_horno = {
	"Random_Forest": RandomForestRegressor(
		n_estimators=50, max_depth=10, random_state=42
	),
	"Extra_Trees": ExtraTreesRegressor(
		n_estimators=50, max_depth=10, random_state=42
	),
	"Gradient_Boosting": GradientBoostingRegressor(
		n_estimators=50, random_state=42
	),
	"HistGradient_Boosting": HistGradientBoostingRegressor(
		max_iter=50, random_state=42
	),
}

# Ejecución en bucle para registrar a los 4 competidores independientes
for nombre, modelo in modelos_horno.items():
	with mlflow.start_run(run_name=nombre):
		print(f"Entrenando y registrando {nombre} en MLflow...")
		modelo.fit(X_train, y_train)
		preds = modelo.predict(X_test)

		# Registrar métricas en tu panel de control
		mlflow.log_metric("R2_Test", r2_score(y_test, preds))

		# Guardar el modelo desactivando bloqueos con formato pickle nativo
		mlflow.sklearn.log_model(modelo, "modelo", serialization_format="pickle")
		print(f"¡{nombre} guardado exitosamente!")
