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

# 1. Conexión a la infraestructura relacional de MLflow local (Puerto libre 8080)
mlflow.set_tracking_uri("sqlite:///../../mlflow_horno.db")
mlflow.set_experiment("Optimizacion_Horno_Actividad5")

# 2. Carga de datos y aseguramiento del ordenamiento cronológico estricto
df = pd.read_csv("../datos/datos_limp/horno_limpio.csv").dropna()
df = df.sort_index()

X = df.drop(columns=["Fecha", "%Gases NO"])
y = df["%Gases NO"]

# 3. Partición manual por bloques temporales para erradicar el overfitting (70% Train / 30% Test)
limite_entrenamiento = int(len(df) * 0.70)

X_train, X_test = X.iloc[:limite_entrenamiento], X.iloc[limite_entrenamiento:]
y_train, y_test = y.iloc[:limite_entrenamiento], y.iloc[limite_entrenamiento:]

# 4. Declaración formal de la ráfaga de los 4 competidores tabulares multi-salida
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

# 5. Ejecución en bucle automatizado para auditoría y registro inmutable en MLflow
for nombre, modelo in modelos_horno.items():
    with mlflow.start_run(run_name=nombre):
        print(f"Entrenando y registrando {nombre} en MLflow...")
        
        # Ajuste y predicción sobre la línea base operacional
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)

        # Registro dinámico de métricas de rendimiento en el set de prueba
        mlflow.log_metric("R2_Test", r2_score(y_test, preds))

        # Serialización segura nativa en formato pickle para evadir bloqueos de Windows
        mlflow.sklearn.log_model(
            sk_model=modelo, 
            artifact_path="modelo", 
            serialization_format="pickle"
        )
        print(f"¡{nombre} guardado exitosamente!")

