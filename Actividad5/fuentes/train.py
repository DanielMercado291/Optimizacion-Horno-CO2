import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Conexión a la infraestructura relacional de MLflow local (Puerto 8080)
mlflow.set_tracking_uri("sqlite:///../../mlflow_horno.db")
mlflow.set_experiment("Optimizacion_Horno_Actividad5")

# 2. Carga de datos directa del CSV limpio
df = pd.read_csv("../datos/datos_limp/horno_limpio.csv")

# === LOGICA DE CEMEX: PASO 3 (PROCESAR TIEMPO E INDEXAR) ===
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df = df.set_index('Fecha')

# === LÓGICA DE CEMEX: PASO 4 (FILTROS OPERATIVOS) ===
df = df[
    (df['Alimentacion'] > 50) & 
    (df['Potencia Motor'] > 80) & 
    (df['Flujo de Carbon quemador'] > 0) & 
    (df['Flujo Carbon Calcinador'] > 0) & 
    (df['%O2'] > 0.5) & (df['%O2'] < 10) & 
    (df['%Gases NO'] > 0) & (df['%Gases NO'] < 1200) & 
    (df['%CO'] >= 0) & (df['%CO'] < 2)
].copy()

# Ordenamiento cronológico estricto sobre el índice de TIEMPO real
df = df.sort_index()

# === LÓGICA DE CEMEX: PASO 5 (INGENIERÍA DE FEATURES - LAGS) ===
cols_lag = [
    'Alimentacion', 'VTI Precalentador', 'VTI Enfriador',
    'Flujo de Carbon quemador', 'Flujo Carbon Calcinador',
    'Potencia Motor', '%Gases NO', '%Gases SO2', '%CO', '%O2'
]

for col in cols_lag:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)

df['Ratio_Carbon'] = df['Flujo de Carbon quemador'] / (df['Flujo Carbon Calcinador'] + 0.1)
df['Ratio_Carbon_lag1'] = df['Ratio_Carbon'].shift(1)

df = df.dropna()

# === LÓGICA DE CEMEX: PASO 6 (MATRICES EXACTAS) ===
features = [
    'Alimentacion', 'VTI Precalentador', 'VTI Enfriador',
    'Flujo de Carbon quemador', 'Flujo Carbon Calcinador', 'Potencia Motor',
    'Alimentacion_lag1', 'VTI Precalentador_lag1', 'VTI Enfriador_lag1',
    'Flujo de Carbon quemador_lag1', 'Flujo Carbon Calcinador_lag1', 'Potencia Motor_lag1',
    '%Gases NO_lag1', '%Gases SO2_lag1', '%CO_lag1', '%O2_lag1',
    'Alimentacion_lag2', 'VTI Precalentador_lag2', 'Flujo de Carbon quemador_lag2',
    'Ratio_Carbon', 'Ratio_Carbon_lag1'
]

targets = ['%Gases NO', '%Gases SO2', '%CO', '%O2', 'Potencia Motor']

X = df[features]
y = df[targets]

# === LOGICA DE CEMEX: CORRECCIÓN DE PARTICIÓN TEMPORAL DE 3 BLOQUES ===
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Separación idéntica a tus imágenes originales para entrenamiento y prueba final
X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]  # Evaluamos sobre el último 15%

print(f"Estructura CEMEX Reconstruida -> Train: {len(X_train)} | Test (Evaluación Final): {len(X_test)}")

# === LÓGICA DE CEMEX: HIPERPARÁMETROS ORIGINALES (200 estimadores / profundidad 14) ===
modelos_horno = {
    "Random_Forest": RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_split=8, random_state=42, n_jobs=-1),
    "Extra_Trees": ExtraTreesRegressor(n_estimators=200, max_depth=14, min_samples_split=8, random_state=42, n_jobs=-1),
    "Gradient_Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)),
    "HistGradient_Boosting": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=10, random_state=42)),
}

# 5. Bucle de tracking en MLflow con evaluación exacta
for nombre, modelo in modelos_horno.items():
    with mlflow.start_run(run_name=nombre):
        print(f"\nEntrenando modelo: {nombre}...")
        
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        preds_df = pd.DataFrame(preds, columns=targets, index=X_test.index)
        
        lista_r2 = []
        print(f"--- Métricas de Validación Final para {nombre} ---")
        for col in targets:
            r2_col = r2_score(y_test[col], preds_df[col])
            mae_col = mean_absolute_error(y_test[col], preds_df[col])
            
            nombre_metrica = col.replace("%", "Porcentaje_").replace(" ", "_")
            mlflow.log_metric(f"R2_{nombre_metrica}", r2_col)
            mlflow.log_metric(f"MAE_{nombre_metrica}", mae_col)
            
            lista_r2.append(r2_col)
            print(f" Variable: {col:<23} | R2: {r2_col:.3f} | MAE: {mae_col:.3f}")
        
        r2_promedio = np.mean(lista_r2)
        mlflow.log_metric("R2_Promedio_Global", r2_promedio)
        print(f"--> R2 PROMEDIO GLOBAL: {r2_promedio:.3f}")

        mlflow.sklearn.log_model(sk_model=modelo, artifact_path="modelo", serialization_format="pickle")
        print(f"¡{nombre} registrado con éxito en MLflow!")



