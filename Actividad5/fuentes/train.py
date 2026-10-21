import os
import hashlib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================================================================
# 1. INFRAESTRUCTURA DE GOBERNANZA (MLFLOW & SQLITE)
# ==============================================================================
mlflow.set_tracking_uri("sqlite:///../../mlflow_horno.db")
mlflow.set_experiment("Optimizacion_Horno_Alineado_Rubrica")

# ==============================================================================
# 2. CARGA, AUDITORÍA Y LIMPIEZA CONSOLIDADA DEL DATASET
# ==============================================================================
ruta_csv = "../datos/datos_limp/horno_limpio.csv"
df_raw = pd.read_csv(ruta_csv)

# Eliminación drástica y consolidada de duplicados y nulos (Requisito de Rúbrica)
df = df_raw.drop_duplicates().dropna().copy()

# Indexación temporal estricta para evitar desfases cronológicos
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df = df.set_index('Fecha').sort_index()

# Filtros operativos estrictos para estados estables del horno CEMEX Planta Monterrey
df = df[
    (df['Alimentacion'] > 50) & 
    (df['Potencia Motor'] > 80) & 
    (df['Flujo de Carbon quemador'] > 0) & 
    (df['Flujo Carbon Calcinador'] > 0) & 
    (df['%O2'] > 0.5) & (df['%O2'] < 10) & 
    (df['%Gases NO'] > 0) & (df['%Gases NO'] < 1200) & 
    (df['%CO'] >= 0) & (df['%CO'] < 2)
].copy()

# Calcular huella digital (SHA-256) del dataset para asegurar reproducibilidad e integridad
with open(ruta_csv, "rb") as f:
    dataset_hash = hashlib.sha256(f.read()).hexdigest()

# ==============================================================================
# 3. INGENIERÍA DE VARIABLES Y CORRECCIÓN DE LA FUGA DE INFORMACIÓN
# ==============================================================================
# Variables operativas de control que sirven como predictores puros
cols_operativas = ['Alimentacion', 'VTI Precalentador', 'VTI Enfriador', 'Flujo de Carbon quemador', 'Flujo Carbon Calcinador']

# Generamos Lags únicamente de las variables operativas y de los gases históricos
# NOTA EXPLICATIVA: Se elimina por completo 'Potencia Motor' de las variables de entrada actuales.
# Esto erradica la fuga de información, obligando al modelo a predecir la potencia basándose en leyes operativas.
for col in cols_operativas + ['%Gases NO', '%Gases SO2', '%CO', '%O2']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

df['Ratio_Carbon'] = df['Flujo de Carbon quemador'] / (df['Flujo Carbon Calcinador'] + 0.1)
df['Ratio_Carbon_lag1'] = df['Ratio_Carbon'].shift(1)
df = df.dropna()

# Definición estricta de las matrices de entrada y salida
features = [
    'Alimentacion', 'VTI Precalentador', 'VTI Enfriador', 'Flujo de Carbon quemador', 'Flujo Carbon Calcinador',
    'Alimentacion_lag1', 'VTI Precalentador_lag1', 'VTI Enfriador_lag1', 'Flujo de Carbon quemador_lag1', 'Flujo Carbon Calcinador_lag1',
    '%Gases NO_lag1', '%Gases SO2_lag1', '%CO_lag1', '%O2_lag1',
    'Alimentacion_lag2', 'VTI Precalentador_lag2', 'Flujo de Carbon quemador_lag2',
    'Ratio_Carbon', 'Ratio_Carbon_lag1'
]
# Colocamos 'Potencia Motor' ÚNICAMENTE como variable de salida (Target) junto con los gases
targets = ['%Gases NO', '%Gases SO2', '%CO', '%O2', 'Potencia Motor']

X = df[features]
y = df[targets]

# ==============================================================================
# 4. REPARTICIÓN CRONOLÓGICA CIENTÍFICA POR BLOQUES
# ==============================================================================
n = len(df)
val_end = int(n * 0.85)

X_train_val, y_train_val = X.iloc[:val_end], y.iloc[:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# Esquema de validación cruzada para series de tiempo (Evita la mezcla de pasado y futuro)
tscv = TimeSeriesSplit(n_splits=3)

# ==============================================================================
# 5. RÁFAGA DE ENTRENAMIENTO CON BÚSQUEDA SISTEMÁTICA DE HIPERPARÁMETROS (GRIDSEARCH)
# ==============================================================================
motores_busqueda = {
    "Random_Forest": GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid={'n_estimators': [50, 100], 'max_depth': [10, 14]},
        cv=tscv, scoring='r2', n_jobs=-1
    ),
    "Extra_Trees": GridSearchCV(
        estimator=ExtraTreesRegressor(random_state=42, n_jobs=-1),
        param_grid={'n_estimators': [50, 100], 'max_depth': [10, 14]},
        cv=tscv, scoring='r2', n_jobs=-1
    ),
        "Gradient_Boosting": GridSearchCV(
        estimator=MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        param_grid={'estimator__n_estimators': [50, 100], 'estimator__max_depth': [5]},
        cv=tscv, scoring='r2', n_jobs=-1
    ),
    "HistGradient_Boosting": GridSearchCV(
        estimator=MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
        param_grid={'estimator__max_iter': [50, 100], 'estimator__max_depth': [10, 14]},
        cv=tscv, scoring='r2', n_jobs=-1
    )
}

# ==============================================================================
# 6. EJECUCIÓN, EVALUACIÓN ESTADÍSTICA Y REGISTRO MLOPS CORREGIDO
# ==============================================================================
for nombre, grid in motores_busqueda.items():
    with mlflow.start_run(run_name=nombre):
        print(f"\nEjecutando Grid Search Sistemático para: {nombre}...")
        
        # Ajuste de la grilla de búsqueda utilizando validación cruzada temporal
        grid.fit(X_train_val, y_train_val)
        
        # Extracción del mejor estimador validado
        mejor_modelo = grid.best_estimator_
        
        # Registro de tags de trazabilidad inmutable (Dataset e Infraestructura de Código)
        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("pipeline_version", "v4.0.0-fuga-corregida")
        
        # REGISTRO EXPLÍCITO DE PARÁMETROS EN MLFLOW (Exigido en la auditoría de la profesora)
                # REGISTRO EXPLÍCITO DE PARÁMETROS EN MLFLOW
        if nombre == "HistGradient_Boosting":
            mlflow.log_param("best_max_iter", grid.best_params_['estimator__max_iter'])
            mlflow.log_param("best_max_depth", grid.best_params_['estimator__max_depth'])
        elif nombre == "Gradient_Boosting":
            mlflow.log_param("best_n_estimators", grid.best_params_['estimator__n_estimators'])
            mlflow.log_param("best_max_depth", grid.best_params_['estimator__max_depth'])
        else:
            mlflow.log_param("best_n_estimators", grid.best_params_['n_estimators'])
            mlflow.log_param("best_max_depth", grid.best_params_['max_depth'])

            
        print(f"   Mejores hiperparámetros seleccionados por CV: {grid.best_params_}")
        
        # Evaluación en el conjunto de prueba oculto independiente (15% final)
        preds = mejor_modelo.predict(X_test)
        preds_df = pd.DataFrame(preds, columns=targets, index=X_test.index)
        
        lista_r2 = []
        print(f"--- Métricas en Set de Prueba Oculto para {nombre} ---")
        for col in targets:
            r2_col = r2_score(y_test[col], preds_df[col])
            mae_col = mean_absolute_error(y_test[col], preds_df[col])
            rmse_col = np.sqrt(mean_squared_error(y_test[col], preds_df[col])) # Inclusión sistemática de RMSE
            
            # Formateo compatible para llaves en la interfaz de MLflow
            nombre_metrica = col.replace("%", "Porcentaje_").replace(" ", "_")
            mlflow.log_metric(f"Test_R2_{nombre_metrica}", r2_col)
            mlflow.log_metric(f"Test_MAE_{nombre_metrica}", mae_col)
            mlflow.log_metric(f"Test_RMSE_{nombre_metrica}", rmse_col)
            
            lista_r2.append(r2_col)
            print(f" Variable: {col:<16} | R2: {r2_col:.3f} | RMSE: {rmse_col:.3f} | MAE: {mae_col:.3f}")
        
        r2_promedio = np.mean(lista_r2)
        mlflow.log_metric("Test_R2_Promedio_Global", r2_promedio)
        print(f"--> R2 PROMEDIO GLOBAL EN TEST: {r2_promedio:.3f}")
        
        # Guardado seguro habilitando TODOS los tipos nativos requeridos por la ráfaga en skops
        mlflow.sklearn.log_model(
            sk_model=mejor_modelo, 
            artifact_path="modelo",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ]
        )
        print(f"¡{nombre} registrado exitosamente con metadatos completos!")

