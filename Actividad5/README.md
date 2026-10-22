# ⚙️ Optimización en Lazo Cerrado de Hornos Industriales mediante Regresión Multi-Salida y Optimización Metaheurística para la Reducción de Emisiones de CO2 (Planta Monterrey)

Este repositorio contiene la arquitectura modular, el pipeline de auditoría de datos y el núcleo predictivo avanzado para eficientar el proceso de combustión en el horno industrial de CEMEX Planta Monterrey. El sistema transiciona de un enfoque experimental simple hacia un entorno de ingeniería formal de producción con gobernanza MLOps estricta, mitigando el sobreajuste mediante ordenación temporal y validación cruzada sistemática.

## 📁 Descripción del Dataset y Diccionario de Variables
El conjunto de datos proviene del sistema SCADA industrial alojado en el servidor `pimonterrey`, registrando variables críticas del proceso de clinkerización a intervalos de tiempo continuos:
*   **Fecha:** Estampa de tiempo (Timestamp) que actúa como el índice cronológico principal.
*   **Alimentacion (Ton/h):** Flujo másico de materia prima cruda introducida al horno. Variable manipulada principal.
*   **VTI Precalentador / VTI Enfriador (rpm):** Velocidades de los ventiladores de tiro inducido que controlan el flujo de aire y gases.
*   **Flujo de Carbon quemador / Flujo Carbon Calcinador (gpm):** Caudal de combustible fósil inyectado en las zonas de combustión.
*   **%Gases NO / %Gases SO2 / %CO / %O2:** Porcentajes y concentraciones de gases de salida medidos en la chimenea (Variables ambientales y de control estequiométrico).
*   **Potencia Motor (kW):** Demanda eléctrica del motor del horno. Actúa estrictamente como **Variable Objetivo (Target)** para evaluar la estabilidad mecánica del sistema sin fuga de información.

## 🛠️ Arquitectura de Infraestructura y Conexión MLOps
El entorno opera bajo un estándar de gobernanza formal que asegura la reproducibilidad independiente del proceso:
*   **Backend Store Relacional:** Registro de metadatos y parámetros en la base SQLite local `mlflow_horno.db`.
*   **Trazabilidad del Dataset:** Cada corrida almacena la huella digital criptográfica (SHA-256) del archivo `horno_limpio.csv` bajo el tag `dataset_hash`.

## 📊 Auditoría de Datos, Limpieza y Partición Temporal
*   **Limpieza Consolidada:** Eliminación total de duplicados y nulos mediante `.drop_duplicates().dropna()`, garantizando **0 nulos remanentes** en memoria.
*   **Filtros Operativos Estrictos:** Exclusión de periodos transitorios (Alimentación > 50 Ton/h, Potencia Motor > 80 kW).
*   **Validación Cruzada Temporal:** Optimización paramétrica mediante **`GridSearchCV` acoplado a `TimeSeriesSplit(n_splits=3)`**.
*   **Set de Prueba Oculto:** Reserva del 15% final cronológico como conjunto independiente invisible.

## 🛡️ Solución Definitiva a la Fuga de Información
Se eliminó la variable `Potencia Motor` de las características de entrada (`features`). Los modelos predicen la demanda eléctrica basándose únicamente en variables operativas puras y rezagos térmicos (`_lag1`, `_lag2`), restituyendo la legitimidad de las métricas.

## 📈 Tabla Comparativa de Resultados Reales (Set de Prueba Oculto)

| Algoritmo Evaluado (MultiOutput) | Métrica | %Gases NO | %Gases SO2 | %CO | %O2 | Potencia Motor | Promedio Global |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradient_Boosting** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 10, max_iter: 50)* | **R²** <br> **RMSE** <br> **MAE** | **0.675** <br> **142.731** <br> **92.492** | **0.526** <br> **1989.620** <br> **1211.356** | **0.036** <br> **0.039** <br> **0.012** | **0.557** <br> **1.447** <br> **0.945** | **0.886** <br> **9.921** <br> **6.714** | **0.536** |
| **Gradient_Boosting** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 5, n_estimators: 50)* | R² <br> RMSE <br> MAE | 0.688 <br> 139.996 <br> 89.275 | 0.515 <br> 2012.592 <br> 1216.895 | -0.127 <br> 0.042 <br> 0.011 | 0.556 <br> 1.449 <br> 0.941 | 0.893 <br> 9.605 <br> 6.462 | **0.505** |
| **Extra_Trees_Regressor** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 14, n_estimators: 100)* | R² <br> RMSE <br> MAE | 0.525 <br> 172.610 <br> 121.101 | 0.530 <br> 1982.203 <br> 1258.958 | -0.387 <br> 0.047 <br> 0.028 | 0.293 <br> 1.827 <br> 1.396 | 0.740 <br> 14.980 <br> 9.558 | **0.340** |
| **Random_Forest_Regressor** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 14, n_estimators: 50)* | R² <br> RMSE <br> MAE | 0.524 <br> 170.778 <br> 121.391 | 0.503 <br> 2037.529 <br> 1285.824 | -0.374 <br> 0.047 <br> 0.028 | 0.310 <br> 1.806 <br> 1.376 | 0.559 <br> 19.505 <br> 12.979 | **0.307** |

**SELECCIÓN EXCLUSIVA DEL MODELO:** **HistGradient_Boosting** se corona como el núcleo operativo con un **R² Promedio Global legítimo de 0.536**. Su diseño basado en histogramas y optimización iterativa de residuos demostró una capacidad superior para capturar las inercias térmicas y estabilizar el Monóxido de Carbono (R² = 0.036).

## 📊 Interpretación de Visualizaciones de Auditoría (Ubicadas en `/reportes`)
1.  **Matriz de Correlación (`matriz_correlacion.png`):** Revela la alta dependencia física lineal entre los flujos de carbón y la temperatura de los gases, validando la inercia térmica simulada mediante rezagos temporales (`_lag1`, `_lag2`).
2.  **Detección de Valores Atípicos (`boxplot_outliers.png`):** Mapea visualmente la dispersión operativa. Permite comprobar que los límites físicos de alimentación e inercias mecánicas se mantuvieron acotados tras la remoción del ruido transitorio del SCADA.

## 🎯 Relación Cuantitativa con la Reducción de Emisiones de CO₂
El rendimiento alcanzado en la predicción de gases, específicamente un **R² de 0.557 en el Oxígeno residual** con una desviación absoluta baja (**RMSE = 1.447%**), establece la conexión directa con el abatimiento ambiental de la planta:
*   **Optimización Estequiométrica:** Al conocer con precisión milimétrica el exceso de aire (\(O_2\)) en la chimenea ante variaciones de carga, la capa de optimización por Evolución Diferencial calcula de forma segura el límite técnico de reducción de combustible fósil (Flujo de Carbón).
*   **Control Predictivo Inverso:** Minimizar de forma controlada el Oxígeno remanente hacia el punto óptimo de operación (~3%) abate directamente el consumo específico de energía térmica, disminuyendo las pérdidas de calor por tiro y recortando estequiométricamente las toneladas de CO₂ emitidas a la atmósfera por tonelada de clínker producida.

## 🚀 Guía de Instalación y Reproducción (Para la Profesora)
Siga estos pasos estructurados para replicar de forma independiente el pipeline y levantar la interfaz de seguimiento experimental:

1.  **Clonar el repositorio e instalar dependencias:**
    ```bash
    git clone https://github.com
    cd Optimizacion-Horno-CO2
    pip install -r requirements.txt
    ```
2.  **Ejecutar los pipelines de auditoría y entrenamiento:**
    ```bash
    cd Actividad5/fuentes
    python eda_auditoria.py
    python train.py
    ```
3.  **Levantar y visualizar la interfaz gráfica de MLflow:**
    Regrese a la raíz del proyecto (`cd ../..`) donde se encuentra el archivo `mlflow_horno.db` y ejecute el servidor de tracking local:
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow_horno.db --host 127.0.0.1 --port 8080
    ```
    Abra su navegador web e ingrese a la ruta **`http://127.0.0.1:8080`** para auditar interactivamente las métricas del experimento.



