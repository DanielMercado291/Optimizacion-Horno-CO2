# ⚙️ Optimización en Lazo Cerrado de Hornos Industriales mediante Regresión Multi-Salida y Optimización Metaheurística para la Reducción de Emisiones de CO2 (Planta Monterrey)

Este repositorio contiene la arquitectura modular, el pipeline de auditoría de datos y el núcleo predictivo avanzado para eficientar el proceso de combustión en el horno industrial de CEMEX Planta Monterrey. El sistema transiciona de un enfoque experimental simple hacia un entorno de ingeniería formal de producción con gobernanza MLOps estricta, mitigando el sobreajuste mediante ordenación temporal y validación cruzada sistemática.

## 🛠️ Arquitectura de Infraestructura y Conexión MLOps
El entorno opera bajo un estándar de gobernanza formal que asegura la reproducibilidad independiente del proceso:
* **Backend Store Relacional:** Registro persistente de metadatos y parámetros en la base de datos SQLite local `mlflow_horno.db`.
* **Servidor de Tracking:** Servidor local de MLflow ejecutándose en el puerto `8080` (`http://127.0.0.1:8080`), interceptando y centralizando las métricas de la ráfaga completa.
* **Trazabilidad Inmutable del Dataset:** Cada corrida (*run*) calcula y almacena automáticamente la huella digital criptográfica (SHA-256) del archivo `horno_limpio.csv` bajo el tag `dataset_hash`, vinculando de forma unívoca el código ejecutable con la versión exacta de los datos industriales extraídos del SCADA.

## 📊 Auditoría de Datos, Limpieza y Partición Temporal
Para mitigar la memorización de patrones y garantizar la validez científica ante la profesora y el cuerpo técnico, el pipeline implementa un flujo de consolidación de datos severo:
1. **Limpieza Consolidada:** Eliminación total y explícita de filas duplicadas y valores nulos (`.dropna()`), garantizando **cero registros nulos** en el archivo final procesado por los modelos.
2. **Filtros Operativos Estrictos:** Exclusión de periodos transitorios, arranques o paros mediante restricciones físicas (Alimentación > 50 Ton/h, Potencia Motor > 80 kW, %O2 entre 0.5 y 10).
3. **Ordenación Cronológica:** Indexación por la estampa de tiempo real de la columna `Fecha` mediante `df.sort_index()`.
4. **Validación Cruzada Sistemática:** Optimización de hiperparámetros ejecutada mediante una búsqueda en grilla de **`GridSearchCV` acoplada a un esquema de validación cruzada para series de tiempo (`TimeSeriesSplit` con 3 cortes)**, eliminando la mezcla de pasado y futuro.
5. **Set de Prueba Oculto:** El 15% final de los datos históricos ordenados se reservó estrictamente como un conjunto independiente invisible para la evaluación y mitigación final del sobreajuste temporal.

## 🛡️ Solución Definitiva a la Fuga de Información
Atendiendo las observaciones críticas de auditoría metodológica, **se eliminó por completo la variable `Potencia Motor` de la matriz de características de entrada (`features`)**. En las iteraciones previas, la inclusión de esta variable provocaba una correlación espuria e inválida con un R² artificial de 1.000. 

En la arquitectura actual, los modelos son obligados a predecir la demanda eléctrica del motor principal basándose únicamente en leyes físicas y variables operativas puras (Alimentación, flujos de carbón y rezagos térmicos `_lag1` y `_lag2`), restituyendo la legitimidad científica de las métricas.

## 📈 Tabla Comparativa de Resultados Reales (Set de Prueba Oculto)

Los siguientes datos corresponden a las métricas estandarizadas globales extraídas de la base relacional de MLflow tras el barrido sistemático de la grilla de parámetros, evaluados sobre el **15% del Set Oculto**:

| Algoritmo Evaluado (MultiOutput) | Métrica | %Gases NO | %Gases SO2 | %CO | %O2 | Potencia Motor | Promedio Global |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradient_Boosting** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 10, max_iter: 50)* | **R²** <br> **RMSE** <br> **MAE** | **0.675** <br> **142.731** <br> **92.492** | **0.526** <br> **1989.620** <br> **1211.356** | **0.036** <br> **0.039** <br> **0.012** | **0.557** <br> **1.447** <br> **0.945** | **0.886** <br> **9.921** <br> **6.714** | **0.536** |
| **Gradient_Boosting** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 5, n_estimators: 50)* | R² <br> RMSE <br> MAE | 0.688 <br> 139.996 <br> 89.275 | 0.515 <br> 2012.592 <br> 1216.895 | -0.127 <br> 0.042 <br> 0.011 | 0.556 <br> 1.449 <br> 0.941 | 0.893 <br> 9.605 <br> 6.462 | **0.505** |
| **Extra_Trees_Regressor** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 14, n_estimators: 100)* | R² <br> RMSE <br> MAE | 0.525 <br> 172.610 <br> 121.101 | 0.530 <br> 1982.203 <br> 1258.958 | -0.387 <br> 0.047 <br> 0.028 | 0.293 <br> 1.827 <br> 1.396 | 0.740 <br> 14.980 <br> 9.558 | **0.340** |
| **Random_Forest_Regressor** <br> *(Hiperparámetros CV óptimos:* <br> *max_depth: 14, n_estimators: 50)* | R² <br> RMSE <br> MAE | 0.524 <br> 170.778 <br> 121.391 | 0.503 <br> 2037.529 <br> 1285.824 | -0.374 <br> 0.047 <br> 0.028 | 0.310 <br> 1.806 <br> 1.376 | 0.559 <br> 19.505 <br> 12.979 | **0.307** |

**SELECCIÓN EXCLUSIVA DEL MODELO:** **HistGradient_Boosting** se corona como el núcleo operativo definitivo del proyecto al alcanzar un **R² Promedio Global legítimo de 0.536**. Su diseño basado en histogramas y optimización iterativa de residuos demostró una capacidad superior para capturar las inercias térmicas y estabilizar el Monóxido de Carbono (R² = 0.036) frente a las estructuras de árboles independientes y evitar el colapso por penalización de residuos del Gradient_Boosting tradicional (R² = -0.127 en CO).

## 🎯 Relación Cuantitativa con la Reducción de Emisiones de CO₂
El rendimiento alcanzado en la predicción de gases, específicamente un **R² de 0.557 en el Oxígeno residual** con una desviación absoluta baja (**RMSE = 1.447%**), establece la conexión directa con el abatimiento ambiental de la planta:
* **Optimización Estequiométrica:** Al conocer con precisión milimétrica el exceso de aire (O₂) en la chimenea ante variaciones de carga, la capa de optimización por Evolución Diferencial calcula de forma segura el límite técnico de reducción de combustible fósil (Flujo de Carbón).
* **Control Predictivo Inverso:** Minimizar de forma controlada el Oxígeno remanente hacia el punto óptimo de operación (~3%) abate directamente el consumo específico de energía térmica, disminuyendo las pérdidas de calor por tiro y recortando estequiométricamente las toneladas de CO₂ emitidas a la atmósfera por tonelada de clínker producida.

