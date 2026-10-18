# Optimizacion Horno CEMEX - Actividad 5

## Resumen del Proyecto
Este proyecto implementa un pipeline modular para la optimizacion del proceso de combustion en un horno industrial, enfocado en la mitigacion de emisiones de CO2 y calibracion de setpoints operativos basados en 9,241 filas de datos historicos.

## Decisiones sobre el Dataset
Se tomo la decision analitica de **mantener el 100%% de las variables operativas** (Alimentacion, VTI Precalentador, VTI Enfriador, Flujos de Carbon, Potencia, Llenado y Gases), ya que contienen informacion critica de sensores industriales indispensables para el proceso.

## Validacion Cruzada y Tabla Comparativa de Modelos
Se realizo el ajuste de hiperparametros mediante **Grid Search con Validacion Cruzada (K-Fold)**, registrando las metricas estandarizadas de rendimiento directamente en el servidor local de MLflow:

| Algoritmo Evaluado | Hiperparametros Ajustados | R2 Test (Metrica) | Estado en MLflow |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | n_estimators=50, max_depth=10 | 0.2145 | Registrado |
| **HistGradient Boosting** | max_iter=100, lr=0.1 | **0.3482** | **Ganador Registrado** |

## Conclusiones
El modelo **HistGradient Boosting Regressor** se consolido como el algoritmo ganador con un **R2 de 0.3482**, siendo el modelo base exportado hacia el optimizador de Evolucion Diferencial para el control eficiente de setpoints.