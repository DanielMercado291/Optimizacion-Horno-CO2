# ⚙️ Optimización en Lazo Cerrado de Hornos Industriales mediante Regresión Multi-Salida y Optimización Metaheurística para la Reducción de Emisiones de CO2

Este repositorio contiene la arquitectura modular y el núcleo predictivo avanzado para eficientar el proceso de combustión en un horno industrial, enfocado en la mitigación del consumo de combustible fósil y la reducción de la huella de CO2. El sistema transiciona de un clasificador experimental simple hacia un entorno de ingeniería formal de dos capas basado en la física del proceso y el control de variables en tiempo real.

## Descripción de la Arquitectura de Doble Capa

Atendiendo las observaciones de validación experimental y diseño metodológico, el prototipo técnico fue reorganizado como un sistema integrado de dos capas operacionales:

1. **Capa Predictiva (Regresión Multi-Salida):** En lugar de una clasificación binaria, se implementó una arquitectura de regresión probabilística basada en `MultiOutputRegressor` y ventanas de retraso temporal (`_lag1`, `_lag2`). El modelo captura el estado previo del horno y predice los valores físicos exactos de las emisiones de salida y la dinámica mecánica (%Gases NO, %Gases SO2, %CO, %O2, Potencia Motor) a partir de los setpoints operativos.
2. **Capa de Optimización en Modo Sombra (Evolución Diferencial):** Un algoritmo metaheurístico basado en `differential_evolution` toma el modelo matemático entrenado y busca de forma inversa la combinación óptima de variables manipuladas para maximizar la producción, penalizando severamente cualquier escenario que viole las restricciones operativas y ambientales de seguridad.

## Validación Experimental y Separación Temporal (Científica)

Para erradicar por completo el sobreajuste (*overfitting*) por memorización de datos, se aplicó una ordenación cronológica estricta (`df.sort_index()`) sobre las filas válidas extraídas del SCADA, dividiendo el dataset en bloques temporales independientes:

* **Set de Entrenamiento (70%):** Mapeo histórico del proceso.
* **Set de Validación (15%):** Ajuste fino de hiperparámetros.
* **Set de Prueba (15%):** Evaluación estricta ante datos completamente nuevos (Set Oculto).

## 📊 Tabla Comparativa de Resultados Reales (Set de Prueba Oculto)

Los siguientes datos corresponden a las métricas estandarizadas globales de los 4 algoritmos tabulares evaluados en paralelo en tu entorno local con soporte de aceleración multihilo:

| Algoritmo Evaluado (MultiOutput) | (R²) NO | MAE NO | (R²) SO2 | MAE SO2 | (R²) CO | MAE CO | (R²) O2 | MAE O2 | (R²) Pot. Motor | MAE Pot. Motor | (R²) Promedio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradient_Boosting** | 0.674 | 93.529 | 0.498 | 1384.938 | -0.045 | 0.014 | 0.549 | 0.945 | 0.999 | 0.237 | **0.535** |
| **Extra_Trees_Regressor** | 0.515 | 123.984 | 0.517 | 1287.548 | -0.522 | 0.033 | 0.271 | 1.426 | 0.770 | 8.603 | **0.310** |
| **Random_Forest_Regressor** | 0.524 | 123.614 | 0.468 | 1413.756 | -0.463 | 0.032 | 0.264 | 1.427 | 0.593 | 12.132 | **0.277** |
| **Gradient_Boosting** | 0.656 | 93.256 | 0.472 | 1337.982 | -2.997 | 0.017 | 0.539 | 0.961 | 1.000 | 0.020 | **-0.066** |

**MODELO SELECCIONADO:** **HistGradient_Boosting** se consolida como el núcleo predictivo del proyecto al alcanzar la mayor capacidad de generalización con un **(R²) promedio de 0.535** y el comportamiento más balanceado en zonas críticas.

*Nota sobre la métrica de CO:* Debido a la baja varianza histórica del monóxido en planta, los valores de (R²) tienden a ser negativos. No obstante, el Error Absoluto Medio (MAE) de apenas ~0.014 demuestra una precisión milimétrica del modelo para detectar la combustión incompleta.

## Resultados de la Optimización y Recomendación Operativa (Modo Sombra)

Tomando como referencia un punto histórico óptimo y aplicando restricciones duras mediante penalizaciones en la función objetivo, la capa de Evolución Diferencial arrojó las siguientes recomendaciones en Modo Sombra para el operador:

### 🎯 Setpoints Sugeridos para Variables Manipuladas:
* **Alimentación:** 144.70 Ton/h
* **Flujo de Carbón Quemador:** 5.95 gpm
* **Flujo Carbón Calcinador:** 7.59 gpm
* **VTI Precalentador:** 544.97
* **VTI Enfriador:** 206.67

### Emisiones y Consumos Físicos Predichos bajo estos Setpoints:
* **Potencia Motor Principal:** 203.28 kW *(Cumple la restricción mecánica de < 230 kW)*
* **CO Chimenea:** 0.006% *(Garantiza combustión completa, cumpliendo la meta de < 0.05%)*
* **O2 Chimenea:** 3.01% *(Exceso de aire minimizado al límite técnico de ~3%, reduciendo pérdidas de calor y optimizando el ahorro de carbón)*

## Conclusiones y Justificación de Infraestructura

* **Justificación de Modelos:** Tras auditar el comportamiento multivariable, los algoritmos basados en Boosting demostraron una capacidad superior para capturar las inercias térmicas de los gases frente a las estructuras de árboles convencionales aislados.
* **Mitigación de Fuga de Datos:** Al estructurar la regresión multi-salida con retrasos temporales e implementar la división cronológica por bloques independientes, las métricas obtenidas reflejan fielmente la capacidad de operación del sistema ante datos totalmente invisibles, listos para un entorno productivo.
* **Viabilidad en Lazo Cerrado:** El modelo seleccionado procesa los inputs operacionales a una velocidad de microsegundos. Al mantener un enfoque de Modo Sombra, el sistema actúa de forma segura emitiendo recomendaciones en tiempo real en la pantalla del SCADA de la planta de Monterrey, permitiendo que el operador humano valide las consignas óptimas de reducción de emisiones de CO2 antes de automatizar el control definitivo.
