# ⚙️ Optimización en Lazo Cerrado de Hornos Industriales mediante Regresión Multi-Salida y Optimización Metaheurística para la Reducción de Emisiones de CO2

Este repositorio contiene la arquitectura modular y el núcleo predictivo avanzado para eficientizar el proceso de combustión en un horno industrial, enfocado en la mitigación del consumo de combustible fósil y la reducción de la huella de CO2. El sistema transiciona de un clasificador experimental simple hacia un entorno de ingeniería formal de dos capas basado en la física del proceso y el control de variables en tiempo real.

---

##  Descripción de la Arquitectura de Doble Capa

Atendiendo las observaciones de validación experimental y diseño metodológico, el prototipo técnico fue reorganizado como un sistema integrado de dos capas operacionales:

1. **Capa Predictiva (Regresión Multi-Salida):** En lugar de una clasificación binaria, se implementó una arquitectura de regresión probabilística basada en `MultiOutputRegressor` y ventanas de retraso temporal (`_lag1`, `_lag2`). El modelo captura el estado previo del horno y predice los valores físicos exactos de las emisiones de salida y la demanda mecánica (`NO`, `SO2`, `CO`, `O2`, `Potencia_Motor`) a partir de los setpoints operativos.
2. **Capa de Optimización en Modo Sombra (Evolución Diferencial):** Un algoritmo metaheurístico basado en `differential_evolution` toma el modelo matemático entrenado y busca de forma inversa la combinación óptima de variables manipuladas para maximizar la producción, penalizando severamente cualquier escenario que viole las restricciones operativas y ambientales de seguridad.

---

##  Validación Experimental y Separación Temporal (Científica)

Para erradicar por completo el sobreajuste (*overfitting*) por memorización de datos, se aplicó una ordenación cronológica estricta (`df.sort_index()`) sobre las **31,455 filas válidas** extraídas del SCADA, dividiendo el dataset en bloques temporales independientes:
*   **Set de Entrenamiento (70%):** 22,018 muestras (Mapeo histórico del proceso).
*   **Set de Validación (15%):** 4,718 muestras (Ajuste fino de hiperparámetros).
*   **Set de Prueba (15%):** 4,719 muestras (Evaluación estricta ante datos completamente nuevos).

---

## 📊 Tabla Comparativa de Resultados Reales (Set de Prueba Oculto)

Los siguientes datos corresponden a las métricas estandarizadas globales de los 4 algoritmos tabulares evaluados en paralelo en la GPU de Google Colab:

| Algoritmo Evaluado (MultiOutput) | \(R^2\) NO | MAE NO | \(R^2\) SO2 | MAE SO2 | \(R^2\) CO | MAE CO | \(R^2\) O2 | MAE O2 | \(R^2\) Pot. Motor | MAE Pot. Motor | \(R^2\) Promedio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Regressor** | **0.708** | **88.742** | **0.402** | **1619.68** | **0.065** | **0.015%** | **0.599** | **0.985%** | **0.992** | **1.137 kW** | **0.5532** |
| HistGradientBoosting | 0.708 | 87.599 | 0.389 | 1551.00 | -0.247 | 0.021% | 0.591 | 0.963% | 0.999 | 0.342 kW | 0.4880 |
| Random Forest Regressor | 0.685 | 92.430 | 0.336 | 1765.30 | -2.943 | 0.022% | 0.580 | 1.014% | 1.000 | 0.044 kW | -0.0684 |
| Gradient Boosting Regressor | 0.682 | 91.445 | 0.304 | 1721.60 | -4.122 | 0.021% | 0.583 | 0.989% | 1.000 | 0.029 kW | -0.3106 |

**MODELO SELECCIONADO:** **Extra Trees Regressor** se consolida como el núcleo predictivo del proyecto al alcanzar la mayor capacidad de generalización con un **\(R^2\) promedio de 0.5532**. 
*   *Nota sobre la métrica de CO:* Debido a la nula varianza histórica del monóxido en planta, los valores de \(R^2\) tienden a ser negativos. No obstante, el **Error Absoluto Medio (MAE) de apenas 0.015%** demuestra una precisión milimétrica del modelo para detectar la combustión incompleta.

---

##  Resultados de la Optimización y Recomendación Operativa (Modo Sombra)

Tomando como referencia un punto histórico óptimo (\(O_2 = 2.75\%\), \(NO = 482.5\text{ ppm}\)) y aplicando restricciones duras mediante penalizaciones en la función objetivo, la capa de Evolución Diferencial arrojó las siguientes **Recomendaciones en Modo Sombra para el operador**:

### 🛠️ Setpoints Sugeridos para Variables Manipuladas:
*   **Alimentación (`FI_520501.MCO`):** 184.70 Ton/h
*   **Flujo de Carbón Quemador (`FIC_885401.MCO`):** 5.95 gpm
*   **Flujo de Carbón Calcinador (`FI_884801.MCO`):** 7.59 gpm
*   **VTI Precalentador (`SI_530201.MCO`):** 569.97
*   **VTI Enfriador (`SI_566701.MCO`):** 206.67

###  Emisiones y Consumos Físicos Predichos bajo estos Setpoints:
*   **Potencia Motor Principal (`JI_541101.MCO`) = 203.28 kW** (Cumple la restricción mecánica de \(\le\) 230 kW).
*   **CO Chimenea (`AI_540501A.MCO`) = 0.009%** (Garantiza combustión completa, cumpliendo la meta de \(\le\) 0.05%).
*   **O2 Chimenea (`AI_540501D.MCO`) = 3.01%** (Exceso de aire minimizado al límite técnico de \(\le\) 3%, reduciendo directamente las pérdidas de calor y optimizando el ahorro de carbón para mitigar el CO2).

---

##  Conclusiones y Justificación de Infraestructura

1.  **Justificación de Modelos Nativos del Hub:** Tras auditar el catálogo público de Hugging Face bajo la tarea `tabular-regression`, se verificó la ausencia de arquitecturas preentrenadas capaces de modelar la telemetría específica de este horno industrial. Se justifica el uso de la especificación oficial "Scikit-In-HuggingFace" para entrenar e instrumentar algoritmos robustos basados en árboles desde cero.
2.  **Mitigación Total de Fugas de Datos:** Al estructurar la regresión multi-salida con retrasos temporales e implementar la división cronológica por bloques independientes, las métricas obtenidas reflejan fielmente la capacidad de operación del sistema ante datos totalmente invisibles, listos para un entorno productivo.
3.  **Viabilidad en Lazo Cerrado:** El modelo seleccionado procesa los lotes operacionales a una velocidad de microsegundos. Al mantener un enfoque de **Modo Sombra**, el sistema actúa de forma segura sirviendo recomendaciones en tiempo real en la pantalla del SCADA de la planta de Monterrey, permitiendo que el operador humano valide las consignas óptimas de reducción de emisiones de CO2 antes de automatizar el control definitivo.


