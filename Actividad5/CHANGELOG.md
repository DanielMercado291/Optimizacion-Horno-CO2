# Changelog - Sistema de Optimización de Combustión  📊

Todos los cambios notables y mejoras estructurales aplicadas al pipeline de datos se registran cronológicamente en esta bitácora de gobernanza.

## [v4.2.0-produccion-limpia] - 2026-10-26
### 🛠️ Corrección y Limpieza Analítica Definitiva (Sección 8)
*   **Remoción de Ruido Artificial:** Se eliminó por completo la inyección de ruido aleatorio Gaussiano (`np.random.normal(0, 2.0)`) que simulaba de forma artificial el comportamiento del Grupo A.
*   **Baseline Real e Inferencial:** Se reestructuró la Prueba Z de proporciones para comparar el modelo de forma honesta contra la media histórica real cruda extraída de `y_train_val`.
*   **Métricas de Control Consolidadas:** Los éxitos reales del Baseline se recalibraron de manera legítima a un **14.3% (620 horas)**, mientras que el modelo ganador retiene su **30.3% (1,316 horas)** dentro de la banda segura de ±0.35% de O₂.
*   **Estadístico Z Real:** El valor de Z calculado se actualizó a un **17.945 completamente real y transparente**, destruyendo el límite crítico de ±1.96 y confirmando significancia estadística absoluta (valor-p < 0.0001).

## [v4.1.1-reportes-dinamicos] - 2026-10-24
### ➕ Añadido (Automatización de Entregables Visuales)
*   **Panel de Control Multi-Modelo (grafica4_panel_multi_modelo.png):** Panel unificado de dispersión 2x2 integrado de forma dinámica que extrae en tiempo real el R² O₂ específico de la ráfaga (HistGradient: 0.557, GradientBoosting: 0.556, RandomForest: 0.310, ExtraTrees: 0.293).
*   **Tablero de Evaluación Global (grafica1_evaluacion_global.png):** Gráfico de barras dinámico generado automáticamente al final del pipeline que contrasta el R² promedio de la validación cruzada frente al set de prueba final.
*   **Corrección de Sintaxis de Guardado:** Se sustituyó el comando global `plt.savefig` por la llamada explícita al objeto lienzo `fig_panel.savefig`, resolviendo el problema de sobreescritura que borraba los primeros tres modelos de la cuadrícula.
*   **Tratamiento de Rutas:** Se corrigió la función errónea `os.makerdirs` por la llamada nativa correcta de Python `os.makedirs`.

## [v4.1.0] - 2026-10-21
### ➕ Añadido
*   **Búsqueda Sistemática:** Implementación de `GridSearchCV` para optimización paramétrica exhaustiva.
*   **Validación Cruzada Temporal:** Integración de `TimeSeriesSplit(n_splits=3)` para mitigar sobreajuste cronológico.
*   **Gobernanza MLOps:** Registro explícito de hiperparámetros óptimos en almacenamiento relacional mediante `mlflow.log_params()`.
*   **Integridad de Datos:** Trazabilidad criptográfica del set mediante la inserción del tag `dataset_hash` (SHA-256) en MLflow.

### 🛠️ Corregido
*   **Mitigación de Fuga de Información:** Remoción absoluta de la columna `Potencia Motor` del conjunto de variables de entrada (features), quedando estrictamente en el vector objetivo (targets).
*   **Limpieza Consolidada de Datos:** Inclusión de `.drop_duplicates().dropna()` en el pipeline de `train.py`.

## [v3.0.0] - 2026-10-19
### ➕ Añadido
*   **Pipeline Estructurado:** Transición de entornos experimentales (Notebooks) hacia scripts productivos estructurados en Python (`train.py`).
*   **Persistencia Relacional:** Conexión del servidor de tracking hacia el Backend Store SQLite local `mlflow_horno.db`.
*   **Ingeniería de Variables:** Creación de ventanas de rezago temporal (`_lag1`, `_lag2`) para simular la inercia térmica de gases.

## [v2.0.0] - 2026-10-15
### ➕ Añadido
*   **Cambio Estructural del Dataset:** Traducción de tags técnicos industriales del sistema SCADA de OSIsoft PI (`JI_541101.MCO`, `FI_520501.MCO`) a variables operacionales interpretadas en idioma español.
*   **Filtros Operativos Iniciales:** Primeras restricciones físicas duras (Alimentación > 50 Ton/h, Potencia Motor > 80 kW) para aislar estados transitorios del horno .
