# #️⃣ Changelog

## [v4.1.0] - 2026-10-21
### Añadido
- **Búsqueda Sistemática:** Implementación de `GridSearchCV` para optimización paramétrica exhaustiva.
- **Validación Cruzada Temporal:** Integración de `TimeSeriesSplit(n_splits=3)` para mitigar sobreajuste cronológico.
- **Métricas de Auditoría:** Inclusión de la evaluación cuadrática media (**RMSE**) por variable en set de prueba oculto.
- **Gobernanza MLOps:** Registro explícito de hiperparámetros óptimos en almacenamiento relacional mediante `mlflow.log_params()`.
- **Integridad de Datos:** Trazabilidad criptográfica del set mediante la inserción del tag `dataset_hash` (SHA-256) en MLflow.
- **Auditoría Visual:** Creación de script `eda_auditoria.py` para la generación autónoma de diagramas de caja y correlaciones.

### Cambiado
- **Evolución de Fuentes:** Reincorporación del algoritmo `Gradient_Boosting` para completar la ráfaga comparativa simétrica de 4 modelos tabulares.

### Corregido
- **Mitigación de Fuga de Información:** Remoción absoluta de la columna `Potencia Motor` del conjunto de variables de entrada (`features`), quedando estrictamente en el vector objetivo (`targets`).
- **Limpieza Consolidada de Datos:** Inclusión de `.drop_duplicates().dropna()` en el pipeline de `train.py`, reduciendo drásticamente la presencia de registros nulos y duplicados en memoria.

## [v3.0.0] - 2026-10-19
### Añadido
- **Pipeline Estructurado:** Transición de entornos experimentales (Notebooks) hacia scripts productivos estructurados en Python (`train.py`).
- **Persistencia Relacional:** Conexión del servidor de tracking hacia el Backend Store SQLite local en el puerto 8080.
- **Ingeniería de Variables:** Creación de ventanas de rezago temporal (`_lag1`, `_lag2`) para simular la inercia térmica de gases.

## [v2.0.0] - 2026-10-15
### Añadido
- **Cambio Estructural del Dataset:** Depuración del SCADA y traducción de tags técnicos industriales del sistema OSIsoft PI (`JI_541101.MCO`, `FI_520501.MCO`) a variables operacionales interpretadas en idioma español.
- **Filtros Operativos Iniciales:** Primeras restricciones físicas duras (Alimentación > 50 Ton/h, Potencia Motor > 80 kW) para aislar estados transitorios del horno Monterrey.
