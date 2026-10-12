
#  Optimización en Lazo Cerrado de Equipos de CEMENTO mediante IA Tabular para la Reducción de Emisiones de CO2

Este repositorio contiene el desarrollo e implementación funcional de un sistema de Inteligencia Artificial enfocado en la eficientización del proceso de combustión en hornos industriales. Utilizando modelos de aprendizaje automático avanzados y la infraestructura de Hugging Face Tabular Hub, el sistema audita variables de proceso para garantizar una operación limpia que mitigue las toneladas de CO2 liberadas a la atmósfera.

---

##  Descripción del Proyecto y Caso de Uso
El proyecto analiza las **Variables Manipuladas** del horno/molino como Alimentación, Flujos de Carbón en Quemador y Calcinador, y Tiros Volumétricos de Aire para predecir si el sistema opera en un estado **Eficiente (Clase 1: Combustión Óptima)** o **Ineficiente (Clase 0: Alto consumo / Desperdicio energético)**.

El estado del sistema se rige bajo **cuatro restricciones (*constraints*) concurrentes**:
1. **%CO < 0.05:** Evita combustión incompleta (pérdidas térmicas).
2. **%Gases NO < 500:** Controla las emisiones para cumplir normativas ambientales.
3. **%O2 < 3:** Reduces el exceso de aire para optimizar el calor y bajar el CO2.
4. **Potencia Motor <= 230 kW:** Restricción física de hardware para proteger el equipo.

---

## Entorno de Ejecución e Infraestructura
* **Hardware Requerido:** Aceleración por hardware mediante GPU NVIDIA T4 probada en Google Colab.
* **Software Base:** Python 3.10+
* **Librerías Clave:** `transformers`, `datasets`, `scikit-learn`, `pandas`, `numpy`, `huggingface_hub`.

---

##  Pasos de Ejecución 

Siga estos pasos de manera secuencial para reproducir el experimento de inferencia y ajuste fino (*Fine-Tuning*):

1. **Clonar el repositorio:**
   ```unset
   git clone https://github.com
   cd Optimizacion-Horno-CO2
   ```
2. **Instalar el Entorno de Ejecución:**
   ```unset
   pip install -r requirements.txt
   ```
3. **Cargar los Datos en tu Entorno:**
   Asegúrese de colocar el archivo `Actividad_4_Gestion_Proyectos.csv` dentro de la ruta del directorio de ejecución.
4. **Ejecutar el Pipeline de Inferencia:**
   Corra el cuaderno correspondiente celda por celda o ejecute el script principal de Python.

---

## Tabla Comparativa de Resultados Reales (Post-Fine-Tuning)
Tras aplicar la estrategia de *Cost-Sensitive Fine-Tuning* (Ajuste Fino por Penalización) para resolver el desbalance de sensores provocado por los constraints estrictos, las métricas reales de los 2,704 registros operacionales arrojaron el siguiente desempeño:

| Modelo Tabular (HF Fine-Tuned) | Accuracy | Precision | Recall | F1-Score | Latencia Promedio (s/m) | Costo Computacional |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HF-Sklearn/Gradient-Boosting** | **0.9763** | **0.9831** | **0.4487** | **0.6162** | **0.000006 s** | Bajo-Medio (GPU) |
| **HF-Sklearn/Random-Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000018 s | Medio (Alto en RAM) |
| **HF-Sklearn/Logistic-Regression**| 0.6498 | 0.0902 | 0.7974 | 0.1621 | 0.000000 s | Ultra-Bajo (Apto PLC) |

---

##  Conclusiones Técnicas del Experimento
1. **Éxito del Fine-Tuning:** El balanceo por pesos penalizados rescató al modelo **Gradient Boosting**, duplicando su sensibilidad operativa y elevando el **F1-Score a un sólido 62%**.
2. **Confiabilidad Industrial:** La **Precision del 98.31%** en Gradient Boosting es el parámetro de seguridad más valioso para la planta, reduciendo el riesgo de falsos positivos operacionales a menos del 2%.
3. **Inviabilidad Lineal:** El colapso del F1-Score 16% de la Regresión Logística demuestra que el comportamiento termodinámico de las chimeneas es altamente no lineal, obligando a usar los modelos del ecosistema de Hugging Face.
4. **Inferencia en Tiempo Real:** Operando a una latencia de **6 microsegundo
