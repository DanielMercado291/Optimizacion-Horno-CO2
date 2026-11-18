import mlflow
from datetime import datetime
import logging

# ===================== CONFIGURACIÓN =====================
logging.basicConfig(
    filename="incidentes.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

mlflow.set_tracking_uri("sqlite:///mlflow_horno.db")
mlflow.set_experiment("Simulacion_Incidentes_Horno")

print("=" * 60)
print("SIMULACIÓN DE INCIDENTES - HORNO CEMEX MONTERREY")
print("=" * 60)

# ===================== INCIDENTE 1 =====================
print("\n[INCIDENTE 1] Alta tasa de errores del servicio")
print("-" * 50)

pasos_runbook_1 = """
Pasos ejecutados del Runbook 1 (Alta tasa de errores):
1. Se verificó el estado del servicio en Render.
2. Se revisaron los logs de la aplicación en busca de errores.
3. Se confirmó una tasa de error del 4%.
4. Se documentó la acción recomendada: redespliegue o rollback si el problema persiste.
5. Se registró el incidente y las acciones tomadas.
"""

with mlflow.start_run(run_name="Incidente_1_Alta_Tasa_Errores"):
    tasa_error = 4.0
    severidad = "Crítica"
    
    print(f"Tasa de error detectada: {tasa_error}%")
    print(f"Severidad: {severidad}")
    print(pasos_runbook_1)
    
    # Registro en log
    logging.info(f"INCIDENTE 1 | Tipo: Alta tasa de errores | Tasa: {tasa_error}% | Severidad: {severidad}")
    logging.info(pasos_runbook_1)
    
    # Registro en MLflow
    mlflow.log_metric("tasa_error_porcentaje", tasa_error)
    mlflow.log_param("severidad", severidad)
    mlflow.log_param("runbook_ejecutado", "Runbook 1 - Alta tasa de errores")
    mlflow.set_tag("incidente", "Alta tasa de errores")
    mlflow.set_tag("estado", "Documentado y contenido")
    
    # Guardar los pasos como artifact de texto
    with open("pasos_runbook_1.txt", "w", encoding="utf-8") as f:
        f.write(pasos_runbook_1)
    mlflow.log_artifact("pasos_runbook_1.txt")
    
    print("→ Incidente y pasos del runbook registrados en MLflow y en incidentes.log")

# ===================== INCIDENTE 2 =====================
print("\n[INCIDENTE 2] Data Drift en %O₂")
print("-" * 50)

pasos_runbook_2 = """
Pasos ejecutados del Runbook 3 (Data Drift):
1. Se confirmó el cambio en la media de %O₂ (de 3.23% a 5.08%).
2. Se revisó la gráfica de %O₂ para validar visualmente el drift.
3. Se analizaron los datos recientes del periodo.
4. Se documentó la recomendación de revisar condiciones operativas del horno.
5. Se recomendó evaluar un posible reentrenamiento del modelo.
6. Se registró el incidente y las acciones tomadas.
"""

with mlflow.start_run(run_name="Incidente_2_Data_Drift_O2"):
    media_inicial = 3.23
    media_reciente = 5.08
    severidad = "Alta"
    
    print(f"Media inicial de %O₂: {media_inicial}")
    print(f"Media reciente de %O₂: {media_reciente}")
    print(pasos_runbook_2)
    
    # Registro en log
    logging.info(f"INCIDENTE 2 | Tipo: Data Drift %O2 | Media inicial: {media_inicial} | Media reciente: {media_reciente}")
    logging.info(pasos_runbook_2)
    
    # Registro en MLflow
    mlflow.log_metric("media_o2_inicial", media_inicial)
    mlflow.log_metric("media_o2_reciente", media_reciente)
    mlflow.log_metric("diferencia_o2", round(media_reciente - media_inicial, 2))
    mlflow.log_param("severidad", severidad)
    mlflow.log_param("runbook_ejecutado", "Runbook 3 - Data Drift")
    mlflow.set_tag("incidente", "Data Drift en %O2")
    mlflow.set_tag("estado", "Recomendación de revisión / reentrenamiento")
    
    # Guardar los pasos como artifact
    with open("pasos_runbook_2.txt", "w", encoding="utf-8") as f:
        f.write(pasos_runbook_2)
    mlflow.log_artifact("pasos_runbook_2.txt")
    
    print("→ Incidente y pasos del runbook registrados en MLflow y en incidentes.log")

print("\n" + "=" * 60)
print("SIMULACIÓN FINALIZADA")
print("Revisa MLflow → experimento: Simulacion_Incidentes_Horno")
print("En cada run, entra a la pestaña Artifacts para ver los pasos del runbook.")
print("=" * 60)