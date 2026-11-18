import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

print("=== Sistema de Monitoreo - Horno CEMEX Monterrey ===\n")

# ===================== SIMULACIÓN DE DATOS DE MONITOREO =====================
# En un caso real estos datos vendrían de los logs de la API
np.random.seed(42)
n_registros = 200

timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(n_registros)][::-1]

latencias = np.random.normal(120, 40, n_registros)
latencias = np.clip(latencias, 20, 2000)

# Predicciones simuladas (valores cercanos a los rangos normales del horno)
o2 = np.random.normal(3.2, 0.4, n_registros)
so2 = np.random.normal(450, 80, n_registros)
no = np.random.normal(180, 30, n_registros)
potencia = np.random.normal(210, 25, n_registros)

# Simular algunos errores y un poco de drift al final
errores = np.random.choice([0, 1], size=n_registros, p=[0.97, 0.03])
o2[-30:] = o2[-30:] + 1.8  # Simular drift en O2 al final

df = pd.DataFrame({
    "timestamp": timestamps,
    "latency_ms": latencias.round(1),
    "o2_pred": o2.round(2),
    "so2_pred": so2.round(1),
    "no_pred": no.round(1),
    "potencia_pred": potencia.round(1),
    "error": errores
})

# ===================== CÁLCULO DE MÉTRICAS =====================
latencia_p95 = np.percentile(df["latency_ms"], 95)
tasa_error = df["error"].mean() * 100
o2_fuera_rango = ((df["o2_pred"] < 2.0) | (df["o2_pred"] > 5.0)).mean() * 100

print(f"Latencia P95: {latencia_p95:.1f} ms")
print(f"Tasa de error: {tasa_error:.2f}%")
print(f"% de predicciones de O2 fuera de rango: {o2_fuera_rango:.1f}%")

# ===================== DETECCIÓN SIMPLE DE ALERTAS =====================
alertas = []

if latencia_p95 > 800:
    alertas.append("ALERTA ALTA: Latencia P95 supera 800 ms")

if tasa_error > 1:
    alertas.append("ALERTA CRÍTICA: Tasa de error superior al 1%")

if o2_fuera_rango > 10:
    alertas.append("ALERTA MEDIA: Más del 10% de predicciones de O2 fuera de rango")

# Detección simple de drift en O2 (comparando últimas 30 vs primeras 100)
media_inicial = df["o2_pred"].iloc[:100].mean()
media_reciente = df["o2_pred"].iloc[-30:].mean()
if abs(media_reciente - media_inicial) > 1.0:
    alertas.append(f"ALERTA ALTA: Posible Data Drift en %O2 (media inicial: {media_inicial:.2f}, reciente: {media_reciente:.2f})")

print("\n=== ALERTAS DETECTADAS ===")
if alertas:
    for a in alertas:
        print("-", a)
else:
    print("No se detectaron alertas.")

# ===================== GENERACIÓN DE GRÁFICAS (DASHBOARD) =====================
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(df["timestamp"], df["latency_ms"], color="steelblue")
plt.axhline(800, color="red", linestyle="--", label="Umbral 800 ms")
plt.title("Latencia del servicio")
plt.ylabel("ms")
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(df["timestamp"], df["o2_pred"], color="green")
plt.axhline(2.0, color="orange", linestyle="--")
plt.axhline(5.0, color="orange", linestyle="--")
plt.title("%O₂ Predicho")
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.plot(df["timestamp"], df["so2_pred"], color="purple")
plt.title("%Gases SO₂ Predicho")
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
plt.plot(df["timestamp"], df["potencia_pred"], color="darkorange")
plt.title("Potencia Motor Predicha")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("dashboard_monitoreo.png", dpi=150)
print("\nGráfica guardada como: dashboard_monitoreo.png")

# Guardar datos para evidencia
df.to_csv("datos_monitoreo.csv", index=False)
print("Datos de monitoreo guardados en: datos_monitoreo.csv")