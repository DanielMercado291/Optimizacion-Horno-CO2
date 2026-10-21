import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carga de datos
df = pd.read_csv("../datos/datos_limp/horno_limpio.csv")

# Crear carpeta de reportes si no existe
os.makedirs("../reportes", exist_ok=True)

# 1. Auditoría de nulos y duplicados iniciales
duplicados = df.duplicated().sum()
nulos_totales = df.isnull().sum().sum()
print(f"Auditoría Inicial SCADA: {duplicados} filas duplicadas. {nulos_totales} nulos detectados.")

# CORRECCIÓN RÚBRICA: Consolidación y eliminación total en memoria para la auditoría
df = df.drop_duplicates().dropna().copy()
print(f"Dataset Consolidado post-limpieza: {df.isnull().sum().sum()} nulos remanentes.")

# 2. Generación de Gráfica: Matriz de Correlación Operativa (Filtrando solo datos numéricos)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación Multivariable - Horno Monterrey")
plt.tight_layout()
plt.savefig("../reportes/matriz_correlacion.png")
plt.close()

# 3. Generación de Gráfica: Detección de Outliers (Boxplot)
plt.figure(figsize=(12, 6))
df.boxplot(column=['Alimentacion', 'Potencia Motor', 'Flujo de Carbon quemador'])
plt.title("Auditoría de Valores Atípicos Operativos (Outliers)")
plt.tight_layout()
plt.savefig("../reportes/boxplot_outliers.png")
plt.close()

print("¡Evidencias visuales de auditoría generadas exitosamente en la carpeta /reportes!")
