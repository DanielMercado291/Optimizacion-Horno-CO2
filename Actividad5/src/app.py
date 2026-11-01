from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
import pickle

app = FastAPI(
    title="CEMEX Monterrey - API Control de Horno",
    description="Motor de inferencia en tiempo real para variables operativas y control de emisiones",
    version="4.2.0-produccion-limpia"
)

# Ruta del archivo pkl que tu entrenamiento acaba de actualizar con éxito
ruta_modelo = "model.pkl"  # <-- Así directo, borra cualquier 'model/' que tenga enfrente
try:
    with open(ruta_modelo, "rb") as f:
        modelo_horno = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar el archivo model.pkl: {str(e)}")

# EL ORDEN MATEMÁTICO EXACTO DE TU SCRIPT DE ENTRENAMIENTO (Línea 74)
TARGET_NAMES = ['%Gases NO', '%Gases SO2', '%CO', '%O2', 'Potencia Motor']

# CAPA DE VALIDACIÓN REQUERIDA POR LA RÚBRICA (19 Variables en orden)
class PayloadHorno(BaseModel):
    variables: list = Field(..., description="Lista ordenada con exactamente 19 valores de proceso y lags")

@app.get("/")
def home():
    return {"status": "activo", "planta": "Planta Monterrey"}

@app.post("/predict")
def ejecutar_inferencia(payload: PayloadHorno):
    # Candado de seguridad para asegurar que entren las 19 variables del entrenamiento real
    if len(payload.variables) != 19:
        raise HTTPException(
            status_code=422, 
            detail=f"Estructura incorrecta. Se requieren exactamente 19 entradas. Recibidas: {len(payload.variables)}"
        )
    try:
        # Convertimos la lista de entrada en una matriz de NumPy (1 fila x 19 columnas)
        inputs_matriz = np.array(payload.variables).reshape(1, -1)
        
        # Ejecutamos la predicción multi-salida con Scikit-Learn
        predicciones = modelo_horno.predict(inputs_matriz)
        
        # Extraemos la primera fila de resultados de forma segura
        valores_predichos = predicciones[0] if predicciones.ndim > 1 else predicciones
        
        # Mapeamos los 5 números resultantes contra sus sensores reales sin cruzar etiquetas
        resultado_mapeado = {TARGET_NAMES[i]: float(valores_predichos[i]) for i in range(len(TARGET_NAMES))}
        
        return {
            "status": "exitoso",
            "resultado": resultado_mapeado
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

