from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn
import pickle

app = FastAPI(
    title="CEMEX Monterrey - API Control de Horno",
    description="Motor de inferencia en tiempo real para variables operativas y control de emisiones",
    version="4.2.0-produccion-limpia"
)

# Las 19 variables operacionales exactas de tu proceso industrial
FEATURE_NAMES = [
    'Alimentacion', 'VTI Precalentador', 'VTI Enfriador', 'Flujo de Carbon quemador', 'Flujo Carbon Calcinador',
    'Alimentacion_lag1', 'VTI Precalentador_lag1', 'VTI Enfriador_lag1', 'Flujo de Carbon quemador_lag1', 'Flujo Carbon Calcinador_lag1',
    'Gases NO_lag1', 'Gases SO2_lag1', 'CO_lag1', 'O2_lag1',
    'Alimentacion_lag2', 'VTI Precalentador_lag2', 'Flujo de Carbon quemador_lag2',
    'Ratio_Carbon', 'Ratio_Carbon_lag1'
]

TARGET_NAMES = ['Gases NO', 'Gases SO2', 'CO', 'O2', 'Potencia Motor']

# Ruta directa del archivo dentro del contenedor Docker
ruta_modelo = "model.pkl"

try:
    with open(ruta_modelo, "rb") as f:
        modelo_horno = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar el archivo model.pkl: {str(e)}")

class PayloadHorno(BaseModel):
    variables: list = Field(..., description="Lista ordenada con los 19 valores de los sensores")

@app.get("/")
def home():
    return {"status": "activo", "planta": "Planta Monterrey"}

@app.post("/predict")
def ejecutar_inferencia(payload: PayloadHorno):
    if len(payload.variables) != 19:
        raise HTTPException(
            status_code=422, 
            detail=f"Estructura incorrecta. Se requieren exactamente 19 entradas. Recibidas: {len(payload.variables)}"
        )
    try:
        df_inputs = pd.DataFrame([payload.variables], columns=FEATURE_NAMES)
        predicciones = modelo_horno.predict(df_inputs)
        
        resultado_mapeado = {TARGET_NAMES[i]: float(predicciones[i]) for i in range(len(TARGET_NAMES))}
        return {
            "status": "exitoso",
            "resultado": resultado_mapeado
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

