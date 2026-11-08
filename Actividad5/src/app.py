from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
import pickle
import logging
import time
import uuid

# ===================== CONFIGURACIÓN DE LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CEMEX Monterrey - API Control de Horno",
    description="Motor de inferencia en tiempo real para variables operativas y control de emisiones",
    version="4.2.0-produccion-limpia"
)

# Ruta del modelo
ruta_modelo = "model.pkl"
try:
    with open(ruta_modelo, "rb") as f:
        modelo_horno = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar el archivo model.pkl: {str(e)}")

TARGET_NAMES = ['%Gases NO', '%Gases SO2', '%CO', '%O2', 'Potencia Motor']

class PayloadHorno(BaseModel):
    variables: list = Field(..., description="Lista ordenada con exactamente 19 valores de proceso y lags")

@app.get("/")
def home():
    return {"status": "activo", "planta": "Planta Monterrey"}

@app.post("/predict")
def ejecutar_inferencia(payload: PayloadHorno):
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    # Validación de las 19 variables
    if len(payload.variables) != 19:
        latency = round((time.time() - start_time) * 1000, 2)
        logger.warning(
            f"trace_id={trace_id} | status=422 | latency_ms={latency} | "
            f"error=Se requieren 19 variables, recibidas: {len(payload.variables)}"
        )
        raise HTTPException(
            status_code=422,
            detail=f"Estructura incorrecta. Se requieren exactamente 19 entradas. Recibidas: {len(payload.variables)}"
        )

    try:
        inputs_matriz = np.array(payload.variables).reshape(1, -1)
        predicciones = modelo_horno.predict(inputs_matriz)
        valores_predichos = predicciones[0] if predicciones.ndim > 1 else predicciones
        resultado_mapeado = {TARGET_NAMES[i]: float(valores_predichos[i]) for i in range(len(TARGET_NAMES))}

        latency = round((time.time() - start_time) * 1000, 2)

        # Log exitoso
        logger.info(
            f"trace_id={trace_id} | status=200 | latency_ms={latency} | "
            f"inputs={payload.variables} | predictions={resultado_mapeado}"
        )

        return {
            "status": "exitoso",
            "resultado": resultado_mapeado,
            "trace_id": trace_id,
            "latency_ms": latency
        }

    except Exception as e:
        latency = round((time.time() - start_time) * 1000, 2)
        logger.error(
            f"trace_id={trace_id} | status=500 | latency_ms={latency} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
