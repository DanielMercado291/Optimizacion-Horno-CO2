import os
import skops.io as sio

def cargar_modelo_industrial():
    """
    Carga el modelo inmutable del horno CEMEX de forma directa.
    """
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(ruta_base, "model", "model.skops")
    
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {ruta_modelo}")
        
    # Carga directa forzando la confianza absoluta del entorno local
    return sio.load(ruta_modelo, trusted=True)


