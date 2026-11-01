# Optimización de Horno Industrial CO₂ - CEMEX Monterrey
### Fase II: API REST + Contenedor Docker + Despliegue en la Nube

Sistema de inferencia en tiempo real para predicción multi-salida de variables críticas del horno (gases y potencia) usando un modelo **HistGradientBoosting** entrenado con 19 features.

## Arquitectura

Usuario / Cliente → API (FastAPI) → Modelo (model.pkl) → Predicción

Docker empaqueta la API + el modelo + las dependencias.

- API: FastAPI con endpoint /predict
- Modelo: MultiOutputRegressor (HistGradientBoosting) - 19 features de entrada → 5 salidas
- Contenedor: Docker
- Despliegue: Local + Nube (Render.com)

## Endpoints

- GET / → Health check
- POST /predict → Predicción (requiere exactamente 19 variables)
- GET /docs → Documentación interactiva (Swagger)

## Cómo ejecutar localmente

### Con Docker (recomendado)

docker build -t horno_mlops:latest .
docker run -d --name horno_mlops_container -p 8000:8000 horno_mlops:latest

Abrir: http://localhost:8000/docs

### Sin Docker

pip install -r requirements.txt
cd src
uvicorn app:app --host 0.0.0.0 --port 8000

## Despliegue en la Nube (Render)

La API está desplegada en:

https://optimizacion-horno-co2.onrender.com

Documentación en vivo:

https://optimizacion-horno-co2.onrender.com/docs

Nota: En el plan Free la instancia se duerme después de inactividad. La primera petición puede tardar 30-50 segundos.

## Estructura del Repositorio

Actividad5/
- src/app.py
- src/model_loader.py
- model/model.pkl
- tests/test_api.py
- docs/manual_despliegue.md
- docs/validacion_pruebas.md
- Dockerfile
- requirements.txt
- README.md

## Ejemplo de petición

POST /predict

{
  "variables": [65.0, 320.0, 210.0, 3.5, 2.8, 450.0, 18.0, 0.15, 2.5, 64.8, 64.5, 318.5, 317.0, 208.0, 206.5, 3.4, 3.3, 1.206, 1.172]
}

## Estrategia de Despliegue

- Local: Docker Desktop
- Nube: Render.com (PaaS) con Runtime Docker
- Ventajas: Portabilidad, reproducibilidad y bajo costo

## Documentación adicional

- Manual de Despliegue → docs/manual_despliegue.md
- Validación y Pruebas → docs/validacion_pruebas.md

Autor: Daniel Mercado  
Proyecto: Optimización Horno CO₂ - CEMEX Planta Monterrey  
Fecha: Octubre 2026