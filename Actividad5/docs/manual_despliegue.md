# Manual de Despliegue - Optimización Horno CO2 CEMEX

## 1. Estrategia de Despliegue

Se eligió una estrategia basada en contenedores Docker + PaaS (Platform as a Service).

- Desarrollo y pruebas: Docker local
- Producción recomendada: Render.com (PaaS gratuito)
- Alternativa empresarial: Azure Container Apps o AWS ECS

Ventajas:
- Portabilidad total
- Fácil escalamiento
- Mismo entorno en local y en la nube
- Bajo costo

## 2. Requerimientos Técnicos

- Docker Desktop instalado
- Python 3.11
- Cuenta en Render.com (gratis)
- Git + GitHub

## 3. Despliegue Local

docker build -t horno_mlops:latest .
docker run -d -p 8000:8000 --name horno_mlops_container horno_mlops:latest

Acceder a: http://localhost:8000/docs

## 4. Despliegue en la Nube (Render.com)

1. Subir el proyecto a GitHub
2. Entrar a https://render.com y crear cuenta
3. New → Web Service
4. Conectar el repositorio de GitHub
5. Configurar Runtime: Docker, Root Directory: Actividad5, Branch: main, Plan: Free
6. Clic en Create Web Service
7. Esperar a que construya
8. La API queda disponible en una URL pública

## 5. URL de Producción

https://optimizacion-horno-co2.onrender.com

Documentación: https://optimizacion-horno-co2.onrender.com/docs

## 6. Conclusión

Docker + PaaS permite llevar el modelo a producción de forma rápida, reproducible y económica.