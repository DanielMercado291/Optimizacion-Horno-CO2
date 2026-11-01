# Documento de Validación y Pruebas
**Proyecto:** Optimización de Horno CO2 - CEMEX Monterrey  
**Fecha:** 31 de octubre de 2026

## 1. Pruebas Funcionales Realizadas

### Prueba 1: Predicción Normal (Caso Exitoso)
- **Endpoint:** POST /predict
- **Entrada:** 19 variables operativas + lags
- **Resultado esperado:** Status 200 + predicción de 5 variables (%Gases NO, %Gases SO2, %CO, %O2, Potencia Motor)
- **Resultado obtenido:** Correcto. La API responde con los valores predichos.

### Prueba 2: Validación de cantidad de variables
- **Entrada:** Menos de 19 variables
- **Resultado esperado:** Error 422
- **Resultado obtenido:** Correcto. La API rechaza la petición.

## 2. Casos Extremos Evaluados

| Caso | Descripción | Resultado |
|------|-------------|-----------|
| Menos de 19 variables | Se envían solo 3 valores | Error 422 (correcto) |
| Más de 19 variables | Se envían 25 valores | Error 422 (correcto) |
| Valores negativos / extremos | Valores fuera de rango operativo | La API acepta y predice (el modelo es robusto) |
| Lista vacía | variables: [] | Error 422 |

## 3. Conclusiones

- La API valida correctamente que se reciban exactamente 19 variables.
- El modelo (HistGradientBoosting multi-output) responde de forma estable.
- El contenedor Docker funciona de manera confiable en entorno local.
- Se recomienda en el futuro agregar validación de rangos físicos de cada variable.