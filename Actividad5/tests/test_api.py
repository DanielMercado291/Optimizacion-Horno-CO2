import requests
import json

# URL de tu API local
URL = "http://localhost:8000/predict"

# Caso normal (19 variables)
payload_normal = {
    "variables": [
        65.0, 320.0, 210.0, 3.5, 2.8,
        450.0, 18.0, 0.15, 2.5, 64.8,
        64.5, 318.5, 317.0, 208.0, 206.5,
        3.4, 3.3, 1.206, 1.172
    ]
}

# Caso extremo: menos variables
payload_pocas = {
    "variables": [65.0, 320.0, 210.0]
}

# Caso extremo: valores negativos raros
payload_extremos = {
    "variables": [-10, 0, 9999, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

def test_prediccion_normal():
    response = requests.post(URL, json=payload_normal)
    print("Status Code:", response.status_code)
    print("Respuesta:", response.json())
    assert response.status_code == 200

def test_error_pocas_variables():
    response = requests.post(URL, json=payload_pocas)
    print("Status Code (pocas variables):", response.status_code)
    print("Detalle:", response.json())
    assert response.status_code == 422

if __name__ == "__main__":
    print("=== PRUEBA NORMAL ===")
    test_prediccion_normal()
    
    print("\n=== PRUEBA CASO EXTREMO (pocas variables) ===")
    test_error_pocas_variables()
    
    print("\n=== Todas las pruebas terminaron ===")