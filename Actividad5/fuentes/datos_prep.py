import pandas as pd
def limpiar():
    df = pd.read_csv("../datos/datos_ini/Actividad_4_Gestion_Proyectos.csv")
    for col in df.columns:
        if col != "Fecha": df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv("../datos/datos_limp/horno_limpio.csv", index=False)
    print("¡Limpieza completada y guardada en datos_limp!")
if __name__ == "__main__": limpiar()