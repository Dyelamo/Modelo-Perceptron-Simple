# loader_utils.py
import pandas as pd

def cargar_tabla(ruta):
    """
    Carga un dataset en formato CSV, JSON o Excel.
    Retorna:
    - df: DataFrame original
    - X: matriz de entradas
    - y: vector de salidas
    - meta: diccionario con información del dataset
    """
    if ruta.endswith(".csv"):
        df = pd.read_csv(ruta)
    elif ruta.endswith(".json"):
        df = pd.read_json(ruta)
    elif ruta.endswith(".xls") or ruta.endswith(".xlsx"):
        df = pd.read_excel(ruta)
    else:
        raise ValueError("Formato de archivo no soportado.")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    meta = {
        "n_patrones": len(df),
        "n_entradas": X.shape[1],
        "n_salidas": 1
    }
    return df, X, y, meta

def preprocesar_academico(df):
    """
    Preprocesa el dataset académico:
    - Normaliza horas de estudio y asistencia.
    - Codifica la participación (0 a 3).
    """
    df["horas_estudio"] = df["horas_estudio"] / df["horas_estudio"].max()
    df["asistencia"] = df["asistencia"] / 100
    return df
