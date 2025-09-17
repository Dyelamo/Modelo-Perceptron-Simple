# loader_utils.py
import pandas as pd
import numpy as np
import os

def load_table(path_or_buffer):
    """
    Carga dataset en formato CSV, JSON o Excel.
    Devuelve: df, X, y, meta
    Asume que la última columna es la salida.
    """
    if isinstance(path_or_buffer, str):  # archivo en disco
        ext = os.path.splitext(path_or_buffer)[1].lower()
    else:  # archivo subido desde Streamlit
        ext = os.path.splitext(path_or_buffer.name)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path_or_buffer)
    elif ext == ".json":
        df = pd.read_json(path_or_buffer)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path_or_buffer)
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    n_patterns = df.shape[0]
    n_inputs = df.shape[1] - 1
    n_outputs = 1
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    meta = {
        "Nº Patrones": n_patterns,
        "Nº Entradas": n_inputs,
        "Nº Salidas": n_outputs,
        "Columnas": df.columns.tolist()
    }
    return df, X, y, meta


def preprocess_academic(df):
    """
    Preprocesamiento específico para dataset3 (u otros similares):
     - horas_estudio -> MinMax divide por max
     - asistencia (%) -> divide por 100
     - participacion (0-3) -> divide por 3
    Retorna dataframe nuevo.
    """
    df = df.copy()
    if "horas_estudio" in df.columns:
        max_h = df["horas_estudio"].max() if df["horas_estudio"].max() != 0 else 1
        df["horas_estudio"] = df["horas_estudio"] / float(max_h)
    if "asistencia" in df.columns:
        df["asistencia"] = df["asistencia"] / 100.0
    if "participacion" in df.columns:
        df["participacion"] = df["participacion"] / 3.0
    return df
