# utils.py
import numpy as np

def evaluar_modelo(modelo, X, y):
    """
    Evalúa el perceptrón sobre un conjunto de datos.
    Retorna:
    - y_pred: salidas predichas por el modelo
    - exactitud: porcentaje de aciertos
    
    Fórmula de exactitud:
    exactitud = (predicciones correctas / total de patrones) × 100
    """
    y_pred = modelo.predecir(X)
    exactitud = np.mean(y_pred == y)
    return y_pred, exactitud
