import pandas as pd
from perceptron import Perceptron
from utils import evaluate

# Cargar dataset1
df = pd.read_csv("data/dataset1_3entradas.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Crear modelo y cargar pesos
model = Perceptron(n_inputs=X.shape[1])
model.load("pesos_dataset1.json")

# Evaluar simulación
y_pred, acc = evaluate(model, X, y)

# Probar un nuevo patrón
nuevo_patron = [1, 0, 1]
resultado = model.forward(nuevo_patron)
print(f"Nuevo patrón {nuevo_patron} → salida: {resultado}")
