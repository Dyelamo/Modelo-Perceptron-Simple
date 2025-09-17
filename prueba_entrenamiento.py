import pandas as pd
from perceptron import Perceptron
from utils import evaluate
from visualize import plot_error

# Cargar dataset1
df = pd.read_csv("data/dataset1_3entradas.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Entrenar modelo
model = Perceptron(n_inputs=X.shape[1], lr=0.1, max_iter=100, error_threshold=0.01)
model.train(X, y)

# Evaluar
y_pred, acc = evaluate(model, X, y)

# Graficar error
plot_error(model.errors, "Error RMS - Dataset1")

# Guardar pesos
model.save("pesos_dataset1.json")
