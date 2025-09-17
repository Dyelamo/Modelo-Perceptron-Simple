# utils.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def print_weight_matrix(weights, bias):
    arr = np.array(weights)
    s = "Pesos (w):\n" + np.array2string(arr, precision=6, separator=", ")
    s += f"\nBias (umbral b): {bias:.6f}"
    print(s)
    return s

def evaluate_model(model, X, y, verbose=True):
    y_pred = model.predict(X)
    acc = (y_pred == y).mean()
    if verbose:
        print("Accuracy: {:.2f}%".format(acc*100))
    return y_pred, float(acc)

def confusion_report(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)
    return cm, report
