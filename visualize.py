# visualize.py
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors  # asegúrate de instalarlo: pip install mplcursors

def plot_rms(history, max_error, converged_epoch=None, dataset_name=""):
    epochs = [h["epoch"] for h in history]
    rms_errors = [h["rms"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Curva RMS
    ax.plot(epochs, rms_errors, marker='o', linestyle='-', label="Error RMS")

    # Línea horizontal del error máximo permitido
    ax.axhline(y=max_error, color='r', linestyle='--',
               label=f"Error máximo ({max_error})")

    # Punto de convergencia
    if converged_epoch is not None:
        ax.plot(converged_epoch, rms_errors[converged_epoch - 1],
                marker='*', color='green', markersize=12, label="Convergencia")

    # Etiquetas y título
    ax.set_xlabel("Iteraciones (épocas)")
    ax.set_ylabel("Error RMS")
    ax.set_title(f"Entrenamiento del Perceptrón Simple {dataset_name}")
    ax.legend()
    ax.grid(True)

    # Tooltips interactivos con el mouse
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Época {epochs[sel.index]}\nError {rms_errors[sel.index]:.4f}"
    ))

    return fig


def epoch_table(history):
    recs = [{"epoch": h["epoch"], "rms": h["rms"]} for h in history]
    return pd.DataFrame(recs)
