# visualizar.py
import matplotlib.pyplot as plt

def graficar_error(historia, error_maximo, epoca_convergencia, nombre_dataset):
    """
    Grafica el error RMS por época de entrenamiento.
    
    Fórmula de error por época:
    Ep = Σ |EL_i| / N_salidas
    RMS = Σ Ep / N_patrones
    
    Parámetros:
    - historia: lista con el error RMS en cada época
    - error_maximo: umbral de error definido por el usuario
    - epoca_convergencia: época donde se alcanzó el mejor resultado
    - nombre_dataset: dataset utilizado
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(historia)), historia, marker="o", label="Error RMS")

    # Línea horizontal de error máximo
    ax.axhline(y=error_maximo, color="r", linestyle="--", label="Error máximo permitido")

    # Línea vertical de convergencia
    ax.axvline(x=epoca_convergencia, color="g", linestyle="--", label="Época de convergencia")

    ax.set_title(f"Entrenamiento del perceptrón ({nombre_dataset})")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Error RMS")
    ax.legend()
    ax.grid(True)

    return fig
