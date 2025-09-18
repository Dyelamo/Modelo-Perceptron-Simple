# perceptron.py
import numpy as np
import json
import os

class Perceptron:
    def __init__(self, entradas, tasa_aprendizaje=0.1, max_epocas=1000, error_maximo=0.01):
        """
        Inicializa el perceptrón simple.
        Parámetros:
        - entradas: número de variables de entrada
        - tasa_aprendizaje: α (valor entre 0 y 1)
        - max_epocas: número máximo de iteraciones
        - error_maximo: error permitido para detener el entrenamiento
        """

        self.entradas = entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.error_maximo = error_maximo

        # Inicialización aleatoria de los pesos y umbral en rango [-1,1]
        # Fórmula: W = Entradas × Salidas (en este caso Salidas = 1)
        self.pesos = np.random.uniform(-1, 1, entradas)
        # Fórmula: U = Salidas (en este caso 1)
        self.umbral = np.random.uniform(-1, 1)

    def funcion_activacion(self, z):
        """
        Función de activación tipo Escalón:
        Si z >= 0 => salida = 1
        Si z <  0 => salida = 0
        """
        return 1 if z >= 0 else 0

    def propagar(self, x):
        """
        Calcula la salida del perceptrón para una entrada x.
        Fórmula de la función soma:
        z = Σ (x_j * w_j) - U
        """
        z = np.dot(x, self.pesos) - self.umbral
        salida = self.funcion_activacion(z)
        return salida, z

    def entrenar(self, X, y):
        """
        Entrena el perceptrón usando la Regla Delta.
        Solo guarda los pesos óptimos si el error final ≤ error_maximo.
        """
        historia = []
        mejor_epoca = None
        convergio = False

        for epoca in range(self.max_epocas):
            errores_patron = []

            for i in range(len(X)):
                xi = X[i]
                yd = y[i]

                # Propagación
                y_calc, z = self.propagar(xi)

                # Error lineal
                error_lineal = yd - y_calc

                # Actualización de pesos y umbral
                self.pesos = self.pesos + self.tasa_aprendizaje * error_lineal * xi
                self.umbral = self.umbral + self.tasa_aprendizaje * error_lineal * 1

                # Error por patrón
                Ep = abs(error_lineal) / 1
                errores_patron.append(Ep)

            # Error RMS de la época
            # error_rms = sum(errores_patron) / len(errores_patron)
            # historia.append(error_rms)

            # Error RMS (versión cuadrática, más estable)
            errores_cuadrados = [(yd - self.propagar(xi)[0])**2 for xi, yd in zip(X, y)]
            error_rms = np.sqrt(np.mean(errores_cuadrados))
            historia.append(error_rms)


            # Guardar mejor época (mínimo error RMS)
            if mejor_epoca is None or error_rms < mejor_epoca["error_rms"]:
                mejor_epoca = {
                    "epoca": epoca,
                    "error_rms": error_rms,
                    "pesos": self.pesos.copy(),
                    "umbral": self.umbral
                }

            # Condición de parada por error
            if error_rms <= self.error_maximo:
                convergio = True
                break

        return {
            "historia": historia,
            "mejor_epoca": mejor_epoca,
            "convergio": convergio
        }

    def predecir(self, X):
        """
        Predice la salida para un conjunto de entradas X.
        """
        salidas = []
        for i in range(len(X)):
            y_calculada, _ = self.propagar(X[i])
            salidas.append(y_calculada)
        return np.array(salidas)

    # def guardar(self, nombre_archivo, epoca_mejor):
    #     """
    #     Guarda los pesos y umbral óptimos en un archivo JSON.
    #     """
    #     carpeta = "pesos_optimos"
    #     os.makedirs(carpeta, exist_ok=True)

    #     ruta = os.path.join(carpeta, nombre_archivo)
    #     data = {
    #         "pesos": self.pesos.tolist(),
    #         "umbral": float(self.umbral),
    #         "epoca_mejor": epoca_mejor
    #     }
    #     with open(ruta, "w") as f:
    #         json.dump(data, f, indent=4)

    #     return ruta

    def guardar(self, nombre_archivo, epoca_mejor):
        """
        Guarda los pesos y el umbral óptimos en un archivo JSON.
        Convierte todos los valores NumPy a tipos nativos de Python.
        """
        carpeta = "pesos_optimos"
        os.makedirs(carpeta, exist_ok=True)

        ruta = os.path.join(carpeta, nombre_archivo)

        # Convertir epoca_mejor a tipos nativos
        epoca_mejor_serializable = {
            "epoca": int(epoca_mejor["epoca"]),
            "error_rms": float(epoca_mejor["error_rms"]),
            "pesos": [float(w) for w in epoca_mejor["pesos"]],
            "umbral": float(epoca_mejor["umbral"])
        }

        data = {
            "pesos": [float(w) for w in self.pesos],
            "umbral": float(self.umbral),
            "epoca_mejor": epoca_mejor_serializable
        }

        with open(ruta, "w") as f:
            json.dump(data, f, indent=4)

        return ruta


    def cargar(self, archivo):
        """
        Carga pesos y umbral desde un archivo JSON.
        """
        with open(archivo, "r") as f:
            data = json.load(f)

        # Restauramos pesos y umbral como arrays NumPy
        self.pesos = np.array(data["pesos"])
        self.umbral = float(data["umbral"])

    # def cargar(self, archivo):
    #     """
    #     Carga pesos y umbral desde un archivo JSON.
    #     """
    #     with open(archivo, "r") as f:
    #         data = json.load(f)
    #     self.pesos = np.array(data["pesos"])
    #     self.umbral = data["umbral"]
