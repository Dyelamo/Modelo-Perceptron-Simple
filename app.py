# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from perceptron import Perceptron
from loader_utils import cargar_tabla, preprocesar_academico
from utils import evaluar_modelo
from visualizar import graficar_error

st.set_page_config(layout="wide", page_title="Perceptrón Simple")
st.title("Modelo Perceptrón Simple — Interfaz Gráfica")

# Función para listar archivos
def listar_archivos(carpeta, extensiones):
    return [f for f in os.listdir(carpeta) if any(f.endswith(e) for e in extensiones)]

# Rutas de carpetas
CARPETA_DATOS = "data"
CARPETA_PESOS = "pesos_optimos"

# Menú lateral: elegir modo
modo = st.sidebar.selectbox("Seleccionar modo", ["Entrenamiento", "Simulación"])

# =====================================================
# ENTRENAMIENTO
# =====================================================
if modo == "Entrenamiento":
    st.header("Entrenamiento del Perceptrón")

    # Selección del dataset
    datasets = listar_archivos(CARPETA_DATOS, [".csv", ".json", ".xls", ".xlsx"])
    dataset_nombre = st.selectbox("Selecciona un dataset", [""] + datasets)

    if dataset_nombre != "":
        df, X, y, meta = cargar_tabla(os.path.join(CARPETA_DATOS, dataset_nombre))
        st.subheader("Resumen del dataset")
        st.write(meta)
        st.dataframe(df.head())

        # Preprocesamiento académico opcional
        if set(["horas_estudio", "asistencia", "participacion"]).issubset(set(df.columns)):
            if st.checkbox("Aplicar preprocesamiento académico"):
                df = preprocesar_academico(df)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                st.success("Preprocesamiento aplicado correctamente")

        # División en entrenamiento (80%) y prueba (20%)
        split_idx = int(0.8 * len(df))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # -----------------------------
        # Inicialización de pesos y umbral
        # -----------------------------
        st.subheader("Inicialización de pesos y umbral")
        if st.button("Inicializar aleatoriamente"):
            modelo = Perceptron(entradas=X.shape[1])
            st.session_state['modelo'] = modelo

            st.write(f"**Matriz de pesos inicial** (1 x {X.shape[1]}):")
            st.write(np.round(modelo.pesos.reshape(1, -1), 2))

            st.write("**Vector de umbral:**")
            st.write(np.array([round(modelo.umbral, 2)]))

            st.info("Pesos inicializados de forma aleatoria en el rango [-1, 1]")

        # -----------------------------
        # Parámetros de entrenamiento
        # -----------------------------
        st.subheader("Parámetros de entrenamiento")
        tasa_aprendizaje = st.number_input("Tasa de aprendizaje (α)", min_value=0.01, max_value=1.0, value=0.1, format="%.2f")
        max_epocas = st.number_input("Número máximo de épocas", min_value=1, max_value=10000, value=100)
        error_maximo = st.number_input("Error máximo permitido (ε)", min_value=0.0, max_value=1.0, value=0.01, format="%.2f")
        nombre_pesos = st.text_input("Nombre para guardar pesos y umbral", value="pesos_dataset.json")

        # -----------------------------
        # Iniciar entrenamiento
        # -----------------------------
        if st.button("Iniciar entrenamiento") and 'modelo' in st.session_state:
            modelo = st.session_state['modelo']
            modelo.tasa_aprendizaje = tasa_aprendizaje
            modelo.max_epocas = int(max_epocas)
            modelo.error_maximo = float(error_maximo)

            st.info("Entrenando el perceptrón...")
            resultado = modelo.entrenar(X_train, y_train)

            if resultado["convergio"]:
                mejor = resultado["mejor_epoca"]
                st.success(f"✅ El modelo convergió en la época {mejor['epoca']} con RMS = {mejor['error_rms']:.2f}")
                st.write("**Pesos óptimos:**", np.round(mejor["pesos"], 2))
                st.write("**Umbral óptimo:**", round(mejor["umbral"], 2))

                fig = graficar_error(resultado["historia"], error_maximo, mejor['epoca'], dataset_nombre)
                st.pyplot(fig)

                y_pred, acc = evaluar_modelo(modelo, X, y)
                st.write(f"**Exactitud total del modelo:** {acc*100:.2f}%")

                # Guardar solo si aprendió
                ruta_guardado = modelo.guardar(nombre_pesos, mejor)
                st.success(f"Pesos y umbral guardados en {ruta_guardado}")
            else:
                st.warning("⚠️ El modelo no logró converger con los parámetros dados. "
                           "Intente aumentar el número de épocas o ajustar la tasa de aprendizaje.")



# =====================================================
# SIMULACIÓN
# =====================================================
elif modo == "Simulación":
    st.header("Simulación del Perceptrón")

    datasets = listar_archivos(CARPETA_DATOS, [".csv", ".json", ".xls", ".xlsx"])
    pesos_archivos = listar_archivos(CARPETA_PESOS, [".json"])

    dataset_nombre = st.selectbox("Selecciona un dataset", [""] + datasets, key="dataset_sim")
    pesos_archivo = st.selectbox("Selecciona archivo de pesos", [""] + pesos_archivos, key="pesos_sim")

    if dataset_nombre != "" and pesos_archivo != "":
        if st.button("Iniciar simulación"):
            df, X, y, meta = cargar_tabla(os.path.join(CARPETA_DATOS, dataset_nombre))
            st.subheader("Dataset cargado")
            st.write(meta)
            st.dataframe(df.head())

            modelo = Perceptron(entradas=X.shape[1])
            modelo.cargar(os.path.join(CARPETA_PESOS, pesos_archivo))
            st.session_state['modelo_sim'] = modelo
            st.session_state['df_sim'] = df
            st.session_state['X_sim'] = X
            st.session_state['y_sim'] = y

        # Mostrar resultados si ya hay modelo cargado
        if 'modelo_sim' in st.session_state:
            modelo = st.session_state['modelo_sim']
            df = st.session_state['df_sim']
            X = st.session_state['X_sim']
            y = st.session_state['y_sim']

            y_pred, acc = evaluar_modelo(modelo, X, y)
            st.write(f"**Exactitud en dataset:** {acc*100:.2f}%")

            resultados_df = pd.DataFrame(X, columns=df.columns[:-1])
            resultados_df["Salida del Perceptrón"] = y_pred
            resultados_df["Salida Deseada"] = y
            st.dataframe(resultados_df)

            # Probar un patrón manual
            st.subheader("Probar un patrón manual")
            entradas = []
            columnas = st.columns(X.shape[1])
            for i in range(X.shape[1]):
                nombre_col = df.columns[i] if i < len(df.columns) else f"X{i}"
                with columnas[i]:
                    v = st.number_input(f"{nombre_col}", value=float(X[0, i]), key=f"input_{i}")
                    entradas.append(v)

            if st.button("Calcular salida", key="calc_out"):
                salida_predicha, _ = modelo.propagar(np.array(entradas))
                st.write(f"Patrón {list(map(int, entradas))} → Salida del perceptrón = {salida_predicha}")
