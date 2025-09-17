# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from perceptron import Perceptron
from loader_utils import load_table, preprocess_academic
from utils import print_weight_matrix, evaluate_model
from visualize import plot_rms

st.set_page_config(layout="wide", page_title="Perceptrón Simple - App")
st.title("Perceptrón Simple — Interfaz")

# Función para listar archivos
def list_files(folder, exts):
    return [f for f in os.listdir(folder) if any(f.endswith(e) for e in exts)]






DATA_PATH = "data"
WEIGHTS_PATH = "pesos"

# Sidebar: elegir modo
mode = st.sidebar.selectbox("Modo", ["Entrenamiento", "Simulación"])

if mode == "Entrenamiento":
    st.header("Entrenamiento")

    # datasets = list_files(DATA_PATH, ".csv")
    # en Entrenamiento y Simulación:
    datasets = list_files(DATA_PATH, [".csv", ".json", ".xls", ".xlsx"])
    dataset_name = st.selectbox("Selecciona un dataset", [""] + datasets)

    if dataset_name != "":
        df, X, y, meta = load_table(os.path.join(DATA_PATH, dataset_name))
        st.subheader("Resumen del dataset")
        st.write(meta)
        st.dataframe(df.head())

        # preprocesamiento académico opcional
        if set(["horas_estudio", "asistencia", "participacion"]).issubset(set(df.columns)):
            if st.checkbox("Aplicar preprocesamiento Académico"):
                df = preprocess_academic(df)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                st.success("Preprocesamiento aplicado")

        # split 80/20
        split_idx = int(0.8 * len(df))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        st.subheader("Inicialización de pesos y umbral")
        if st.button("Inicializar aleatoriamente"):
            model = Perceptron(n_inputs=X.shape[1])
            st.session_state['model'] = model

            # Mostrar dimensiones + valores redondeados
            st.write(f"Matriz de pesos inicial (1 x {X.shape[1]}):")
            st.write(np.round(model.w.reshape(1, -1), 2))

            st.write("Vector umbral:")
            st.write(np.array([round(model.b, 2)]))


        st.subheader("Parámetros de entrenamiento")
        lr = st.number_input("Tasa de aprendizaje η", min_value=0.01, max_value=1.0, value=0.1, format="%.2f")
        max_iter = st.number_input("Máx iteraciones", min_value=1, max_value=10000, value=100)
        error_max = st.number_input("Error máximo permitido (ε)", min_value=0.0, max_value=1.0, value=0.01, format="%.2f")
        pesos_name = st.text_input("Nombre para guardar pesos", value="pesos_dataset")

        if st.button("Iniciar entrenamiento") and 'model' in st.session_state:
            model = st.session_state['model']
            model.lr = lr
            model.max_iter = int(max_iter)
            model.error_threshold = float(error_max)

            st.info("Entrenando...")
            result = model.train(X_train, y_train, dataset_name=None, verbose=False)

            best = result["best_epoch"]
            st.success(f"Entrenamiento terminado — Época {best['epoch']} RMS = {best['rms']:.2f}")
            st.write("Pesos óptimos:", np.round(best["weights"], 2))
            st.write("Umbrales:", round(best["bias"], 2))

            fig = plot_rms(result["history"], max_error=error_max, converged_epoch=best['epoch'], dataset_name=dataset_name)
            st.pyplot(fig)

            # evaluar en todo el dataset
            y_pred, acc = evaluate_model(model, X, y)
            st.write(f"Accuracy total: {acc*100:.2f}%")

            # guardar pesos
            # fname = model.save(f"{pesos_name}.json", epoch_record=best)
            # st.success(f"Pesos guardados en {fname}")
            # guardar pesos en carpeta 'pesos'
            os.makedirs(WEIGHTS_PATH, exist_ok=True)
            fname = os.path.join(WEIGHTS_PATH, f"{pesos_name}.json")
            model.save(fname, epoch_record=best)
            st.success(f"Pesos y umbrales guardados exitosemente")



elif mode == "Simulación":
    st.header("Simulación")

    # datasets = list_files(DATA_PATH, ".csv")
    # en Entrenamiento y Simulación:
    datasets = list_files(DATA_PATH, [".csv", ".json", ".xls", ".xlsx"])
    weights_files = list_files(WEIGHTS_PATH, ".json")

    dataset_name = st.selectbox("Selecciona un dataset", [""] + datasets, key="dataset_sim")
    weights_file = st.selectbox("Selecciona archivo de pesos", [""] + weights_files, key="weights_sim")

    if dataset_name != "" and weights_file != "":
        if st.button("Iniciar simulación"):
            df, X, y, meta = load_table(os.path.join(DATA_PATH, dataset_name))
            st.subheader("Dataset cargado")
            st.write(meta)
            st.dataframe(df.head())

            model = Perceptron(n_inputs=X.shape[1])
            model.load(os.path.join(WEIGHTS_PATH, weights_file))
            st.session_state['sim_model'] = model
            st.session_state['sim_df'] = df
            st.session_state['sim_X'] = X
            st.session_state['sim_y'] = y

        # Solo mostrar resultados si ya hay modelo cargado
        if 'sim_model' in st.session_state:
            model = st.session_state['sim_model']
            df = st.session_state['sim_df']
            X = st.session_state['sim_X']
            y = st.session_state['sim_y']

            y_pred, acc = evaluate_model(model, X, y)
            st.write(f"Accuracy en dataset: {acc*100:.2f}%")

            result_df = pd.DataFrame(X, columns=df.columns[:-1])
            result_df["Salida Red"] = y_pred
            result_df["Salida Deseada"] = y
            st.dataframe(result_df)

            st.subheader("Probar patrón manual")
            inputs = []
            cols = st.columns(X.shape[1])
            for i in range(X.shape[1]):
                col_name = df.columns[i] if i < len(df.columns) else f"X{i}"
                with cols[i]:
                    v = st.number_input(f"{col_name}", value=float(X[0, i]), key=f"input_{i}")
                    inputs.append(v)
            if st.button("Calcular salida", key="calc_out"):
                y_hat = model.forward_single(np.array(inputs))[1]
                st.write(f"Patrón {list(map(int, inputs))} = {y_hat}")

