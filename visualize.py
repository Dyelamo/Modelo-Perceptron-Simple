# visualize.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_rms(history, title="Error RMS vs Iteraciones"):
    rms = [h["rms"] for h in history]
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(epochs, rms, marker='o', linestyle='-')
    ax.set_xlabel("Iteración (época)")
    ax.set_ylabel("Error RMS")
    ax.set_title(title)
    ax.grid(True)
    return fig

def epoch_table(history):
    recs = [{"epoch": h["epoch"], "rms": h["rms"]} for h in history]
    return pd.DataFrame(recs)
