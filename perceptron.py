# perceptron.py
import numpy as np
import json
from copy import deepcopy

class Perceptron:
    def __init__(self, n_inputs, lr=0.1, max_iter=100, error_threshold=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_inputs = n_inputs
        self.w = np.random.uniform(-0.5, 0.5, n_inputs)
        self.b = np.random.uniform(-0.5, 0.5)
        self.lr = lr
        self.max_iter = int(max_iter)
        self.error_threshold = float(error_threshold)
        self.history = []  # lista de dicts por epoca

    def activation_step(self, z):
        return 1 if z >= 0 else 0

    def forward_single(self, x):
        z = np.dot(self.w, x) + self.b
        y_hat = self.activation_step(z)
        return z, y_hat

    def predict(self, X):
        return np.array([self.forward_single(xi)[1] for xi in X])

    def train(self, X, y, dataset_name=None, verbose=False, save_intermediate=False):
        """
        Entrena y guarda en self.history:
         - epoch, rms, weights, bias, per-pattern details
        Si converge (rms <= error_threshold) marca convergido True.
        Si no converge, identifica epoch cuyo rms está más cercano al error_threshold.
        Si dataset_name se pasa, puede usarse por save() al final.
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        n_samples = X.shape[0]

        self.history = []

        converged = False
        for epoch in range(1, self.max_iter + 1):
            total_error_sq = 0.0
            pattern_records = []

            # online update (por patrón)
            for xi, target in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                y_hat = self.activation_step(z)
                error = target - y_hat  # error lineal (t - yhat)

                # update weights and bias (Regla Delta for discrete-step perceptron)
                # we use the perceptron/delta style update with step activation as requested:
                self.w = self.w + self.lr * error * xi
                self.b = self.b + self.lr * error

                total_error_sq += error**2

                pattern_records.append({
                    "x": xi.tolist(),
                    "target": int(target),
                    "z": float(z),
                    "y_hat": int(y_hat),
                    "error": int(error),
                    "error_sq": float(error**2)
                })

            rms = float(np.sqrt(total_error_sq / n_samples))

            # save snapshot for this epoch
            self.history.append({
                "epoch": epoch,
                "rms": rms,
                "weights": self.w.copy().tolist(),
                "bias": float(self.b),
                "patterns": pattern_records
            })

            if verbose:
                print(f"Epoch {epoch} - RMS: {rms:.6f}")

            # condición de parada
            if rms <= self.error_threshold:
                converged = True
                if verbose:
                    print(f"Convergió en epoch {epoch} con RMS {rms:.6f}")
                break

        # determinar mejor epoch (si no convergió) como la más cercana a error_threshold
        best_epoch_idx = None
        if converged:
            # buscar la primera época donde rms <= threshold
            for i, rec in enumerate(self.history):
                if rec["rms"] <= self.error_threshold:
                    best_epoch_idx = i
                    break
        else:
            # escoger epoch con min |rms - error_threshold|
            diffs = [abs(rec["rms"] - self.error_threshold) for rec in self.history]
            best_epoch_idx = int(np.argmin(diffs))

        best_epoch = self.history[best_epoch_idx]

        # Si save_intermediate True o convergió -> guardar automáticamente pesos "óptimos"
        if dataset_name is not None:
            if converged:
                fname = f"weights_{dataset_name}_converged.json"
            else:
                fname = f"weights_{dataset_name}_closest.json"
            self.save(fname, epoch_record=best_epoch)

        return {
            "converged": converged,
            "best_epoch_index": best_epoch_idx,
            "best_epoch": best_epoch,
            "history": deepcopy(self.history)
        }

    def save(self, filename, epoch_record=None):
        """
        Guarda en JSON: weights actuales + bias.
        Si epoch_record se pasa, guarda esos valores (pesos de esa época).
        """
        if epoch_record is not None:
            data = {
                "weights": epoch_record["weights"],
                "bias": epoch_record["bias"],
                "epoch": int(epoch_record["epoch"]),
                "rms": float(epoch_record["rms"])
            }
        else:
            data = {
                "weights": self.w.tolist(),
                "bias": float(self.b),
                "epoch": None,
                "rms": None
            }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SAVE] Pesos guardados en {filename}")
        return filename

    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.w = np.array(data["weights"])
        self.b = float(data["bias"])
        print(f"[LOAD] Pesos cargados desde {filename} (epoch:{data.get('epoch')}, rms:{data.get('rms')})")
        return data
