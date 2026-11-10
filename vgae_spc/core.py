"""Entrenamiento y puntuación del VGAE, compartidos por run.py y tune.py."""
import os

import numpy as np
import torch

from .data import Scaler, load_all, time_of_day, windows
from .graph import build_graph
from .model import VGAE, normalize_adj, vgae_loss

# GPU si está disponible (Colab), si no CPU. En CPU el comportamiento es idéntico al original.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# En GPU la CPU sólo lanza operaciones: 1 hilo por proceso evita que varios workers se peleen
# por los pocos núcleos de Colab (suele haber 2). En CPU se usan hasta 4 hilos.
torch.set_num_threads(1 if DEVICE.type == "cuda" else min(4, os.cpu_count() or 1))
if DEVICE.type == "cuda":
    # TF32 en las multiplicaciones (A100/L4; en T4 no tiene efecto)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# Presupuesto de entrenamiento común a la búsqueda (tune.py), las semillas y el modelo final (run.py).
# En GPU la búsqueda usa batches más grandes (menos pasos por época), así que se dan más épocas.
# (VGAE_MAX_EPOCHS sólo para pruebas rápidas del pipeline)
ON_GPU = DEVICE.type == "cuda"
EPOCHS = int(os.environ.get("VGAE_MAX_EPOCHS", 300 if ON_GPU else 150))
PATIENCE = 25 if ON_GPU else 15
BATCH_SIZES = [128, 256, 512] if ON_GPU else [32, 64, 128]

# Configuración usada antes de la optimización (referencia "por defecto")
DEFAULT_HP = dict(W=24, k=3, hidden=64, latent=16, emb=8, lr=1e-3, bs=64,
                  alpha=0.1, beta=1e-3)


def prepare(d, W, k):
    """Grafo, escalado (ajustado sólo con la parte de entrenamiento de dataset03) y ventanas."""
    A_np, _, _ = build_graph(d["sensors"], k=k)
    A = torch.from_numpy(A_np).to(DEVICE)
    cut = int(0.8 * len(d["d03"]))
    scaler = Scaler().fit(d["d03"].iloc[:cut])
    x03 = scaler.transform(d["d03"])
    t03 = time_of_day(d["d03"].index)
    tod = lambda t: windows(t, W).transpose(0, 2, 1).copy()   # (T', W, 2)
    return {
        "A_np": A_np, "A": A, "a_hat": normalize_adj(A), "scaler": scaler, "cut": cut,
        "Xtr": windows(x03[:cut], W), "Xva": windows(x03[cut:], W),
        "x_ph1": x03[cut:], "x04": scaler.transform(d["d04"]), "xte": scaler.transform(d["test"]),
        # hora del día por ventana (sólo la usa la variante temporal="tcn")
        "Ttr": tod(t03[:cut]), "Tva": tod(t03[cut:]),
        "tod_ph1": t03[cut:], "tod04": time_of_day(d["d04"].index), "todte": time_of_day(d["test"].index),
    }


def build_model(n_nodes, hp):
    return VGAE(n_nodes, hp["W"], hidden=hp["hidden"], latent=hp["latent"], emb=hp["emb"],
                temporal=hp.get("temporal", "none"), tcn_ch=hp.get("tcn_ch", 16)).to(DEVICE)


def train(model, Xtr, Xva, A, a_hat, epochs=EPOCHS, bs=64, lr=1e-3, patience=PATIENCE,
          alpha=0.1, beta=1e-3, epoch_callback=None, verbose=True, Ttr=None, Tva=None):
    """Entrena con early stopping sobre la pérdida de validación.
    epoch_callback(ep, model) se llama al final de cada época (p. ej. para pruning)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Los datos se cargan una sola vez en el device (en GPU se evita copiar cada batch)
    Xtr, Xva = torch.from_numpy(Xtr).to(DEVICE), torch.from_numpy(Xva).to(DEVICE)
    if Ttr is not None:
        Ttr, Tva = torch.from_numpy(Ttr).to(DEVICE), torch.from_numpy(Tva).to(DEVICE)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        tot = torch.zeros((), device=DEVICE)   # se acumula en el device; una sola lectura por época
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]
            x_hat, a_log, mu, lv = model(xb, a_hat, None if Ttr is None else Ttr[idx])
            loss, _ = vgae_loss(xb, x_hat, a_log, A, mu, lv, alpha=alpha, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach() * len(xb)
        model.eval()
        with torch.no_grad():
            x_hat, a_log, mu, lv = model(Xva, a_hat, Tva)
            vl, parts = vgae_loss(Xva, x_hat, a_log, A, mu, lv, alpha=alpha, beta=beta)
        rx, ra, kl = (v.item() for v in parts)
        vl = vl.item()
        hist.append((tot.item() / len(Xtr), vl, rx, ra, kl))
        if verbose:
            print(f"ep {ep:3d} train {hist[-1][0]:.4f} val {vl:.4f} "
                  f"(rec_x {rx:.4f} rec_A {ra:.4f} KL {kl:.2f})")
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch_callback is not None:
            epoch_callback(ep, model)
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return np.array(hist)


@torch.no_grad()
def score(model, x, a_hat, W, tod=None):
    """Error cuadrático por ventana (T', N, W) y embedding latente del grafo (T', latent).
    La ventana k termina en el instante k+W-1; las primeras W-1 horas no tienen puntaje."""
    was_training = model.training
    model.eval()
    X = torch.from_numpy(windows(x, W)).to(DEVICE)
    T = None if tod is None else torch.from_numpy(windows(tod, W).transpose(0, 2, 1).copy()).to(DEVICE)
    mu, _ = model.encode(X, a_hat, T)
    x_hat = model.decode_features(mu, a_hat, T)
    model.train(was_training)
    return {"E": ((X - x_hat) ** 2).cpu().numpy(), "latent": mu.mean(1).cpu().numpy()}


def pad(a, W, fill=0):
    return np.concatenate([np.full((W - 1,) + a.shape[1:], fill, dtype=float), a])
