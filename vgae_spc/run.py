"""Pipeline completo: VGAE entrenado con operación normal + SPC sobre sus salidas.

Uso:  python -m vgae_spc.run
Protocolo:
  dataset03 (normal) 80 % -> entrenamiento del VGAE
  dataset03 (normal) 20 % -> validación / Fase I del SPC (límites de control)
  dataset04 (ataques)     -> ajuste de hiperparámetros del SPC (L, agregación, λ, cuantil)
  --optuna: usa los hiperparámetros del VGAE y del SPC elegidos por tune.py
  --tcn:    variante con convolución temporal por sensor y hora del día (resultados en <carpeta>_tcn)
  test      (ataques)     -> evaluación final, sin reajuste
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .core import DEFAULT_HP, DEVICE, build_model, prepare, score, train
from .model import normalize_adj
from .core import pad as _pad
from .data import ROOT, load_all
from .spc import AGG, SPCMonitor, batadal_score

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# --optuna: usa los hiperparámetros encontrados por tune.py y guarda en results/<carpeta>_final
# --optuna-dir results/optuna_colab: lee best_params.json de otra búsqueda (por defecto results/optuna)
USE_OPTUNA = "--optuna" in sys.argv
OPTUNA_DIR = ROOT / (sys.argv[sys.argv.index("--optuna-dir") + 1] if "--optuna-dir" in sys.argv
                     else "results/optuna")
if USE_OPTUNA:
    _best = json.load(open(OPTUNA_DIR / "best_params.json"))
    HP = {k: _best[k] for k in DEFAULT_HP}
    SPC_OPTUNA = {k: _best[k] for k in ("L", "agg", "lam", "q")}
    OUT = OPTUNA_DIR.parent / f"{OPTUNA_DIR.name}_final"
else:
    HP, SPC_OPTUNA = dict(DEFAULT_HP), None
    OUT = ROOT / "results"
if "--tcn" in sys.argv:
    HP["temporal"] = "tcn"
    OUT = OUT.parent / f"{OUT.name}_tcn"
OUT.mkdir(parents=True, exist_ok=True)
W = HP["W"]


def pad(a, fill=0):
    return _pad(a, W, fill)


def shade(ax, index, y):
    m = np.diff(np.concatenate([[0], y, [0]]))
    for s, e in zip(np.where(m == 1)[0], np.where(m == -1)[0]):
        ax.axvspan(index[s], index[e - 1], color="tab:red", alpha=0.15, lw=0)


LABELS = {"ewma": "EWMA", "shewhart": "EWMA λ=1\n(Shewhart)", "t2": "Hotelling T²\n(latente)", "spe": "SPE (Q)", "ewma_optuna": "EWMA\n(Optuna)"}


def plot_charts(name, index, y, series, path):
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.5 * len(series)), sharex=True)
    for ax, (chart, (v, ucl, al)) in zip(axes, series.items()):
        ax.plot(index, v, lw=0.7, color="tab:blue")
        ax.axhline(ucl, color="k", ls="--", lw=0.8, label="UCL")
        a = al.astype(bool)
        ax.scatter(index[a], v[a], s=4, color="tab:orange", zorder=3, label="alarma")
        shade(ax, index, y)
        ax.set_ylabel(LABELS[chart], fontsize=9)
        if chart in ("t2", "spe"):
            ax.set_yscale("symlog", linthresh=1)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"VGAE + SPC — {name} (sombreado = ataque real)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# Shewhart no se calcula aparte: es la EWMA con λ = 1, incluida en la rejilla. Como referencia
# sin memoria se reporta la mejor configuración con λ = 1 (fila "shewhart").
# CUSUM tampoco: para desplazamientos pequeños es equivalente a la EWMA (Lucas & Saccucci, 1990).
QS = (0.99, 0.995, 0.999)
GRID = {
    "ewma": [dict(L=L, agg=g, lam=l, q=q) for L in (1, 3, 6, 12) for g in AGG
             for l in (0.05, 0.1, 0.2, 0.5, 1.0) for q in QS],
    "t2": [dict(q=q) for q in QS],
    "spe": [dict(L=L, q=q) for L in (1, 3, 6, 12, 24) for q in QS],
}


def main():
    d = load_all()
    sensors = d["sensors"]
    prep = prepare(d, W, HP["k"])
    A, A_np, a_hat, cut = prep["A"], prep["A_np"], prep["a_hat"], prep["cut"]
    Xtr, Xva, x04, xte = prep["Xtr"], prep["Xva"], prep["x04"], prep["xte"]
    print(f"device: {DEVICE}  hiperparámetros: {HP}")
    print(f"sensores={len(sensors)} aristas={int(A_np.sum() / 2)} train={len(Xtr)} val={len(Xva)}")

    model = build_model(len(sensors), HP)
    ckpt = OUT / "vgae.pt"
    if ckpt.exists() and "--retrain" not in sys.argv:
        ck = torch.load(ckpt, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ck["state"])
        # El grafo se evalúa con la adyacencia guardada en el entrenamiento: build_graph puede
        # resolver empates de distancia de forma distinta según la versión de numpy.
        A_np = ck["A"]
        A = torch.from_numpy(A_np).to(DEVICE)
        a_hat = normalize_adj(A)
        print(f"modelo cargado de {ckpt} (usa --retrain para reentrenar); aristas={int(A_np.sum() / 2)}")
    else:
        hist = train(model, Xtr, Xva, A, a_hat, bs=HP["bs"], lr=HP["lr"],
                     alpha=HP["alpha"], beta=HP["beta"], Ttr=prep["Ttr"], Tva=prep["Tva"])
        torch.save({"state": model.state_dict(), "sensors": sensors, "A": A_np, "hp": HP,
                    "mu": prep["scaler"].mu, "sd": prep["scaler"].sd, "W": W}, ckpt)
        pd.DataFrame(hist, columns=["train", "val", "rec_x", "rec_A", "KL"]).to_csv(OUT / "training_history.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist[:, 0], label="train"); ax.plot(hist[:, 1], label="val")
        ax.set_xlabel("época"); ax.set_ylabel("pérdida VGAE"); ax.legend(); fig.tight_layout()
        fig.savefig(OUT / "training_loss.png", dpi=120); plt.close(fig)

    ph1 = score(model, prep["x_ph1"], a_hat, W, prep["tod_ph1"])
    sets = {"dataset04": (score(model, x04, a_hat, W, prep["tod04"]), d["y04"], d["id04"], d["d04"].index),
            "test": (score(model, xte, a_hat, W, prep["todte"]), d["ytest"], d["idtest"], d["test"].index)}

    def evaluate(mon, name):
        s, y, aid, _ = sets[name]
        v, al = mon.run(s["E"], s["latent"])
        return pad(v, np.nan), pad(al).astype(int), batadal_score(y, pad(al).astype(int), aid)

    # 1) Ajuste de hiperparámetros de cada carta en dataset04 (el test no se mira)
    rows, best = [], {}
    for chart, grid in GRID.items():
        for p in grid:
            mon = SPCMonitor(chart, **p).fit(ph1["E"], ph1["latent"])
            m = evaluate(mon, "dataset04")[2]
            rows.append({"chart": chart, **p, **{k: v for k, v in m.items() if k != "ttd"}})
            if chart not in best or m["S"] > best[chart][1]:
                best[chart] = (mon, m["S"])
            if chart == "ewma" and p["lam"] == 1.0 and ("shewhart" not in best or m["S"] > best["shewhart"][1]):
                best["shewhart"] = (mon, m["S"])
    pd.DataFrame(rows).to_csv(OUT / "tuning_dataset04.csv", index=False)
    if SPC_OPTUNA is not None:
        p = SPC_OPTUNA
        best["ewma_optuna"] = (SPCMonitor("ewma", L=p["L"], agg=p["agg"], lam=p["lam"], q=p["q"])
                               .fit(ph1["E"], ph1["latent"]), None)

    # 2) Evaluación con la configuración elegida
    results, series, summ = {}, {"dataset04": {}, "test": {}}, []
    for chart, (mon, _) in best.items():
        results[chart] = {"params": mon.describe()}
        for name in sets:
            v, al, m = evaluate(mon, name)
            results[chart][name] = m
            series[name][chart] = (v, mon.ucl, al)
    # Regla combinada: alarma si EWMA o T² superan su límite
    for name, (s, y, aid, _) in sets.items():
        al = series[name]["ewma"][2] | series[name]["t2"][2]
        results.setdefault("ewma|t2", {"params": "EWMA OR T²"})[name] = batadal_score(y, al, aid)
        series[name]["_comb"] = al
    for chart, r in results.items():
        for name in sets:
            m = r[name]
            summ.append({"carta": chart, "dataset": name,
                         **{k: round(m[k], 3) for k in ("S", "S_TTD", "S_CM", "TPR", "TNR", "Precision", "F1")},
                         "detectados": f'{m["detectados"]}/{m["n_ataques"]}', "params": str(r["params"])})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "summary.csv", index=False)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(summ.to_string(index=False))

    for name, (s, y, aid, idx) in sets.items():
        ser = {k: v for k, v in series[name].items() if not k.startswith("_")}
        plot_charts(name, idx, y, ser, OUT / f"spc_{name}.png")
        pd.DataFrame({"y": y, "attack_id": aid, **{f"stat_{k}": v[0] for k, v in ser.items()},
                      **{f"alarm_{k}": v[2] for k, v in ser.items()}, "alarm_ewma|t2": series[name]["_comb"]},
                     index=idx).to_csv(OUT / f"scores_{name}.csv")

    # 3) Diagnóstico: z-score por sensor (horizonte de la carta EWMA elegida) en el test
    ns = best["ewma"][0].ns
    s, y, aid, idx = sets["test"]
    z = pad(ns.transform(s["E"]), np.nan)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(np.nan_to_num(z, nan=0).T, aspect="auto", cmap="magma", vmin=-2, vmax=8,
                   extent=[0, len(idx), len(sensors) - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(len(sensors))); ax.set_yticklabels(sensors, fontsize=7)
    m = np.diff(np.concatenate([[0], y, [0]]))
    for st, en, a_ in zip(np.where(m == 1)[0], np.where(m == -1)[0], range(8, 15)):
        ax.axvline(st, color="cyan", lw=0.8); ax.axvline(en, color="cyan", lw=0.8, ls=":")
        ax.text(st, -0.8, f"#{a_}", color="tab:cyan", fontsize=8)
    ticks = np.linspace(0, len(idx) - 1, 8).astype(int)
    ax.set_xticks(ticks); ax.set_xticklabels([idx[t].strftime("%d/%m") for t in ticks])
    fig.colorbar(im, label="z-score del error de reconstrucción")
    ax.set_title("Contribución de cada sensor al error del VGAE — test")
    fig.tight_layout(); fig.savefig(OUT / "sensor_contrib_test.png", dpi=130); plt.close(fig)

    top = {}
    for a_ in range(8, 15):
        top[a_] = list(pd.Series(np.nanmean(z[aid == a_], 0), index=sensors).nlargest(4).round(1).items())
    print("\nSensores con mayor z-score medio por ataque (test):")
    for a_, t in top.items():
        print(f"  #{a_}: {t}")
    with open(OUT / "top_sensors_test.json", "w") as f:
        json.dump({str(k): v for k, v in top.items()}, f, indent=1)


if __name__ == "__main__":
    main()
