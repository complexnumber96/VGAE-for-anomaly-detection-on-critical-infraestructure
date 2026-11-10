"""Comparación VGAE base vs. VGAE + convolución temporal, con el mismo protocolo.

Uso:  python -m vgae_spc.compare_temporal [--seeds 10] [--worker i --workers n] [--threads 2]
      python -m vgae_spc.compare_temporal --report

Ambas variantes usan los hiperparámetros elegidos por Optuna (best_params.json), el mismo
grafo que el modelo final (la adyacencia guardada en su checkpoint), el mismo presupuesto de
entrenamiento que la búsqueda y el modelo final (300 épocas, paciencia 25, también en CPU) y
las mismas semillas (100, 101, ...). La única diferencia es temporal="tcn". Se evalúa la carta EWMA con los parámetros de Optuna, igual que
en la etapa de semillas de tune.py, y la prueba se mira sólo aquí.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from .data import ROOT

OUT = ROOT / "resultados_batadal" / "temporal"
BEST = ROOT / "resultados_batadal" / "optuna_colab" / "best_params.json"
CKPT = ROOT / "resultados_batadal" / "optuna_colab_final" / "vgae.pt"
CONFIGS = {"VGAE base": "none", "VGAE + conv. temporal": "tcn"}
METRICS = ("S", "S_TTD", "S_CM", "TPR", "TNR", "Precision", "F1", "detectados")


def run(n_seeds, worker, n_workers, threads=0, epochs=300, patience=25):
    from . import tune   # importa los datos; sólo hace falta al entrenar
    from .core import DEVICE
    from .model import normalize_adj
    if threads:   # después del import: core.py fija su propio número de hilos al importarse
        torch.set_num_threads(threads)
    # Presupuesto explícito: en CPU core.py usa por defecto uno menor (150 épocas, paciencia 15)
    tune.MAX_EPOCHS, tune.PATIENCE = epochs, patience
    best = json.load(open(BEST))
    # Mismo grafo que el modelo final: build_graph puede resolver empates de distancia de otra forma
    A_np = torch.load(CKPT, weights_only=False, map_location="cpu")["A"]
    prep = tune.get_prep(best["W"], best["k"])
    prep["A_np"], prep["A"] = A_np, torch.from_numpy(A_np).to(DEVICE)
    prep["a_hat"] = normalize_adj(prep["A"])
    jobs = [(c, s) for c in CONFIGS for s in range(100, 100 + n_seeds)][worker::n_workers]
    part = OUT / f"part_{worker}.csv"
    done = set(zip(*pd.read_csv(part)[["config", "seed"]].values.T)) if part.exists() else set()
    for name, seed in jobs:
        if (name, seed) in done:
            continue
        r = tune.fit_and_eval({**best, "temporal": CONFIGS[name]}, seed=seed, eval_test=True)
        row = {"config": name, "seed": seed, "val_loss": r["val_loss"], "epochs": r["epochs"],
               **{f"{ds}_{k}": r[ds][k] for ds in ("dataset04", "test") for k in METRICS},
               **{f"ttd_{a}": v[0] for a, v in r["test"]["ttd"].items()}}
        pd.DataFrame([row]).to_csv(part, mode="a", header=not part.exists(), index=False)
        print(f"[{name}] seed={seed} épocas={r['epochs']} S_d04={r['dataset04']['S']:.3f} "
              f"S_test={r['test']['S']:.3f}", flush=True)


def report():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("part_*.csv"))]).sort_values(["config", "seed"])
    df.to_csv(OUT / "seeds_temporal.csv", index=False)
    cols = [f"{ds}_{k}" for ds in ("dataset04", "test") for k in ("S", "TPR", "TNR", "F1")] + ["val_loss", "epochs"]
    summ = df.groupby("config")[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(OUT / "resumen_temporal.csv")
    piv = {ds: df.pivot(index="seed", columns="config", values=f"{ds}_S").dropna() for ds in ("dataset04", "test")}
    tests = {}
    for ds, p in piv.items():
        a, b = p["VGAE + conv. temporal"], p["VGAE base"]
        tests[ds] = {"n": int(len(p)), "diff_media": float((a - b).mean()),
                     "wilcoxon_p_bilateral": float(stats.wilcoxon(a, b).pvalue) if len(p) > 1 else None,
                     "gana_tcn": int((a > b).sum())}
    ttd = {c: g[[c2 for c2 in df.columns if c2.startswith("ttd_")]].mean().round(1).to_dict() for c, g in df.groupby("config")}
    json.dump({"pruebas": tests, "ttd_medio_test": ttd}, open(OUT / "pruebas_temporal.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (ds, p) in zip(axes, piv.items()):
        for s in p.index:
            ax.plot([0, 1], [p.loc[s, "VGAE base"], p.loc[s, "VGAE + conv. temporal"]], color="0.7", lw=1)
        ax.scatter(np.zeros(len(p)), p["VGAE base"], color="0.45", zorder=3)
        ax.scatter(np.ones(len(p)), p["VGAE + conv. temporal"], color="#008C9E", zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "+ conv. temporal"]); ax.set_xlim(-0.4, 1.4)
        pv = tests[ds]["wilcoxon_p_bilateral"]
        ax.set_title(f"S {ds}" + (f" (Wilcoxon p = {pv:.3g})" if pv is not None else ""))
    axes[0].set_ylabel("S de BATADAL")
    fig.tight_layout(); fig.savefig(OUT / "seeds_temporal.png", dpi=130); plt.close(fig)
    print(summ.to_string()); print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    report() if a.report else run(a.seeds, a.worker, a.workers, a.threads, a.epochs, a.patience)
