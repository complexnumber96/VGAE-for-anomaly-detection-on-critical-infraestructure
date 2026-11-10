"""Optimización de hiperparámetros del VGAE + SPC con Optuna, con evidencia de que
la configuración elegida no es fruto del azar.

Uso:  python -m vgae_spc.tune [--tpe 100] [--random 50] [--seeds 10] [--stage all|search|seeds|report]
      [--out resultados/optuna] [--storage ruta.db|ruta.log] [--parallel]
      [--seed-worker i --seed-workers n]   (etapa seeds repartida en n procesos)

  --storage *.log usa JournalStorage (archivo con bloqueo, seguro para varios procesos);
  --parallel activa constant_liar en TPE y corta en el total de trials pedido, para
  lanzar varios workers sobre el mismo estudio (p. ej. en una GPU de Colab).

Protocolo (el test NO se usa durante la búsqueda):
  - Objetivo: S de BATADAL en dataset04 con la carta EWMA sobre z-scores por sensor.
    Optuna busca a la vez los hiperparámetros del VGAE (arquitectura, entrenamiento,
    grafo) y los de la carta EWMA (horizonte L, agregación, λ, cuantil del UCL).
  - Estudio principal: TPE multivariante (semilla fija).
  - Control: búsqueda aleatoria con el mismo espacio y el mismo objetivo.
  - Robustez: la mejor configuración de TPE, la mejor aleatoria y la configuración
    por defecto se reentrenan con varias semillas; recién aquí se evalúa el test.

Evidencia que se genera en resultados/optuna/:
  1. history.png         curva del mejor valor acumulado TPE vs aleatorio
  2. tpe_vs_random.png   distribución de S por trial + test de Mann-Whitney
  3. importances.png     importancia de cada hiperparámetro (fANOVA)
  4. slices.png          S vs valor de cada hiperparámetro importante
  5. seeds.png           S en dataset04 y test con 10 semillas por configuración
  6. REPORTE.md          resumen con todos los números
"""
import argparse
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from scipy import stats

from .core import (BATCH_SIZES, DEFAULT_HP, DEVICE, EPOCHS, PATIENCE, build_model, pad, prepare,
                   score, train)
from .data import ROOT, load_all
from .spc import SPCMonitor, batadal_score

OUT = ROOT / "resultados" / "optuna"
OUT.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{OUT / 'optuna.db'}"
PARALLEL = False


def make_storage(path):
    if str(path).endswith(".log"):
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(str(path)))
    return f"sqlite:///{path}"
MAX_EPOCHS = EPOCHS
DEFAULT_SPC = dict(L=6, agg="mean", lam=0.2, q=0.99)   # elegida en run.py por grid search
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = load_all()
_PREP = {}


def get_prep(W, k):
    if (W, k) not in _PREP:
        _PREP[(W, k)] = prepare(DATA, W, k)
    return _PREP[(W, k)]


def suggest(trial):
    return {
        # grafo y entrada
        "W": trial.suggest_categorical("W", [12, 24, 48]),
        "k": trial.suggest_int("k", 1, 5),
        # arquitectura
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
        "latent": trial.suggest_categorical("latent", [4, 8, 16, 32]),
        "emb": trial.suggest_categorical("emb", [4, 8, 16]),
        # entrenamiento
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "bs": trial.suggest_categorical("bs", BATCH_SIZES),   # 128–512 en GPU, 32–128 en CPU
        "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),   # peso de la reconstrucción de A
        "beta": trial.suggest_float("beta", 1e-5, 1e-1, log=True),    # peso del KL
        # carta EWMA
        "L": trial.suggest_categorical("L", [1, 3, 6, 12]),
        "agg": trial.suggest_categorical("agg", ["mean", "max", "top3"]),
        "lam": trial.suggest_float("lam", 0.05, 1.0, log=True),
        "q": trial.suggest_categorical("q", [0.99, 0.995, 0.999]),
    }


def fit_and_eval(p, seed, eval_test=False):
    """Entrena el VGAE con los hiperparámetros p y evalúa la carta EWMA."""
    torch.manual_seed(seed); np.random.seed(seed)
    W = p["W"]
    prep = get_prep(W, p["k"])
    model = build_model(len(DATA["sensors"]), p)
    hist = train(model, prep["Xtr"], prep["Xva"], prep["A"], prep["a_hat"], epochs=MAX_EPOCHS,
                 patience=PATIENCE, bs=p["bs"], lr=p["lr"], alpha=p["alpha"], beta=p["beta"],
                 verbose=False, Ttr=prep["Ttr"], Tva=prep["Tva"])
    ph1 = score(model, prep["x_ph1"], prep["a_hat"], W, prep["tod_ph1"])
    mon = SPCMonitor("ewma", L=p["L"], agg=p["agg"], lam=p["lam"], q=p["q"]).fit(ph1["E"], ph1["latent"])
    out = {"val_loss": float(hist[:, 1].min()), "epochs": len(hist)}
    sets = [("dataset04", "x04", "y04", "id04", "tod04")] + ([("test", "xte", "ytest", "idtest", "todte")] if eval_test else [])
    for name, xk, yk, ik, tk in sets:
        s = score(model, prep[xk], prep["a_hat"], W, prep[tk])
        al = pad(mon.run(s["E"], s["latent"])[1], W).astype(int)
        out[name] = batadal_score(DATA[yk], al, DATA[ik])
    return out


def objective(trial):
    p = suggest(trial)
    r = fit_and_eval(p, seed=trial.number)
    m = r["dataset04"]
    for k in ("S_TTD", "S_CM", "TPR", "TNR", "F1", "detectados"):
        trial.set_user_attr(k, float(m[k]))
    trial.set_user_attr("val_loss", r["val_loss"]); trial.set_user_attr("epochs", r["epochs"])
    return m["S"]


def search(n_tpe, n_random):
    tpe = optuna.create_study(
        study_name="vgae_tpe", storage=STORAGE, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=None if PARALLEL else 42, multivariate=True,
                                           n_startup_trials=10, constant_liar=PARALLEL))
    rnd = optuna.create_study(
        study_name="vgae_random", storage=STORAGE, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=None if PARALLEL else 2024))  # semilla distinta a TPE: muestras independientes
    for study, n in ((tpe, n_tpe), (rnd, n_random)):
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        cb = lambda s, t: print(f"[{s.study_name}] trial {t.number:3d}  S_d04={t.value:.3f}  "
                                f"mejor={s.best_value:.3f}", flush=True)
        if n - done > 0:
            if PARALLEL:
                # cada worker sigue pidiendo trials hasta que el estudio llega a n en total
                stop = optuna.study.MaxTrialsCallback(n, states=(optuna.trial.TrialState.COMPLETE,
                                                                 optuna.trial.TrialState.RUNNING))
                study.optimize(objective, callbacks=[cb, stop])
            else:
                study.optimize(objective, n_trials=n - done, callbacks=[cb])
    best = dict(tpe.best_params)
    best["S_dataset04"] = tpe.best_value
    best["trial"] = tpe.best_trial.number
    json.dump(best, open(OUT / "best_params.json", "w"), indent=2)
    return tpe, rnd


def seeds_check(tpe, rnd, n_seeds, worker=0, n_workers=1):
    """Reentrena las configuraciones finalistas con semillas nuevas y evalúa el test.
    Con n_workers > 1 cada proceso toma un subconjunto de (configuración, semilla).
    Cada resultado se agrega a seeds_part_<worker>.csv apenas termina, así que un
    corte no pierde lo hecho: al relanzar se saltan los pares ya calculados."""
    configs = {
        "Optuna TPE (mejor)": tpe.best_params,
        "Aleatoria (mejor)": rnd.best_params,
        "Por defecto": {**DEFAULT_HP, **DEFAULT_SPC},
    }
    jobs = [(n, s) for n in configs for s in range(100, 100 + n_seeds)][worker::n_workers]
    part = OUT / f"seeds_part_{worker}.csv"
    done = set()
    if part.exists():
        prev = pd.read_csv(part)
        done = set(zip(prev.config, prev.seed))
    for name, seed in jobs:
        if (name, seed) in done:
            continue
        r = fit_and_eval(configs[name], seed=seed, eval_test=True)
        row = {"config": name, "seed": seed, "val_loss": r["val_loss"],
               **{f"{ds}_{k}": r[ds][k] for ds in ("dataset04", "test")
                  for k in ("S", "S_TTD", "S_CM", "TPR", "TNR", "F1", "detectados")}}
        pd.DataFrame([row]).to_csv(part, mode="a", header=not part.exists(), index=False)
        print(f"[semillas] {name:20s} seed={seed}  S_d04={r['dataset04']['S']:.3f}  "
              f"S_test={r['test']['S']:.3f}", flush=True)


def collect_seeds(n_seeds):
    """Une los seeds_part_*.csv en seeds.csv (o lee seeds.csv si no hay partes)."""
    parts = sorted(OUT.glob("seeds_part_*.csv"))
    if not parts:
        return pd.read_csv(OUT / "seeds.csv")
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["config", "seed"])
    df = df[df.seed < 100 + n_seeds].sort_values(["config", "seed"])
    df.to_csv(OUT / "seeds.csv", index=False)
    return df


def report(tpe, rnd, seeds):
    df_t = tpe.trials_dataframe(); df_r = rnd.trials_dataframe()
    df_t.to_csv(OUT / "trials_tpe.csv", index=False); df_r.to_csv(OUT / "trials_random.csv", index=False)
    vt = df_t.loc[df_t.state == "COMPLETE", "value"].values
    vr = df_r.loc[df_r.state == "COMPLETE", "value"].values

    # 1) historia: mejor valor acumulado
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for v, lab, c in ((vt, "TPE (Optuna)", "tab:blue"), (vr, "Aleatoria", "tab:gray")):
        ax.scatter(range(len(v)), v, s=12, color=c, alpha=0.5)
        ax.plot(np.maximum.accumulate(v), color=c, lw=2, label=f"{lab} — mejor acumulado")
    ax.axvline(9.5, color="tab:blue", ls=":", lw=1)
    ax.text(10, ax.get_ylim()[0] + 0.01, " fin del arranque aleatorio de TPE", fontsize=8, color="tab:blue")
    ax.set_xlabel("trial"); ax.set_ylabel("S en dataset04"); ax.legend(loc="lower right")
    ax.set_title("Historia de optimización")
    fig.tight_layout(); fig.savefig(OUT / "history.png", dpi=130); plt.close(fig)

    # 2) distribución de S por trial: TPE guiado (tras el arranque) vs aleatorio
    vt_guided = vt[10:]
    mw = stats.mannwhitneyu(vt_guided, vr, alternative="greater")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([vr, vt_guided], labels=["Aleatoria", "TPE (trials ≥ 10)"], showmeans=True)
    for i, v in enumerate([vr, vt_guided], 1):
        ax.scatter(np.random.normal(i, 0.04, len(v)), v, s=10, alpha=0.5)
    ax.set_ylabel("S en dataset04")
    ax.set_title(f"Mann-Whitney U (TPE > aleatoria): p = {mw.pvalue:.2g}")
    fig.tight_layout(); fig.savefig(OUT / "tpe_vs_random.png", dpi=130); plt.close(fig)

    # 3) importancia de hiperparámetros (fANOVA sobre los trials de TPE + aleatorios)
    imp = optuna.importance.get_param_importances(
        tpe, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    imp_r = optuna.importance.get_param_importances(
        rnd, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    json.dump({"tpe": imp, "random": imp_r}, open(OUT / "importances.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(imp)[::-1]
    ax.barh(names, [imp[n] for n in names], color="tab:blue")
    ax.set_xlabel("importancia relativa (fANOVA)"); ax.set_title("¿Qué hiperparámetros importan?")
    fig.tight_layout(); fig.savefig(OUT / "importances.png", dpi=130); plt.close(fig)

    # 4) slices de los 6 hiperparámetros más importantes (todos los trials completos)
    top = list(imp)[:6]
    both = pd.concat([df_t.assign(estudio="TPE"), df_r.assign(estudio="Aleatoria")])
    both = both[both.state == "COMPLETE"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    best_p = tpe.best_params
    for ax, n in zip(axes.ravel(), top):
        col = both[f"params_{n}"]
        for est, c in (("Aleatoria", "tab:gray"), ("TPE", "tab:blue")):
            m = both.estudio == est
            xv = col[m]
            if xv.dtype == object:
                cats = sorted(col.unique(), key=str)
                xv = xv.map({c_: i for i, c_ in enumerate(cats)})
                ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
            ax.scatter(xv + (np.random.normal(0, 0.05, len(xv)) if col.dtype == object else 0),
                       both.value[m], s=12, alpha=0.6, color=c, label=est)
        if col.dtype != object and n in ("lr", "alpha", "beta", "lam"):
            ax.set_xscale("log")
        bv = best_p[n]
        if col.dtype == object:
            bv = sorted(col.unique(), key=str).index(bv)
        ax.axvline(bv, color="tab:red", ls="--", lw=1)
        ax.set_title(f"{n}  (importancia {imp[n]:.2f})", fontsize=10)
        ax.set_ylabel("S dataset04")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("S vs hiperparámetro (línea roja = mejor configuración TPE)")
    fig.tight_layout(); fig.savefig(OUT / "slices.png", dpi=130); plt.close(fig)

    # 5) robustez con semillas
    order = ["Optuna TPE (mejor)", "Aleatoria (mejor)", "Por defecto"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, ds in zip(axes, ("dataset04", "test")):
        data = [seeds.loc[seeds.config == c, f"{ds}_S"].values for c in order]
        ax.boxplot(data, labels=order, showmeans=True)
        for i, v in enumerate(data, 1):
            ax.scatter(np.full(len(v), i) + np.random.normal(0, 0.04, len(v)), v, s=14, alpha=0.7)
        ax.set_title(f"S en {ds} ({len(data[0])} semillas por configuración)")
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("S de BATADAL")
    fig.tight_layout(); fig.savefig(OUT / "seeds.png", dpi=130); plt.close(fig)

    agg = seeds.groupby("config")[[c for c in seeds.columns if c.endswith(("_S", "_TPR", "_TNR", "_F1"))]]
    summary = agg.agg(["mean", "std"]).round(3).reindex(order)
    summary.to_csv(OUT / "seeds_summary.csv")
    st = seeds.pivot(index="seed", columns="config", values="test_S")
    s04 = seeds.pivot(index="seed", columns="config", values="dataset04_S")

    def wilcoxon_p(a, b):
        if len(a) < 5 or np.allclose(a, b):
            return float("nan")
        return stats.wilcoxon(a, b, alternative="greater").pvalue
    w_def = wilcoxon_p(st["Optuna TPE (mejor)"], st["Por defecto"])
    w_rnd = wilcoxon_p(st["Optuna TPE (mejor)"], st["Aleatoria (mejor)"])
    w_def04 = wilcoxon_p(s04["Optuna TPE (mejor)"], s04["Por defecto"])

    # 6) reporte
    def ms(c, col):
        v = seeds.loc[seeds.config == c, col]
        return f"{v.mean():.3f} ± {v.std():.3f}"
    lines = [
        "# Optimización de hiperparámetros del VGAE + SPC con Optuna", "",
        "## Protocolo", "",
        "- **Objetivo:** S de BATADAL en `dataset04` (carta EWMA sobre z-scores por sensor). "
        "El test no se usó en ninguna decisión de la búsqueda.",
        f"- **Estudio principal:** TPE multivariante, {len(vt)} trials (10 de arranque aleatorio), semilla 42.",
        f"- **Control:** búsqueda aleatoria, {len(vr)} trials, mismo espacio y objetivo.",
        f"- **Entrenamiento por trial:** hasta {MAX_EPOCHS} épocas, early stopping (paciencia {PATIENCE}) "
        "sobre la pérdida de validación de `dataset03`.",
        f"- **Robustez:** cada configuración finalista se reentrena con {seeds.seed.nunique()} semillas nuevas "
        f"({seeds.seed.min()}–{seeds.seed.max()}) y sólo entonces se evalúa el test.", "",
        "## Mejor configuración (TPE)", "",
        "| Hiperparámetro | Valor |", "|---|---|",
        *[f"| `{k}` | {v:.4g} |" if isinstance(v, float) else f"| `{k}` | {v} |" for k, v in tpe.best_params.items()],
        "", f"S en dataset04 en la búsqueda: **{tpe.best_value:.3f}** (trial {tpe.best_trial.number}).", "",
        "## Evidencia de que los parámetros no son al azar", "",
        "### 1. TPE supera a la búsqueda aleatoria", "",
        f"- Mejor S aleatoria: {vr.max():.3f} · mejor S TPE: {vt.max():.3f}",
        f"- Mediana de S por trial — aleatoria: {np.median(vr):.3f} · TPE (trials ≥ 10): {np.median(vt_guided):.3f}",
        f"- Mann-Whitney U unilateral (TPE guiado > aleatorio): U = {mw.statistic:.0f}, **p = {mw.pvalue:.2g}**",
        "- Gráficos: `history.png`, `tpe_vs_random.png`", "",
        "### 2. El objetivo depende de forma sistemática de algunos hiperparámetros", "",
        "Importancia fANOVA (fracción de la varianza de S explicada):", "",
        "| Hiperparámetro | TPE | Aleatoria |", "|---|---|---|",
        *[f"| `{k}` | {imp[k]:.3f} | {imp_r.get(k, float('nan')):.3f} |" for k in imp],
        "", "Gráfico: `importances.png`, `slices.png`.", "",
        "### 3. La mejora se mantiene con otras semillas y en el test", "",
        "| Configuración | S dataset04 | S test | TPR test | TNR test |", "|---|---|---|---|---|",
        *[f"| {c} | {ms(c, 'dataset04_S')} | {ms(c, 'test_S')} | {ms(c, 'test_TPR')} | {ms(c, 'test_TNR')} |"
          for c in order],
        "",
    ]
    if len(st) >= 5:
        lines += [
            f"- Wilcoxon pareado por semilla, S dataset04, TPE > por defecto: p = {w_def04:.3g}",
            f"- Wilcoxon pareado por semilla, S test, TPE > por defecto: p = {w_def:.3g}",
            f"- Wilcoxon pareado por semilla, S test, TPE > mejor aleatoria: p = {w_rnd:.3g}",
            f"- Con {len(st)} semillas, el menor p posible en un Wilcoxon unilateral es {1 / 2 ** len(st):.3g}.",
        ]
    lines += ["", "Gráfico: `seeds.png`. Datos: `trials_tpe.csv`, `trials_random.csv`, `seeds.csv`, `journal.log`."]
    (OUT / "REPORTE.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpe", type=int, default=100)
    ap.add_argument("--random", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--stage", default="all", choices=["all", "search", "seeds", "report"])
    ap.add_argument("--out", default="resultados/optuna")
    ap.add_argument("--storage", default=None, help="ruta .db (SQLite) o .log (Journal); por defecto <out>/journal.log")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--seed-worker", type=int, default=0, help="índice de este worker en la etapa seeds")
    ap.add_argument("--seed-workers", type=int, default=1, help="workers totales en la etapa seeds")
    a = ap.parse_args()
    global OUT, STORAGE, PARALLEL
    OUT = ROOT / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    STORAGE = make_storage(a.storage or OUT / "journal.log")
    PARALLEL = a.parallel
    print(f"device: {DEVICE}  salida: {OUT}", flush=True)
    if a.stage in ("all", "search"):
        tpe, rnd = search(a.tpe, a.random)
    else:
        tpe = optuna.load_study(study_name="vgae_tpe", storage=STORAGE)
        rnd = optuna.load_study(study_name="vgae_random", storage=STORAGE)
    if a.stage in ("all", "seeds"):
        seeds_check(tpe, rnd, a.seeds, a.seed_worker, a.seed_workers)
        if a.seed_workers > 1:
            return          # el reporte se genera con --stage report cuando terminan todos
    if a.stage in ("all", "seeds", "report"):
        report(tpe, rnd, collect_seeds(a.seeds))


if __name__ == "__main__":
    main()
