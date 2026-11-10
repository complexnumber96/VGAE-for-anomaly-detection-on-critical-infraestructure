"""Figuras descriptivas del modelo y de la búsqueda con Optuna.

Uso:  python -m vgae_spc.figures [--optuna-dir results/optuna]   (se puede correr mientras el estudio avanza)
Salida: results/figuras/
  grafo_sensores.png      el grafo del VGAE sobre la red C-Town y como grafo abstracto
  arquitectura_vgae.png   diagrama del modelo con los hiperparámetros elegidos por Optuna
  optuna_progreso.png     S en dataset04 por trial (TPE y aleatoria) frente a la configuración por defecto
  nodos_anomalos_test.png z-score de cada sensor durante cada ataque del test, sobre el grafo
                          (requiere el modelo final: python -m vgae_spc.run --optuna --retrain)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optuna
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import torch
from matplotlib.colors import LinearSegmentedColormap

from .core import DEFAULT_HP, DEVICE, build_model, prepare, score
from .model import normalize_adj
from .spc import NodeScores
from .data import ROOT, load_all
from .graph import build_graph, parse_inp, sensor_location

import sys

OPTUNA_DIR = ROOT / (sys.argv[sys.argv.index("--optuna-dir") + 1] if "--optuna-dir" in sys.argv
                     else "results/optuna")
FINAL_DIR = OPTUNA_DIR.parent / f"{OPTUNA_DIR.name}_final"
OUT = OPTUNA_DIR.parent / f"figuras_{OPTUNA_DIR.name}"
OUT.mkdir(parents=True, exist_ok=True)


def optuna_storage():
    journal = OPTUNA_DIR / "journal.log"
    if journal.exists():
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(str(journal)))
    return f"sqlite:///{OPTUNA_DIR / 'optuna.db'}"


def default_reference():
    """S en dataset04 de la carta EWMA del modelo por defecto (results/summary.csv), si existe."""
    import pandas as pd
    f = ROOT / "results" / "summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        m = df[(df.carta == "ewma") & (df.dataset == "dataset04")]
        if len(m):
            return float(m.S.iloc[0])
    return None

# Paleta de referencia (validada: 3 categorías, todos los pares, modo claro)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

GROUP = {"L": ("Tanque (nivel)", BLUE), "F": ("Bomba / válvula", ORANGE),
         "S": ("Bomba / válvula", ORANGE), "P": ("Presión", AQUA)}
EDGE_STYLE = {"control": ("-", 1.6, INK2, "Control tanque → actuador (PLC)"),
              "estación": ((0, (1, 1.5)), 1.4, INK2, "Misma estación (presión ↔ equipo)"),
              "dispositivo": ("-", 3.0, GRID, "Mismo equipo (caudal ↔ estado)"),
              "hidráulica": ((0, (4, 2)), 1.0, MUTED, "k vecinos hidráulicos más cercanos")}


def coordinates():
    coords, cur = {}, None
    for line in open(ROOT / "CTOWN.INP"):
        line = line.split(";")[0].strip()
        if line.startswith("["):
            cur = line
            continue
        if cur == "[COORDINATES]" and line:
            n, x, y = line.split()[:3]
            coords[n] = (float(x), float(y))
    return coords


def model_graph(sensors, k):
    """Grafo del modelo final: la adyacencia guardada en su checkpoint (build_graph puede resolver
    empates de distancia de otra forma). Las aristas que build_graph no tipifica son hidráulicas."""
    A, D, kinds = build_graph(sensors, k=k)
    ckpt_path = FINAL_DIR / "vgae.pt"
    if ckpt_path.exists():
        A = torch.load(ckpt_path, weights_only=False, map_location="cpu")["A"]
        kinds = {(int(i), int(j)): kinds.get((int(i), int(j)), "hidráulica") for i, j in zip(*np.where(np.triu(A)))}
    return A, D, kinds


def fig_grafo(sensors, k):
    A, D, kinds = model_graph(sensors, k)
    adj, links = parse_inp()
    xy = coordinates()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) ubicación física de los sensores sobre la red de tuberías
    for a, nbrs in adj.items():
        for b, _ in nbrs:
            if a < b and a in xy and b in xy:
                ax1.plot(*zip(xy[a], xy[b]), color=GRID, lw=0.8, zorder=1)
    placed = {}
    for s in sensors:
        loc = sensor_location(s, links)
        if loc not in xy:
            continue
        label, color = GROUP[s[0]]
        placed.setdefault(loc, []).append(s)
        ax1.scatter(*xy[loc], s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for loc, names in placed.items():
        devs = sorted({n.split("_", 1)[1] for n in names})
        ax1.annotate(", ".join(devs), xy[loc], xytext=(5, 5), textcoords="offset points",
                     fontsize=7.5, color=INK, zorder=4)
    ax1.set_title("(a) Sensores sobre la red C-Town (CTOWN.INP)", loc="left", fontsize=11)
    ax1.set_aspect("equal"); ax1.axis("off")

    # (b) el grafo que ve el VGAE
    G = nx.Graph()
    G.add_nodes_from(range(len(sensors)))
    for (i, j), t in kinds.items():
        G.add_edge(i, j, kind=t)
    pos = nx.kamada_kawai_layout(G)
    for kind, (ls, lw, col, _) in EDGE_STYLE.items():
        for i, j in [e for e, t in kinds.items() if t == kind]:
            ax2.plot(*zip(pos[i], pos[j]), ls=ls, lw=lw, color=col, zorder=1 if kind == "dispositivo" else 2)
    for i, s in enumerate(sensors):
        _, color = GROUP[s[0]]
        ax2.scatter(*pos[i], s=260, color=color, marker="s" if s[0] == "S" else "o",
                    edgecolor=SURFACE, linewidth=2, zorder=3)
        ax2.annotate(s, pos[i], xytext=(0, 11), textcoords="offset points", ha="center",
                     fontsize=7.5, color=INK, zorder=4)
    ax2.set_title(f"(b) Grafo de entrada del VGAE: {len(sensors)} nodos, {int(A.sum() / 2)} aristas (k = {k})",
                  loc="left", fontsize=11)
    ax2.axis("off")

    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=9, label="Nivel de tanque (L_)"),
               Line2D([], [], marker="o", ls="", color=ORANGE, ms=9, label="Caudal bomba/válvula (F_)"),
               Line2D([], [], marker="s", ls="", color=ORANGE, ms=9, label="Estado bomba/válvula (S_)"),
               Line2D([], [], marker="o", ls="", color=AQUA, ms=9, label="Presión (P_)")]
    handles += [Line2D([], [], ls=ls, lw=max(lw, 1.2), color=col, label=lab)
                for ls, lw, col, lab in EDGE_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "grafo_sensores.png", dpi=150); plt.close(fig)


def fig_arquitectura(hp, spc, n_nodes, title_extra):
    W, H, Z, E = hp["W"], hp["hidden"], hp["latent"], hp["emb"]
    fig, ax = plt.subplots(figsize=(16, 6.9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.9); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=SURFACE, ec=color, lw=2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", color="white",
                fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center", fontsize=8.6,
                color=INK, linespacing=1.45)

    def arrow(x0, y0, x1, y1, text=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=INK2, lw=1.3))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.14, text, ha="center", fontsize=8, color=INK2)

    box(0.2, 2.2, 2.2, 2.2, "Entrada", f"Ventana de {W} h\npor sensor\n\nX: {n_nodes} × {W}\n+ embedding\nde nodo ({E})", MUTED)
    box(3.0, 2.2, 2.4, 2.2, "Encoder GCN", f"GCN {W + E} → {H}\nReLU\n\nÂ = D⁻½(A+I)D⁻½\ngrafo C-Town", BLUE)
    box(6.0, 3.5, 2.0, 1.5, "μ", f"GCN {H} → {Z}", BLUE)
    box(6.0, 1.4, 2.0, 1.5, "log σ²", f"GCN {H} → {Z}", BLUE)
    box(8.6, 2.2, 1.9, 2.2, "Latente z", f"z = μ + σ·ε\n\n{n_nodes} × {Z}\n\nKL × β\nβ = {hp['beta']:.1e}", BLUE)
    box(11.1, 3.7, 2.4, 2.0, "Decoder atributos", f"GCN {Z + E} → {H}\nReLU\nLineal {H} → {W}\n→ X̂", ORANGE)
    box(11.1, 0.6, 2.4, 2.0, "Decoder estructura", f"Â_ij = σ(zᵢ · zⱼ)\n\nBCE × α\nα = {hp['alpha']:.3g}", ORANGE)
    box(14.0, 2.2, 1.8, 2.2, "SPC", f"error (X−X̂)²\núltimas {spc['L']} h\nz-score por sensor\n→ {spc['agg']}\nEWMA λ={spc['lam']:.2f}\nUCL q={spc['q']}", AQUA)

    arrow(2.4, 3.3, 3.0, 3.3)
    arrow(5.4, 3.6, 6.0, 4.2); arrow(5.4, 3.0, 6.0, 2.2)
    arrow(8.0, 4.2, 8.6, 3.6); arrow(8.0, 2.2, 8.6, 3.0)
    arrow(10.5, 3.6, 11.1, 4.6); arrow(10.5, 3.0, 11.1, 1.7)
    arrow(13.5, 4.6, 14.0, 3.6, None)
    ax.text(0.2, 6.6, f"VGAE + SPC — {title_extra}", fontsize=13, weight="bold", color=INK)
    ax.text(0.2, 6.25, f"Entrenamiento: Adam lr = {hp['lr']:.2g}, batch = {hp['bs']}, early stopping "
                      f"sobre la pérdida de validación · pérdida = MSE(X, X̂) + α·BCE(A, Â) + β·KL",
            fontsize=9, color=INK2)
    ax.text(14.9, 1.9, "alarma si\nEWMA > UCL", ha="center", va="top", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "arquitectura_vgae.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_optuna():
    storage = optuna_storage()
    fig, ax = plt.subplots(figsize=(11, 5))
    best_overall = None
    for name, label, color in (("vgae_tpe", "TPE (Optuna)", BLUE), ("vgae_random", "Búsqueda aleatoria", ORANGE)):
        try:
            st = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            continue
        x = np.arange(len(done)); v = np.array([t.value for t in done])
        ax.scatter(x, v, s=34, color=color, edgecolor=SURFACE, linewidth=1, alpha=0.75, zorder=3)
        best = np.maximum.accumulate(v)
        ax.step(x, best, where="post", color=color, lw=2, zorder=4)
        ax.annotate(f"{label}: mejor {best[-1]:.3f} ({len(done)} trials)", (x[-1], best[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        if name == "vgae_tpe":
            best_overall = st.best_trial
    ref = default_reference()
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(0.99, ref - 0.006, f"configuración por defecto ({ref:.3f})", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=8.5, color=INK2)
    ax.axvspan(-0.5, 9.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(4.5, 0.99, "arranque aleatorio de TPE", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color=INK2)
    if best_overall is not None:
        ax.annotate(f"trial {best_overall.number}", (best_overall.number, best_overall.value),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xlabel("trial (en orden de ejecución)")
    ax.set_ylabel("S de BATADAL en dataset04")
    ax.set_title("Puntos: S de cada trial · líneas: mejor S acumulado", loc="left", fontsize=10, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(-0.5, None)
    fig.suptitle(f"Búsqueda de hiperparámetros con Optuna ({OPTUNA_DIR.name})", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "optuna_progreso.png", dpi=150); plt.close(fig)
    return best_overall


# Sensores del componente atacado según Attacks_TestDataset.png
ATTACKED_TEST = {
    8: ["L_T3", "F_PU4", "S_PU4"],
    9: ["L_T2", "F_V2", "S_V2"],
    10: [], 11: [],                       # PU3 no tiene sensores propios en los datos
    12: ["L_T2", "F_V2", "S_V2", "P_J14", "P_J422"],
    13: ["L_T7", "F_PU10", "S_PU10", "F_PU11", "S_PU11"],
    14: ["L_T4", "L_T6"],
}
DESC_TEST = {8: "umbrales de T3 (PU4/PU5)", 9: "lectura de L_T2 → V2 abierta", 10: "activación de PU3",
             11: "activación de PU3", 12: "como #9 + replay", 13: "umbrales de T7 (PU10/PU11)",
             14: "señal de T4 → desborde de T6"}
# Rampa secuencial azul de la paleta de referencia (cerca de cero ≈ superficie)
SEQ = LinearSegmentedColormap.from_list("seq", ["#eceae4", "#cde2fb", "#86b6ef", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])


def fig_nodos_anomalos():
    ckpt_path = FINAL_DIR / "vgae.pt"
    if not ckpt_path.exists():
        print("sin modelo final; omito nodos_anomalos_test.png")
        return
    ck = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    hp = ck["hp"]
    p = json.load(open(OPTUNA_DIR / "best_params.json"))
    d = load_all(); sensors = d["sensors"]
    prep = prepare(d, hp["W"], hp["k"])
    prep["a_hat"] = normalize_adj(torch.from_numpy(ck["A"]).to(DEVICE))   # grafo del entrenamiento
    model = build_model(len(sensors), hp); model.load_state_dict(ck["state"])
    ns = NodeScores(p["L"]).fit(score(model, prep["x_ph1"], prep["a_hat"], hp["W"])["E"])
    z = ns.transform(score(model, prep["xte"], prep["a_hat"], hp["W"])["E"])
    z = np.concatenate([np.full((hp["W"] - 1, len(sensors)), np.nan), z])

    A, _, kinds = model_graph(sensors, hp["k"])
    G = nx.Graph(); G.add_nodes_from(range(len(sensors))); G.add_edges_from(kinds)
    pos = nx.kamada_kawai_layout(G)
    vmax = 6
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for ax, aid in zip(axes.ravel(), range(8, 15)):
        zm = np.nanmean(z[d["idtest"] == aid], 0)
        for i, j in kinds:
            ax.plot(*zip(pos[i], pos[j]), color=GRID, lw=0.8, zorder=1)
        ax.scatter([pos[i][0] for i in range(len(sensors))], [pos[i][1] for i in range(len(sensors))],
                   c=np.clip(zm, 0, vmax), cmap=SEQ, vmin=0, vmax=vmax, s=170,
                   marker="o", edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for name in ATTACKED_TEST[aid]:
            i = sensors.index(name)
            ax.scatter(*pos[i], s=360, facecolor="none", edgecolor=ORANGE, linewidth=2.2, zorder=4)
        top = list(np.argsort(-zm)[:4])
        for i in top:
            ax.annotate(f"{sensors[i]}\n{zm[i]:.1f}", pos[i], xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color=INK, zorder=5, weight="bold")
        for name in ATTACKED_TEST[aid]:
            i = sensors.index(name)
            if i not in top:
                ax.annotate(f"{name} {zm[i]:.1f}", pos[i], xytext=(0, -17), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK2, zorder=5)
        ax.margins(0.12)
        hours = int((d["idtest"] == aid).sum())
        ax.set_title(f"Ataque #{aid} ({hours} h): {DESC_TEST[aid]}", loc="left", fontsize=10)
        ax.axis("off")
    ax = axes.ravel()[-1]; ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.0, aspect=18)
    cb.set_label("z-score medio del error durante el ataque (recortado a 0–6)", color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=13, mfc="none", mec=ORANGE, mew=2.2,
                              label="sensor del componente atacado\n(tabla oficial del test)"),
                       Line2D([], [], ls="", label="etiquetas: los 4 sensores con mayor z")],
              loc="center", frameon=False, fontsize=9.5)
    ax.text(0.5, 0.85, "#10 y #11 atacan PU3, que no tiene\nsensores propios: se ve su efecto\n"
            "en la estación PU1/PU2 y en T1.", ha="center", va="top", transform=ax.transAxes,
            fontsize=9, color=INK2)
    fig.suptitle("¿Qué sensores marca como anómalos el VGAE? — test, modelo final (Optuna)",
                 x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "nodos_anomalos_test.png", dpi=140); plt.close(fig)


def main():
    d = load_all()
    best_params = OPTUNA_DIR / "best_params.json"
    if best_params.exists():
        p = json.load(open(best_params)); extra = f"mejor configuración Optuna (trial {p['trial']})"
    else:
        st = optuna.load_study(study_name="vgae_tpe", storage=optuna_storage())
        p = st.best_params; extra = f"mejor configuración Optuna hasta ahora (trial {st.best_trial.number})"
    hp = {k: p[k] for k in DEFAULT_HP}
    spc = {k: p[k] for k in ("L", "agg", "lam", "q")}
    fig_grafo(d["sensors"], hp["k"])
    fig_arquitectura(hp, spc, len(d["sensors"]), extra)
    fig_optuna()
    fig_nodos_anomalos()
    print("figuras en", OUT)


if __name__ == "__main__":
    main()
