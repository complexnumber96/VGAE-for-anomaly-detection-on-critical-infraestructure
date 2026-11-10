"""Cartas de control estadístico de procesos sobre las salidas del VGAE.

Siguiendo el esquema del MSPC basado en PCA, se vigilan dos aspectos del VGAE:

  - Residuo (pérdida de coherencia física entre sensores):
      ewma: para cada sensor i se toma el error de reconstrucción de las últimas L horas
            de la ventana, se aplica log y se estandariza con su media/desviación en
            Fase I (z_i). Así cada sensor pesa lo mismo y un ataque local (p. ej. sólo
            L_T2 + V2) no se diluye entre los 36 nodos. Los z_i se agregan con 'mean',
            'max' o 'top3' y sobre esa serie se aplica una EWMA (con λ = 1 se reduce a
            una carta de Shewhart, por eso Shewhart no se calcula aparte).
      spe:  error cuadrático crudo sumado sobre sensores (estadístico Q), con el límite
            de Box (1954) / Nomikos & MacGregor (1995): UCL = g·χ²_h(q), con g = v / 2m y
            h = 2m² / v, donde m y v son la media y la varianza del SPE en Fase I.
  - Espacio latente (operación coherente pero en un estado anormal):
      t2:   T² de Hotelling sobre el embedding del grafo.

Los límites de ewma y t2 son cuantiles empíricos del estadístico en Fase I (datos
normales no vistos en el entrenamiento), porque la serie está autocorrelacionada y los
límites teóricos i.i.d. generarían demasiadas falsas alarmas.
"""
import numpy as np
from scipy import stats

AGG = {
    "mean": lambda z: z.mean(1),
    "max": lambda z: z.max(1),
    "top3": lambda z: np.sort(z, 1)[:, -3:].mean(1),
}


def ewma(x, lam, x0):
    out = np.empty_like(x)
    prev = x0
    for t, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        out[t] = prev
    return out


class NodeScores:
    """z-scores por sensor a partir del error cuadrático por ventana (T, N, W)."""

    def __init__(self, L):
        self.L = L

    def _log_err(self, E):
        return np.log(E[:, :, -self.L:].mean(-1) + 1e-6)

    def fit(self, E_ph1):
        le = self._log_err(E_ph1)
        self.mu, self.sd = le.mean(0), le.std(0)
        return self

    def transform(self, E):
        return (self._log_err(E) - self.mu) / self.sd


class SPCMonitor:
    """chart ∈ {ewma, t2, spe}."""

    def __init__(self, chart, L=6, agg="mean", lam=0.2, q=0.99):
        self.chart, self.L, self.agg, self.lam, self.q = chart, L, agg, lam, q

    def fit(self, E_ph1, latent_ph1):
        if self.chart == "t2":
            self.lat_mu = latent_ph1.mean(0)
            self.lat_icov = np.linalg.pinv(np.cov(latent_ph1, rowvar=False))
        elif self.chart == "ewma":
            self.ns = NodeScores(self.L).fit(E_ph1)
            self.s_mu = AGG[self.agg](self.ns.transform(E_ph1)).mean()
        if self.chart == "spe":
            spe = self.statistic(E_ph1, latent_ph1)
            m, v = spe.mean(), spe.var()
            self.g, self.h = v / (2 * m), 2 * m ** 2 / v
            self.ucl = self.g * stats.chi2.ppf(self.q, self.h)
        else:
            self.ucl = np.quantile(self.statistic(E_ph1, latent_ph1), self.q)
        return self

    def statistic(self, E, latent):
        if self.chart == "t2":
            d = latent - self.lat_mu
            return np.einsum("ij,jk,ik->i", d, self.lat_icov, d)
        if self.chart == "spe":
            # SPE / estadístico Q: suma sobre sensores del error cuadrático (últimas L horas)
            return E[:, :, -self.L:].mean(-1).sum(1)
        return ewma(AGG[self.agg](self.ns.transform(E)), self.lam, self.s_mu)

    def run(self, E, latent):
        v = self.statistic(E, latent)
        return v, v > self.ucl

    def describe(self):
        if self.chart == "t2":
            return {"q": self.q}
        if self.chart == "spe":
            return {"L": self.L, "q": self.q, "g": round(float(self.g), 4), "h": round(float(self.h), 2)}
        return {"L": self.L, "agg": self.agg, "lambda": self.lam, "q": self.q}


def batadal_score(y, yhat, att_id):
    """Métrica oficial de BATADAL: S = 0.5·S_TTD + 0.5·S_CM."""
    tp = int(((y == 1) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    ttd = {}
    for aid in np.unique(att_id[att_id > 0]):
        idx = np.where(att_id == aid)[0]
        hits = np.where(yhat[idx] == 1)[0]
        ttd[int(aid)] = (int(hits[0]) if len(hits) else len(idx), len(idx), len(hits) > 0)
    s_ttd = 1 - np.mean([t / d for t, d, _ in ttd.values()])
    s_cm = (tpr + tnr) / 2
    return {"S": 0.5 * s_ttd + 0.5 * s_cm, "S_TTD": s_ttd, "S_CM": s_cm,
            "TPR": tpr, "TNR": tnr, "Precision": prec, "F1": f1,
            "detectados": sum(d for *_, d in ttd.values()), "n_ataques": len(ttd), "ttd": ttd}
