# Optimización de hiperparámetros del VGAE + SPC con Optuna

## Protocolo

- **Objetivo:** S de BATADAL en `dataset04` (carta EWMA sobre z-scores por sensor). El test no se usó en ninguna decisión de la búsqueda.
- **Estudio principal:** TPE multivariante, 100 trials (10 de arranque aleatorio), semilla 42.
- **Control:** búsqueda aleatoria, 50 trials, mismo espacio y objetivo.
- **Entrenamiento por trial:** hasta 300 épocas, early stopping (paciencia 25) sobre la pérdida de validación de `dataset03`.
- **Robustez:** cada configuración finalista se reentrena con 10 semillas nuevas (100–109) y sólo entonces se evalúa el test.

## Mejor configuración (TPE)

| Hiperparámetro | Valor |
|---|---|
| `W` | 12 |
| `k` | 1 |
| `hidden` | 128 |
| `latent` | 32 |
| `emb` | 16 |
| `lr` | 0.003363 |
| `bs` | 128 |
| `alpha` | 0.07764 |
| `beta` | 2.01e-05 |
| `L` | 12 |
| `agg` | mean |
| `lam` | 0.6447 |
| `q` | 0.995 |

S en dataset04 en la búsqueda: **0.879** (trial 57).

## Evidencia de que los parámetros no son al azar

### 1. TPE supera a la búsqueda aleatoria

- Mejor S aleatoria: 0.751 · mejor S TPE: 0.879
- Mediana de S por trial — aleatoria: 0.586 · TPE (trials ≥ 10): 0.802
- Mann-Whitney U unilateral (TPE guiado > aleatorio): U = 4294, **p = 3.1e-19**
- Gráficos: `history.png`, `tpe_vs_random.png`

### 2. El objetivo depende de forma sistemática de algunos hiperparámetros

Importancia fANOVA (fracción de la varianza de S explicada):

| Hiperparámetro | TPE | Aleatoria |
|---|---|---|
| `agg` | 0.455 | 0.051 |
| `W` | 0.174 | 0.032 |
| `latent` | 0.150 | 0.052 |
| `L` | 0.039 | 0.007 |
| `lam` | 0.036 | 0.567 |
| `k` | 0.034 | 0.024 |
| `lr` | 0.028 | 0.154 |
| `beta` | 0.022 | 0.001 |
| `emb` | 0.021 | 0.028 |
| `hidden` | 0.017 | 0.009 |
| `alpha` | 0.011 | 0.014 |
| `q` | 0.010 | 0.059 |
| `bs` | 0.003 | 0.003 |

Gráfico: `importances.png`, `slices.png`.

### 3. La mejora se mantiene con otras semillas y en el test

| Configuración | S dataset04 | S test | TPR test | TNR test |
|---|---|---|---|---|
| Optuna TPE (mejor) | 0.852 ± 0.014 | 0.750 ± 0.017 | 0.426 ± 0.026 | 0.976 ± 0.005 |
| Aleatoria (mejor) | 0.704 ± 0.034 | 0.496 ± 0.062 | 0.197 ± 0.092 | 0.972 ± 0.004 |
| Por defecto | 0.722 ± 0.052 | 0.529 ± 0.035 | 0.185 ± 0.022 | 0.968 ± 0.002 |

- Wilcoxon pareado por semilla, S dataset04, TPE > por defecto: p = 0.000977
- Wilcoxon pareado por semilla, S test, TPE > por defecto: p = 0.000977
- Wilcoxon pareado por semilla, S test, TPE > mejor aleatoria: p = 0.000977
- Con 10 semillas, el menor p posible en un Wilcoxon unilateral es 0.000977.

Gráfico: `seeds.png`. Datos: `trials_tpe.csv`, `trials_random.csv`, `seeds.csv`, `journal.log`.