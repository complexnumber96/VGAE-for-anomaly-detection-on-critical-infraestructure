# Detección de ciberataques con un Variational Graph Autoencoder y control estadístico de procesos

**Autores:** Jorge Proaño, Salvador Naya, Javier Tarrío, Miguel Flores (Universidad de La Coruña)

📄 **Paper (borrador):** [`documentos/paper.pdf`](documentos/paper.pdf) ·
**Metodología detallada:** [`documentos/metodologia.pdf`](documentos/metodologia.pdf)

Detector de ciberataques para la red de agua C-Town de la competición
[BATADAL](https://www.batadal.net). Los 36 sensores de la red se representan como nodos de
un grafo construido a partir de la topología hidráulica y de la lógica de control de los PLC.
Un **VGAE** (Variational Graph Autoencoder) entrenado solo con operación normal reconstruye
cada sensor a partir de sus vecinos, y tres **cartas de control** vigilan sus salidas:

- **EWMA** sobre el error de reconstrucción estandarizado de cada sensor;
- **T² de Hotelling** sobre el espacio latente;
- **SPE** sobre el residuo.

El detector propuesto da alarma cuando la carta EWMA o la T² la dan (regla EWMA ∨ T²).

## Resultados principales (conjunto de prueba de BATADAL)

| Detector | Ataques detectados | S | TPR | TNR | F1 |
|---|---|---|---|---|---|
| **EWMA ∨ T² (propuesto)** | **7/7** | **0.819** | 0.651 | 0.929 | 0.669 |
| EWMA | 7/7 | 0.765 | 0.575 | 0.934 | 0.622 |
| T² | 7/7 | 0.678 | 0.243 | 0.983 | 0.370 |
| SPE | 2/7 | 0.356 | 0.044 | 0.996 | 0.083 |

S es la métrica oficial de BATADAL (0.5 · tiempo de detección + 0.5 · clasificación). Con
S = 0.819 el detector quedaría en 5.º lugar entre los 8 participantes de la competición.

## Estructura

```
├── documentos/          paper y metodología (.tex y .pdf)
├── vgae_spc/            código
├── datos/               datos de BATADAL (copia de batadal.net)
├── resultados/          todo lo que aparece en el paper
│   ├── optuna/          búsqueda de hiperparámetros y prueba de semillas
│   ├── optuna_final/    modelo final entrenado (vgae.pt) y sus métricas
│   ├── figuras_optuna/  figuras del grafo, la arquitectura y los nodos anómalos
│   └── temporal/        comparación con una variante con convolución temporal
├── colab_busqueda_optuna.ipynb   repite la búsqueda en Google Colab (GPU)
└── requirements.txt
```

### Código (`vgae_spc/`)

| Archivo | Qué hace |
|---|---|
| `data.py` | Lee los CSV, etiqueta los ataques y arma ventanas temporales |
| `graph.py` | Construye el grafo de sensores a partir de `CTOWN.INP` |
| `model.py` | El VGAE: codificador GCN, decodificador de atributos y de estructura |
| `spc.py` | Cartas de control (EWMA, T², SPE) y métrica S de BATADAL |
| `core.py` | Entrenamiento y cálculo de puntuaciones, comunes a los demás scripts |
| `run.py` | **Script principal**: entrena o carga el modelo y evalúa las cartas |
| `tune.py` | Búsqueda de hiperparámetros con Optuna (TPE frente a aleatoria) y prueba de semillas |
| `figures.py` | Genera las figuras del paper |
| `compare_temporal.py` | Compara el VGAE base con la variante con convolución temporal |

### Resultados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `optuna_final/summary.csv` | Métricas de cada carta en `dataset04` y en prueba |
| `optuna_final/spc_test.png` | Cartas de control sobre el conjunto de prueba |
| `optuna_final/vgae.pt` | Modelo final entrenado (pesos y grafo) |
| `optuna/best_params.json` | Hiperparámetros elegidos |
| `optuna/REPORTE.md` | Resumen de la búsqueda: TPE frente a aleatoria, importancias y semillas |
| `temporal/resumen_temporal.csv` | VGAE base frente a VGAE + convolución temporal (10 semillas) |

## Cómo usarlo

Requiere Python 3.11.

```bash
git clone https://github.com/complexnumber96/VGAE-for-anomaly-detection-on-critical-infraestructure.git
cd VGAE-for-anomaly-detection-on-critical-infraestructure
pip install -r requirements.txt
```

**1. Evaluar el modelo final** (≈ 20 s en CPU; no entrena, usa `resultados/optuna_final/vgae.pt`):

```bash
python -m vgae_spc.run --optuna
```

Imprime la tabla de resultados y reescribe `resultados/optuna_final/` con los mismos valores.

**2. Regenerar las figuras del paper:**

```bash
python -m vgae_spc.figures
```

**3. Reproducir todo desde cero** (horas; conviene GPU):

```bash
python -m vgae_spc.run --optuna --retrain   # reentrena el modelo final
python -m vgae_spc.tune                      # búsqueda con Optuna: 100 TPE + 50 aleatorios + 10 semillas
python -m vgae_spc.compare_temporal          # variante temporal, 10 semillas
python -m vgae_spc.compare_temporal --report
```

`tune.py` retoma el estudio guardado en `resultados/optuna/journal.log`. Para una búsqueda
nueva, usa otra carpeta con `--out resultados/otra_busqueda` o el cuaderno de Colab.
En CPU el entrenamiento usa por defecto 150 épocas y paciencia 15; los resultados del paper
se obtuvieron con 300 épocas y paciencia 25, en GPU.

**Compilar el paper:** `cd documentos && pdflatex paper.tex && pdflatex paper.tex`

## Datos

`datos/` contiene los conjuntos públicos de BATADAL (`dataset03`: operación normal;
`dataset04`: con ataques, usado para elegir hiperparámetros; `test`: evaluación final) y el
modelo de la red `CTOWN.INP`, descargados de <https://www.batadal.net/data.html>. Fuente:

> R. Taormina et al., "Battle of the Attack Detection Algorithms: Disclosing cyber attacks on
> water distribution networks", *Journal of Water Resources Planning and Management*,
> 144(8), 2018.
