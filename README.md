# Detección de ciberataques con un Variational Graph Autoencoder y control estadístico de procesos

Código, resultados y borrador del paper sobre detección de ciberataques en la red de agua
C-Town (competición [BATADAL](https://www.batadal.net/data.html)). Un VGAE con codificador
GCN sobre el grafo de sensores (topología y lógica de control de `CTOWN.INP`) aprende la
operación normal, y sobre sus salidas se aplican cartas de control: EWMA por sensor,
T² de Hotelling en el espacio latente y SPE. El detector propuesto es la regla EWMA ∨ T².

**Autores:** Jorge Proaño, Salvador Naya, Javier Tarrío, Miguel Flores (Universidad de La Coruña).

## Estructura

| Carpeta | Contenido |
|---|---|
| `vgae_spc/` | Código: datos, grafo, modelo, cartas SPC, búsqueda con Optuna, figuras, comparación temporal |
| `paper/` | Borrador del paper (`vgae_spc_paper.tex` y PDF) |
| `informe/` | Informe de metodología (`metodologia.tex` y PDF) |
| `colab/` | Cuaderno para la búsqueda de hiperparámetros en Google Colab |
| `resultados_batadal/optuna_colab/` | Búsqueda TPE vs. aleatoria, importancias y estabilidad por semillas (100–109) |
| `resultados_batadal/optuna_colab_final/` | Modelo final (`vgae.pt`, incluye el grafo), métricas y cartas en dataset04 y test |
| `resultados_batadal/figuras_optuna_colab/` | Figuras del grafo, la arquitectura y los nodos anómalos |
| `resultados_batadal/temporal/` | VGAE base vs. VGAE + convolución temporal (10 semillas) |

## Datos

Los datos de BATADAL no se incluyen. Hay que descargarlos de <https://www.batadal.net/data.html>
y dejar en la raíz del repositorio:

```
BATADAL_training_dataset03.csv
BATADAL_training_dataset04.csv
BATADAL_test_dataset.csv   (o BATADAL_test_dataset.zip)
CTOWN.INP
```

## Uso

```bash
pip install -r requirements.txt

# Búsqueda de hiperparámetros (TPE vs. aleatoria) y estabilidad por semillas
python -m vgae_spc.tune --out resultados_batadal/optuna_colab --storage resultados_batadal/optuna_colab/journal.log --seeds 10

# Modelo final con los hiperparámetros elegidos y evaluación en test
python -m vgae_spc.run --optuna --optuna-dir resultados_batadal/optuna_colab --retrain

# Figuras
python -m vgae_spc.figures --optuna-dir resultados_batadal/optuna_colab

# Comparación con la variante temporal (TCN)
python -m vgae_spc.compare_temporal --seeds 10
python -m vgae_spc.compare_temporal --report
```

En CPU, `core.py` usa por defecto un presupuesto menor (150 épocas, paciencia 15); los
resultados del paper usan 300 épocas y paciencia 25.

Para compilar el paper: `cd paper && pdflatex vgae_spc_paper.tex` (dos veces). Las figuras
se leen de `resultados_batadal/`.
