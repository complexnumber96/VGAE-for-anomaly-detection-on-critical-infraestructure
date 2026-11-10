"""Carga de los CSV de BATADAL, etiquetas de ataque y ventanas deslizantes."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Ventanas de ataque completas (inicio y fin inclusivos), transcritas de
# Attacks_TrainingDataset4.png y Attacks_TestDataset.png.
ATTACKS_TRAIN2 = [
    (1, "13/09/2016 23", "16/09/2016 00"),
    (2, "26/09/2016 11", "27/09/2016 10"),
    (3, "09/10/2016 09", "11/10/2016 20"),
    (4, "29/10/2016 19", "02/11/2016 16"),
    (5, "26/11/2016 17", "29/11/2016 04"),
    (6, "06/12/2016 07", "10/12/2016 04"),
    (7, "14/12/2016 15", "19/12/2016 04"),
]
ATTACKS_TEST = [
    (8, "16/01/2017 09", "19/01/2017 06"),
    (9, "30/01/2017 08", "02/02/2017 00"),
    (10, "09/02/2017 03", "10/02/2017 09"),
    (11, "12/02/2017 01", "13/02/2017 07"),
    (12, "24/02/2017 05", "28/02/2017 08"),
    (13, "10/03/2017 14", "13/03/2017 21"),
    (14, "25/03/2017 20", "27/03/2017 01"),
]


def _read(name):
    df = pd.read_csv(ROOT / name, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    return df.set_index("DATETIME")


def _label(index, attacks):
    y = np.zeros(len(index), dtype=int)
    att_id = np.zeros(len(index), dtype=int)
    for aid, start, end in attacks:
        s = pd.to_datetime(start, format="%d/%m/%Y %H")
        e = pd.to_datetime(end, format="%d/%m/%Y %H")
        m = (index >= s) & (index <= e)
        y[m] = 1
        att_id[m] = aid
    return y, att_id


def load_all():
    """Devuelve dict con DataFrames de señales, etiquetas y la lista de columnas usadas."""
    d03 = _read("BATADAL_training_dataset03.csv")
    d04 = _read("BATADAL_training_dataset04.csv")
    test_csv = ROOT / "BATADAL_test_dataset.csv"
    if not test_csv.exists():
        import zipfile
        with zipfile.ZipFile(ROOT / "BATADAL_test_dataset.zip") as z:
            z.extract("BATADAL_test_dataset.csv", ROOT)
    dte = _read("BATADAL_test_dataset.csv")

    sensors = [c for c in d03.columns if c != "ATT_FLAG"]
    # Variables con varianza cero en operación normal (PU3, PU5, PU9 y S_PU1)
    sensors = [c for c in sensors if d03[c].std() > 0]

    y04, id04 = _label(d04.index, ATTACKS_TRAIN2)
    yte, idte = _label(dte.index, ATTACKS_TEST)
    return {
        "sensors": sensors,
        "d03": d03[sensors], "y03": np.zeros(len(d03), dtype=int),
        "d04": d04[sensors], "y04": y04, "id04": id04,
        "test": dte[sensors], "ytest": yte, "idtest": idte,
    }


class Scaler:
    def fit(self, df):
        self.mu = df.mean().values
        self.sd = df.std().values
        return self

    def transform(self, df):
        return ((df.values - self.mu) / self.sd).astype(np.float32)


def time_of_day(index):
    """Hora del día codificada como (sen, cos): (T, 2)."""
    a = 2 * np.pi * index.hour.values / 24
    return np.stack([np.sin(a), np.cos(a)], 1).astype(np.float32)


def windows(x, w):
    """x: (T, N) -> (T-w+1, N, w); la ventana k termina en el instante k+w-1."""
    idx = np.arange(w)[None, :] + np.arange(len(x) - w + 1)[:, None]
    return np.transpose(x[idx], (0, 2, 1)).copy()
