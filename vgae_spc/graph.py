"""Grafo de sensores construido a partir de la topología de CTOWN.INP."""
import heapq
import re
from collections import defaultdict

import numpy as np

from .data import DATA

# Lógica de control del [CONTROLS] del .INP: tanque -> actuadores que gobierna
CONTROLS = {
    "T1": ["PU1", "PU2"], "T2": ["V2"], "T3": ["PU4", "PU5"],
    "T4": ["PU6", "PU7"], "T5": ["PU8"], "T7": ["PU10", "PU11"],
}


def parse_inp(path=DATA / "CTOWN.INP"):
    sections, cur = defaultdict(list), None
    for line in open(path):
        line = line.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("["):
            cur = line.strip("[]").upper()
            continue
        sections[cur].append(line.split())
    adj = defaultdict(list)
    links = {}
    for p in sections["PIPES"]:
        a, b, length = p[1], p[2], float(p[3])
        adj[a].append((b, length)); adj[b].append((a, length))
    for kind in ("PUMPS", "VALVES"):
        for p in sections[kind]:
            links[p[0]] = (p[1], p[2])
            adj[p[1]].append((p[2], 1.0)); adj[p[2]].append((p[1], 1.0))
    return adj, links


def dijkstra(adj, src):
    dist = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def sensor_location(name, links):
    kind, comp = name.split("_", 1)
    if kind == "L":
        return comp
    if kind == "P":
        return comp
    return links[comp][0]  # bomba / válvula: nodo de entrada


def device(name):
    return name.split("_", 1)[1]


def build_graph(sensors, k=3):
    """Adyacencia binaria simétrica (sin bucles) y tipo de arista para documentación."""
    adj, links = parse_inp()
    n = len(sensors)
    loc = [sensor_location(s, links) for s in sensors]
    dists = {l: dijkstra(adj, l) for l in set(loc)}
    D = np.array([[dists[loc[i]].get(loc[j], np.inf) for j in range(n)] for i in range(n)])

    A = np.zeros((n, n), dtype=np.float32)
    kinds = {}

    def add(i, j, kind):
        if i != j and not A[i, j]:
            A[i, j] = A[j, i] = 1
            kinds[(min(i, j), max(i, j))] = kind

    # 1) mismo dispositivo (flujo <-> estado)
    for i in range(n):
        for j in range(i + 1, n):
            if device(sensors[i]) == device(sensors[j]):
                add(i, j, "dispositivo")
    # 2) presión de entrada/salida <-> bomba/válvula a la que pertenece
    #    (estación de bombeo: distancia hidráulica casi nula)
    for i, s in enumerate(sensors):
        if not s.startswith("P_"):
            continue
        for j, t in enumerate(sensors):
            if t[0] in "FS" and D[i, j] <= 5.0:
                add(i, j, "estación")
    pressures = [i for i, s in enumerate(sensors) if s.startswith("P_")]
    for i in pressures:
        for j in pressures:
            if i < j and D[i, j] <= 5.0:
                add(i, j, "estación")
    # 3) lazos de control tanque -> actuador (vía PLC)
    for tank, devs in CONTROLS.items():
        if f"L_{tank}" not in sensors:
            continue
        i = sensors.index(f"L_{tank}")
        for j, t in enumerate(sensors):
            if t[0] in "FS" and device(t) in devs:
                add(i, j, "control")
    # 4) k vecinos hidráulicos más cercanos (longitud de tubería); orden estable para que los
    #    empates de distancia se resuelvan igual en cualquier versión de numpy
    for i in range(n):
        order = [j for j in np.argsort(D[i], kind="stable") if j != i and device(sensors[j]) != device(sensors[i])]
        for j in order[:k]:
            add(i, j, "hidráulica")
    return A, D, kinds
