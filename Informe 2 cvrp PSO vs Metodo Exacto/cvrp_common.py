"""
Código compartido para el Informe 2 (CVRP con PSO, representación SR-1 de
Ai & Kachitvichyanukul, 2009): lectura de instancias CVRPLIB, decodificación
de partícula a rutas (Algoritmo 2 del paper), función de aptitud, PSO canónico
y utilidades de experimentación/análisis.

Todas las funciones asumen que el nodo 1 del archivo .vrp es el depósito.
Los "clientes" se re-indexan internamente como 0..n-1 (n = DIMENSION - 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

_HERE = Path(__file__).resolve().parent
INSTANCES_DIR = _HERE / "Instances"


# ---------------------------------------------------------------------------
# 1. Lectura de instancias CVRPLIB (.vrp / .sol)
# ---------------------------------------------------------------------------

@dataclass
class CVRPInstance:
    name: str
    n_customers: int
    capacity: float
    coords: np.ndarray      # (n_customers + 1, 2), índice 0 = depósito
    demands: np.ndarray     # (n_customers + 1,), demands[0] = 0
    dist: np.ndarray        # (n_customers + 1, n_customers + 1), distancias reales de la instancia
    bks: float | None = None
    n_vehicles_ref: int | None = None  # nº de vehículos usado en el .sol de referencia
    coords_are_exact: bool = True  # False si `coords` viene de un embedding MDS (ver `_mds_embed`)


def _euclidean_dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return np.round(d)  # convención EUC_2D de TSPLIB/CVRPLIB


def _mds_embed(dist: np.ndarray) -> np.ndarray:
    """Embedding 2D aproximado (classical MDS / escalamiento multidimensional clásico)
    a partir de una matriz de distancias, para instancias sin coordenadas (EXPLICIT).

    Solo se usa para el mecanismo de puntos de referencia de vehículos del decode
    SR-1 (que necesita un plano 2D); los costos siempre se calculan con `dist`
    original, nunca con distancias reconstruidas desde este embedding."""
    d2 = dist ** 2
    n = dist.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    top2 = np.clip(eigvals[:2], a_min=0, a_max=None)
    return eigvecs[:, :2] * np.sqrt(top2)[None, :]


def _parse_lower_row(values: list[float], dimension: int) -> np.ndarray:
    """Reconstruye una matriz de distancias simétrica completa a partir del
    formato LOWER_ROW de TSPLIB/CVRPLIB (triangular inferior, sin diagonal,
    aplanada por filas: fila 2 tiene 1 valor, fila 3 tiene 2, ..., fila n tiene n-1)."""
    dist = np.zeros((dimension, dimension))
    it = iter(values)
    for i in range(1, dimension):
        for j in range(i):
            v = next(it)
            dist[i, j] = v
            dist[j, i] = v
    return dist


def read_vrp(path: str | Path) -> CVRPInstance:
    path = Path(path)
    text = path.read_text().splitlines()

    name = None
    dimension = None
    capacity = None
    edge_weight_type = None
    coords_raw: dict[int, tuple[float, float]] = {}
    demands_raw: dict[int, float] = {}
    edge_weight_values: list[float] = []

    section = None
    for line in text:
        line = line.strip()
        if not line or line == "EOF":
            continue
        if line.startswith("NAME"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1])
        elif line.startswith("CAPACITY"):
            capacity = float(line.split(":", 1)[1])
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            pass  # se asume LOWER_ROW (única variante presente en las instancias usadas)
        elif line.startswith("NODE_COORD_SECTION"):
            section = "coord"
        elif line.startswith("DEMAND_SECTION"):
            section = "demand"
        elif line.startswith("DEPOT_SECTION"):
            section = "depot"
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            section = "edge_weight"
        elif line.startswith("DISPLAY_DATA_TYPE"):
            pass
        elif section == "coord":
            parts = line.split()
            idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            coords_raw[idx] = (x, y)
        elif section == "demand":
            parts = line.split()
            idx, q = int(parts[0]), float(parts[1])
            demands_raw[idx] = q
        elif section == "depot":
            pass  # se asume depósito = nodo 1 (convención CVRPLIB estándar)
        elif section == "edge_weight":
            edge_weight_values.extend(float(v) for v in line.split())

    n_customers = dimension - 1
    demands = np.zeros(dimension)
    for idx in range(1, dimension + 1):
        demands[idx - 1] = demands_raw[idx]

    if edge_weight_type == "EXPLICIT":
        # Sin coordenadas reales: la matriz de distancias es la fuente de verdad
        # (todos los costos se calculan con ella); las "coordenadas" son un
        # embedding 2D aproximado solo para el mecanismo de referencia de
        # vehículos del decode SR-1 (ver `_mds_embed`).
        dist = _parse_lower_row(edge_weight_values, dimension)
        coords = _mds_embed(dist)
        coords_are_exact = False
    else:
        coords = np.zeros((dimension, 2))
        for idx in range(1, dimension + 1):
            coords[idx - 1] = coords_raw[idx]
        dist = _euclidean_dist_matrix(coords)
        coords_are_exact = True

    return CVRPInstance(
        name=name,
        n_customers=n_customers,
        capacity=capacity,
        coords=coords,
        demands=demands,
        dist=dist,
        coords_are_exact=coords_are_exact,
    )


def read_sol(path: str | Path) -> tuple[list[list[int]], float]:
    """Lee un .sol de CVRPLIB. Devuelve (rutas, costo).

    La numeración de clientes en el .sol es 1..n (el depósito no se numera),
    lo que coincide directamente con el índice de array usado en `dist`
    (índice 0 = depósito, índice i = cliente i) — no requiere desplazamiento."""
    path = Path(path)
    routes: list[list[int]] = []
    cost = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Route"):
            nodes = [int(tok) for tok in line.split(":", 1)[1].split()]
            routes.append(nodes)
        elif line.startswith("Cost"):
            cost = float(line.split()[1])
    return routes, cost


def route_cost(route: list[int], dist: np.ndarray) -> float:
    """Costo de una ruta de clientes (0-indexados), incluyendo ida/vuelta al depósito (nodo 0)."""
    if not route:
        return 0.0
    c = dist[0, route[0]] + dist[route[-1], 0]
    for a, b in zip(route[:-1], route[1:]):
        c += dist[a, b]
    return c


def total_cost(routes: list[list[int]], dist: np.ndarray) -> float:
    return sum(route_cost(r, dist) for r in routes)


# ---------------------------------------------------------------------------
# 2. Decodificación SR-1 (Ai & Kachitvichyanukul, 2009 — Algoritmo 2)
# ---------------------------------------------------------------------------
#
# Partícula: vector de (n + 2m) reales.
#   - x[0:n]         -> claves de prioridad de los n clientes (SPV).
#   - x[n:n+m]        -> coordenada x del punto de referencia del vehículo j.
#   - x[n+m:n+2m]     -> coordenada y del punto de referencia del vehículo j.
#
# La decodificación puede dejar clientes sin asignar si, para un cliente,
# ninguno de los m vehículos admite insertarlo sin violar la capacidad
# (esto también le ocurre al paper original: Tabla 2, vrpnc9 con SR-1,
# "All replications yielded infeasible solution"). En ese caso se retorna
# la lista de clientes no asignados junto con las rutas parciales, y quien
# llama decide la penalización (ver `evaluate_particle`).

PENALTY_UNASSIGNED = 1e7  # penalización dura por cliente sin asignar


def _priority_list(x_customers: np.ndarray) -> np.ndarray:
    """SPV: ordena los clientes ascendentemente por su clave real. Devuelve
    índices de array (1..n, con 0 = depósito), no números de cliente 0-indexados,
    para que sean directamente usables en `dist`/`demands`."""
    return np.argsort(x_customers, kind="stable") + 1


def _vehicle_priority_matrix(x_vehicles: np.ndarray, coords: np.ndarray, m: int) -> np.ndarray:
    """Para cada cliente, el orden de vehículos por distancia a su punto de referencia.
    Devuelve un array (n_customers, m) con índices de vehículo (0..m-1)."""
    ref_x = x_vehicles[:m]
    ref_y = x_vehicles[m:2 * m]
    ref_points = np.stack([ref_x, ref_y], axis=1)  # (m, 2)
    customer_coords = coords[1:]  # excluye depósito
    diff = customer_coords[:, None, :] - ref_points[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))  # (n_customers, m)
    return np.argsort(d, axis=1, kind="stable")


def _best_insertion_cost(route: list[int], customer: int, dist: np.ndarray) -> tuple[int, float]:
    """Mejor posición (índice) e incremento de costo al insertar `customer` en `route`
    (clientes 0-indexados, sin incluir el depósito)."""
    if not route:
        return 0, dist[0, customer] + dist[customer, 0]
    best_pos, best_delta = None, None
    prev = 0
    for i, node in enumerate(route):
        delta = dist[prev, customer] + dist[customer, node] - dist[prev, node]
        if best_delta is None or delta < best_delta:
            best_pos, best_delta = i, delta
        prev = node
    # posición final (después del último cliente, antes de volver al depósito)
    delta_end = dist[route[-1], customer] + dist[customer, 0] - dist[route[-1], 0]
    if delta_end < best_delta:
        best_pos, best_delta = len(route), delta_end
    return best_pos, best_delta


def _two_opt(route: list[int], dist: np.ndarray) -> list[int]:
    """2-opt clásico sobre una ruta (clientes 0-indexados), incluyendo el depósito
    como nodo fijo en los extremos (Algoritmo 4 del paper)."""
    if len(route) < 3:
        return route
    improved = True
    route = route[:]
    n = len(route)
    ext = [0] + route + [0]
    while improved:
        improved = False
        for i in range(0, n - 1):
            for j in range(i + 2, n + 1):
                a, b, c, d = ext[i], ext[i + 1], ext[j], ext[j + 1]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    ext[i + 1:j + 1] = ext[i + 1:j + 1][::-1]
                    improved = True
        n = len(ext) - 2
    return ext[1:-1]


def decode_sr1(
    x: np.ndarray,
    instance: CVRPInstance,
    n_vehicles: int,
    apply_local_search: bool = True,
) -> tuple[list[list[int]], list[int]]:
    """Decodifica una partícula SR-1 a un conjunto de a lo más `n_vehicles` rutas.

    Devuelve (rutas, clientes_sin_asignar), con clientes como índices de array
    (1..n; 0 es el depósito), directamente usables en `instance.dist`.
    """
    n = instance.n_customers
    m = n_vehicles
    dist = instance.dist
    demands = instance.demands  # demands[0] = depósito, demands[1:] = clientes

    priority_list = _priority_list(x[:n])                       # U, orden de clientes (índices 1..n)
    vehicle_priority = _vehicle_priority_matrix(x[n:n + 2 * m], instance.coords, m)  # V

    routes: list[list[int]] = [[] for _ in range(m)]
    loads = np.zeros(m)
    unassigned: list[int] = []

    for customer in priority_list:  # customer: índice de array (1..n)
        demand = demands[customer]
        placed = False
        for j in vehicle_priority[customer - 1]:
            if loads[j] + demand > instance.capacity:
                continue
            pos, _delta = _best_insertion_cost(routes[j], customer, dist)
            routes[j].insert(pos, customer)
            loads[j] += demand
            if apply_local_search:
                routes[j] = _two_opt(routes[j], dist)
            placed = True
            break
        if not placed:
            unassigned.append(customer)

    return routes, unassigned


def evaluate_particle_reference(
    x: np.ndarray,
    instance: CVRPInstance,
    n_vehicles: int,
    apply_local_search: bool = True,
) -> float:
    """Versión de referencia (Python puro, listas) de la aptitud. Se conserva
    solo para validar que la versión Numba (`evaluate_particle`, más abajo)
    da resultados idénticos; no se usa en las corridas reales por lentitud."""
    routes, unassigned = decode_sr1(x, instance, n_vehicles, apply_local_search)
    cost = total_cost(routes, instance.dist)
    if unassigned:
        cost += PENALTY_UNASSIGNED * len(unassigned)
    return cost


# ---------------------------------------------------------------------------
# 2b. Decodificación SR-1 acelerada con Numba (arrays fijos, sin listas de Python)
# ---------------------------------------------------------------------------
#
# Misma lógica que `decode_sr1`/`_best_insertion_cost`/`_two_opt` de arriba, pero
# reescrita sobre arrays NumPy de tamaño fijo (con -1 como centinela de "slot
# vacío") para poder compilarse en modo nopython de Numba. Es la versión que
# efectivamente usan `evaluate_particle` y `run_pso`.

@njit(cache=True)
def _two_opt_inplace_nb(route: np.ndarray, rlen: int, dist: np.ndarray) -> None:
    """2-opt in-place sobre `route[0:rlen]` (Algoritmo 4 del paper), con el
    depósito (nodo 0) como extremo fijo implícito."""
    if rlen < 3:
        return
    improved = True
    while improved:
        improved = False
        for i in range(0, rlen - 1):
            a = route[i - 1] if i > 0 else 0
            b = route[i]
            for j in range(i + 2, rlen + 1):
                c = route[j - 1]
                d = route[j] if j < rlen else 0
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    lo, hi = i, j - 1
                    while lo < hi:
                        route[lo], route[hi] = route[hi], route[lo]
                        lo += 1
                        hi -= 1
                    improved = True
                    b = route[i]


@njit(cache=True)
def _decode_sr1_nb(
    x: np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    dist: np.ndarray,
    capacity: float,
    n: int,
    m: int,
    apply_local_search: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decode SR-1 (Algoritmo 2 del paper) en modo nopython.

    Devuelve (routes, route_len, n_unassigned). `routes` es (m, n) con -1 como
    relleno; la ruta del vehículo j son sus primeros `route_len[j]` elementos.
    """
    priority = np.argsort(x[:n]) + 1  # customer ids 1..n, en orden de prioridad

    ref_x = x[n:n + m]
    ref_y = x[n + m:n + 2 * m]

    routes = np.full((m, n), -1, dtype=np.int64)
    route_len = np.zeros(m, dtype=np.int64)
    loads = np.zeros(m, dtype=np.float64)
    n_unassigned = 0

    veh_order = np.empty(m, dtype=np.int64)
    veh_dist = np.empty(m, dtype=np.float64)

    for idx in range(n):
        customer = priority[idx]
        cx = coords[customer, 0]
        cy = coords[customer, 1]
        for j in range(m):
            dx = cx - ref_x[j]
            dy = cy - ref_y[j]
            veh_dist[j] = dx * dx + dy * dy
            veh_order[j] = j
        # insertion sort de veh_order por veh_dist (m es chico: 5-14 vehículos)
        for a in range(1, m):
            key_d = veh_dist[a]
            key_o = veh_order[a]
            b = a - 1
            while b >= 0 and veh_dist[b] > key_d:
                veh_dist[b + 1] = veh_dist[b]
                veh_order[b + 1] = veh_order[b]
                b -= 1
            veh_dist[b + 1] = key_d
            veh_order[b + 1] = key_o

        demand = demands[customer]
        placed = False
        for oi in range(m):
            j = veh_order[oi]
            if loads[j] + demand > capacity:
                continue
            rlen = route_len[j]
            if rlen == 0:
                best_pos = 0
            else:
                best_pos = 0
                best_delta = 1e18
                prev = 0
                for p in range(rlen):
                    node = routes[j, p]
                    delta = dist[prev, customer] + dist[customer, node] - dist[prev, node]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = p
                    prev = node
                last = routes[j, rlen - 1]
                delta_end = dist[last, customer] + dist[customer, 0] - dist[last, 0]
                if delta_end < best_delta:
                    best_pos = rlen
            for p in range(rlen, best_pos, -1):
                routes[j, p] = routes[j, p - 1]
            routes[j, best_pos] = customer
            route_len[j] += 1
            loads[j] += demand
            if apply_local_search:
                _two_opt_inplace_nb(routes[j], route_len[j], dist)
            placed = True
            break
        if not placed:
            n_unassigned += 1

    return routes, route_len, n_unassigned


@njit(cache=True)
def _total_cost_nb(routes: np.ndarray, route_len: np.ndarray, dist: np.ndarray, m: int) -> float:
    cost = 0.0
    for j in range(m):
        rlen = route_len[j]
        if rlen == 0:
            continue
        cost += dist[0, routes[j, 0]]
        for p in range(rlen - 1):
            cost += dist[routes[j, p], routes[j, p + 1]]
        cost += dist[routes[j, rlen - 1], 0]
    return cost


@njit(cache=True)
def evaluate_particle(
    x: np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    dist: np.ndarray,
    capacity: float,
    n: int,
    m: int,
    apply_local_search: bool = True,
) -> float:
    """Aptitud (Numba) de una partícula: costo total de las rutas decodificadas,
    más una penalización dura por cada cliente sin asignar (ver nota sobre
    infactibilidad más arriba). Firma "plana" (arrays sueltos, no `CVRPInstance`)
    porque Numba en modo nopython no puede recibir un `dataclass` de Python."""
    routes, route_len, n_unassigned = _decode_sr1_nb(
        x, coords, demands, dist, capacity, n, m, apply_local_search
    )
    cost = _total_cost_nb(routes, route_len, dist, m)
    if n_unassigned > 0:
        cost += PENALTY_UNASSIGNED * n_unassigned
    return cost


@njit(cache=True, parallel=True)
def evaluate_batch(
    X: np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    dist: np.ndarray,
    capacity: float,
    n: int,
    m: int,
    apply_local_search: bool = True,
) -> np.ndarray:
    """Evalúa un enjambre completo (N partículas) en paralelo. `X` es (N, n+2m)."""
    N = X.shape[0]
    costs = np.empty(N, dtype=np.float64)
    for i in prange(N):
        costs[i] = evaluate_particle(X[i], coords, demands, dist, capacity, n, m, apply_local_search)
    return costs


def decode_sr1_fast(
    x: np.ndarray, instance: CVRPInstance, n_vehicles: int, apply_local_search: bool = True
) -> tuple[list[list[int]], list[int]]:
    """Envoltorio en Python de `_decode_sr1_nb` que devuelve rutas como listas
    (para inspección/graficado), en el mismo formato que `decode_sr1`."""
    routes_arr, route_len, _ = _decode_sr1_nb(
        x, instance.coords, instance.demands, instance.dist, instance.capacity,
        instance.n_customers, n_vehicles, apply_local_search,
    )
    routes = [list(routes_arr[j, :route_len[j]]) for j in range(n_vehicles)]
    assigned = {c for r in routes for c in r}
    unassigned = [c for c in range(1, instance.n_customers + 1) if c not in assigned]
    return routes, unassigned


# ---------------------------------------------------------------------------
# 3. PSO canónico (informe2/chapters/4/metodo.tex, Ecs. m2_velocidad / m2_posicion)
# ---------------------------------------------------------------------------

@dataclass
class PSOParams:
    n_particles: int = 50
    n_iter: int = 1000
    w0: float = 0.9
    wf: float = 0.4
    c1: float = 1.5
    c2: float = 1.5
    v_max: float | None = None  # si None, se fija a x_max - x_min


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_cost: float
    best_routes: list[list[int]]
    history: np.ndarray  # (n_iter,), mejor costo global por iteración


def run_pso(
    instance: CVRPInstance,
    n_vehicles: int,
    params: PSOParams,
    seed: int = 0,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> PSOResult:
    rng = np.random.default_rng(seed)
    n = instance.n_customers
    m = n_vehicles
    d = n + 2 * m
    N = params.n_particles
    v_max = params.v_max if params.v_max is not None else (x_max - x_min)

    X = rng.uniform(x_min, x_max, size=(N, d))
    V = np.zeros((N, d))

    coords, demands, dist, capacity = instance.coords, instance.demands, instance.dist, instance.capacity

    fitness = evaluate_batch(X, coords, demands, dist, capacity, n, m)
    P = X.copy()
    p_fitness = fitness.copy()
    g_idx = int(np.argmin(p_fitness))
    g = P[g_idx].copy()
    g_fitness = p_fitness[g_idx]

    history = np.empty(params.n_iter)

    for t in range(params.n_iter):
        w = params.w0 + (t / max(params.n_iter - 1, 1)) * (params.wf - params.w0)
        r1 = rng.uniform(0, 1, size=(N, d))
        r2 = rng.uniform(0, 1, size=(N, d))
        V = w * V + params.c1 * r1 * (P - X) + params.c2 * r2 * (g[None, :] - X)
        V = np.clip(V, -v_max, v_max)
        X = X + V
        # clamping de posición (Ecs. 5-8 del paper: reflejar velocidad a 0 en el borde)
        below = X < x_min
        above = X > x_max
        X = np.clip(X, x_min, x_max)
        V[below | above] = 0.0

        fitness = evaluate_batch(X, coords, demands, dist, capacity, n, m)
        improved = fitness < p_fitness
        P[improved] = X[improved]
        p_fitness[improved] = fitness[improved]

        g_idx = int(np.argmin(p_fitness))
        if p_fitness[g_idx] < g_fitness:
            g = P[g_idx].copy()
            g_fitness = p_fitness[g_idx]

        history[t] = g_fitness

    best_routes, _ = decode_sr1_fast(g, instance, m)
    return PSOResult(best_x=g, best_cost=g_fitness, best_routes=best_routes, history=history)


# ---------------------------------------------------------------------------
# 4. Utilidades de experimentación
# ---------------------------------------------------------------------------

def gap_pct(obtained: float, reference: float) -> float:
    return (obtained - reference) / reference * 100.0


def load_instance(name: str) -> CVRPInstance:
    """Carga una instancia por nombre desde `Instances/`, adjuntando su BKS y
    nº de vehículos de referencia si existe un .sol homónimo."""
    inst = read_vrp(INSTANCES_DIR / f"{name}.vrp")
    sol_path = INSTANCES_DIR / f"{name}.sol"
    if sol_path.exists():
        routes, cost = read_sol(sol_path)
        inst.bks = cost
        inst.n_vehicles_ref = len(routes)
    return inst
