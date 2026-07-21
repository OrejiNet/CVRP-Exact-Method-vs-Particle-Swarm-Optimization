"""
Genera las figuras propias de la presentacion (rutas, evolucion del enjambre,
acantilado de tiempo del exacto). Reutiliza cvrp_common del Informe 2.

Ejecutar desde la carpeta Presentación:  python generar_figuras.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / ".." / "Informe 2 cvrp PSO vs Metodo Exacto").resolve()))

import numpy as np
import matplotlib.pyplot as plt

import cvrp_common as cc

OUT = Path(__file__).parent / "figuras"
OUT.mkdir(exist_ok=True)

BEST_PARAMS = cc.PSOParams(
    n_particles=100, n_iter=500,
    w0=0.9116, wf=0.2782, c1=2.4045, c2=1.1843, v_max=0.3762,
)

ROUTE_COLORS = ["#0E5C4A", "#C77C1E", "#5E2150", "#1F5C92", "#B5179E",
                "#3A7D44", "#8C4A2F", "#37474F", "#7A1F3D", "#2E6F95"]


def plot_routes(ax, inst, routes, title):
    coords = inst.coords
    for j, route in enumerate(routes):
        if not route:
            continue
        seq = [0] + list(route) + [0]
        xs = coords[seq, 0]
        ys = coords[seq, 1]
        ax.plot(xs, ys, "-", color=ROUTE_COLORS[j % len(ROUTE_COLORS)], lw=1.6, zorder=1)
    ax.scatter(coords[1:, 0], coords[1:, 1], s=26, c="#37474F", zorder=2)
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=170, c="#C77C1E", marker="*",
               edgecolors="black", linewidths=0.6, zorder=3, label="depósito")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#BBBBBB")


# ---------------------------------------------------------------------------
# 1. Rutas: optimo certificado vs. mejor corrida de PSO (A-n32-k5)
# ---------------------------------------------------------------------------
inst = cc.load_instance("A-n32-k5")
m = inst.n_vehicles_ref

routes_opt, cost_opt = cc.read_sol(cc.INSTANCES_DIR / "A-n32-k5.sol")

res = cc.run_pso(inst, m, BEST_PARAMS, seed=0)  # seed 0 = mejor corrida (costo 786)
routes_pso = res.best_routes
cost_pso = res.best_cost

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
plot_routes(axes[0], inst, routes_opt, f"Óptimo certificado (Informe 1): z = {cost_opt:.0f}")
plot_routes(axes[1], inst, routes_pso, f"Mejor corrida de PSO: z = {cost_pso:.0f}  (GAP 0,26%)")
fig.suptitle("A-n32-k5: rutas del óptimo vs. la mejor solución de PSO", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "rutas_opt_vs_pso.png", dpi=170)
plt.close()
print(f"rutas_opt_vs_pso.png  (opt={cost_opt:.0f}, pso={cost_pso:.0f})")

# ---------------------------------------------------------------------------
# 2. Evolucion del enjambre: mejor global decodificado en iteraciones 1/25/500
#    (bucle PSO local identico a cvrp_common.run_pso, capturando snapshots)
# ---------------------------------------------------------------------------
SNAP_ITERS = [1, 25, 500]

rng = np.random.default_rng(0)
n = inst.n_customers
d = n + 2 * m
N = BEST_PARAMS.n_particles
v_max = BEST_PARAMS.v_max
x_min, x_max = 0.0, 1.0

X = rng.uniform(x_min, x_max, size=(N, d))
V = np.zeros((N, d))
coords_, demands_, dist_, cap_ = inst.coords, inst.demands, inst.dist, inst.capacity
fitness = cc.evaluate_batch(X, coords_, demands_, dist_, cap_, n, m)
P = X.copy()
p_fitness = fitness.copy()
g_idx = int(np.argmin(p_fitness))
g = P[g_idx].copy()
g_fitness = p_fitness[g_idx]

snapshots = {}
for t in range(BEST_PARAMS.n_iter):
    w = BEST_PARAMS.w0 + (t / max(BEST_PARAMS.n_iter - 1, 1)) * (BEST_PARAMS.wf - BEST_PARAMS.w0)
    r1 = rng.uniform(0, 1, size=(N, d))
    r2 = rng.uniform(0, 1, size=(N, d))
    V = w * V + BEST_PARAMS.c1 * r1 * (P - X) + BEST_PARAMS.c2 * r2 * (g[None, :] - X)
    V = np.clip(V, -v_max, v_max)
    X = X + V
    below = X < x_min
    above = X > x_max
    X = np.clip(X, x_min, x_max)
    V[below | above] = 0.0
    fitness = cc.evaluate_batch(X, coords_, demands_, dist_, cap_, n, m)
    improved = fitness < p_fitness
    P[improved] = X[improved]
    p_fitness[improved] = fitness[improved]
    g_idx = int(np.argmin(p_fitness))
    if p_fitness[g_idx] < g_fitness:
        g = P[g_idx].copy()
        g_fitness = p_fitness[g_idx]
    if (t + 1) in SNAP_ITERS:
        snapshots[t + 1] = (g.copy(), g_fitness)

for it, (gx, gcost) in snapshots.items():
    routes_g, _ = cc.decode_sr1_fast(gx, inst, m)
    gap = cc.gap_pct(gcost, inst.bks)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    plot_routes(ax, inst, routes_g, f"Iteración {it}: z = {gcost:.0f}  (GAP {gap:.1f}%)")
    plt.tight_layout()
    plt.savefig(OUT / f"evolucion_iter{it:03d}.png", dpi=170)
    plt.close()
    print(f"evolucion_iter{it:03d}.png  (z={gcost:.0f}, gap={gap:.2f}%)")

# ---------------------------------------------------------------------------
# 3. Acantilado de tiempo: exacto vs. PSO (escala log)
# ---------------------------------------------------------------------------
inst_names = ["E-n13-k4", "E-n22-k4", "E-n23-k3", "A-n32-k5", "A-n33-k5"]
t_exacto = [0.9, 0.8, 1.2, 576.1, 17096.4]
t_pso = [0.066, 0.086, 0.096, 0.106, 0.106]

x = np.arange(len(inst_names))
width = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.4))
b1 = ax.bar(x - width / 2, t_exacto, width, label="Exacto (Informe 1)", color="#37474F")
b2 = ax.bar(x + width / 2, t_pso, width, label="PSO (prom. 31 corridas)", color="#C77C1E")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(inst_names)
ax.set_ylabel("Tiempo (s, escala log)")
ax.legend()
ax.set_ylim(top=2e6)
ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=3)
ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=3)
ax.set_title("Tiempo de resolución: el exacto pasa de segundos a horas; PSO se mantiene bajo 0,11 s")
ax.annotate("4,7 h\n(sin certificar)", xy=(4 - width / 2, 17096.4), xytext=(2.55, 250000),
            fontsize=9, color="#7A1F3D", ha="center",
            arrowprops=dict(arrowstyle="->", color="#7A1F3D"))
plt.tight_layout()
plt.savefig(OUT / "acantilado_tiempo.png", dpi=170)
plt.close()
print("acantilado_tiempo.png")

print("\nListo. Figuras en:", OUT)
