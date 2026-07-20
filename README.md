# CVRP: Método Exacto vs. Particle Swarm Optimization

Repositorio del trabajo de investigación del curso Optimización en Ingeniería
(Magíster en Ingeniería Informática, USACH, Sem 1-2026), sobre el
*Capacitated Vehicle Routing Problem* (CVRP), en dos informes:

- **Informe 1** — método exacto: formulación de Laporte & Nobert (1983) con cortes RCI,
  resuelta con PuLP + COIN-OR CBC. (También en su repositorio original:
  <https://github.com/OrejiNet/CVRP_BNB>.)
- **Informe 2** — metaheurística: *Particle Swarm Optimization* con representación SR-1
  (Ai & Kachitvichyanukul, 2009), parametrizada con Optuna y acelerada con Numba,
  contrastada contra el método exacto sobre un conjunto común de instancias de CVRPLIB.

## Informe final (Informe 2)

El informe compilado está en
[`Informe 2 cvrp PSO vs Metodo Exacto/informe2/main.pdf`](<Informe 2 cvrp PSO vs Metodo Exacto/informe2/main.pdf>).

## Estructura del repositorio

```
.
├── Informe 2 cvrp PSO vs Metodo Exacto/   # entrega final
│   ├── cvrp_common.py                     # parser CVRPLIB, decode SR-1 (Numba), PSO, utilidades
│   ├── Instances/                         # instancias .vrp/.sol (evaluación + tuning)
│   ├── 1_Parametrizacion/                 # Optuna, 60 trials (notebook + resultados)
│   ├── 2_Experimentacion/                 # 31 repeticiones x 9 instancias (notebook + CSV)
│   ├── 3_Analisis/                        # comparación exacto vs. PSO (notebook + figuras)
│   └── informe2/                          # fuente LaTeX del informe + PDF
│
└── Informe 1 cvrp Metodo Exacto/          # entrega previa (método exacto)
    ├── cvrp_laporte1983.ipynb             # notebook Branch and Bound + cortes RCI
    ├── A/, E/                             # instancias CVRPLIB Sets A y E
    └── Informe_1_cvrp_Metodo_Exacto/      # fuente LaTeX del Informe 1
```

## Orden de ejecución (Informe 2)

1. `1_Parametrizacion/Parametrizacion_PSO.ipynb` — búsqueda de hiperparámetros
   (Optuna, TPE, 60 trials, sobre 3 instancias de tuning separadas de las de evaluación).
2. `2_Experimentacion/Experimentacion_PSO.ipynb` — 31 corridas independientes por
   instancia con los parámetros encontrados (presupuesto: 50.000 evaluaciones por corrida).
3. `3_Analisis/Analisis_Comparativo.ipynb` — tablas y figuras comparativas contra los
   resultados del método exacto del Informe 1 (no lo reejecuta).

Dependencias: `numpy`, `numba`, `pandas`, `matplotlib`, `optuna` (Python 3.12).
