# Energy-profiling apps

Four workloads (montecarlo, titanic, ml, video) profiled through
`DAGExecutor.profile()`, which writes to
`<exec_path>/profiling/<dag_id>/<stage_id>.json` in the format
`AnaPerfModel.train()` consumes.

## Usage

From the repository root:

```bash
python examples/energy_apps/run.py profile
python examples/energy_apps/run.py profile --apps video --workers 1 2 4 8 --reps 5
python examples/energy_apps/run.py validate --apps ml
python examples/energy_apps/run.py holdout --apps video --workers 6
```

With a specific backend:

```bash
LITHOPS_CONFIG_FILE=config/localhost_sudo.yaml python examples/energy_apps/run.py profile
```

Profiles accumulate across runs; running `profile` twice adds repetitions.

## DAGs

| app | stages | shape |
|---|---|---|
| `montecarlo` | 1 | total work constant, per-worker share shrinks with W |
| `titanic` | 1 | fixed work per worker, total work grows with W |
| `ml` | 4 | `stage0 >> [stage1, stage2, stage3]`, `stage1 >> stage2 >> stage3` |
| `video` | 4 | `stage0 >> stage1 >> [stage2, stage3]`, `stage2 >> stage3` |

`ml` and `video` mix serial and parallel stages, which is what a per-stage
allocator acts on. `SERIAL_STAGES` in `dags.py` pins the serial ones to one
worker during profiling.

Stages are compute-only (`inputs=[]`, `outputs=[]`), so no storage bucket is
needed on any backend. Dependencies: numpy and scikit-learn.

## Caveats

- `AnaPerfModel.train` sums `energy` across a stage's workers. That is correct
  for the psutil model but not for RAPL, whose per-worker value is a
  whole-package counter. `validate.py` reports `energy_source` per row.
- On localhost, `worker_processes` in the Lithops config must be at least the
  largest worker count swept, or invocations serialise. See
  `config/localhost_sudo.yaml`.
- `DAGExecutor.predict()` calls `perf_model.predict`, which does not exist
  (the method is `predict_time`). `validate.py` calls `predict_time` directly.
