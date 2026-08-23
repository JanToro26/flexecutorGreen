"""Profiling harness: sweeps a configuration space through DAGExecutor.profile().

Results are written to <exec_path>/profiling/<dag_id>/<stage_id>.json and
accumulate across runs.
"""

from __future__ import annotations

import logging
import time

from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.executor import DAGExecutor

from examples.energy_apps.dags import APPS, SERIAL_STAGES

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = (1, 2, 4, 8)
# Memory only varies the measurement on a backend that enforces it (Lambda, or
# k8s with limits). On localhost extra values just duplicate profile keys.
DEFAULT_MEMORY = (1024,)
DEFAULT_CPU = (1,)


def _apply_params(dag, params_by_stage):
    """Set per-worker-count params on each stage before a run."""
    for stage in dag.stages:
        new = params_by_stage.get(stage.stage_id)
        if new is not None:
            stage._params.clear()
            stage._params.update(new)


def profile_app(
    app: str,
    workers=DEFAULT_WORKERS,
    memory=DEFAULT_MEMORY,
    cpu=DEFAULT_CPU,
    num_reps: int = 5,
):
    """Profile one app across the configuration space.

    Each point gets its own profile() call because the workload size per worker
    changes with the worker count, and profile() fixes the params for the
    duration of a call. It reloads and appends to the stored JSON each time.
    """
    if app not in APPS:
        raise ValueError(f"Unknown app {app!r}; expected one of {sorted(APPS)}")

    dag, params_fn = APPS[app]()
    serial = SERIAL_STAGES[app]
    executor = DAGExecutor(dag)

    total = len(workers) * len(memory) * len(cpu)
    done = 0
    t_start = time.time()

    try:
        for c in cpu:
            for m in memory:
                for w in workers:
                    _apply_params(dag, params_fn(w))

                    combination = {
                        stage.stage_id: StageConfig(
                            cpu=c,
                            memory=m,
                            workers=1 if stage.stage_id in serial else w,
                        )
                        for stage in dag.stages
                    }

                    done += 1
                    logger.info(
                        "[%s] config %d/%d: cpu=%s memory=%s workers=%s x%d reps",
                        app, done, total, c, m, w, num_reps,
                    )
                    executor.profile([combination], num_reps=num_reps)

        logger.info(
            "[%s] profiling finished: %d configurations in %.1fs",
            app, total, time.time() - t_start,
        )
    finally:
        executor.shutdown()

    return dag


def profile_all(apps=None, **kwargs):
    """Profile several apps in sequence, continuing past individual failures."""
    apps = apps or list(APPS)
    results = {}
    for app in apps:
        try:
            profile_app(app, **kwargs)
            results[app] = True
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] profiling FAILED: %s", app, exc, exc_info=True)
            results[app] = False

    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    for app, ok in results.items():
        print(f"  {app:<12} {'OK' if ok else 'FAILED'}")
    return results
