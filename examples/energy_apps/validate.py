"""Model accuracy: predicted vs measured energy and time.

cross_validate      leave-one-configuration-out over the stored profile.
validate_against_run  train on the profile, predict a configuration, run it,
                      compare.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.utils.dataclass import StageConfig
from flexecutor.utils.utils import load_profiling_results
from flexecutor.workflow.executor import AssetType, DAGExecutor, get_asset_path

from examples.energy_apps.dags import APPS, SERIAL_STAGES, apply_pinning
from examples.energy_apps.harness import _apply_params

logger = logging.getLogger(__name__)

TIME_PHASES = ("cold_start", "read", "compute", "write")


def _numeric(values) -> List[float]:
    """Flatten to numeric leaves, dropping None and non-numbers."""
    out = []
    stack = list(values)
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, (int, float)):
            out.append(float(item))
    return out


def _flatten_str(values) -> List[str]:
    out = []
    stack = list(values)
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str):
            out.append(item)
    return out


def _measured_energy(config_data: dict) -> Optional[float]:
    """Mean over repetitions of the per-repetition sum across workers.

    Matches how AnaPerfModel.train aggregates energy, so the comparison
    measures model error and not an aggregation difference.
    """
    runs = config_data.get("energy")
    if not runs:
        return None
    per_rep = []
    for repetition in runs:
        vals = _numeric([repetition])
        if vals:
            per_rep.append(sum(vals))
    return sum(per_rep) / len(per_rep) if per_rep else None


def _measured_total_time(config_data: dict) -> Optional[float]:
    """Sum of the four phase means, which is what predict_time returns."""
    total = 0.0
    seen = False
    for phase in TIME_PHASES:
        vals = _numeric([config_data.get(phase, [])])
        if vals:
            total += sum(vals) / len(vals)
            seen = True
    return total if seen else None


def _energy_source(config_data: dict) -> str:
    sources = {s for s in _flatten_str(config_data.get("energy_source", [])) if s}
    return "/".join(sorted(sources)) if sources else "unknown"


def _pct_err(predicted: Optional[float], measured: Optional[float]) -> Optional[float]:
    if predicted is None or measured is None or not measured:
        return None
    return abs(predicted - measured) / abs(measured) * 100.0


def cross_validate(app: str) -> List[dict]:
    """Leave-one-config-out error per stage. Needs at least 3 profiled configs."""
    if app not in APPS:
        raise ValueError(f"Unknown app {app!r}; expected one of {sorted(APPS)}")

    dag, _ = APPS[app]()
    apply_pinning(app, dag)
    executor = DAGExecutor(dag)
    rows: List[dict] = []

    try:
        for stage in dag.stages:
            path = get_asset_path(executor._base_path, dag, stage, AssetType.PROFILE)
            profile = load_profiling_results(path)
            if len(profile) < 3:
                logger.warning(
                    "[%s/%s] %d profiled configuration(s); need at least 3. Skipping.",
                    app, stage.stage_id, len(profile),
                )
                continue

            model = stage.init_perf_model(PerfModelEnum.ANALYTIC)

            for held_out_key in list(profile):
                train_subset = {k: v for k, v in profile.items() if k != held_out_key}
                try:
                    model.train(train_subset)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[%s/%s] training without %s failed: %s",
                        app, stage.stage_id, held_out_key, exc,
                    )
                    continue

                cpu, memory, workers = held_out_key
                predicted = model.predict_time(
                    StageConfig(cpu=cpu, memory=memory, workers=workers)
                )
                held = profile[held_out_key]
                meas_e = _measured_energy(held)
                meas_t = _measured_total_time(held)

                rows.append({
                    "app": app,
                    "stage": stage.stage_id,
                    "cpu": cpu,
                    "memory": memory,
                    "workers": workers,
                    "energy_pred": predicted.energy,
                    "energy_meas": meas_e,
                    "energy_err_pct": _pct_err(predicted.energy, meas_e),
                    "time_pred": predicted.total,
                    "time_meas": meas_t,
                    "time_err_pct": _pct_err(predicted.total, meas_t),
                    "energy_source": _energy_source(held),
                })
    finally:
        executor.shutdown()

    return rows


def validate_against_run(
    app: str,
    workers: int,
    cpu: float = 1,
    memory: float = 1024,
) -> List[dict]:
    """Train on the stored profile, predict this configuration, run it, compare.

    Use a configuration that was not profiled, otherwise this only measures the
    fit residual.
    """
    if app not in APPS:
        raise ValueError(f"Unknown app {app!r}; expected one of {sorted(APPS)}")

    dag, params_fn = APPS[app]()
    serial = SERIAL_STAGES[app]
    apply_pinning(app, dag)
    executor = DAGExecutor(dag)
    rows: List[dict] = []

    try:
        predictions: Dict[str, object] = {}
        for stage in dag.stages:
            path = get_asset_path(executor._base_path, dag, stage, AssetType.PROFILE)
            profile = load_profiling_results(path)
            if len(profile) < 2:
                if getattr(stage, "pinned", False):
                    # Pinned to one worker by construction, so a single
                    # profile key is correct and there is nothing to fit.
                    # The stage still executes, because downstream stages
                    # depend on it; it is only excluded from the
                    # comparison. leave_one_config_out treats it the same
                    # way.
                    logger.info(
                        "[%s/%s] pinned, %d profiled configuration(s); "
                        "excluded from the comparison.",
                        app, stage.stage_id, len(profile),
                    )
                    continue
                # Not pinned, so this really is missing data rather than a
                # design decision, and it should stop the run.
                raise RuntimeError(
                    f"[{app}/{stage.stage_id}] needs at least 2 profiled "
                    f"configurations, found {len(profile)}. Run the harness first."
                )
            model = stage.init_perf_model(PerfModelEnum.ANALYTIC)
            model.train(profile)
            w = 1 if stage.stage_id in serial else workers
            predictions[stage.stage_id] = model.predict_time(
                StageConfig(cpu=cpu, memory=memory, workers=w)
            )

        _apply_params(dag, params_fn(workers))
        for stage in dag.stages:
            stage.resource_config = StageConfig(
                cpu=cpu,
                memory=memory,
                workers=1 if stage.stage_id in serial else workers,
            )

        t0 = time.time()
        futures = executor.execute()
        wall = time.time() - t0

        for stage in dag.stages:
            if stage.stage_id not in predictions:
                # No model was fitted for this stage, so there is nothing
                # to compare against. It still executed, because downstream
                # stages depend on it.
                continue
            future = futures.get(stage.stage_id)
            if future is None or future.error():
                logger.error("[%s/%s] execution failed", app, stage.stage_id)
                continue

            timings = future.get_timings()
            energies = [t.energy for t in timings if t.energy is not None]
            measured_energy = sum(energies) if energies else None
            measured_time = sum(
                sum(getattr(t, p, 0.0) or 0.0 for t in timings) / len(timings)
                for p in TIME_PHASES
            ) if timings else None

            predicted = predictions[stage.stage_id]
            sources = {t.energy_source for t in timings if t.energy_source}

            rows.append({
                "app": app,
                "stage": stage.stage_id,
                "cpu": cpu,
                "memory": memory,
                "workers": 1 if stage.stage_id in serial else workers,
                "energy_pred": predicted.energy,
                "energy_meas": measured_energy,
                "energy_err_pct": _pct_err(predicted.energy, measured_energy),
                "time_pred": predicted.total,
                "time_meas": measured_time,
                "time_err_pct": _pct_err(predicted.total, measured_time),
                "energy_source": "/".join(sorted(sources)) if sources else "none",
                "wall_s": round(wall, 3),
            })
    finally:
        executor.shutdown()

    return rows


def print_report(rows: List[dict], title: str = "VALIDATION") -> None:
    if not rows:
        print(f"\n{title}: no rows, profile the app first.")
        return

    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    def fmt(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)

    table = [
        [
            r["stage"], r["workers"],
            fmt(r["energy_pred"]), fmt(r["energy_meas"]), fmt(r["energy_err_pct"], ".1f"),
            fmt(r["time_pred"]), fmt(r["time_meas"]), fmt(r["time_err_pct"], ".1f"),
            r["energy_source"],
        ]
        for r in rows
    ]
    headers = ["Stage", "W", "E_pred(J)", "E_meas(J)", "E_err%",
               "t_pred(s)", "t_meas(s)", "t_err%", "source"]

    print(f"\n=== {title} : {rows[0]['app']} ===")
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    else:
        print(" | ".join(headers))
        for row in table:
            print(" | ".join(str(c) for c in row))

    e_errs = [r["energy_err_pct"] for r in rows if r["energy_err_pct"] is not None]
    t_errs = [r["time_err_pct"] for r in rows if r["time_err_pct"] is not None]
    if e_errs:
        print(f"  MAPE energy: {sum(e_errs) / len(e_errs):.1f}%  (n={len(e_errs)})")
    else:
        print("  MAPE energy: n/a, no stage carried a fitted energy model.")
    if t_errs:
        print(f"  MAPE time  : {sum(t_errs) / len(t_errs):.1f}%  (n={len(t_errs)})")

    if any("rapl" in (r["energy_source"] or "") for r in rows):
        print(
            "  NOTE: RAPL values are whole-package counters and the training "
            "path sums them across workers, so these figures scale with the "
            "worker count."
        )
