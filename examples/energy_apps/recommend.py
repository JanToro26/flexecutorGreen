"""Pick a configuration. validate and holdout only check one you name.

profiled   Fit the stored profile, sweep the parallelism budget, predict DAG
           energy and makespan at each point, return the cheapest that fits
           the time budget. Ditto splits each budget across the stages.

zero-shot  No profile for this app. Classify each stage from its params
           function (does per-worker work shrink with W or not) and borrow the
           normalised curves measured on the other apps. Shape only, no scale,
           so no joules and no seconds: --time-budget becomes a multiple of the
           single-worker time.

--loao     Hold each app out of the prior and check the borrowed shapes pick
           what its own profile picks.

    run.py recommend --apps video --wmax 8
    run.py recommend --apps video --time-budget 12
    run.py recommend --apps video --zero-shot --time-budget 0.8
    run.py recommend --loao
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from flexecutor.scheduling.ditto import Ditto
from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.executor import DAGExecutor

from examples.energy_apps.dags import APPS, apply_pinning
from examples.energy_apps.validate import fit_stage_models

logger = logging.getLogger(__name__)

DIVIDES = "divides"   # per-worker work shrinks as the worker count grows
FIXED = "fixed"       # per-worker work is constant, so total work grows with W

DEFAULT_WMAX = 8


# --------------------------------------------------------------------------
# static classification
# --------------------------------------------------------------------------

def construction_class(params_fn, stage_ids, probe=(1, 4)) -> Dict[str, str]:
    """Classify each stage from params_fn, without running anything.

    A numeric param that shrinks between params_fn(1) and params_fn(4) means the
    work is divided among workers. Otherwise it is fixed per worker and total
    work grows with W. The two give different energy and time shapes; see
    validacion-modelo.md section 6.
    """
    lo, hi = params_fn(probe[0]), params_fn(probe[1])
    classes = {}
    for stage_id in stage_ids:
        before = lo.get(stage_id) or {}
        after = hi.get(stage_id) or {}
        shrinks = any(
            _is_number(before.get(key)) and _is_number(value) and value < before[key]
            for key, value in after.items()
        )
        classes[stage_id] = DIVIDES if shrinks else FIXED
    return classes


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def energy_ratios(app: str) -> Dict[str, float]:
    """Ditto's per-stage split of the budget, objective=energy.

    Own DAG instance: the constructor replaces every perf model and schedule()
    mutates the virtual DAG, so it cannot share one with the prediction models.
    Ratios do not depend on the budget size, so one call covers the sweep. An
    empty dict means one worker per stage.
    """
    dag, _ = APPS[app]()
    apply_pinning(app, dag)
    free = [s.stage_id for s in dag.stages if not getattr(s, "pinned", False)]
    executor = None
    try:
        # Inside the try: the constructor replaces the perf models and can fail
        # on its own, and _schedule_for_energy raises when an unpinned stage has
        # no fitted energy model.
        scheduler = Ditto(dag, total_parallelism=1, cpu_per_worker=1,
                          objective="energy")
        executor = DAGExecutor(dag)
        for stage in dag.stages:
            if getattr(stage, "pinned", False):
                continue
            executor.train(stage)
        scheduler.schedule()
        return dict(scheduler.parallelism_ratios)
    except Exception as exc:  # noqa: BLE001
        print(f"[recommend] Ditto could not split the budget for {app!r}: {exc}")
        print("[recommend] equal split across unpinned stages instead. The sweep "
              "still holds; the allocation is just not energy-aware.")
        return {sid: 1.0 / len(free) for sid in free} if free else {}
    finally:
        if executor is not None:
            executor.shutdown()


def workers_per_stage(dag, ratios: Dict[str, float], total: int) -> Dict[str, int]:
    """Apply the ratios to a budget, exactly as Ditto.schedule() does."""
    allocation = {}
    for stage in dag.stages:
        ratio = ratios.get(stage.stage_id)
        if getattr(stage, "pinned", False) or ratio is None:
            allocation[stage.stage_id] = 1
        else:
            allocation[stage.stage_id] = max(1, round(total * ratio))
    return allocation


def makespan(dag, stage_time: Dict[str, float]) -> float:
    """Longest path by stage time. Stages with no model count as zero, so the
    result is a lower bound whenever `unfitted` is non-empty."""
    finished: Dict[str, float] = {}

    def finish(stage) -> float:
        if stage.stage_id in finished:
            return finished[stage.stage_id]
        # Cycle guard.
        finished[stage.stage_id] = 0.0
        earliest = max((finish(p) for p in stage.parents), default=0.0)
        finished[stage.stage_id] = earliest + (stage_time.get(stage.stage_id) or 0.0)
        return finished[stage.stage_id]

    return max((finish(s) for s in dag.stages), default=0.0)


# --------------------------------------------------------------------------
# profiled sweep
# --------------------------------------------------------------------------

def sweep(app: str, wmax: int = DEFAULT_WMAX, cpu: float = 1, memory: float = 1024):
    """Predicted energy and makespan for every parallelism budget in 1..wmax."""
    ratios = energy_ratios(app)
    dag, _, executor, models, unfitted = fit_stage_models(app)
    rows: List[dict] = []

    try:
        for total in range(1, wmax + 1):
            allocation = workers_per_stage(dag, ratios, total)
            energy = 0.0
            times: Dict[str, float] = {}
            no_energy: List[str] = []

            for stage in dag.stages:
                model = models.get(stage.stage_id)
                if model is None:
                    no_energy.append(stage.stage_id)
                    continue
                predicted = model.predict_time(
                    StageConfig(cpu=cpu, memory=memory,
                                workers=allocation[stage.stage_id])
                )
                times[stage.stage_id] = predicted.total
                if predicted.energy is None:
                    no_energy.append(stage.stage_id)
                else:
                    energy += predicted.energy

            rows.append({
                "app": app,
                "budget": total,
                "wmax": wmax,
                "allocation": allocation,
                "energy_j": energy,
                "time_s": makespan(dag, times),
                "incomplete": sorted(set(no_energy)),
            })
    finally:
        executor.shutdown()

    # Rounding collapses several budgets onto the same allocation. Keep the
    # smallest: the rest pay for workers the allocation never uses.
    seen, unique = set(), []
    for row in rows:
        signature = tuple(sorted(row["allocation"].items()))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)

    return unique, dag, unfitted


# --------------------------------------------------------------------------
# zero-shot
# --------------------------------------------------------------------------

def normalised_shapes(app: str, wmax: int, cpu: float = 1, memory: float = 1024):
    """Per-stage e(W)/e(1) and t(W)/t(1) from that stage's own fitted model.

    Dividing by the W=1 value drops the workload's scale and leaves the shape,
    which is the only part that transfers between apps.
    """
    _, _, executor, models, _ = fit_stage_models(app)
    shapes: Dict[str, tuple] = {}
    try:
        for stage_id, model in models.items():
            energies, times = [], []
            for workers in range(1, wmax + 1):
                predicted = model.predict_time(
                    StageConfig(cpu=cpu, memory=memory, workers=workers)
                )
                energies.append(predicted.energy)
                times.append(predicted.total)
            if any(e is None for e in energies) or not energies[0] or not times[0]:
                continue
            shapes[stage_id] = (
                [e / energies[0] for e in energies],
                [t / times[0] for t in times],
            )
    finally:
        executor.shutdown()
    return shapes


def build_prior(wmax: int, cpu: float = 1, memory: float = 1024,
                exclude: Optional[str] = None) -> Dict[str, tuple]:
    """Class-mean normalised shapes over every stage of every app but `exclude`."""
    collected: Dict[str, List[tuple]] = {}
    for other in APPS:
        if other == exclude:
            continue
        dag, params_fn = APPS[other]()
        classes = construction_class(params_fn, [s.stage_id for s in dag.stages])
        try:
            shapes = normalised_shapes(other, wmax, cpu, memory)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[prior] skipping %s: %s", other, exc)
            continue
        for stage_id, shape in shapes.items():
            collected.setdefault(classes[stage_id], []).append(shape)

    prior = {}
    for cls, shapes in collected.items():
        n = len(shapes)
        prior[cls] = (
            [sum(s[0][i] for s in shapes) / n for i in range(wmax)],
            [sum(s[1][i] for s in shapes) / n for i in range(wmax)],
        )
    return prior


def sweep_zero_shot(app: str, wmax: int = DEFAULT_WMAX, cpu: float = 1,
                    memory: float = 1024, prior: Optional[Dict] = None):
    """Relative energy and makespan, using no profile of `app`.

    Unpinned stages are weighted equally at W=1: nothing has measured this app,
    so there is nothing else to weight them by. Exact when the stages really do
    cost the same at W=1, wrong in proportion to how far off that is.
    """
    dag, params_fn = APPS[app]()
    apply_pinning(app, dag)
    classes = construction_class(params_fn, [s.stage_id for s in dag.stages])
    prior = prior if prior is not None else build_prior(wmax, cpu, memory, exclude=app)

    free = [s for s in dag.stages if not getattr(s, "pinned", False)]
    if not free:
        raise ValueError(f"{app!r}: every stage is pinned, nothing to choose.")
    missing = sorted({classes[s.stage_id] for s in free} - set(prior))
    if missing:
        raise ValueError(
            f"{app!r}: no prior for construction class(es) {missing}. "
            "Profile at least one other app whose stages fall in that class."
        )

    rows: List[dict] = []
    base_energy = base_time = None
    for index in range(wmax):
        energy = sum(prior[classes[s.stage_id]][0][index] for s in free) / len(free)
        # Pinned stages hold at 1.0: their worker count never changes.
        times = {
            s.stage_id: (prior[classes[s.stage_id]][1][index]
                         if not getattr(s, "pinned", False) else 1.0)
            for s in dag.stages
        }
        span = makespan(dag, times)
        if index == 0:
            base_energy, base_time = energy, span
        rows.append({
            "app": app,
            "budget": index + 1,
            "wmax": wmax,
            "allocation": {s.stage_id: (1 if getattr(s, "pinned", False)
                                        else index + 1) for s in dag.stages},
            "energy_rel": energy / base_energy if base_energy else None,
            "time_rel": span / base_time if base_time else None,
        })
    return rows, dag, classes


# --------------------------------------------------------------------------
# selection and reporting
# --------------------------------------------------------------------------

def pareto(rows, e_key: str, t_key: str) -> set:
    """Budgets not dominated in BOTH energy and time."""
    keep = set()
    for row in rows:
        dominated = any(
            other is not row
            and other[e_key] <= row[e_key] and other[t_key] <= row[t_key]
            and (other[e_key] < row[e_key] or other[t_key] < row[t_key])
            for other in rows
        )
        if not dominated:
            keep.add(row["budget"])
    return keep


def pick(rows, e_key: str, t_key: str, budget: Optional[float]):
    """Lowest predicted energy among the rows meeting the time constraint."""
    feasible = [r for r in rows if budget is None or r[t_key] <= budget]
    return min(feasible, key=lambda r: r[e_key]) if feasible else None


def _alloc_str(allocation: Dict[str, int]) -> str:
    return " ".join(f"{k}={v}" for k, v in allocation.items())


def print_sweep(rows, dag, unfitted, time_budget=None, relative=False) -> None:
    e_key, t_key = ("energy_rel", "time_rel") if relative else ("energy_j", "time_s")
    e_head = "E (rel)" if relative else "E_pred(J)"
    t_head = "t (rel)" if relative else "t_pred(s)"
    mode = "ZERO-SHOT" if relative else "PROFILED"

    front = pareto(rows, e_key, t_key)
    best_energy = min(rows, key=lambda r: r[e_key])
    fastest = min(rows, key=lambda r: r[t_key])
    chosen = pick(rows, e_key, t_key, time_budget)

    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    table = []
    for row in rows:
        tags = []
        if row["budget"] in front:
            tags.append("PARETO")
        if row is best_energy:
            tags.append("min-energy")
        if row is fastest:
            tags.append("min-time")
        if chosen is not None and row is chosen:
            tags.append("CHOSEN")
        table.append([
            row["budget"], _alloc_str(row["allocation"]),
            f"{row[e_key]:.3f}", f"{row[t_key]:.3f}", " ".join(tags),
        ])
    # profiled: Ditto's total budget, split unevenly per stage. zero-shot: no
    # per-stage slopes to split by, so it is workers per unpinned stage.
    headers = ["workers/stage" if relative else "budget",
               "allocation", e_head, t_head, ""]

    print(f"\n=== RECOMMEND ({mode}) : {rows[0]['app']} ===")
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    else:
        print(" | ".join(headers))
        for line in table:
            print(" | ".join(str(c) for c in line))

    if time_budget is None:
        print(f"  -> lowest energy: budget={best_energy['budget']}  "
              f"{_alloc_str(best_energy['allocation'])}")
        print("  -> no --time-budget given, so the time constraint is not binding.")
    elif chosen is None:
        searched = rows[0].get("wmax", rows[-1]["budget"])
        print(f"  -> nothing in 1..{searched} meets "
              f"{'t/t(1)' if relative else 't'} <= {time_budget}.")
    else:
        print(f"  -> greenest within {'t/t(1)' if relative else 't'} <= {time_budget}"
              f": budget={chosen['budget']}  {_alloc_str(chosen['allocation'])}")

    if relative:
        print("  NOTE: shape only. The columns are ratios against the W=1 run, "
              "not joules and not seconds, and --time-budget is a multiple of "
              "the W=1 time.")
        print("  NOTE: unpinned stages weighted equally at W=1; nothing has "
              "measured this app.")
        print("  NOTE: same-class stages share one borrowed shape, so there is "
              "nothing to split a budget by and every unpinned stage gets the "
              "same worker count. Per-stage allocation needs the profiled mode.")
    else:
        incomplete = sorted({s for r in rows for s in r["incomplete"]})
        if incomplete:
            print(f"  NOTE: no energy model for stage(s) {incomplete}; their "
                  "energy is missing and their time counts as zero, so the "
                  "makespan is a lower bound.")
        if unfitted:
            print(f"  NOTE: unfitted stages: {unfitted}")
        flat = all(abs(r["energy_j"] - rows[0]["energy_j"]) < 1e-12 for r in rows)
        rising = all(rows[i]["energy_j"] <= rows[i + 1]["energy_j"]
                     for i in range(len(rows) - 1))
        if rising and not flat:
            print("  NOTE: energy rises monotonically at this size, so the "
                  "minimum is at the low end of the range. The fitted optimum "
                  "W* = sqrt(a/(b*s)) falls below 1 here; smaller s raises it.")


# --------------------------------------------------------------------------
# leave-one-app-out
# --------------------------------------------------------------------------

def leave_one_app_out(wmax: int = DEFAULT_WMAX, cpu: float = 1, memory: float = 1024,
                      budgets=(0.80, 0.85, 0.95, 1.0, 1.20)) -> List[dict]:
    """Does a prior built without an app pick that app's own best budget?

    Both sides use the same equal-weight normalised construction; only the
    source of the shapes differs. Per budget: match (same pick), penalty_pct
    (what the zero-shot pick costs on the real curve), feasible (whether it
    actually meets the real time constraint).
    """
    rows: List[dict] = []
    for app in APPS:
        dag, _ = APPS[app]()
        apply_pinning(app, dag)
        free = [s for s in dag.stages if not getattr(s, "pinned", False)]

        try:
            shapes = normalised_shapes(app, wmax, cpu, memory)
        except Exception as exc:  # noqa: BLE001
            print(f"[loao] {app}: no usable profile ({exc})")
            continue
        usable = [s for s in free if s.stage_id in shapes]
        if not usable:
            print(f"[loao] {app}: no stage carried a fitted energy model.")
            continue

        real = []
        base_energy = base_time = None
        for index in range(wmax):
            energy = sum(shapes[s.stage_id][0][index] for s in usable) / len(usable)
            # Pinned stages hold at 1.0, same as the zero-shot side, so the
            # two makespans are comparable.
            times = {s.stage_id: (shapes[s.stage_id][1][index]
                                  if s.stage_id in shapes else 1.0)
                     for s in dag.stages}
            span = makespan(dag, times)
            if index == 0:
                base_energy, base_time = energy, span
            real.append({
                "budget": index + 1,
                "e": energy / base_energy if base_energy else energy,
                "t": span / base_time if base_time else span,
            })

        try:
            zero, _, _ = sweep_zero_shot(
                app, wmax, cpu, memory,
                prior=build_prior(wmax, cpu, memory, exclude=app),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[loao] {app}: no zero-shot prediction ({exc})")
            continue
        zero = [{"budget": r["budget"], "e": r["energy_rel"], "t": r["time_rel"]}
                for r in zero]

        for budget in budgets:
            profiled = pick(real, "e", "t", budget)
            borrowed = pick(zero, "e", "t", budget)
            row = {"app": app, "budget_rel": budget,
                   "w_profiled": profiled["budget"] if profiled else None,
                   "w_zero_shot": borrowed["budget"] if borrowed else None}
            if borrowed:
                # Judged on the real curve, so an optimistic borrowed shape is
                # caught, including when the profile says nothing is feasible
                # and zero-shot still names something.
                on_real = real[borrowed["budget"] - 1]
                row["feasible"] = on_real["t"] <= budget
                if profiled:
                    row["match"] = profiled["budget"] == borrowed["budget"]
                    row["penalty_pct"] = (
                        (on_real["e"] - profiled["e"]) / profiled["e"] * 100
                    )
            rows.append(row)
    return rows


def print_loao(rows) -> None:
    if not rows:
        print("\nLEAVE-ONE-APP-OUT: no rows. Profile at least two apps first.")
        return
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    def fmt(value, spec=".1f"):
        return "n/a" if value is None else format(value, spec)

    table = [[
        r["app"], f"{r['budget_rel']:.2f}",
        r["w_profiled"] if r["w_profiled"] else "none",
        r["w_zero_shot"] if r["w_zero_shot"] else "none",
        r.get("match", "n/a"),
        fmt(r.get("penalty_pct")),
        r.get("feasible", "n/a"),
    ] for r in rows]
    headers = ["app", "t/t(1) budget", "W profiled", "W zero-shot",
               "match", "penalty %", "feasible"]

    print("\n=== LEAVE-ONE-APP-OUT : zero-shot vs the app's own profile ===")
    if tabulate:
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    else:
        print(" | ".join(headers))
        for line in table:
            print(" | ".join(str(c) for c in line))

    decided = [r for r in rows if "match" in r]
    if decided:
        hits = sum(1 for r in decided if r["match"])
        print(f"  same budget chosen: {hits}/{len(decided)}")
        penalties = [r["penalty_pct"] for r in decided
                     if r.get("penalty_pct") is not None and r.get("feasible")]
        if penalties:
            print(f"  mean energy penalty of the feasible zero-shot choices: "
                  f"{sum(penalties) / len(penalties):.1f}%")

    answered = [r for r in rows if r["w_zero_shot"] is not None]
    if answered:
        ok = sum(1 for r in answered if r.get("feasible"))
        print(f"  zero-shot choice meets the real time constraint: "
              f"{ok}/{len(answered)}")
    overclaim = [r for r in rows
                 if r["w_zero_shot"] is not None and r["w_profiled"] is None]
    if overclaim:
        print(f"  zero-shot named a budget where the profile says none is "
              f"feasible: {len(overclaim)} case(s) "
              f"({', '.join(sorted({r['app'] for r in overclaim}))})")
    print("  NOTE: four apps, six fitted stages. Indicative, not conclusive.")
