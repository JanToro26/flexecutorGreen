#!/usr/bin/env python3
"""Entry point for the energy-profiling experiments.

    python examples/energy_apps/run.py profile
    python examples/energy_apps/run.py profile --apps video --workers 1 2 4 8 --reps 5
    python examples/energy_apps/run.py validate --apps ml
    python examples/energy_apps/run.py holdout --apps video --workers 6 --reps 5
    python examples/energy_apps/run.py recommend --apps video --time-budget 12
    python examples/energy_apps/run.py recommend --apps video --zero-shot --time-budget 0.8
    python examples/energy_apps/run.py recommend --loao

Run from the repository root.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flexecutor.utils.utils import flexorchestrator, setup_logging  # noqa: E402

from examples.energy_apps.dags import APPS  # noqa: E402
from examples.energy_apps.harness import (  # noqa: E402
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    DEFAULT_WORKERS,
    profile_all,
)
from examples.energy_apps.recommend import (  # noqa: E402
    DEFAULT_WMAX,
    _alloc_str,
    leave_one_app_out,
    print_loao,
    print_sweep,
    sweep,
    sweep_zero_shot,
)
from examples.energy_apps.validate import (  # noqa: E402
    cross_validate,
    print_report,
    validate_against_run,
)

def _resolve_config():
    """
    Point Lithops at the repo config unless the caller overrode it, then verify
    the runtime interpreter exists.

    Without this, an unset LITHOPS_CONFIG_FILE makes Lithops fall back to
    `runtime: python.exe`, which does not resolve from the worker's working
    directory. Every worker then dies at exec with no log, and the localhost
    backend waits on results that never arrive.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if not os.environ.get("LITHOPS_CONFIG_FILE"):
        default = os.path.join(root, "config", "localhost_sudo.yaml")
        if not os.path.exists(default):
            sys.exit(f"No LITHOPS_CONFIG_FILE set and {default} does not exist.")
        os.environ["LITHOPS_CONFIG_FILE"] = default
        print(f"[config] using {default}")

    import lithops.config
    cfg = lithops.config.default_config()
    backend = cfg["lithops"]["backend"]
    if backend != "localhost":
        return

    runtime = cfg.get("localhost", {}).get("runtime", "")
    if os.path.sep in runtime or "/" in runtime:
        if not os.path.exists(runtime):
            sys.exit(
                f"Configured runtime does not exist: {runtime}\n"
                f"Config: {os.environ['LITHOPS_CONFIG_FILE']}\n"
                "Workers would fail at exec with no log. Fix the path first."
            )
        print(f"[config] runtime ok: {runtime}")

def main() -> None:
    _resolve_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["profile", "validate", "holdout", "recommend"])
    ap.add_argument("--apps", nargs="*", default=None,
                    help=f"subset of {sorted(APPS)}; default: all")
    ap.add_argument("--workers", nargs="*", type=int, default=list(DEFAULT_WORKERS))
    ap.add_argument("--memory", nargs="*", type=float, default=list(DEFAULT_MEMORY))
    ap.add_argument("--cpu", nargs="*", type=float, default=list(DEFAULT_CPU))
    # Also used by holdout: it must repeat as many times as the profile did,
    # otherwise the single measurement is biased against the profiled mean.
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--log", default="INFO")
    # recommend only
    ap.add_argument("--wmax", type=int, default=DEFAULT_WMAX,
                    help="recommend: largest total parallelism budget to consider")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="recommend: seconds; with --zero-shot, a multiple of "
                         "the single-worker time instead")
    ap.add_argument("--zero-shot", action="store_true",
                    help="recommend: ignore the target app's own profile and "
                         "borrow the normalised shapes measured on the others")
    ap.add_argument("--loao", action="store_true",
                    help="recommend: leave-one-app-out check of the zero-shot "
                         "prior against each app's own profile")
    args = ap.parse_args()

    if args.command != "recommend" and (args.zero_shot or args.loao):
        ap.error("--zero-shot and --loao only apply to the recommend command")

    setup_logging(level=getattr(logging, args.log.upper(), logging.INFO))
    apps = args.apps or list(APPS)

    @flexorchestrator(bucket=args.bucket)
    def run() -> None:
        if args.command == "profile":
            if len(args.memory) > 1:
                print("NOTE: memory only varies the measurement on a backend "
                      "that enforces it.")
            profile_all(
                apps,
                workers=tuple(args.workers),
                memory=tuple(args.memory),
                cpu=tuple(args.cpu),
                num_reps=args.reps,
            )

        elif args.command == "validate":
            for app in apps:
                print_report(cross_validate(app), title="LEAVE-ONE-CONFIG-OUT")

        elif args.command == "holdout":
            if len(args.workers) != 1:
                ap.error("holdout takes exactly one --workers value")
            for app in apps:
                print_report(
                    validate_against_run(
                        app,
                        workers=args.workers[0],
                        cpu=args.cpu[0],
                        memory=args.memory[0],
                        num_reps=args.reps,
                    ),
                    title="PREDICTED vs MEASURED",
                )

        elif args.command == "recommend":
            if args.loao:
                print_loao(
                    leave_one_app_out(
                        wmax=args.wmax, cpu=args.cpu[0], memory=args.memory[0]
                    )
                )
                return
            for app in apps:
                if args.zero_shot:
                    rows, dag, _ = sweep_zero_shot(
                        app, wmax=args.wmax, cpu=args.cpu[0], memory=args.memory[0]
                    )
                    print_sweep(rows, dag, {}, time_budget=args.time_budget,
                                relative=True)
                    continue

                # The budget sweep covers W; this loop covers s, so the
                # recommendation is a (workers, size) pair.
                sizes = [(c, m) for c in args.cpu for m in args.memory]
                best = None
                for cpu_v, mem_v in sizes:
                    if len(sizes) > 1:
                        print(f"\n--- per-worker size s = {cpu_v:g} x {mem_v:g} "
                              f"= {cpu_v * mem_v:g} ---")
                    rows, dag, unfitted = sweep(
                        app, wmax=args.wmax, cpu=cpu_v, memory=mem_v
                    )
                    print_sweep(rows, dag, unfitted, time_budget=args.time_budget)

                    feasible = [r for r in rows if r["energy_j"]
                                and (args.time_budget is None
                                     or r["time_s"] <= args.time_budget)]
                    if not feasible:
                        continue
                    cand = min(feasible, key=lambda r: r["energy_j"])
                    if best is None or cand["energy_j"] < best[0]["energy_j"]:
                        best = (cand, cpu_v, mem_v)

                if len(sizes) > 1:
                    print(f"\n=== MINIMUM OVER BOTH AXES : {app} ===")
                    if best is None:
                        print("  no configuration met the constraints.")
                    else:
                        row, cpu_v, mem_v = best
                        print(f"  per-worker size s = {cpu_v * mem_v:g} "
                              f"(cpu={cpu_v:g}, memory={mem_v:g})")
                        print(f"  parallelism budget = {row['budget']}  "
                              f"{_alloc_str(row['allocation'])}")
                        print(f"  E = {row['energy_j']:.1f} J    "
                              f"t = {row['time_s']:.1f} s")

    run()


if __name__ == "__main__":
    main()
