#!/usr/bin/env python3
"""Entry point for the energy-profiling experiments.

    python examples/energy_apps/run.py profile
    python examples/energy_apps/run.py profile --apps video --workers 1 2 4 8 --reps 5
    python examples/energy_apps/run.py validate --apps ml
    python examples/energy_apps/run.py holdout --apps video --workers 6

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
    ap.add_argument("command", choices=["profile", "validate", "holdout"])
    ap.add_argument("--apps", nargs="*", default=None,
                    help=f"subset of {sorted(APPS)}; default: all")
    ap.add_argument("--workers", nargs="*", type=int, default=list(DEFAULT_WORKERS))
    ap.add_argument("--memory", nargs="*", type=float, default=list(DEFAULT_MEMORY))
    ap.add_argument("--cpu", nargs="*", type=float, default=list(DEFAULT_CPU))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

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
                    ),
                    title="PREDICTED vs MEASURED",
                )

    run()


if __name__ == "__main__":
    main()
