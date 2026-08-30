"""Run Ditto's energy objective on a profiled DAG and print the allocation."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.scheduling.ditto import Ditto
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.utils.utils import flexorchestrator

from examples.energy_apps.dags import APPS, apply_pinning


@flexorchestrator(bucket="")
def main(app="video", total_parallelism=12):
    dag, _ = APPS[app]()
    apply_pinning(app, dag)

    # Ditto's constructor replaces each stage's perf model, so it has to come
    # before training.
    sched = Ditto(dag, total_parallelism=total_parallelism,
                  cpu_per_worker=1, objective="energy")

    executor = DAGExecutor(dag)
    for stage in dag.stages:
        if getattr(stage, "pinned", False):
            print(f"  {stage.stage_id}: pinned, not trained")
            continue
        executor.train(stage)
        # dE/dx, not the raw coefficient -- see AnaPerfModel.energy_marginal.
        m = stage.perf_model.energy_marginal()
        if m is None:
            print(f"  {stage.stage_id}: no energy model")
        elif m > 0:
            print(f"  {stage.stage_id}: dE/dx={m:.6g}  weight=1/m={1/m:.4f}")
        else:
            print(f"  {stage.stage_id}: dE/dx={m:.6g}  (flat or falling)")

    configs = sched.schedule()

    print(f"\n=== Ditto allocation, objective=energy, budget={total_parallelism} ===")
    for stage, cfg in zip(dag.stages, configs):
        tag = " (pinned)" if getattr(stage, "pinned", False) else ""
        print(f"  {stage.stage_id:<8} workers={cfg.workers}{tag}")
    print(f"  total allocated: {sum(c.workers for c in configs)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", default="video", choices=sorted(APPS))
    ap.add_argument("--budget", type=int, default=12)
    args = ap.parse_args()
    main(app=args.app, total_parallelism=args.budget)
