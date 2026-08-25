"""DAG definitions for the energy-profiling workloads."""

from flexecutor.workflow.dag import DAG
from flexecutor.workflow.stage import Stage

from examples.energy_apps import functions as F


def build_montecarlo():
    dag = DAG("energy_montecarlo")
    stage = Stage(
        stage_id="montecarlo",
        func=F.montecarlo_stage,
        inputs=[],
        outputs=[],
        params={"points_per_worker": F.TOTAL_POINTS},
    )
    dag.add_stage(stage)

    def params(workers):
        return {"montecarlo": {"points_per_worker": F.TOTAL_POINTS // workers}}

    return dag, params


def build_titanic():
    dag = DAG("energy_titanic")
    stage = Stage(
        stage_id="titanic",
        func=F.titanic_stage,
        inputs=[],
        outputs=[],
        params={"seed": 0, "samples": F.SAMPLES_PER_WORKER},
    )
    dag.add_stage(stage)

    def params(workers):
        return {"titanic": {"seed": 0, "samples": F.SAMPLES_PER_WORKER}}

    return dag, params


def build_ml():
    dag = DAG("energy_ml")

    stage0 = Stage("stage0", func=F.ml_stage0_pca, inputs=[], outputs=[],
                   params={"seed": 42})
    stage1 = Stage("stage1", func=F.ml_stage1_train, inputs=[], outputs=[],
                   params={"seed": 0})
    stage2 = Stage("stage2", func=F.ml_stage2_aggregate, inputs=[], outputs=[],
                   params={"n_models": 1}, max_concurrency=1)
    stage3 = Stage("stage3", func=F.ml_stage3_test, inputs=[], outputs=[],
                   params={}, max_concurrency=1)

    stage0 >> [stage1, stage2, stage3]
    stage1 >> stage2
    stage2 >> stage3

    dag.add_stages([stage0, stage1, stage2, stage3])

    def params(workers):
        return {
            "stage0": {"seed": 42},
            "stage1": {"seed": 0},
            "stage2": {"n_models": workers},
            "stage3": {},
        }

    return dag, params


def build_video():
    dag = DAG("energy_video")

    stage0 = Stage("stage0", func=F.video_stage0_segment, inputs=[], outputs=[],
                   params={"workers": 1}, max_concurrency=1)
    stage1 = Stage("stage1", func=F.video_stage1_extract, inputs=[], outputs=[],
                   params={"seed": 0, "frames_per_worker": F.TOTAL_FRAMES})
    stage2 = Stage("stage2", func=F.video_stage2_enhance, inputs=[], outputs=[],
                   params={"seed": 0, "frames_per_worker": F.TOTAL_FRAMES})
    stage3 = Stage("stage3", func=F.video_stage3_analyze, inputs=[], outputs=[],
                   params={"seed": 0, "frames_per_worker": F.TOTAL_FRAMES})

    stage0 >> stage1 >> [stage2, stage3]
    stage2 >> stage3

    dag.add_stages([stage0, stage1, stage2, stage3])

    def params(workers):
        per = max(1, F.TOTAL_FRAMES // workers)
        return {
            "stage0": {"workers": workers},
            "stage1": {"seed": 0, "frames_per_worker": per},
            "stage2": {"seed": 0, "frames_per_worker": per},
            "stage3": {"seed": 0, "frames_per_worker": per},
        }

    return dag, params


APPS = {
    "montecarlo": build_montecarlo,
    "titanic": build_titanic,
    "ml": build_ml,
    "video": build_video,
}

# Stages pinned to one worker while profiling: they are serial by construction,
# so extra workers would only add idle startup cost to the training data.
SERIAL_STAGES = {
    "ml": {"stage0", "stage2", "stage3"},
    "video": {"stage0"},
    "montecarlo": set(),
    "titanic": set(),
}

def apply_pinning(app: str, dag) -> None:
    """Mark serial stages so schedulers know they have no parallelism decision."""
    serial = SERIAL_STAGES[app]
    for stage in dag.stages:
        stage.pinned = stage.stage_id in serial
