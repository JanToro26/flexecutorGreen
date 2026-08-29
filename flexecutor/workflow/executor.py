import logging
import os
from enum import Enum
from typing import Dict, Set, List, Optional, Callable

from tabulate import tabulate

from flexecutor.utils.dataclass import FunctionTimes, StageConfig
from flexecutor.utils.utils import (
    load_profiling_results,
    save_profiling_results,
    get_my_exec_path,
)
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.processors import ThreadPoolProcessor
from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture
from flexecutor.scheduling.scheduler import Scheduler

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """
    Enum class for asset types
    """

    MODEL = ("model", ".pkl")
    PROFILE = ("profiling", ".json")
    IMAGE = ("image", ".png")


def get_asset_path(base_path, dag, stage: Stage, asset_type: AssetType):
    dir_name, file_extension = asset_type.value
    os.makedirs(f"{base_path}/{dir_name}/{dag.dag_id}", exist_ok=True)
    return f"{base_path}/{dir_name}/{dag.dag_id}/{stage.stage_id}{file_extension}"


class DAGExecutor:
    """
    Executor class that is responsible for executing the DAG

    :param dag: DAG to execute
    :param executor: Executor to use for the stages
    """

    def __init__(
        self,
        dag: DAG,
        scheduler: Optional[Scheduler] = None,
    ):
        self._dag = dag

        self._processor = ThreadPoolProcessor()

        self._futures: Dict[str, StageFuture] = dict()
        self._num_final_stages = 0
        self._base_path = get_my_exec_path()
        self._dependence_free_stages: List[Stage] = list()
        self._running_stages: List[Stage] = list()
        self._finished_stages: Set[Stage] = set()

        self._scheduler = scheduler

    def _store_profiling(
        self,
        profile_data: dict,
        new_profile_data: List[FunctionTimes],
        resource_config: StageConfig,
    ) -> None:
        config_key = resource_config.key
        if config_key not in profile_data:
            profile_data[config_key] = {}

        # Fittable series, per-mechanism energy series, and provenance are all
        # persisted. The last two are not fitted, but without them a stored
        # profile cannot be audited afterwards: there would be no way to tell
        # a RAPL-backed run from one that fell back to the modelled estimate
        # on a host whose TDP never resolved.
        all_keys = (
            FunctionTimes.profile_keys()
            + FunctionTimes.energy_keys()
            + FunctionTimes.metadata_keys()
        )
        for key in all_keys:
            if key not in profile_data[config_key]:
                profile_data[config_key][key] = []
            profile_data[config_key][key].append([])
        for profiling in new_profile_data:
            for key in all_keys:
                profile_data[config_key][key][-1].append(
                    getattr(profiling, key, None)
                )
        print(f"Profile data: {profile_data}")

    def profile(
        self,
        # TODO: add a profile id (also on training) to allow having different
        # trained models, mostly for different backends (k8s, lambda, etc.)
        config_space: List[Dict[str, StageConfig]],
        callback: Optional[Callable] = None,
        num_reps: int = 1,
    ) -> None:

        logger.info(f"Profiling DAG {self._dag.dag_id}")

        for config in config_space:
            if len(config) != len(self._dag.stages):
                raise ValueError(
                    "Each configuration dictionary must have the same length as the number of stages in the DAG"
                )
            for stage_id in self._dag.stages:
                if stage_id._stage_id not in config:
                    raise ValueError(
                        f"Configuration for stage {stage_id._stage_id} is missing"
                    )
        profile_data = {}
        for stage in self._dag.stages:
            profiling_file = get_asset_path(
                self._base_path, self._dag, stage, AssetType.PROFILE
            )
            profile_data[stage.stage_id] = load_profiling_results(profiling_file)

        for iteration in range(num_reps):
            logger.info(f"Starting iteration {iteration + 1} of {num_reps}")

            for config_combination in config_space:

                config_description = ", ".join(
                    f"{stage_id} config: {config}"
                    for stage_id, config in config_combination.items()
                )
                logger.info(f"Applying configuration combination: {config_description}")

                for stage in self._dag.stages:
                    stage.resource_config = config_combination[stage._stage_id]
                    stage.state = StageState.NONE
                    logger.info(
                        f"Configured {stage.stage_id} with {config_combination[stage._stage_id]}"
                    )

                futures = self.execute()

                for stage in self._dag.stages:
                    future = futures.get(stage.stage_id)
                    if future and not future.error():
                        timings = future.get_timings()
                        self._store_profiling(
                            profile_data[stage.stage_id],
                            timings,
                            config_combination[stage._stage_id],
                        )
                        logger.info(
                            f"Profiling data for {stage.stage_id} updated in memory"
                        )
                        profiling_file = get_asset_path(
                            self._base_path, self._dag, stage, AssetType.PROFILE
                        )
                        save_profiling_results(
                            profiling_file, profile_data[stage.stage_id]
                        )
                        logger.info(
                            f"Profiling data for {stage.stage_id} saved in {profiling_file}"
                        )
                    elif future and future.error():
                        logger.error(
                            f"Error processing stage {stage.stage_id}: {future.error()}"
                        )

                if callback:
                    callback()

        logger.info("Profiling completed for all configurations")

    # Unused, validate.py actually uses method called predict_time defined in models
    def predict(
        self, resource_config: List[StageConfig], stage: Optional[Stage] = None
    ) -> List[FunctionTimes]:
        """Predicts the latency of the entire dag for a given dictionary of resource configurations.
        If a stage is provided, it will predict the latency of that stage only.


        Args:
            resource_config (List[StageConfig]): Dictionary with the resource configuration for each stage.
            stage (Optional[Stage], optional): Stage to profile. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            List[FunctionTimes]: _description_
        """
        # TODO: predict latency/cost of the full dag. Return an object with the
        # breakdown of latencies per stage.

        # FIXME: (?) predict makes sense to move as method of DAG/Stage since models
        # are stored there. Train too?
        # Keep this method as a convenient wrapper for self._dag.predict()

        # FIXME: (?) resource_config as a list or as a dict by stage_id?
        # assert it contains config for all stages

        if stage is not None and len(resource_config) > 1:
            raise ValueError(
                "predict() requires single Stage when only one StageConfig is provided and vice versa."
            )
        elif stage is None and len(resource_config) != len(self._dag.stages):
            raise ValueError(
                "predict() requires a list of StageConfig equal to the number of Stage in the DAG."
            )
        result = []
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage, resource_config in zip(stages_list, resource_config):
            result.append(stage.perf_model.predict(resource_config))
        return result

    def train(self, stage: Optional[Stage] = None) -> None:
        """Train the DAG."""
        stages_list = [stage] if stage is not None else self._dag.stages
        for stage in stages_list:
            profile_data = load_profiling_results(
                get_asset_path(self._base_path, self._dag, stage, AssetType.PROFILE)
            )
            if stage.perf_model:
                stage.perf_model.train(profile_data)
                stage.perf_model.save_model()

    def execute(self, num_workers=None) -> Dict[str, StageFuture]:
        """
        Execute the DAG

        @param num_workers: **DEV** parameter to set the number of workers to use in the stages in the DAG.
         If defined, overrides the number of workers defined in the resource configuration of the stages.
        @return A dictionary with the output data of the DAG stages with the stage ID as key
        """
        logger.info(f"Executing DAG {self._dag.dag_id}")

        self._num_final_stages = len(self._dag.leaf_stages)
        logger.info(f"DAG {self._dag.dag_id} has {self._num_final_stages} final stages")

        # Before the execution, get the optimal configurations for all stages in the DAG
        # FIXME: actually optimize, hardcoded for now

        # self.train()
        # FIXME: the optimal config seems to be an array, why is that?
        # self.optimize(ConfigBounds(*[(1, 6), (512, 4096), (1, 3)]))
        if num_workers is not None:
            for stage in self._dag.stages:
                stage.resource_config = StageConfig(
                    cpu=1, memory=2048, workers=num_workers
                )

        self._futures = dict()

        # Start by executing the root stages
        self._dependence_free_stages = set(self._dag.root_stages)
        self._running_stages = set()
        self._finished_stages = set()

        # Execute stages until all stages have been executed
        while self._dependence_free_stages or self._running_stages:
            # Select the stages to execute
            batch = list(self._dependence_free_stages)

            # Add the batch to the running stages
            set_batch = set(batch)
            self._running_stages |= set_batch

            # Call the processor to execute the batch
            futures = self._processor.process(batch)

            self._running_stages -= set_batch
            self._dependence_free_stages -= set_batch
            self._finished_stages |= set_batch

            for stage in batch:
                self._futures[stage.stage_id] = futures[stage.stage_id]
                for child in stage.children:
                    if child.parents.issubset(self._finished_stages):
                        self._dependence_free_stages.add(child)

        return self._futures

    def optimize(self):
        """
        Sets the optimal configuration for each stage.
        """

        print(f"Optimizing DAG {self._dag.dag_id}")

        if self._scheduler is None:
            raise ValueError("Scheduler not defined")

        self.train()
        resource_config_list = self._scheduler.schedule()

        stages_data = []

        for stage, resource_config in zip(self._dag.stages, resource_config_list):
            stages_data.append(
                [
                    stage.stage_id,
                    resource_config.cpu,
                    resource_config.memory,
                    resource_config.workers,
                ]
            )

        headers = ["Stage #", "CPU (vCPU)", "Memory (MB)", "Workers"]
        print(tabulate(stages_data, headers=headers, tablefmt="fancy_grid"))

    def shutdown(self):
        """
        Shutdown the executor
        """
        self._processor.shutdown()
