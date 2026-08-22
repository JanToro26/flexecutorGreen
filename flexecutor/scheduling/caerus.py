from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.scheduling.scheduler import Scheduler

from flexecutor.scheduling.utils import get_size_by_s3_prefix
from flexecutor.utils.dataclass import StageConfig


class Caerus(Scheduler):
    def __init__(self, dag, total_parallelism: int, cpu_per_worker: float):
        """
        When Caerus is initialized, it requires that the user specifies the total parallelism
        and the number of CPUs per function.
        @param dag: the DAG associated with the scheduler
        @param total_parallelism: sum of worker over all stages in the DAG
        @param cpu_per_worker: number of CPUs per function
        """
        super().__init__(dag, PerfModelEnum.ANALYTIC)
        self.total_parallelism = total_parallelism
        self.cpu_per_worker = cpu_per_worker

    def schedule(self):
        """
        The schedule function in Caerus uses the total parallelism of the DAG to calculate
        the number of workers in each state.
        The total parallelism is distributed into the states based on the input size of
        the data in each state.
        """

        # Calculate the input size of each stage.
        total_size = 0
        size_dict = {}
        for stage in self._dag.stages:
            size_dict[stage.stage_id] = 0
            for flex_input in stage.inputs:
                item_size = get_size_by_s3_prefix(flex_input.bucket, flex_input.prefix)
                size_dict[stage.stage_id] += item_size
                total_size += item_size

        # Calculate the normalized parallelism ratio for each stage.
        parallelism_ratios = {
            stage_id: size / total_size for stage_id, size in size_dict.items()
        }

        # Skipping round_num_funcs() & check_config() Jolteon functions
        pass

        # Setting the number of workers for each stage
        num_workers_dict = {}
        for stage in self._dag.stages:
            num_workers = round(
                self.total_parallelism * parallelism_ratios[stage.stage_id]
            )
            num_workers = 1 if num_workers == 0 else num_workers
            num_workers_dict[stage.stage_id] = num_workers

        resource_config_list = []
        for stage_id, num_workers in num_workers_dict.items():
            stage = self._dag.get_stage_by_id(stage_id)
            # FIXME: resolve that. Currently using fixed ratio of Lambda
            resource_config = StageConfig(
                workers=num_workers,
                cpu=self.cpu_per_worker,
                memory=self.cpu_per_worker * 1769,
            )
            resource_config_list.append(resource_config)
        return resource_config_list
