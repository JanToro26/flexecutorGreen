import heapq
from typing import List

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.scheduling.scheduler import Scheduler
from flexecutor.scheduling.utils import get_size_by_s3_prefix
from flexecutor.utils.dataclass import StageConfig


class MyQueue:
    def __init__(self, init_list=None):
        self.queue = [] if init_list is None else init_list

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        assert len(self.queue) > 0
        return self.queue.pop(0)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)


class MyPriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (-priority, item))

    def pop(self):
        if self.heap:
            priority, item = heapq.heappop(self.heap)
            return item, -priority
        raise IndexError("pop from an empty priority queue")

    def peek(self):
        if self.heap:
            priority, item = self.heap[0]
            return -priority, item
        raise IndexError("peek from an empty priority queue")

    def is_empty(self):
        return len(self.heap) == 0

    def __len__(self):
        return len(self.heap)


class Orion(Scheduler):
    def __init__(
        self, dag, total_parallelism: int, latency: int, memory_step: int = 256
    ):
        super().__init__(dag, PerfModelEnum.DISTRIBUTION)
        self.total_parallelism = total_parallelism
        self.latency = latency
        self.memory_step = memory_step
        # FIXME: Note that AWS Lambda effects are present here
        self.max_memory = 1769 * 6
        self.stage_workers = {k.stage_id: 1 for k in self._dag.stages}

    def schedule(self):
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

        # Setting the right size for each stage
        # FIXME: parametrize fixed vars
        mode = "simple"  # | "priority"

        # FIXME: convert workers_size_list to a dict "stage_id: memory"
        if mode == "simple":
            workers_size_list = self._bfs_simple_queue()
        elif mode == "priority":
            workers_size_list = self._bfs_priority_queue()
        else:
            raise ValueError("Invalid mode")

        resource_config_list = []
        for index, (stage_id, num_workers) in enumerate(num_workers_dict.items()):
            resource_config = StageConfig(
                workers=num_workers,
                cpu=workers_size_list[index] / 1769,
                memory=workers_size_list[index],
            )
            resource_config_list.append(resource_config)
        return resource_config_list

    def _bfs_simple_queue(self):
        confidence = 0.99

        config_list = [self.memory_step for _ in self._dag.stages]

        search_space = MyQueue()
        search_space.push(config_list)

        searched = set()
        searched.add(tuple(config_list))

        dist = self._get_distribution(config_list)
        if dist.probability(self.latency) >= confidence:
            return config_list

        while len(search_space) > 0:
            val = search_space.pop()

            # If the number of functions is less than 6, we can search exhaustively
            if len(self._dag.stages) < 6:
                for i in range(len(val)):
                    new_val = val.copy()
                    new_val[i] += self.memory_step

                    t = tuple(new_val)
                    if t in searched:
                        continue
                    # Max limit
                    if new_val[i] > self.max_memory:
                        continue
                    search_space.push(new_val)
                    searched.add(t)

                    dist = self._get_distribution(new_val)
                    if dist.probability(self.latency) >= confidence:
                        return new_val
            else:  # Fast search for large workflow
                new_val = val.copy()
                for i in range(len(val)):
                    new_val[i] += self.memory_step
                if new_val[0] > self.max_memory:
                    continue
                t = tuple(new_val)
                if t in searched:
                    continue
                search_space.push(new_val)
                searched.add(t)

                dist = self._get_distribution(new_val)
                if dist.probability(self.latency) >= confidence:
                    return new_val

        config_list = [self.max_memory for _ in self._dag.stages]
        return config_list

    def _bfs_priority_queue(self):
        confidence = 0.99

        config_list = [self.memory_step for _ in self._dag.stages]

        search_space = MyPriorityQueue()
        search_space.push(config_list, 0)

        searched = set()
        searched.add(tuple(config_list))

        dist = self._get_distribution(config_list)
        if dist.probability(self.latency) >= confidence:
            return config_list

        while len(search_space) > 0:
            val, _ = search_space.pop()

            for i in range(len(val)):
                new_val = val.copy()
                new_val[i] += self.memory_step

                t = tuple(new_val)
                if t in searched:
                    continue
                # Max limit
                if new_val[i] > self.max_memory:
                    continue
                search_space.push(
                    new_val,
                    -1
                    * self._get_cost(new_val, confidence)
                    * self._get_latency(new_val, confidence),
                )
                searched.add(t)

                dist = self._get_distribution(new_val)
                if dist.probability(self.latency) >= confidence:
                    return new_val

    def _get_distribution(self, config_list: List[float]):
        stage_sizes = {}

        # FIXME: check consistency between stage_workers dict and config list
        for (stage_id, workers), memory in zip(self.stage_workers.items(), config_list):
            stage_sizes[stage_id] = round(memory * workers / 1769, 1)

        dist = None
        for stage in self._dag.leaf_stages:
            # noinspection PyUnresolvedReferences
            tmp_dist = stage.perf_model.calculate(stage_sizes[stage.stage_id])
            if dist is None:
                dist = tmp_dist
            else:
                dist.combine(tmp_dist, "parallel")

        return dist

    def _get_latency(self, config_list, confidence):
        dist = self._get_distribution(config_list)
        return dist.tail_value(confidence)

    def _get_cost(self, config_list, confidence):
        stage_sizes = {}

        # FIXME: check consistency between stage_workers dict and config list
        for (stage_id, workers), memory in zip(self.stage_workers.items(), config_list):
            stage_sizes[stage_id] = round(memory * workers / 1769, 1)

        cost = 0
        for idx, stage in enumerate(self._dag.stages):
            # noinspection PyUnresolvedReferences
            tmp_dist = stage.perf_model._interpolate(self.stage_workers[stage.stage_id])
            cost += tmp_dist.tail_value(confidence) * stage_sizes[stage.stage_id]
        return cost
