from typing import Dict

import numpy as np

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import StageConfig, FunctionTimes
from flexecutor.workflow.stage import Stage


class Distribution:
    def __init__(self, init_list, init_prob=None):

        init_list.sort()
        self.data = np.array(init_list)
        if init_prob is None:
            self.prob = np.ones(len(init_list)) / len(init_list)
        else:
            assert len(init_list) == len(init_prob)
            self.prob = np.array(init_prob)

        self.subset = []

    def combine(self, dist, com_type="in-series"):
        assert isinstance(dist, Distribution)
        assert com_type in ["in-series", "parallel"]

        if com_type == "in-series":
            new_data = []
            for i in range(len(self.data)):
                for j in range(len(dist.data)):
                    new_data.append(
                        (self.data[i] + dist.data[j], self.prob[i] * dist.prob[j])
                    )

            new_data.sort()
            self.data = np.array([item[0] for item in new_data])
            self.prob = np.array([item[1] for item in new_data])
        elif com_type == "parallel":
            cum_prob1 = np.cumsum(self.prob)
            cum_prob2 = np.cumsum(dist.prob)

            new_data = np.concatenate((self.data, dist.data))
            new_data.sort()
            new_probs = []
            prev = 0
            for item in new_data:
                index = np.sum(self.data <= item) - 1
                prob1 = cum_prob1[index] if index >= 0 else 0
                index = np.sum(dist.data <= item) - 1
                prob2 = cum_prob2[index] if index >= 0 else 0
                new_probs.append(prob1 * prob2 - prev)
                prev = prob1 * prob2

            new_data = np.array(new_data)
            new_prob = np.array(new_probs)
            new_prob = new_prob / np.sum(new_prob)

            self.data = np.array([])
            self.prob = np.array([])

            for i in range(len(new_data)):
                if new_prob[i] > 0:
                    self.data = np.append(self.data, new_data[i])
                    self.prob = np.append(self.prob, new_prob[i])
        else:
            raise ValueError("Unknown combine type")

        self._reduce_dim()

    def probability(self, value):
        val = np.sum(np.array(self.data) <= value)
        if val == 0:
            return 0
        if val == len(self.data):
            return 1
        lower_idx = val - 1
        upper_idx = val
        base_prob = np.sum(self.prob[: lower_idx + 1])
        lower_val = self.data[lower_idx]
        upper_val = self.data[upper_idx]

        add_prob = (value - lower_val) / (upper_val - lower_val) * self.prob[upper_idx]

        return base_prob + add_prob

    # $The tail_value function is only used for calculate the cost in Orion scheduler
    def tail_value(self, percentile):
        cum_prob = np.cumsum(self.prob)
        index = None
        for i in range(len(self.prob)):
            if cum_prob[i] >= percentile:
                index = i
        if index is None:
            return self.data[-1]
        if index == 0:
            lower_val = 0
            lower_prob = 0
        else:
            lower_val = self.data[index - 1]
            lower_prob = cum_prob[index - 1]

        upper_val = self.data[index]
        upper_prob = cum_prob[index]

        add_val = (
            (percentile - lower_prob)
            / (upper_prob - lower_prob)
            * (upper_val - lower_val)
        )

        return lower_val + add_val

    # $The reduce_dim function is only used for combine function
    def _reduce_dim(self, dim=100):
        length = len(self.data)
        if length <= dim:
            return
        split_data = np.array_split(self.data, dim)
        split_prob = np.array_split(self.prob, dim)

        new_data = []
        new_prob = []

        for idx in range(dim):
            probs = split_prob[idx]
            prob = float(np.sum(split_prob[idx]))
            probs = probs / prob
            val = float(np.dot(split_data[idx], probs))
            new_prob.append(prob)
            new_data.append(val)

        self.data = np.array(new_data)
        self.prob = np.array(new_prob)

    def copy(self):
        new_data = self.data.copy().tolist()
        new_prob = self.prob.copy().tolist()
        return Distribution(new_data, new_prob)

    def __str__(self):
        return str(list(zip(self.data, self.prob)))


class DistPerfModel(PerfModel):
    def __init__(self, stage: Stage):
        super().__init__("distribution", stage)

        self.distributions = {}  # k*d -> Distribution
        self.up_models = []
        self.max_stage_size = None

    def init_from_dag(self, dag):
        """f
        In DistPerfModel, the initialization consists in load the parent models
        @param dag:
        """
        my_stage = dag.get_stage_by_id(self._stage_id)
        parents = my_stage.parents
        for parent in parents:
            model = parent.perf_model
            assert isinstance(model, DistPerfModel)
            self.up_models.append(model)

    def train(self, stage_profile_data: Dict) -> None:
        """
        This train function in DistPerfModel loads the profile data Distributions into memory
        Distributions only contain total stage duration values
        @param stage_profile_data:
        """
        # FIXME: for compatibility with Jolteon
        accessor = slice(1, None), 1

        for config_tuple, data in stage_profile_data.items():
            _, memory, workers = config_tuple
            config_latencies = (
                np.array(data["read"])
                + np.array(data["compute"])
                + np.array(data["write"])
                + np.array(data["cold_start"])
            )
            config_latencies = config_latencies[accessor]
            # In Jolteon impl, cpu key is calculated via memory*workers/1769
            # TODO: consider that in the future this will change
            stage_size = round(memory * workers / 1769, 1)
            self.distributions[stage_size] = Distribution(config_latencies)
            self.max_stage_size = max(self.distributions.keys())

    def calculate(self, stage_size) -> Distribution:
        cur_dist = self._interpolate(stage_size)

        if len(self.up_models) == 0:
            return cur_dist

        sub_dist = None

        for sub_model in self.up_models:
            dist = sub_model.calculate(stage_size)
            if sub_dist is None:
                sub_dist = dist
            else:
                sub_dist.combine(dist, com_type="parallel")
        cur_dist.combine(sub_dist, com_type="in-series")

        return cur_dist

    def _interpolate(self, stage_size) -> Distribution:
        assert stage_size >= 0

        if stage_size in self.distributions:
            return self.distributions[stage_size].copy()

        if stage_size >= self.max_stage_size:
            return self.distributions[self.max_stage_size].copy()

        stage_size_list = sorted(self.distributions.keys())
        index = None
        for idx, val in enumerate(stage_size_list):
            if val > stage_size:
                index = idx
                break
        lower_idx = index - 1
        upper_idx = index

        lower_dist = (
            self.distributions[stage_size_list[lower_idx]]
            if lower_idx >= 0
            else Distribution([0])
        )
        upper_dist = self.distributions[stage_size_list[upper_idx]]

        lower_size = stage_size_list[lower_idx] if lower_idx >= 0 else 0
        upper_size = stage_size_list[upper_idx]

        assert lower_size < stage_size < upper_size

        # y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        slope = (stage_size - lower_size) / (upper_size - lower_size)
        new_data = []
        new_prob = []

        for i in range(len(lower_dist.data)):
            for j in range(len(upper_dist.data)):
                new_data.append(
                    lower_dist.data[i]
                    + (upper_dist.data[j] - lower_dist.data[i]) * slope
                )
                new_prob.append(lower_dist.prob[i] * upper_dist.prob[j])

        return Distribution(new_data, new_prob)

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass

    def parameters(self):
        pass
