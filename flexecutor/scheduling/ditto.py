import math

from flexecutor.modelling.perfmodel import PerfModelEnum
from flexecutor.scheduling.scheduler import Scheduler
from flexecutor.utils.dataclass import StageConfig

# Below this magnitude a fitted energy slope is indistinguishable from zero.
ENERGY_SLOPE_TOL = 1e-9

class VirtualStage:
    def __init__(self, param_a, idx=None):
        self.param_a = param_a
        self.stage_id = idx
        self.parents = []
        self.children = []

        self.merge_type = None  # 0 is parent-child, 1 is two siblings
        self.merge_stages = None
        self.merge_ratio = None

    def add_parent(self, parent: "VirtualStage"):
        self.parents.append(parent)

    def add_child(self, child: "VirtualStage"):
        self.children.append(child)

    def check_parent(self, parent: "VirtualStage"):
        if len(self.parents) == 0:
            return False
        if parent in self.parents:
            return True
        for p in self.parents:
            if p.check_parent(parent):
                return True
        return False

    def check_child(self, child: "VirtualStage"):
        if len(self.children) == 0:
            return False
        if child in self.children:
            return True
        for c in self.children:
            if c.check_child(child):
                return True
        return False

    # Only support 'v_stage' is a child stage of self
    def max_distance(self, v_stage: "VirtualStage"):
        if v_stage == self:
            return 0

        ret = -1
        for c in self.children:
            val = c.max_distance(v_stage)
            if val >= 0:
                ret = max(ret, val + 1)
        return ret

    def reverse_deepest_stages(self):
        if len(self.parents) == 0:
            return [self]

        ret = []
        for c in self.parents:
            val = c.reverse_deepest_stages()
            ret.extend(val)

        return ret

    def __str__(self):
        return "stage index: " + str(self.stage_id)


def merge_stages(stage0: VirtualStage, stage1: VirtualStage, merge_type: str):
    if merge_type not in ["parent-child", "siblings"]:
        raise ValueError("Invalid merge type")

    a0 = stage0.param_a
    a1 = stage1.param_a

    if merge_type == "parent-child":
        new_a = math.sqrt(a0) + math.sqrt(a1)
        new_a = new_a * new_a
        ratio = a0 / a1
        ratio = math.sqrt(ratio)
    elif merge_type == "siblings":
        new_a = a0 + a1
        ratio = a0 / a1
    else:
        raise ValueError("Invalid merge type")

    new_stage = VirtualStage(new_a)
    new_stage.merge_type = merge_type
    new_stage.merge_stages = [stage0, stage1]
    new_stage.merge_ratio = ratio
    return new_stage


def distance(stage0: VirtualStage, stage1: VirtualStage):
    if stage0 == stage1:
        return 0

    ret = -1
    for c in stage0.children:
        val = distance(c, stage1)
        if val >= 0:
            ret = min(ret, val + 1) if ret >= 0 else val + 1

    return ret


class Ditto(Scheduler):
    def __init__(
        self,
        dag,
        total_parallelism: int,
        cpu_per_worker: float,
        objective: str,
        flat_stage_boost: float = 10.0,
        ):
        super().__init__(dag, PerfModelEnum.ANALYTIC)
        self.total_parallelism = total_parallelism
        self.cpu_per_worker = cpu_per_worker
        self.parallelism_ratios = {}
        self.flat_stage_boost = flat_stage_boost

        # Init the virtual DAG
        self.virtual_stages = {}
        self.roots = []
        self.leafs = []

        if objective not in ["latency", "cost", "energy"]:
            raise ValueError(
                "Invalid objective. Should be 'latency', 'cost' or 'energy'"
            )
        self.objective = objective

    def _assign(self, v_stage: VirtualStage, degree: float):
        assert degree > 0

        if v_stage.stage_id is not None:
            self.parallelism_ratios[v_stage.stage_id] = degree
            return

        con_stages = v_stage.merge_stages
        stage0 = con_stages[0]
        stage1 = con_stages[1]

        assert isinstance(stage0, VirtualStage)
        assert isinstance(stage1, VirtualStage)

        ratio1 = 1 / (v_stage.merge_ratio + 1)
        ratio0 = 1 - ratio1

        degree0 = degree * ratio0
        degree1 = degree * ratio1

        self._assign(stage0, degree0)
        self._assign(stage1, degree1)

    def _schedule_for_cost(self):
        param_a_dict = {}
        # Assume each function under Ditto has the same memory size
        for stage in self._dag.stages:
            param_a, _ = stage.perf_model.parameters
            param_a = math.sqrt(abs(param_a))
            param_a_dict[stage.stage_id] = param_a
        sum_a = sum(param_a_dict.values())
        self.parallelism_ratios = {k: v / sum_a for k, v in param_a_dict.items()}

    def _schedule_for_energy(self):
        """
        Split the parallelism budget by each stage's energy slope.

        The slope is dE/dx from AnaPerfModel.energy_marginal, not the raw
        coefficient a -- a is a slope only for the affine shape; for
        E = a/x + b it is -a/x^2. Each stage's share is 1/m, so the cheaper a
        stage is to widen, the more it gets. |m| within tol counts as flat and
        gets flat_stage_boost; m below -tol is reported and also treated as
        flat.

        Raises if any stage lacks an energy model, rather than falling back to
        a time proxy and calling the result energy scheduling.
        """
        # A pinned stage runs at one worker by construction, so it has no share
        # of the parallelism budget to receive and a missing energy model costs
        # nothing. Only a stage that could have been widened but has no model is
        # a genuine gap.
        schedulable = [s for s in self._dag.stages
                       if not getattr(s, "pinned", False)]

        if not schedulable:
            raise ValueError(
                "Cannot schedule for energy: every stage is pinned, so there is "
                "no parallelism allocation to make."
            )

        unfitted = [
            s.stage_id for s in schedulable
            if not getattr(s.perf_model, "has_energy_model", False)
        ]
        if unfitted:
            raise ValueError(
                "Cannot schedule for energy: no fitted energy model for stage(s) "
                f"{unfitted}. Profile these stages with an energy monitor active "
                "(at least two configurations) before using objective='energy'."
            )

        # Weight by dE/dx, not the raw coefficient a: for the shared-meter
        # shape E = a/x + b the slope is -a/x^2, so a positive a means energy
        # falls. Three distinct cases:
        #   m > tol     stage pays for parallelism -> weight 1/m
        #   |m| <= tol  flat -> free to widen
        #   m < -tol    energy falls as resources grow; possible, but usually a
        #               profile too narrow to constrain the fit
        weights = {}
        flat_stages = []
        for stage in schedulable:
            m = stage.perf_model.energy_marginal()
            if m is None:
                raise ValueError(
                    f"Cannot schedule for energy: stage '{stage.stage_id}' has a "
                    "fitted energy model but no profiled configurations to "
                    "evaluate its slope at."
                )
            if m > ENERGY_SLOPE_TOL:
                weights[stage.stage_id] = 1.0 / m
            else:
                if m < -ENERGY_SLOPE_TOL:
                    print(
                        f"[Ditto] Stage '{stage.stage_id}' fitted a negative energy "
                        f"slope (dE/dx={m:.6g}). Treating it as flat; widen the "
                        "profiled configuration space to confirm."
                    )
                flat_stages.append(stage.stage_id)

        # Pass 2: a stage whose energy is flat in parallelism is worth widening,
        # but the size of that preference has to be expressed relative to the
        # stages it competes with. An absolute constant cannot be: any value
        # large enough to mean "prefer this" for one DAG is large enough to take
        # the entire budget in another, and a flat fit is the normal outcome of
        # profiling over too narrow a range, so this is the common case rather
        # than the exceptional one.
        if flat_stages:
            boost = (
                max(weights.values()) * self.flat_stage_boost if weights else 1.0
            )
            for stage_id in flat_stages:
                weights[stage_id] = boost

        total = sum(weights.values())
        self.parallelism_ratios = {k: v / total for k, v in weights.items()}

    def _schedule_for_latency(self):
        if len(self.leafs) != 1:
            raise ValueError("Do not support the current virtual DAG in Ditto")
        sink_stage = self.leafs[0]
        final_stage = None
        stop = True
        while stop:
            idx = None
            dis_val = -1
            leaves = sink_stage.reverse_deepest_stages()
            for v_stage in leaves:
                dis = distance(v_stage, sink_stage)
                if dis > dis_val:
                    dis_val = dis
                    idx = leaves.index(v_stage)
            assert dis_val >= 0
            cur_stage = leaves[idx]

            if cur_stage == sink_stage:
                stop = False
                final_stage = cur_stage
                break

            assert len(cur_stage.children) <= 1
            if len(cur_stage.children) == 0:
                final_stage = cur_stage
                break

            children_stage = cur_stage.children[0]
            sibling_stages = children_stage.parents
            merge_stage = None

            # Merge sibling stages
            for idx, sibling_stage in enumerate(sibling_stages):
                if idx == 0:
                    merge_stage = sibling_stage
                    continue

                merge_stage = merge_stages(merge_stage, sibling_stage, "siblings")

            # Merge parent-child stage
            merge_stage = merge_stages(merge_stage, children_stage, "parent-child")

            if children_stage == sink_stage:
                sink_stage = merge_stage

            for c in children_stage.children:
                c.parents.remove(children_stage)
                c.parents.append(merge_stage)
                merge_stage.children.append(c)

        self._assign(final_stage, 1)

    def schedule(self):
        for stage in self._dag.stages:
            param_a, _ = stage.perf_model.parameters
            # Fit for not allow_parallel stages, whose "a" is negative
            param_a = abs(param_a)
            new_stage = VirtualStage(param_a, stage.stage_id)
            self.virtual_stages[stage.stage_id] = new_stage
            if stage in self._dag.root_stages:
                self.roots.append(new_stage)
            if stage in self._dag.leaf_stages:
                self.leafs.append(new_stage)

        # Build the virtual DAG
        for stage in self._dag.stages:
            idx = stage.stage_id
            for p in stage.parents:
                self.virtual_stages[idx].add_parent(self.virtual_stages[p.stage_id])

            for c in stage.children:
                self.virtual_stages[idx].add_child(self.virtual_stages[c.stage_id])

        # Tune the virtual DAG, delete the redundant edges
        for stage_id, stage in self.virtual_stages.items():
            maintain_list = []
            for idx, c in enumerate(stage.children):
                if stage.max_distance(c) == 1:
                    maintain_list.append(idx)

            new_children = [stage.children[idx] for idx in maintain_list]
            for idx, c in enumerate(stage.children):
                if idx in maintain_list:
                    continue
                c.parents.remove(stage)
            stage.children = new_children

        # Scheduling using Ditto algorithm
        if self.objective == "latency":
            self._schedule_for_latency()
        elif self.objective == "cost":
            self._schedule_for_cost()
        elif self.objective == "energy":
            self._schedule_for_energy()
        else:
            raise ValueError("Invalid objective")

        # Calculate the normalized parallelism ratio for each stage
        sum_workers = sum(self.parallelism_ratios.values())
        self.parallelism_ratios = {
            k: v / sum_workers for k, v in self.parallelism_ratios.items()
        }

        # Skipping round_num_funcs() & check_config() Jolteon functions
        pass

        num_workers_dict = {}
        for stage in self._dag.stages:
            ratio = self.parallelism_ratios.get(stage.stage_id)
            if ratio is None:
                # No share was allocated to this stage. Two situations
                # produce this, and both are handled the same way. A pinned
                # stage runs at one worker by construction and is
                # deliberately excluded from the split by
                # _schedule_for_energy. A stage missed by the virtual-DAG
                # walk in _schedule_for_latency would also land here, in
                # which case falling back is safer than raising a KeyError
                # from inside the scheduler.
                num_workers = 1
            else:
                num_workers = round(self.total_parallelism * ratio)
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
