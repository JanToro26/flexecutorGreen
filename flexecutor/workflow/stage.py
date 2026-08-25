from __future__ import annotations

from enum import Enum
from typing import Any, Set, List, Optional, Callable

from flexecutor.modelling.perfmodel import PerfModel, PerfModelEnum
from flexecutor.storage.storage import FlexData
from flexecutor.utils.dataclass import StageConfig
from flexecutor.workflow.stagefuture import StageFuture


class StageState(Enum):
    """
    State of a stage
    """

    NONE = 0
    SCHEDULED = 1
    WAITING = 2
    RUNNING = 3
    SUCCESS = 4
    FAILED = 5


class Stage:
    """
    :param stage_id: Stage ID
    :param inputs: List of InputS3Path instances for the operator
    """
    def __init__(
        self,
        stage_id: str,
        func: Callable[..., Any],
        inputs: List[FlexData],
        outputs: List[FlexData],
        params: Optional[dict[str, Any]] = None,
        max_concurrency: int = 1024,
    ):
        if params is None:
            params = {}
        self._stage_unique_id = None
        self._stage_id = stage_id
        self._stage_idx = None
        self._inputs = inputs
        self._outputs = outputs
        self._params = params
        self._children: Set[Stage] = set()
        self._parents: Set[Stage] = set()
        self._state = StageState.NONE
        self._map_func = func
        self._max_concurrency = max_concurrency
        self.dag_id = None
        self._perf_model_type: Optional[PerfModelEnum] = None
        self._perf_model: Optional[PerfModel] = None
        self._resource_config: Optional[StageConfig] = StageConfig(
            cpu=1, memory=2048, workers=1
        )
        # True when this stage is fixed to one worker by construction, so there
        # is no parallelism decision for a scheduler to make. Distinct from a
        # stage whose model simply failed to fit.
        self.pinned = False

    def __repr__(self) -> str:
        return f"Stage({self._stage_id}, resource_config={self.resource_config}) "

    @property
    def dag_id(self) -> str:
        """Return the DAG ID."""
        return self._dag_id

    @property
    def resource_config(self):
        return self._resource_config

    @property
    def stage_unique_id(self) -> str:
        return self._stage_unique_id

    @resource_config.setter
    def resource_config(self, value: StageConfig):
        self._resource_config = value

    @property
    def map_func(self) -> Callable[..., Any]:
        """Return the map function."""
        return self._map_func

    @dag_id.setter
    def dag_id(self, value: str):
        self._dag_id = value
        self._stage_unique_id = f"{self._dag_id}-{self._stage_id}"

    @property
    def stage_idx(self) -> int:
        return self._stage_idx

    @stage_idx.setter
    def stage_idx(self, value: int):
        self._stage_idx = value

    @property
    def perf_model(self) -> PerfModel:
        return self._perf_model

    def init_perf_model(self, perf_model_type: PerfModelEnum) -> PerfModel:
        self._perf_model_type = perf_model_type
        self._perf_model = PerfModel.instance(perf_model_type, self)
        return self._perf_model

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def stage_id(self) -> str:
        """Return the stage ID."""
        return self._stage_id

    @property
    def parents(self) -> Set[Stage]:
        """Return the parents of this operator."""
        return self._parents

    @property
    def children(self) -> Set[Stage]:
        """Return the children of this operator."""
        return self._children

    @property
    def inputs(self) -> List[FlexData]:
        """Return the list of input paths."""
        return self._inputs

    @property
    def outputs(self) -> List[FlexData]:
        """Return the list of output paths."""
        return self._outputs

    @property
    def params(self) -> dict[str, Any]:
        """Return the parameters of the stage."""
        return self._params

    @property
    def state(self) -> StageState:
        """Return the state of the stage."""
        return self._state

    @state.setter
    def state(self, value):
        """Set the state of the stage."""
        self._state = value

    def _set_relation(
        self, operator_or_operators: Stage | List[Stage], upstream: bool = False
    ):
        """
        Set relation between this operator and another operator or list of operator

        :param operator_or_operators: Operator or list of operator
        :param upstream: Whether to set the relation as upstream or downstream
        """
        if isinstance(operator_or_operators, Stage):
            operator_or_operators = [operator_or_operators]

        for operator in operator_or_operators:
            if upstream:
                self.parents.add(operator)
                operator.children.add(self)
            else:
                self.children.add(operator)
                operator.parents.add(self)

    def add_parent(self, operator: Stage | List[Stage]):
        """
        Add a parent to this operator.
        :param operator: Operator or list of operator
        """
        self._set_relation(operator, upstream=True)

    def add_child(self, operator: Stage | List[Stage]):
        """
        Add a child to this operator.
        :param operator: Operator or list of operator
        """
        self._set_relation(operator, upstream=False)

    def __lshift__(self, other: Stage | List[Stage]) -> Stage | List[Stage]:
        """Overload the << operator to add a parent to this operator."""
        self.add_parent(other)
        return other

    def __rshift__(self, other: Stage | List[Stage]) -> Stage | List[Stage]:
        """Overload the >> operator to add a child to this operator."""
        self.add_child(other)
        return other

    def __rrshift__(self, other: Stage | List[Stage]) -> Stage:
        """Overload the >> operator for lists of operator."""
        self.add_parent(other)
        return self

    def __rlshift__(self, other: Stage | List[Stage]) -> Stage:
        """Overload the << operator for lists of operator."""
        self.add_child(other)
        return self

    def execute(self) -> StageFuture:
        """
        Execute the stage: it declares DAGExecutor and then execute the stage
        """
        from flexecutor.workflow.executor import DAGExecutor
        from flexecutor.workflow.dag import DAG

        dag = DAG("single-stage-dag")
        dag.add_stage(self)
        executor = DAGExecutor(dag=dag)
        result = executor.execute()
        executor.shutdown()
        return list(result.values())[0]
