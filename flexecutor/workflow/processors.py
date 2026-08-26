import logging
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Callable, Sequence

from lithops import FunctionExecutor

from flexecutor.storage.wrapper import worker_wrapper
from flexecutor.utils import setup_logging
from flexecutor.workflow.stagecontext import InternalStageContext
from flexecutor.workflow.stage import Stage, StageState
from flexecutor.workflow.stagefuture import StageFuture

logger = setup_logging(level=logging.INFO)


class ThreadPoolProcessor:
    """
    Processor that uses a thread pool to execute stages
    """

    def __init__(self, log_level="DEBUG", max_threadpool_concurrency=256):
        super().__init__()
        self._executor = None
        self.log_level = log_level
        self._max_concurrency = max_threadpool_concurrency
        self._pool = ThreadPoolExecutor(max_workers=max_threadpool_concurrency)

    def process(
        self,
        stages: Sequence[Stage],
        on_future_done: Callable[[Stage, StageFuture], None] = None,
    ) -> dict[str, StageFuture]:
        """
        Process a list of stages
        :param stages: List of stages to process
        :param on_future_done: Callback to execute every time a future is done
        :return: Futures of the stages
        :raises ValueError: If there are no stages to process or if there are
        more stages than the maximum parallelism
        """
        if len(stages) == 0:
            raise ValueError("No stages to process")

        if len(stages) > self._max_concurrency:
            # TODO: don't fail. queue them
            raise ValueError(
                f"Too many stages to process. Max concurrency is {self._max_concurrency}"
            )

        ex_futures = {}

        for stage in stages:
            logger.info(f"Submitting stage {stage.stage_id}")

            stage.state = StageState.RUNNING
            ex_futures[stage.stage_id] = self._pool.submit(
                lambda s=stage: self._process_stage(s, on_future_done)
            )
        wait(ex_futures.values())

        return {
            stage_id: ex_future.result() for stage_id, ex_future in ex_futures.items()
        }

    @staticmethod
    def _retire_executor(executor) -> None:
        """Stop a finished executor's job monitor, then drop its temporary data.

        clean() only removes temporary data. The storage job monitor is a
        separate thread that keeps polling until stop() is called, so without
        this every stage invocation leaks one polling thread and a long sweep
        progressively loads the machine it is measuring.
        """
        if executor is None:
            return
        monitor = getattr(executor, "job_monitor", None)
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                logger.debug("Could not stop job monitor", exc_info=True)
        try:
            executor.clean()
        except Exception:
            logger.debug("Could not clean executor", exc_info=True)

    def shutdown(self):
        # The last executor is not retired by _process_stage, so do it here.
        self._retire_executor(self._executor)
        self._executor = None
        self._pool.shutdown()

    def _process_stage(
        self, stage: Stage, on_future_done: Callable[[Stage, StageFuture], None] = None
    ) -> StageFuture:
        """
        Process a stage

        :param stage: stage to process
        :param on_future_done: Callback to execute every time a future is done
        """

        map_iterdata = []
        num_workers = min(stage.resource_config.workers, stage.max_concurrency)

        # FIXME: big overhead; please only query object storage once

        for flex_data in stage.inputs:
            flex_data.scan_keys()
            flex_data.set_local_paths()
            if flex_data.chunker:
                flex_data.chunker.chunk(flex_data, num_workers)

        for worker_id in range(num_workers):
            copy_inputs = [deepcopy(item) for item in stage.inputs]
            copy_outputs = [deepcopy(item) for item in stage.outputs]
            for input_item in copy_inputs:
                input_item.set_file_indexes(worker_id, num_workers)
            ctx = InternalStageContext(
                worker_id, num_workers, copy_inputs, copy_outputs, stage.params
            )
            map_iterdata.append(ctx)

        self._retire_executor(self._executor)

        self._executor = FunctionExecutor(
            log_level=self.log_level, **{"runtime_cpu": stage.resource_config.cpu}
        )

        future = self._executor.map(
            map_function=worker_wrapper(stage.map_func),
            map_iterdata=map_iterdata,
            runtime_memory=int(stage.resource_config.memory),
        )

        self._executor.wait(future)
        future = StageFuture(stage.stage_id, future)

        # Update the state of the stage based on the future result
        stage.state = StageState.FAILED if future.error() else StageState.SUCCESS

        # Call the callback function if provided
        if on_future_done:
            on_future_done(stage, future)

        return future
