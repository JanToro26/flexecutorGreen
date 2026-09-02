import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, StageConfig
from flexecutor.modelling.energy_agg import stage_energy, _is_shared_meter

logger = logging.getLogger(__name__)

TIME_PHASES = ("cold_start", "read", "compute", "write")
FITTED = TIME_PHASES + ("energy",)
N_PARAMS = 3


def model_func(workers, size, a, b, c):
    """f(W, s) = a/(W*s) + b*W + c, fitted for every quantity.

    W is the worker count, s the per-worker resource size. The a term is work
    divided among the workers, b the fixed cost each worker adds, c an offset.
    b is what a scalar x = cpu*memory*workers cannot express: it makes
    (512MB, W=4) and (2048MB, W=1) the same point when they measurably differ.

    At fixed s this is (a/s)/W + b*W + c, so profiles swept over W alone fit
    exactly as before. Convex for a, b >= 0, minimum at W* = sqrt(a/(b*s)).
    """
    return a / (workers * size) + b * workers + c


def _has_numeric(series) -> bool:
    for run in series or []:
        for value in run if isinstance(run, (list, tuple)) else [run]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
    return False


def _numeric_values(series) -> List[float]:
    """Flatten a stored profile series, dropping None and non-numeric entries."""
    out = []
    for run in series or []:
        for value in run if isinstance(run, (list, tuple)) else [run]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
    return out


class AnaPerfModel(PerfModel):
    """Analytic energy and latency model over (workers, per-worker size).

    One curve, `model_func`, per quantity per stage. Configurations are keyed on
    the pair (s, W), never on their product, so allocations differing in worker
    count stay distinct points.

    s = cpu * memory folds the two per-worker quantities into one size. Lambda
    ties vCPU to memory and exposes only memory; k8s enforces cpu while memory
    limits do not change compute speed. So exactly one of the two varies on
    either platform. predict_time refuses a size the profile never covered.
    """

    def __init__(self, stage) -> None:
        super().__init__("analytic", stage)

        self._params: Dict[str, Optional[List[float]]] = {k: None for k in FITTED}
        self._has_energy = False
        # One shared whole-machine meter (k8s RAPL) vs independent per-worker
        # meters (Lambda). Decides max-vs-sum aggregation, not the curve shape.
        self._energy_shared_meter = False
        self._profiling_results = None
        self._keys: Tuple[Tuple[float, int], ...] = ()

    # ------------------------------------------------------------------
    # configuration key
    # ------------------------------------------------------------------

    @classmethod
    def _config_to_key(cls, num_vcpu, memory, num_func) -> Tuple[float, int]:
        """(per-worker size, worker count)."""
        return (round(float(num_vcpu) * float(memory), 1), int(num_func))

    def _key(self, cpu, memory, workers) -> Tuple[float, int]:
        vcpu = cpu if self.allow_parallel else 1
        return self._config_to_key(vcpu, memory, workers)

    @property
    def profiled_sizes(self) -> Tuple[float, ...]:
        return tuple(sorted({s for s, _ in self._keys}))

    @property
    def profiled_workers(self) -> Tuple[int, ...]:
        return tuple(sorted({w for _, w in self._keys}))

    def _check_size(self, size: float) -> None:
        """Refuse a size the profile cannot speak to; warn when extrapolating."""
        sizes = self.profiled_sizes
        if not sizes:
            return
        if len(sizes) == 1:
            if abs(size - sizes[0]) > 1e-6:
                raise ValueError(
                    f"[{self._stage_name}] profiled at one per-worker size "
                    f"(s={sizes[0]:g}), asked to predict s={size:g}. The size axis "
                    f"was never varied, so no coefficient describes it. Profile "
                    f"more than one cpu or memory value first."
                )
            return
        if size < sizes[0] or size > sizes[-1]:
            logger.warning(
                "[%s] s=%g outside the profiled range [%g, %g]; extrapolating "
                "along the size axis.",
                self._stage_name, size, sizes[0], sizes[-1],
            )

    # ------------------------------------------------------------------
    # persistence (unused; the profile is the model)
    # ------------------------------------------------------------------

    @overrides
    def save_model(self):
        pass

    @overrides
    def load_model(self):
        pass

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _fit(points: Dict[Tuple[float, int], float]) -> Optional[List[float]]:
        """Fit model_func to {(s, W): mean}. None if there are too few points.

        a, b >= 0 is imposed. Unbounded, the fit returns a negative work or
        per-worker term and then predicts negative joules outside the sweep.
        """
        if len(points) < N_PARAMS:
            return None

        keys = list(points)
        size = np.array([k[0] for k in keys], dtype=float)
        workers = np.array([k[1] for k in keys], dtype=float)
        y = np.array([points[k] for k in keys], dtype=float)

        guess = [float(max(y.max(), 0.0) * (workers * size).min()), 0.0, float(y.min())]
        result = scipy_opt.least_squares(
            lambda p: model_func(workers, size, *p) - y,
            guess,
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            max_nfev=20000,
        )
        return result.x.tolist()

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        if len(stage_profile_data) < N_PARAMS:
            raise ValueError(
                f"At least {N_PARAMS} profiled configurations are required to fit "
                f"the model; got {len(stage_profile_data)}."
            )
        self._profiling_results = stage_profile_data

        for config_data in stage_profile_data.values():
            assert all(
                key in config_data for key in TIME_PHASES
            ), f"Each configuration's data must contain {list(TIME_PHASES)} keys."

        self._energy_shared_meter = False
        # Energy is fitted only when every configuration carries it. Older
        # profiles without it stay valid for latency and cost.
        self._has_energy = all(
            "energy" in cd and _has_numeric(cd["energy"])
            for cd in stage_profile_data.values()
        )
        if not self._has_energy:
            logger.info(
                "[%s] no usable energy series; the energy objective is "
                "unavailable for this stage.", self._stage_name,
            )

        points: Dict[str, Dict[Tuple[float, int], List[float]]] = {
            name: {} for name in FITTED
        }

        for config_tuple, data in stage_profile_data.items():
            num_vcpu, memory, num_func = config_tuple
            key = self._key(num_vcpu, memory, num_func)

            for phase in TIME_PHASES:
                points[phase].setdefault(key, []).extend(_numeric_values(data[phase]))

            if not self._has_energy:
                continue
            source_runs = data.get("energy_source") or []
            for i, repetition in enumerate(data["energy"]):
                values = _numeric_values([repetition])
                if not values:
                    continue
                srcs = source_runs[i] if i < len(source_runs) else []
                if not isinstance(srcs, (list, tuple)):
                    srcs = [srcs]
                if _is_shared_meter(srcs):
                    self._energy_shared_meter = True
                points["energy"].setdefault(key, []).append(stage_energy(values, srcs))

        means = {
            name: {k: float(np.mean(v)) for k, v in series.items() if v}
            for name, series in points.items()
        }
        self._keys = tuple(sorted(means["compute"]))

        for name in TIME_PHASES:
            self._params[name] = self._fit(means[name])

        self._params["energy"] = self._fit(means["energy"]) if self._has_energy else None
        # Fewer configurations than free parameters: leave it unfitted rather
        # than pass a curve exactly through every point.
        self._has_energy = self._params["energy"] is not None

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        assert config.workers > 0

        size, workers = self._key(config.cpu, config.memory, config.workers)
        self._check_size(size)

        def value(name: str) -> Optional[float]:
            params = self._params.get(name)
            return float(model_func(workers, size, *params)) if params else None

        read = value("read") or 0.0
        compute = value("compute") or 0.0
        write = value("write") or 0.0
        cold = value("cold_start") or 0.0
        # Energy stays None when unfitted. No proxy fallback: a made-up number
        # is indistinguishable from a real one downstream.
        energy = value("energy")

        return FunctionTimes(
            total=read + compute + write + cold,
            read=read,
            compute=compute,
            write=write,
            cold_start=cold,
            energy=energy,
            energy_source="fitted_from_profile" if energy is not None else "none",
        )

    # ------------------------------------------------------------------
    # coefficients and derived quantities
    # ------------------------------------------------------------------

    @property
    @overrides
    def parameters(self):
        """(a, c) summed over the latency phases. Two values, Ditto unpacks two.

        The bounds keep a non-negative, so the abs() the scheduler applies to
        work around negative fits is no longer load-bearing. For b, use
        time_marginal or energy_marginal.
        """
        fitted = [self._params[name] for name in TIME_PHASES if self._params[name]]
        return (sum(p[0] for p in fitted), sum(p[2] for p in fitted))

    @property
    def energy_parameters(self) -> Optional[Tuple[float, float, float]]:
        """(a, b, c) of the energy curve, or None if unfitted."""
        params = self._params.get("energy")
        return tuple(params) if params else None

    @property
    def per_worker_energy_cost(self) -> Optional[float]:
        """b: joules each extra worker costs regardless of its size."""
        params = self._params.get("energy")
        return float(params[1]) if params else None

    def _median_key(self) -> Optional[Tuple[float, int]]:
        return self._keys[len(self._keys) // 2] if self._keys else None

    def _resolve(self, workers, size):
        """Default missing arguments to the median profiled configuration."""
        median = self._median_key()
        if median is None:
            return None, None
        return (median[1] if workers is None else workers,
                median[0] if size is None else size)

    def energy_marginal(self, workers=None, size=None) -> Optional[float]:
        """dE/dW = -a/(W^2 * s) + b. Negative below the optimum, positive above."""
        params = self._params.get("energy")
        if not params:
            return None
        workers, size = self._resolve(workers, size)
        if workers is None:
            return None
        a, b, _ = params
        return -a / (float(workers) ** 2 * float(size)) + b

    def time_marginal(self, workers=None, size=None) -> Optional[float]:
        """dt/dW summed over the latency phases."""
        workers, size = self._resolve(workers, size)
        if workers is None:
            return None
        total = 0.0
        for name in TIME_PHASES:
            params = self._params[name]
            if params:
                total += -params[0] / (float(workers) ** 2 * float(size)) + params[1]
        return total

    def energy_optimal_workers(self, size=None) -> Optional[float]:
        """W* = sqrt(a/(b*s)), or None.

        None when a coefficient is zero, so the curve is monotone, and None
        outside the profiled worker range, where it is an extrapolation.
        """
        params = self._params.get("energy")
        if not params:
            return None
        if size is None:
            median = self._median_key()
            if median is None:
                return None
            size = median[0]
        a, b, _ = params
        if a <= 0 or b <= 0:
            return None
        optimum = float(np.sqrt(a / (b * float(size))))
        workers = self.profiled_workers
        if not workers or not (workers[0] <= optimum <= workers[-1]):
            return None
        return optimum

    @property
    def has_energy_model(self) -> bool:
        return bool(self._params.get("energy"))

    @property
    def is_shared_meter(self) -> bool:
        return self._energy_shared_meter
