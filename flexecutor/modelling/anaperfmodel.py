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

#: Per-worker size at which one core is saturated; None disables the cap. A
#: single-threaded stage gains nothing above it. Set from run.py: 1024 on k8s
#: with memory fixed and cpu swept, ~1769 on Lambda (one vCPU).
SIZE_SATURATION: Optional[float] = None


def model_func(workers, size, a, b, c, q=1.0):
    """a/(W * u**q) + b*W + c, where u = min(s, SIZE_SATURATION).

    a is work divided among the workers, b the fixed cost each worker adds,
    c an offset. b is what a scalar x = cpu*memory*workers cannot express: it
    makes (512MB, W=4) and (2048MB, W=1) the same point when they differ.

    q is how strongly usable size converts into progress. It fits near 1 for
    time and for energy on per-worker meters, near 2 on a shared package meter,
    where a throttled worker runs longer and keeps the package powered for that
    longer window. Unidentifiable at a single size, where it is held at 1.

    Convex in W for a, b >= 0, minimum at W* = sqrt(a / (b * u**q)).
    """
    return a / (workers * size ** q) + b * workers + c


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
    """Energy and latency over (workers, per-worker size), one curve per quantity.

    Configurations are keyed on the pair (s, W), never their product, so
    allocations differing in worker count stay distinct points.

    s = cpu * memory folds the two per-worker quantities into one size: Lambda
    ties vCPU to memory and exposes only memory, k8s enforces cpu while memory
    limits do not change compute speed, so only one of the two ever varies.
    predict_time refuses a size the profile never covered.
    """

    def __init__(self, stage) -> None:
        super().__init__("analytic", stage)

        # [a, b, c, q] per quantity; q stays 1.0 when it was not fitted.
        self._params: Dict[str, Optional[List[float]]] = {k: None for k in FITTED}
        self._has_energy = False
        # Shared whole-machine meter (k8s RAPL) vs per-worker meters (Lambda).
        # Decides max-vs-sum aggregation.
        self._energy_shared_meter = False
        self._fit_exponent = False
        self._profiling_results = None
        self._keys: Tuple[Tuple[float, int], ...] = ()

    # ------------------------------------------------------------------
    # configuration key
    # ------------------------------------------------------------------

    @staticmethod
    def effective_size(size: float) -> float:
        """Usable per-worker size, capped at one core when configured."""
        return min(size, SIZE_SATURATION) if SIZE_SATURATION else size

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

    @property
    def min_configs(self) -> int:
        return 4 if self._fit_exponent else 3

    def _check_size(self, size: float) -> None:
        """Refuse a size the profile cannot speak to; warn when extrapolating."""
        sizes = self.profiled_sizes
        if not sizes:
            return
        if len(sizes) == 1:
            if abs(size - sizes[0]) > 1e-6:
                raise ValueError(
                    f"[{self._stage_name}] profiled at one size (s={sizes[0]:g}), "
                    f"asked for s={size:g}. Profile more than one cpu or memory "
                    f"value before predicting across sizes."
                )
            return
        if size < sizes[0] or size > sizes[-1]:
            logger.warning(
                "[%s] s=%g outside the profiled range [%g, %g]; extrapolating.",
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

    def _fit(self, points: Dict[Tuple[float, int], float]) -> Optional[List[float]]:
        """Fit model_func to {(s, W): mean}, returning [a, b, c, q] or None.

        a, b >= 0 is imposed; unbounded the fit predicts negative joules outside
        the sweep.
        """
        if len(points) < self.min_configs:
            return None

        keys = list(points)
        size = np.array([self.effective_size(k[0]) for k in keys], dtype=float)
        workers = np.array([k[1] for k in keys], dtype=float)
        y = np.array([points[k] for k in keys], dtype=float)

        guess = [float(max(y.max(), 0.0) * (workers * size).min()), 0.0, float(y.min())]
        lower, upper = [0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]
        if self._fit_exponent:
            guess.append(1.0)
            lower.append(0.0)
            upper.append(4.0)

        result = scipy_opt.least_squares(
            lambda p: model_func(workers, size, *p) - y,
            guess,
            bounds=(lower, upper),
            max_nfev=80000,
        )
        params = result.x.tolist()
        return params if self._fit_exponent else params + [1.0]

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        self._profiling_results = stage_profile_data

        for config_data in stage_profile_data.values():
            assert all(
                key in config_data for key in TIME_PHASES
            ), f"Each configuration's data must contain {list(TIME_PHASES)} keys."

        self._energy_shared_meter = False
        # Fitted only when every configuration carries energy. Older profiles
        # without it stay valid for latency and cost.
        self._has_energy = all(
            "energy" in cd and _has_numeric(cd["energy"])
            for cd in stage_profile_data.values()
        )
        if not self._has_energy:
            logger.info("[%s] no usable energy series; energy objective "
                        "unavailable.", self._stage_name)

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
        # q is identifiable only where the usable size varies: at one effective
        # level a/u**q collapses to a single coefficient. Held at 1 otherwise,
        # so single-size profiles fit exactly as they did before.
        self._fit_exponent = len(
            {self.effective_size(s) for s in self.profiled_sizes}
        ) > 1

        if len(stage_profile_data) < self.min_configs:
            raise ValueError(
                f"At least {self.min_configs} profiled configurations are required "
                f"to fit the model; got {len(stage_profile_data)}."
            )

        for name in TIME_PHASES:
            self._params[name] = self._fit(means[name])

        self._params["energy"] = self._fit(means["energy"]) if self._has_energy else None
        # Too few configurations for the free parameters: leave it unfitted
        # rather than pass a curve through every point.
        self._has_energy = self._params["energy"] is not None

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        assert config.workers > 0

        size, workers = self._key(config.cpu, config.memory, config.workers)
        self._check_size(size)
        usable = self.effective_size(size)

        def value(name: str) -> Optional[float]:
            params = self._params.get(name)
            return float(model_func(workers, usable, *params)) if params else None

        read = value("read") or 0.0
        compute = value("compute") or 0.0
        write = value("write") or 0.0
        cold = value("cold_start") or 0.0
        # None when unfitted. No proxy fallback: a made-up number is
        # indistinguishable from a real one downstream.
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

        The bounds keep a non-negative, so the scheduler's abs() is no longer
        load-bearing. For b, use time_marginal or energy_marginal.
        """
        fitted = [self._params[name] for name in TIME_PHASES if self._params[name]]
        return (sum(p[0] for p in fitted), sum(p[2] for p in fitted))

    @property
    def energy_parameters(self) -> Optional[Tuple[float, float, float, float]]:
        """(a, b, c, q) of the energy curve, or None if unfitted."""
        params = self._params.get("energy")
        return tuple(params) if params else None

    @property
    def per_worker_energy_cost(self) -> Optional[float]:
        """b: joules each extra worker costs regardless of its size."""
        params = self._params.get("energy")
        return float(params[1]) if params else None

    @property
    def size_exponent(self) -> Optional[float]:
        """q of the energy curve. Near 1 for per-worker meters, near 2 shared."""
        params = self._params.get("energy")
        return float(params[3]) if params else None

    def _median_key(self) -> Optional[Tuple[float, int]]:
        return self._keys[len(self._keys) // 2] if self._keys else None

    def _resolve(self, workers, size):
        """Default missing arguments to the median profiled configuration."""
        median = self._median_key()
        if median is None:
            return None, None
        return (median[1] if workers is None else workers,
                self.effective_size(median[0] if size is None else size))

    def energy_marginal(self, workers=None, size=None) -> Optional[float]:
        """dE/dW = -a/(W^2 * u^q) + b. Negative below the optimum, positive above."""
        params = self._params.get("energy")
        if not params:
            return None
        workers, usable = self._resolve(workers, size)
        if workers is None:
            return None
        a, b, _, q = params
        return -a / (float(workers) ** 2 * usable ** q) + b

    def time_marginal(self, workers=None, size=None) -> Optional[float]:
        """dt/dW summed over the latency phases."""
        workers, usable = self._resolve(workers, size)
        if workers is None:
            return None
        total = 0.0
        for name in TIME_PHASES:
            params = self._params[name]
            if params:
                total += -params[0] / (float(workers) ** 2 * usable ** params[3])
                total += params[1]
        return total

    def energy_optimal_workers(self, size=None) -> Optional[float]:
        """W* = sqrt(a / (b * u^q)).

        None when a coefficient is zero, so the curve is monotone, and None
        outside the profiled worker range.
        """
        params = self._params.get("energy")
        if not params:
            return None
        _, usable = self._resolve(None, size)
        if usable is None:
            return None
        a, b, _, q = params
        if a <= 0 or b <= 0:
            return None
        optimum = float(np.sqrt(a / (b * usable ** q)))
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
