import logging
from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
from flexecutor.modelling.energy_agg import stage_energy, _is_shared_meter

logger = logging.getLogger(__name__)


def phase_func(x, a, b):
    return a / x + b


io_func = comp_func = phase_func


def coldstart_func(x, a, b):
    """Cold start against x (= vcpu * memory * workers).

    Affine, not 1/x: read/compute/write are work divided among workers, but
    every worker pays its own start-up and they contend for admission, so
    the term grows with the worker count (6 s at W=1, 268 s at W=64). On
    Lambda, where workers start in parallel, the fit returns a ~ 0.
    """
    return a * x + b


def energy_func(x, a, b):
    """Energy against x, for independent per-worker meters (Lambda).

    Affine, not 1/x: more resources shorten a stage but do not reduce the
    work, so energy does not decay towards a floor.
    """
    return a * x + b

def energy_func_shared(x, a, b, c):
    """Energy against x, for a shared whole-machine meter (k8s RAPL).

    Node power times elapsed time: falls as 1/x while the work is divided,
    rises again as cold start grows. Convex, minimum at sqrt(a/b). A plain
    a/x + b can only fall, so it fits a negative a on workloads whose work
    grows with x and predicts negative joules.
    """
    return a / x + b * x + c


def _has_numeric(series) -> bool:
    """True if a stored profile series contains at least one usable number."""
    for run in series or []:
        for value in run if isinstance(run, (list, tuple)) else [run]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
    return False


def _numeric_values(series):
    """Flatten a stored profile series, dropping None and non-numeric entries."""
    out = []
    for run in series or []:
        for value in run if isinstance(run, (list, tuple)) else [run]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
    return out


class AnaPerfModel(PerfModel):
    """
    AnaPerfModel records the mean parameter value.
    Advantage: it is fast and accurate enough to optimize the average performance.
    Shortcoming: it does not guarantee the bounded performance.

    Ditto, Caerus model.
    Adapted from https://github.com/pkusys/Jolteon/blob/main/workflow/perf_model_analytic.py
    """

    def __init__(self, stage) -> None:
        super().__init__("analytic", stage)

        # Init in train, list with size three
        self._write_params = None
        self._read_params = None
        self._comp_params = None
        self._cold_params = None
        self._energy_params = None
        self._has_energy = False
        # True when the profile came from one shared whole-machine meter
        # (k8s RAPL).
        self._energy_shared_meter = False

        self._profiling_results = None

    @classmethod
    def _config_to_xparam(cls, num_vcpu, memory, num_func):
        return round(num_vcpu * memory * num_func, 1)

    @overrides
    def save_model(self):
        pass

    @overrides
    def load_model(self):
        pass

    @overrides
    def train(self, stage_profile_data: Dict) -> None:
        if len(stage_profile_data) < 2:
            raise ValueError(
                "At least two profiled configurations for each stage are required to train the step model."
            )
        self._profiling_results = stage_profile_data

        # Energy is fitted only when every configuration carries it. Older
        # profiles without it stay valid for latency and cost.
        required = [k for k in FunctionTimes.profile_keys() if k != "energy"]
        for config_data in stage_profile_data.values():
            assert all(
                key in config_data for key in required
            ), f"Each configuration's data must contain {required} keys."

        self._energy_shared_meter = False

        self._has_energy = all(
            "energy" in cd and _has_numeric(cd["energy"])
            for cd in stage_profile_data.values()
        )
        if not self._has_energy:
            print(
                "[AnaPerfModel] No usable energy series in the profile; the energy "
                "objective is unavailable for this stage. Re-profile with an "
                "energy monitor active before scheduling for energy."
            )

        # print(f"Training Analytical performance model for {self._stage_name}")

        size2points_coldstart = {}
        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}
        size2points_energy = {}

        for config_tuple, data in stage_profile_data.items():
            num_vcpu, memory, num_func = config_tuple
            # adapt to parallel mode
            # if the stage does not allow more than one function, ignore num_func and set to 1
            num_vcpu = num_vcpu if self.allow_parallel else 1
            config_key = self._config_to_xparam(num_vcpu, memory, num_func)

            for size2points, phase in zip(
                [
                    size2points_coldstart,
                    size2points_read,
                    size2points_comp,
                    size2points_write,
                ],
                ["cold_start", "read", "compute", "write"],
            ):
                if config_key not in size2points:
                    size2points[config_key] = []
                size2points[config_key].extend(data[phase])

            if self._has_energy:
                if config_key not in size2points_energy:
                    size2points_energy[config_key] = []
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
                    size2points_energy[config_key].append(stage_energy(values, srcs))

        for size2points in [
            size2points_coldstart,
            size2points_read,
            size2points_comp,
            size2points_write,
            size2points_energy,
        ]:
            for config in size2points:
                size2points[config] = np.mean(size2points[config])

        # print(size2points_coldstart)
        # print(size2points_read)
        # print(size2points_comp)
        # print(size2points_write)

        def fit_params(data, func, n_params=2, bounds=None, guess=None):
            assert isinstance(data, dict)
            arr_x = np.array(list(data.keys()))
            arr_y = np.array([data[x] for x in arr_x])

            initial_guess = guess if guess is not None else [1] * n_params

            def residuals(para, x, y):
                return func(x, *para) - y

            if bounds is None:
                params, _ = scipy_opt.leastsq(
                    residuals, initial_guess, args=(arr_x, arr_y)
                )
                return params.tolist()

            res = scipy_opt.least_squares(
                lambda para: residuals(para, arr_x, arr_y),
                initial_guess,
                bounds=bounds,
            )
            return res.x.tolist()

        # Fit the parameters
        # print("Fitting parameters...")
        self._cold_params = fit_params(size2points_coldstart, coldstart_func)
        self._read_params = fit_params(size2points_read, io_func)
        self._comp_params = fit_params(size2points_comp, comp_func)
        self._write_params = fit_params(size2points_write, io_func)
        if self._has_energy and len(size2points_energy) >= self._energy_nparams:
            if self._energy_shared_meter:
                xs = np.array(list(size2points_energy.keys()))
                ys = np.array(list(size2points_energy.values()))
                # a >= 0 (energy divided among workers), b >= 0 (cost per extra
                # unit). Unbounded, the fit goes negative and so do the joules.
                self._energy_params = fit_params(
                    size2points_energy,
                    self._energy_curve,
                    3,
                    bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
                    guess=[float(ys.max() * xs.min()), 0.0, float(ys.min())],
                )
            else:
                self._energy_params = fit_params(
                    size2points_energy, self._energy_curve, 2
                )
        else:
            # Fewer configurations than free parameters: leave it unfitted
            # rather than pass a curve exactly through every point.
            self._energy_params = None
            self._has_energy = False

        # print(
        #     f"COLD START: alpha parameter = {self._cold_params[0]}, beta parameter = {self._cold_params[1]}"
        # )
        # print(
        #     f"READ STEP: alpha parameter = {self._read_params[0]}, beta parameter = {self._read_params[1]}"
        # )
        # print(
        #     f"COMPUTE STEP: alpha parameter = {self._comp_params[0]}, beta parameter = {self._comp_params[1]}"
        # )
        # print(
        #     f"WRITE_STEP: alpha parameter = {self._write_params[0]}, beta parameter = {self._write_params[1]}"
        # )

    @property
    @overrides
    def parameters(self):
        # parameter a (alpha), represents the paralelizable part, while beta is some non-paralellizable constant
        a = sum(
            [
                self._cold_params[0],
                self._read_params[0],
                self._comp_params[0],
                self._write_params[0],
            ]
        )
        b = sum(
            [
                self._cold_params[1],
                self._read_params[1],
                self._comp_params[1],
                self._write_params[1],
            ]
        )
        return a, b

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        assert config.workers > 0

        # FIXME: Use the parameter function to predict the time
        # key = num_vcpu + runtime_memory + num_workers
        key = self._config_to_xparam(config.cpu, config.memory, config.workers)
        predicted_read_time = io_func(key, *self._read_params)
        predicted_comp_time = comp_func(key, *self._comp_params)
        predicted_write_time = io_func(key, *self._write_params)
        predicted_cold_time = coldstart_func(key, *self._cold_params)
        total_predicted_time = (
            predicted_read_time
            + predicted_comp_time
            + predicted_write_time
            + predicted_cold_time
        )

        a, b = self.parameters
        logger.debug(
            "Predicted time: %s / %s + %s = %s", a, key, b, total_predicted_time
        )

        # Predicted only when fitted from measurement. No proxy fallback: a
        # made-up number is indistinguishable from a real one downstream.
        predicted_energy = (
            self._energy_curve(key, *self._energy_params)
            if self._energy_params
            else None
        )
        if predicted_energy is not None:
            logger.debug("Predicted energy: %.4f J at key=%s", predicted_energy, key)

        return FunctionTimes(
            total=total_predicted_time,
            read=predicted_read_time,
            compute=predicted_comp_time,
            write=predicted_write_time,
            cold_start=predicted_cold_time,
            energy=predicted_energy,
            energy_source="fitted_from_profile" if predicted_energy is not None else "none",
        )

    @property
    def _energy_nparams(self):
        """Free parameters of the energy curve for this profile's meter."""
        return 3 if self._energy_shared_meter else 2

    def _energy_curve(self, x, *params):
        """Dispatch to the curve matching the profile's meter.

        Fitting and prediction must use the same shape, so both come here.
        """
        if self._energy_shared_meter:
            return energy_func_shared(x, *params)
        return energy_func(x, *params)

    @property
    def energy_parameters(self):
        """Raw coefficients of the fitted curve, or None if unfitted.

        a is a slope only for the affine shape. Use energy_marginal() for a
        slope.
        """
        return tuple(self._energy_params) if self._energy_params else None
    
    def _profiled_x(self):
        """The x values this stage was actually profiled at, ascending."""
        if not self._profiling_results:
            return []
        xs = []
        for cpu, memory, workers in self._profiling_results:
            vcpu = cpu if self.allow_parallel else 1
            xs.append(self._config_to_xparam(vcpu, memory, workers))
        return sorted(xs)

    def energy_marginal(self, x=None):
        """dE/dx at x: what one more unit of resource costs in energy.

        For the affine shape a is the slope; for a/x + b*x + c it is
        -a/x^2 + b, so reading a directly flips the sign on RAPL profiles.
        x defaults to the median profiled configuration.
        """
        if not self._energy_params:
            return None
        if x is None:
            xs = self._profiled_x()
            if not xs:
                return None
            x = xs[len(xs) // 2]
        if self._energy_shared_meter:
            a, b = self._energy_params[0], self._energy_params[1]
            return -a / (x * x) + b
        return self._energy_params[0]
    
    def energy_optimal_x(self):
        """The x minimising the fitted energy curve, or None.

        sqrt(a/b) for the convex shape. None for the affine one, which has
        no interior minimum, and None outside the profiled range: a b near
        zero puts the minimum at absurd x.
        """
        if not self._energy_params or not self._energy_shared_meter:
            return None
        a, b = self._energy_params[0], self._energy_params[1]
        if a <= 0 or b <= 0:
            return None
        xstar = float(np.sqrt(a / b))
        xs = self._profiled_x()
        if not xs or not (xs[0] <= xstar <= xs[-1]):
            return None
        return xstar

    @property
    def has_energy_model(self) -> bool:
        return bool(self._energy_params)
