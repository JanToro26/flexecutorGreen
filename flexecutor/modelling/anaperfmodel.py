from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import FunctionTimes, StageConfig, ConfigBounds
from flexecutor.modelling.energy_agg import stage_energy


def phase_func(x, a, b):
    return a / x + b


coldstart_func = io_func = comp_func = phase_func


def energy_func(x, a, b):
    """
    Energy against allocated resource x (= vcpu * memory * workers).

    Deliberately affine rather than the 1/x shape used for the time phases.
    Adding resources shortens a stage but does not reduce the work done, so
    energy does not decay towards a floor the way latency does; it typically
    rises, because more parallel capacity means more idle silicon powered for
    the duration. Fitting energy with a latency-shaped curve would force the
    optimiser to conclude that more workers are always more efficient, which
    is the opposite of the effect being measured.
    """
    return a * x + b


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

        # `energy` is fitted only when every configuration actually carries it.
        # Profiles recorded before energy collection existed are still valid
        # for the latency and cost objectives; failing the whole training run
        # for a missing energy series would throw away that data.
        required = [k for k in FunctionTimes.profile_keys() if k != "energy"]
        for config_data in stage_profile_data.values():
            assert all(
                key in config_data for key in required
            ), f"Each configuration's data must contain {required} keys."

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

        def fit_params(data, func):
            assert isinstance(data, dict)
            arr_x = list(data.keys())
            arr_y = [data[x] for x in arr_x]

            arr_x = np.array(arr_x)
            arr_y = np.array(arr_y)

            initial_guess = [1, 1]

            def residuals(para, x, y):
                predicted = func(x, *para)
                residuals = predicted - y
                return residuals

            params, _ = scipy_opt.leastsq(residuals, initial_guess, args=(arr_x, arr_y))

            return params.tolist()

        # Fit the parameters
        # print("Fitting parameters...")
        self._cold_params = fit_params(size2points_coldstart, coldstart_func)
        self._read_params = fit_params(size2points_read, io_func)
        self._comp_params = fit_params(size2points_comp, comp_func)
        self._write_params = fit_params(size2points_write, io_func)
        if self._has_energy and len(size2points_energy) >= 2:
            self._energy_params = fit_params(size2points_energy, energy_func)
        else:
            # A single configuration cannot constrain a two-parameter fit. Leave
            # the model unfitted rather than returning a curve through one point.
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
        print(
            f"Predicted time: {a} / {key} + {b} = {total_predicted_time} = {(a / key) + b}"
        )

        # Energy is predicted only when it was actually fitted from measured
        # data. There is deliberately no proxy fallback here: substituting
        # something like `k * vcpu * workers * time` would return a number that
        # looks like a prediction but carries no measurement, and downstream
        # code cannot tell the two apart. `None` forces the caller to handle
        # the absence explicitly.
        predicted_energy = (
            energy_func(key, *self._energy_params) if self._energy_params else None
        )
        if predicted_energy is not None:
            print(f"Predicted energy: {predicted_energy:.4f} J at key={key}")

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
    def energy_parameters(self):
        """
        (a, b) of the fitted affine energy model, or None when unfitted.

        Exposed so schedulers can allocate on measured energy coefficients
        rather than re-deriving them from a single predicted point.
        """
        return tuple(self._energy_params) if self._energy_params else None

    @property
    def has_energy_model(self) -> bool:
        return bool(self._energy_params)
