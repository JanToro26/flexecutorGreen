from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.optimize as scipy_opt
from overrides import overrides

from flexecutor.modelling.perfmodel import PerfModel
from flexecutor.utils.dataclass import StageConfig, FunctionTimes
from flexecutor.modelling.energy_agg import stage_energy


class ModelStrategy(ABC):
    """
    The ModelStrategy is used to define the strategy to be used to model the performance
    """

    def __init__(self, mixed_model: "MixedPerfModel"):
        self._model = mixed_model

    @abstractmethod
    def run(self) -> None:
        pass


class GetAndSet:
    def get(self, attr: str):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(
                f"Attribute {attr} does not exist in the class MixedModelCoefficients"
            )

    def set(self, attr: str, value: float):
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            raise AttributeError(
                f"Attribute {attr} does not exist in the class MixedModelCoefficients"
            )


@dataclass
class ModelParams(GetAndSet):
    """
    The average parameters of the performance model
    - cold: array with the coefficients of the cold start phase
    - read: array with the coefficients of the read phase
    - compute: array with the coefficients of the compute phase
    - write: array with the coefficients of the write phase
    """

    read: np.array
    compute: np.array
    write: np.array
    cold: np.array

    def __init__(self):
        self.read = np.array([])
        self.compute = np.array([])
        self.write = np.array([])
        self.cold = np.array([])


class ModelCovariance(ModelParams):
    """
    The covariance of the average parameters of the performance model
    - cold: covariance of the cold start phase
    - read: covariance of the read phase
    - compute: covariance of the compute phase
    - write: covariance of the write phase
    """

    pass


@dataclass
class MixedModelCoefficients(GetAndSet):
    """
    The coefficients of the mixed performance model

    The dimension of the parameters is reduced from 8 to 5 (mixing common degree
    coeffs between phases), excluding cold start
    By merging the parameters of read, compute, and write as follows:
    - allow_parallel: a/d + b/(kd) + c*log(x)/x + e/x**2 + f, x can be d or kd
    - not allow_parallel: a/k + b*d + c*log(k)/k + e/k**2 + f
    """

    cold: float  # _ --> cold start time
    x: float  # a --> coefficient of 1/d or 1/k
    kd_d: float  # b --> coefficient of 1/(kd) or d
    logx: float  # c --> coefficient of log(x)/x, x can be d or kd
    x2: float  # e --> coefficient of 1/x**2, x can be d or kd
    const: float  # f --> constant coefficient

    def __init__(self):
        self.cold = 0
        self.x = 0
        self.kd_d = 0
        self.logx = 0
        self.x2 = 0
        self.const = 0

    def __array__(self):
        return np.array([self.cold, self.x, self.kd_d, self.logx, self.x2, self.const])


@dataclass
class CanIntraParallel(GetAndSet):
    """
    The can_intra_parallel is used to check if the phase can be parallelized
    """

    read: bool
    compute: bool
    write: bool

    def __init__(self):
        self.read = False
        self.compute = False
        self.write = False


def eq_vcpu_alloc(mem, num_func):
    """
    The eq_vcpu_alloc is used to convert the memory to vCPU (Lambda fix rate)
    Function inherited from Jolteon
    """
    return round((mem / 1792) * num_func, 1)


def io_func(x, a, b):
    """
    The io_func is used to model the read and write phases of the stage
    Note that the form is a*(1/x) + b
    @param x: array with the computational resource. Can be:
        - k: number of individual cpu units (per worker) --> Only for not allow_parallel
        - kd: number of total cpu units
        - d: number of workers
    @param a: variable coefficient for 1/x
    @param b: the constant coefficient
    @return: the time taken for the phase
    """
    return a / x + b


def io_func_pr(_input, a, b, c):
    """
    io_func2 is used to model the read parent_relevant phase
    Note that the form is a*(1/x) + b*y + c
    @param _input: two-dim array with (specific case):
        _input[0] (x): number of individual cpu units (per worker)
        _input[1] (y): number of workers
    @param a: variable coefficient for 1/x
    @param b: variable coefficient for y
    @param c: the constant coefficient
    @return: the time taken for the phase
    """
    x = _input[0]
    y = _input[1]
    return a / x + b * y + c


def comp_func(x, a, b, c, d):
    """
    The comp_func is used to model the compute phase of the stage
    Two different complexities are considered:
        - logarithmic complexity
        - quadratic complexity
    So, the curve is more adaptable to the real data, being aware of the different complexities
    Note that the form is a*(1/x) + b*log(x)/x + c/x**2 + d
    @param x: array with the computational resource. Can be:
        - k: number of individual cpu units (per worker) --> Only for not allow_parallel
        - kd: number of total cpu units
        - d: number of workers
    @param a: variable coefficient for 1/x
    @param b: variable coefficient for log(x)/x
    @param c: variable coefficient for 1/x**2
    @param d: the constant coefficient
    @return: the time taken for the phase
    """
    return a / x + b * np.log(x) / x + c / x**2 + d


def curve_fit(func, x, y, dims):
    return scipy_opt.curve_fit(func, x, y)[0:dims]


class MixedPerfModel(PerfModel, GetAndSet):
    """
    Mixed performance model that combines the white-box and black-box modelling
    Here, definitions of the notations used in the model:
    - y_s: array with cold start times
    - self.y_r: array with read times
    - y_c: array with compute times
    - y_w: array with write times
    - d: array with number of workers
    - kd: array with number of total cpu units
    - k: array with number of individual cpu units (per worker)
    - k_d: two-dim array with:
        k_d[0]: number of individual cpu units (per worker)
        k_d[1]: number of workers
    - can_intra_parallel:
        if True, we can parallelize the phase (read|compute|write) of the stage
        else otherwise
    - allow_parallel:
        if True, the stage can be parallelized
        else the stage cannot be parallelized (only 1 worker is allowed) for this stage
    - parent_relevant: this attribute deserves a better explanation:
        when profiling, not allow_parallel stages also are profiled with multiple workers
        but the result of the optimization will only output 1 worker
        so, parent_relevant is used to check if the read time of the stages depend on:
            only the number of cpu of the worker (parent_relevant = False)
            or the number of cpu of the worker and the number of workers (parent_relevant = True)
    """

    def __init__(self, stage):
        super().__init__("mixed", stage)

        self.can_intra_parallel = CanIntraParallel()
        self.parent_relevant = False

        # (p_dyn, p_idle) in watts, fitted from measured energy during train().
        # None means the energy objective is unavailable for this stage.
        self._power_params = None

        self.params_avg = ModelParams()
        self.cov_avg = ModelCovariance()

        self.y_s = np.array([])
        self.y_r = np.array([])
        self.y_c = np.array([])
        self.y_w = np.array([])
        self.d = np.array([])
        self.kd = np.array([])
        self.k = np.array([])
        self.k_d = np.empty((0, 2))

        self.coeffs = MixedModelCoefficients()

    def _set_coeff_by_params(self, coeff, params, make_error=False):
        # IMPORTANT: Jolteon contains 1 huge error in its pkysus/Jolteon src repo
        # In not parallel: compute and write const times aren't considered in sampling
        # make_error is a trick to adapt to that
        # Please, remove it when the time arrives
        coeff.cold = params.cold
        coeff.logx += params.compute[1]
        coeff.x2 += params.compute[2]
        if self.allow_parallel:
            for phase in ["read", "compute", "write"]:
                if self.can_intra_parallel.get(phase):
                    coeff.kd_d += params.get(phase)[0]
                else:
                    coeff.x += params.get(phase)[0]
            coeff.const += params.read[1] + params.compute[3] + params.write[1]
        else:
            for phase in ["read", "compute", "write"]:
                coeff.x += params.get(phase)[0]
            if self.parent_relevant:
                coeff.kd_d += params.read[1]
                coeff.const += params.read[2]
            else:
                coeff.const += params.read[1]
            if not make_error:
                coeff.const += params.compute[3] + params.write[1]

    @overrides
    def _fit_power_params(self, stage_profile_data: Dict) -> None:
        """
        Fit a per-stage power model from measured energy:

            P(k, d) = p_dyn * (k * d) + p_idle          [watts]
            E(k, d) = P(k, d) * T(k, d)                 [joules]

        Both coefficients come from least squares over the profiled
        configurations. That is the whole point of the exercise: the structural
        form (energy scales with resources held times time held) is assumed,
        but the watts are measured, not a constant picked to make the numbers
        move. A model of the shape `energy = 0.1 * vcpu * workers * time` with
        a hard-coded 0.1 is not an energy model -- it is latency wearing an
        energy label, and it will rank configurations exactly as latency does.

        Sets ``self._power_params`` to None when the profile carries no usable
        energy, which makes the energy objective refuse to run rather than
        quietly degrade.
        """
        self._power_params = None

        rows_x, rows_y = [], []
        for config_tuple, data in stage_profile_data.items():
            cpu, memory, workers = config_tuple
            energy_runs = data.get("energy")
            duration_runs = data.get("energy_duration")
            if not energy_runs or not duration_runs:
                continue

            source_runs = data.get("energy_source") or []
            for idx, (e_run, d_run) in enumerate(zip(energy_runs, duration_runs)):
                e_vals = [v for v in (e_run or []) if isinstance(v, (int, float))
                          and not isinstance(v, bool)]
                d_vals = [v for v in (d_run or []) if isinstance(v, (int, float))
                          and not isinstance(v, bool)]
                if not e_vals or not d_vals:
                    continue
                srcs = source_runs[idx] if idx < len(source_runs) else []
                if not isinstance(srcs, (list, tuple)):
                    srcs = [srcs]
                # Stage duration is the slowest worker, because workers run
                # concurrently. Stage energy depends on what meter produced the
                # readings -- SUM for independent per-worker meters, MAX for one
                # shared whole-machine meter. See flexecutor.modelling.energy_agg.
                stage_e = stage_energy(e_vals, srcs)
                stage_duration = float(max(d_vals))
                if stage_duration <= 0:
                    continue
                k = float(cpu) if cpu else 1.0
                d = float(workers) if self.allow_parallel else 1.0
                # E = (p_dyn * k*d + p_idle) * T  ->  linear in (k*d*T, T)
                rows_x.append([k * d * stage_duration, stage_duration])
                rows_y.append(stage_e)

        if len(rows_x) < 2:
            print(
                f"[MixedPerfModel:{self._model_name}] Not enough measured energy to fit "
                "a power model; the energy objective is unavailable for this stage."
            )
            return

        A = np.array(rows_x, dtype=float)
        y = np.array(rows_y, dtype=float)
        try:
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError as e:
            print(f"[MixedPerfModel:{self._model_name}] Power fit failed: {e}")
            return

        p_dyn, p_idle = float(coeffs[0]), float(coeffs[1])
        # Negative power is not physical; it means the profile does not
        # constrain the fit. Report it instead of scheduling against it.
        if p_dyn < 0 or p_idle < 0:
            print(
                f"[MixedPerfModel:{self._model_name}] Power fit produced non-physical "
                f"coefficients (p_dyn={p_dyn:.3f}W, p_idle={p_idle:.3f}W). Widen the "
                "profiled configuration space before scheduling for energy."
            )
            return

        self._power_params = (p_dyn, p_idle)
        print(
            f"[MixedPerfModel:{self._model_name}] Power model fitted: "
            f"P = {p_dyn:.3f}W per vcpu-worker + {p_idle:.3f}W static"
        )

    @property
    def has_energy_model(self) -> bool:
        return getattr(self, "_power_params", None) is not None

    def train(self, stage_profile_data: Dict) -> None:
        # STEP 0: Fit the power model from measured energy, if present
        self._fit_power_params(stage_profile_data)

        # STEP 1: Populate the data
        self._populate_data(stage_profile_data)

        # STEP 2: Calculate the average parameters and covariance
        self._calc_params_and_covariances()

        # STEP 3: Compute the average coefficients
        self._set_coeff_by_params(self.coeffs, self.params_avg)

        # STEP 4: Print the accuracy of the model
        self._evaluate_model()

    def _evaluate_model(self):
        y_actual = self.y_r + self.y_c + self.y_w + self.y_s
        if self.allow_parallel:
            y_pred = (
                self.coeffs.x / self.d
                + self.coeffs.kd_d / self.kd
                + self.coeffs.const
                + np.mean(self.y_s)
            )
            if self.can_intra_parallel.compute:
                y_pred += self.coeffs.logx * np.log(self.kd) / self.kd
                y_pred += self.coeffs.x2 / self.kd**2
            else:
                y_pred += self.coeffs.logx * np.log(self.d) / self.d
                y_pred += self.coeffs.x2 / self.d**2
        else:
            y_pred = (
                self.coeffs.x / self.k
                + self.coeffs.kd_d * self.k_d[1]
                + self.coeffs.const
                + np.mean(self.y_s)
                + self.coeffs.logx * np.log(self.k) / self.k
                + self.coeffs.x2 / self.k**2
            )
        print(f"### {self._model_name} ### ")
        err = (y_pred - y_actual) / y_actual
        s_err = np.mean(np.abs(err))
        m_err = np.mean(err)
        print(
            "Stage {} mean abs error:".format(self._stage_name),
            "%.2f" % (s_err * 100),
            "%",
        )
        print(
            "Stage {} mean error:".format(self._stage_name),
            "%.2f" % (m_err * 100),
            "%",
        )
        print()

    def _calc_params_and_covariances(self):
        rw_parallel_choices = [
            {"var": "d", "func": io_func, "dims": 2},
            {"var": "kd", "func": io_func, "dims": 2},
        ]
        parallel_heuristic = {
            "read": {
                "data": self.y_r,
                "choices": rw_parallel_choices,
            },
            "compute": {
                "data": self.y_c,
                "choices": [
                    {"var": "d", "func": comp_func, "dims": 4},
                    {"var": "kd", "func": comp_func, "dims": 4},
                ],
            },
            "write": {
                "data": self.y_w,
                "choices": rw_parallel_choices,
            },
        }
        not_parallel_heuristic = {
            "read": {
                "data": self.y_r,
                "choices": [
                    {"var": "k", "func": io_func, "dims": 2},
                    {
                        "var": "k_d",
                        "func": io_func_pr,
                        "dims": 3,
                        "restriction": self.has_parent,
                    },
                ],
            },
            "compute": {
                "data": self.y_c,
                "choices": [
                    {"var": "k", "func": comp_func, "dims": 4},
                ],
            },
            "write": {
                "data": self.y_w,
                "choices": [
                    {"var": "k", "func": io_func, "dims": 2},
                ],
            },
        }

        heuristic = (
            parallel_heuristic if self.allow_parallel else not_parallel_heuristic
        )

        def choice_is_allowed(var):
            return "restriction" not in var or var["restriction"]

        for phase, items in heuristic.items():
            err_dict = {}
            for choice in items["choices"]:
                choice["params_avg"], choice["cov_avg"] = curve_fit(
                    choice["func"],
                    self.get(choice["var"]),
                    items["data"],
                    choice["dims"],
                )
                y_ = choice["func"](self.get(choice["var"]), *choice["params_avg"])
                err = (y_ - items["data"]) / items["data"]
                s_err = np.mean(np.abs(err))
                if choice_is_allowed(choice):
                    err_dict[choice["var"]] = s_err
            best_choice = min(err_dict, key=err_dict.get)
            avg_data = next(x for x in items["choices"] if (x["var"] == best_choice))
            self.params_avg.set(phase, avg_data["params_avg"])
            self.cov_avg.set(phase, avg_data["cov_avg"])
            # Set special attributes
            self.can_intra_parallel.set(phase, best_choice == "kd")
            if phase == "read":
                self.parent_relevant = best_choice == "k_d"

    def _populate_data(self, stage_profile_data):
        for config_tuple, data in stage_profile_data.items():
            _, memory, num_func = config_tuple

            # Jolteon's conversion from memory to vCPU (lambda fix rate)
            # FIXME: self system that does not use this conversion
            num_vcpu = eq_vcpu_alloc(memory, num_func if self.allow_parallel else 1)

            # Only taken the first item in each round & discarding the first exec (erase cold start effects)
            # FIXME: check if more data improve results
            skip_first_pos = 0
            number_items = len(data["cold_start"]) - skip_first_pos
            self.y_r = np.concatenate(
                [self.y_r, [item[0] for item in data["read"][skip_first_pos:]]]
            )
            self.y_c = np.concatenate(
                [self.y_c, [item[0] for item in data["compute"][skip_first_pos:]]]
            )
            self.y_w = np.concatenate(
                [self.y_w, [item[0] for item in data["write"][skip_first_pos:]]]
            )
            self.d = np.concatenate([self.d, [num_func] * number_items])
            self.k = np.concatenate([self.k, [num_vcpu] * number_items])
            self.kd = np.concatenate(
                [self.kd, [eq_vcpu_alloc(memory, num_func)] * number_items]
            )
            self.k_d = np.concatenate([self.k_d, [[num_vcpu, num_func]] * number_items])
            self.y_s = np.concatenate(
                [self.y_s, [item[0] for item in data["cold_start"][skip_first_pos:]]]
            )
        self.params_avg.cold = self.y_s
        self.k_d = self.k_d.reshape(-1, 2).T

    def predict_time(self, config: StageConfig) -> FunctionTimes:
        pass

    @overrides
    def load_model(self):
        pass

    @overrides
    def save_model(self):
        pass

    @overrides
    def parameters(self):
        self.coeffs.cold = np.percentile(self.params_avg.cold, 60)
        return np.array(self.coeffs)

    def sample_offline(self, num_samples=10000) -> np.ndarray:
        # seed_val = int(time.time())
        seed_val = 0
        rng = np.random.default_rng(seed=seed_val)

        cold_samples = rng.choice(self.params_avg.cold, num_samples)
        read_samples = rng.multivariate_normal(
            self.params_avg.read, self.cov_avg.read, num_samples
        )
        compute_samples = rng.multivariate_normal(
            self.params_avg.compute, self.cov_avg.compute, num_samples
        )
        write_samples = rng.multivariate_normal(
            self.params_avg.write, self.cov_avg.write, num_samples
        )

        params_list = [ModelParams() for _ in range(num_samples)]
        coeffs_list = [MixedModelCoefficients() for _ in range(num_samples)]

        for i in range(num_samples):
            params_list[i].read = read_samples[i]
            params_list[i].compute = compute_samples[i]
            params_list[i].write = write_samples[i]
            params_list[i].cold = cold_samples[i]

            self._set_coeff_by_params(coeffs_list[i], params_list[i], make_error=True)

        return np.array([np.array(coeffs_list[i]) for i in range(num_samples)])

    def generate_func_code(self, mode) -> str:
        config_list = "config_list"
        coeffs_list = "coeffs_list"

        assert mode in ["latency", "cost", "energy"]

        if mode == "energy" and not self.has_energy_model:
            raise ValueError(
                f"Stage '{self._model_name}' has no fitted power model, so an energy "
                "expression cannot be generated. Re-profile this stage with an energy "
                "monitor active over at least two configurations."
            )

        stage_idx = int(self._stage_idx)
        cold_param = f"{coeffs_list}[{stage_idx}][cold]"
        x_param = f"{coeffs_list}[{stage_idx}][x]"
        kd_d_param = f"{coeffs_list}[{stage_idx}][kd_d]"
        logx_param = f"{coeffs_list}[{stage_idx}][logx]"
        x2_param = f"{coeffs_list}[{stage_idx}][x2]"
        const_param = f"{coeffs_list}[{stage_idx}][const]"

        var_d = f"{config_list}[workers({stage_idx})]" if self.allow_parallel else "1"
        var_k = f"{config_list}[cpu({stage_idx})]"
        var_x = (
            f"({var_k} * {var_d})" if self.can_intra_parallel.compute else f"({var_d})"
        )

        if self.allow_parallel:
            code = (
                f"{x_param}/{var_d} + {kd_d_param}/({var_k}*{var_d}) + {logx_param}*np.log({var_x})/{var_x} + "
                f"{x2_param}/{var_x}**2 + {const_param}"
            )
        else:
            code = f"{x_param}/{var_k} + "
            if self.parent_relevant:
                code += f"{kd_d_param}*{config_list}[workers({self.parent_idx})] + "
            code += f"{logx_param}*np.log({var_k})/{var_k} + {x2_param}/{var_k}**2 + {const_param}"
        if mode == "latency":
            code = f"{cold_param} + {code}"
        elif mode == "energy":
            # E = (p_dyn * k * d + p_idle) * T, with p_dyn/p_idle in watts
            # fitted from measured energy in _fit_power_params. Cold start is
            # included in T: the container is powered while it initialises, so
            # excluding it would systematically under-report the energy of
            # wide, short stages -- exactly the configurations the optimiser is
            # deciding between.
            p_dyn, p_idle = self._power_params
            code = (
                f"({cold_param} + {code}) * "
                f"({p_dyn!r} * {var_k} * {var_d} + {p_idle!r})"
            )
        else:
            # 1792 / 1024 * 0.0000000167 * 1000 = 0.000029225
            # 1000 is to convert from ms to s
            # We multiply 1e5 to the cost to make it more readable
            # s = cold_param + ' / 2 + ' + s
            code = f"({code}) * {var_k} * {var_d} * 2.9225 + 0.02 * {var_d}"
        return code
