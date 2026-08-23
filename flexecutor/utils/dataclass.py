from dataclasses import dataclass
from typing import Optional


@dataclass
class StageConfig:
    """
    Resource configuration for one stage.
    """

    cpu: float
    workers: int
    memory: float = 0

    @property
    def key(self) -> tuple[float, float, int]:
        return self.cpu, self.memory, self.workers

    def __array__(self):
        return [self.workers, self.cpu]


@dataclass
class ConfigBounds:
    """
    Configuration bounds for the stage
    """

    cpu: tuple[float, float]
    memory: tuple[float, float]
    workers: tuple[int, int]

    def to_tuple_list(self) -> list[tuple]:
        return [self.cpu, self.memory, self.workers]


@dataclass
class FunctionTimes:
    """
    Per-invocation measurements returned by a stage.

    Energy fields
    -------------
    ``energy`` is the canonical, fittable quantity in Joules: it is what the
    performance models train on and what the energy objective minimises. It is
    selected in ``StageFuture.get_timings`` from the best mechanism actually
    available on the worker, with ``energy_source`` recording which one, so a
    modelled figure is never silently mistaken for a measured one.

    The per-mechanism fields are kept alongside it because the comparison
    between them is itself a result: drift between RAPL and the psutil model
    across configurations is the signal that fitted power-model constants are
    absorbing a systematic error rather than describing the hardware.
    """

    read: Optional[float] = None
    compute: Optional[float] = None
    write: Optional[float] = None
    cold_start: Optional[float] = None
    total: Optional[float] = None

    # --- canonical energy (Joules) -----------------------------------------
    energy: Optional[float] = None
    energy_source: Optional[str] = None  # 'rapl' | 'psutil_model' | 'perf' | 'none'
    energy_duration: Optional[float] = None

    # --- per-mechanism energy ----------------------------------------------
    rapl_energy_pkg: Optional[float] = None
    rapl_energy_cores: Optional[float] = None
    rapl_available: Optional[bool] = None

    # psutil model: the dynamic share is per-worker and additive; the idle
    # floor is a host property and must not be summed across co-located
    # workers. Keeping them apart is what stops the per-worker idle
    # duplication that made local energy look superlinear in worker count.
    psutil_energy_pkg: Optional[float] = None
    psutil_energy_dynamic: Optional[float] = None
    psutil_energy_idle_machine: Optional[float] = None
    psutil_p_idle_machine_w: Optional[float] = None
    psutil_available: Optional[bool] = None

    # perf is system-wide (`perf stat -a`); a package reading, never a
    # per-worker one. Opt-in, used for cross-validating RAPL.
    perf_energy_pkg: Optional[float] = None
    perf_available: Optional[bool] = None
    perf_scope: Optional[str] = None

    # --- utilisation --------------------------------------------------------
    cpu_percent: Optional[float] = None       # share of the package, 0-100
    proc_cpu_percent: Optional[float] = None  # % of one core, may exceed 100
    cores_used: Optional[float] = None
    util_share: Optional[float] = None

    # --- host identity ------------------------------------------------------
    cpu_name: Optional[str] = None
    cpu_brand: Optional[str] = None
    cpu_architecture: Optional[str] = None
    cpu_cores_physical: Optional[int] = None
    cpu_cores_logical: Optional[int] = None
    tdp_ref: Optional[float] = None
    tdp_source: Optional[str] = None
    # True when the TDP could not be resolved and a fallback was used. Any
    # result derived from such a run rests on an unverified constant.
    tdp_is_default: Optional[bool] = None
    cloud_instance_type: Optional[str] = None

    @classmethod
    def profile_keys(cls) -> list[str]:
        """
        Numeric quantities the performance models fit curves against.

        ``energy`` is included so the analytic model fits an energy curve from
        real measurements instead of a proxy. Keep this list numeric:
        everything here is averaged and handed to a least-squares fit.
        """
        return ["read", "compute", "write", "cold_start", "energy"]

    @classmethod
    def energy_keys(cls) -> list[str]:
        """Per-mechanism numeric energy/utilisation. Recorded, not fitted."""
        return [
            "energy_duration",
            "rapl_energy_pkg",
            "rapl_energy_cores",
            "psutil_energy_pkg",
            "psutil_energy_dynamic",
            "psutil_energy_idle_machine",
            "psutil_p_idle_machine_w",
            "perf_energy_pkg",
            "cpu_percent",
            "proc_cpu_percent",
            "cores_used",
            "util_share",
            "tdp_ref",
        ]

    @classmethod
    def metadata_keys(cls) -> list[str]:
        """Non-numeric provenance. Stored with the profile, never fitted."""
        return [
            "energy_source",
            "rapl_available",
            "psutil_available",
            "perf_available",
            "perf_scope",
            "cpu_name",
            "cpu_brand",
            "cpu_architecture",
            "cpu_cores_physical",
            "cpu_cores_logical",
            "tdp_source",
            "tdp_is_default",
            "cloud_instance_type",
        ]

    def __lt__(self, other):
        return self.total < other.total
