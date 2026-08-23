from __future__ import annotations

from typing import Any, Optional, List

from lithops.utils import FuturesList

from flexecutor.utils.dataclass import FunctionTimes


class StageFuture:
    def __init__(self, stage_id: str, future: Optional[FuturesList] = None):
        self.__stage_id = stage_id
        self.__future = future

    def result(self) -> Any:
        return [i[0] for i in self.__future.get_result()]

    def _timings_list(self) -> list[FunctionTimes]:
        return [i[1] for i in self.__future.get_result()]

    @property
    def stats(self):
        return [f.stats for f in self.__future]

    def error(self) -> bool:
        return any([f.error for f in self.__future])

    def __getattr__(self, item):
        if item in vars(self):
            return getattr(self, item)
        elif "__future" in vars(self) and item in vars(self.__future):
            return getattr(self.__future, item)
        raise AttributeError(f"Future object has no attribute {item}")

    # Preference order for the canonical `energy` field. RAPL first because it
    # is a hardware counter; perf second because `perf stat -a` is real but
    # system-wide, so it over-attributes when workers share a host; the psutil
    # power model last because it is modelled, not measured.
    _ENERGY_PREFERENCE = (
        ("rapl", "worker_func_rapl_available", "worker_func_rapl_energy_pkg"),
        ("perf", "worker_func_perf_available", "worker_func_perf_energy_pkg"),
        ("psutil_model", "worker_func_psutil_available", "worker_func_psutil_energy_pkg"),
    )

    @staticmethod
    def _select_energy(s: dict) -> tuple[Optional[float], str]:
        """
        Pick the canonical energy value and record which mechanism produced it.

        Returning the source alongside the number is the point: a run whose
        energy came from the psutil model is a different kind of evidence from
        one backed by RAPL, and collapsing them into a single unlabelled
        column makes that distinction unrecoverable afterwards.
        """
        for name, avail_key, value_key in StageFuture._ENERGY_PREFERENCE:
            if s.get(avail_key) and s.get(value_key):
                return float(s[value_key]), name
        return None, "none"

    def get_timings(self) -> List[FunctionTimes]:
        """Get the timings of the future."""
        timings_list = []
        for r, s in zip(self._timings_list(), self.stats):
            host_submit_tstamp = s["host_submit_tstamp"]
            worker_start_tstamp = s["worker_start_tstamp"]
            r.cold_start = worker_start_tstamp - host_submit_tstamp

            # --- energy -----------------------------------------------------
            r.energy, r.energy_source = self._select_energy(s)
            r.energy_duration = s.get("worker_func_energy_duration", 0.0)

            r.rapl_energy_pkg = s.get("worker_func_rapl_energy_pkg", 0.0)
            r.rapl_energy_cores = s.get("worker_func_rapl_energy_cores", 0.0)
            r.rapl_available = bool(s.get("worker_func_rapl_available", False))

            r.psutil_energy_pkg = s.get("worker_func_psutil_energy_pkg", 0.0)
            r.psutil_energy_dynamic = s.get("worker_func_psutil_energy_pkg_dynamic", 0.0)
            r.psutil_energy_idle_machine = s.get(
                "worker_func_psutil_energy_pkg_idle_machine", 0.0
            )
            r.psutil_p_idle_machine_w = s.get("worker_func_psutil_p_idle_machine_w", 0.0)
            r.psutil_available = bool(s.get("worker_func_psutil_available", False))

            r.perf_energy_pkg = s.get("worker_func_perf_energy_pkg", 0.0)
            r.perf_available = bool(s.get("worker_func_perf_available", False))
            r.perf_scope = s.get("worker_func_perf_scope", "none")

            # --- utilisation -------------------------------------------------
            r.cpu_percent = s.get("worker_func_psutil_avg_cpu_percent", 0.0)
            r.proc_cpu_percent = s.get("worker_func_psutil_proc_cpu_percent", 0.0)
            r.cores_used = s.get("worker_func_psutil_cores_used", 0.0)
            r.util_share = s.get("worker_func_psutil_util_share", 0.0)

            # --- host identity ------------------------------------------------
            # worker_processor_* come from processor_info; the psutil monitor's
            # own view is the fallback when that module is not present.
            processor_info = s.get("worker_processor_info") or {}
            r.cpu_name = s.get(
                "worker_processor_name",
                processor_info.get(
                    "processor_name", s.get("worker_func_psutil_cpu_model", "Unknown")
                ),
            )
            r.cpu_brand = s.get(
                "worker_processor_brand", processor_info.get("processor_brand", "Unknown")
            )
            r.cpu_architecture = s.get(
                "worker_processor_architecture",
                processor_info.get(
                    "architecture",
                    s.get("worker_func_psutil_cpu_architecture", "Unknown"),
                ),
            )
            r.cpu_cores_physical = (
                s.get("worker_processor_cores", processor_info.get("cores", 0)) or 0
            )
            r.cpu_cores_logical = (
                s.get("worker_processor_threads", processor_info.get("threads"))
                or s.get("worker_func_psutil_n_logical", 0)
                or 0
            )
            r.tdp_ref = s.get("worker_func_psutil_cpu_tdp_ref", 0.0)
            r.tdp_source = s.get("worker_func_psutil_cpu_tdp_source", "unresolved")
            r.tdp_is_default = bool(s.get("worker_func_psutil_cpu_tdp_is_default", True))
            r.cloud_instance_type = s.get(
                "worker_cloud_instance_type",
                processor_info.get("cloud_instance_type", "unknown"),
            )

            timings_list.append(r)
        return timings_list
