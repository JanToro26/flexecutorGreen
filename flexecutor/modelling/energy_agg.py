"""

  * Independent per-worker meters (AWS Lambda: one microVM per worker,
    psutil/psutil_model source) -> SUM across workers.
  * One shared whole-machine meter (k8s RAPL: every worker on the node reads
    the same package counters) -> MAX across workers. 
    
"""

# perf stat -a is system-wide: every worker on a host reports the same machine
# energy, so it is a shared meter like RAPL.
SHARED_METER_SOURCES = {"rapl", "perf"}


def _is_shared_meter(sources) -> bool:
    if not sources:
        return False
    vals = [str(s).lower() for s in sources if s]
    return bool(vals) and all(v in SHARED_METER_SOURCES for v in vals)


def stage_energy(values, sources) -> float:
    """Combine one repetition's per-worker energies into a stage total."""
    return float(max(values)) if _is_shared_meter(sources) else float(sum(values))