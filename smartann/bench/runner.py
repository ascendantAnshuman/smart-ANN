"""The query loop: telemetry in, decision, execution, record out.

One loop, used identically by every policy, so a policy comparison differs only
in the policy. The rolling window and the record construction are shared code
precisely so neither can be accidentally specialised for one arm of the
comparison.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from smartann.bench.writer import RunWriter
from smartann.controller.base import Policy
from smartann.index.ecovector import EcoVectorIndex, ProbeResult
from smartann.telemetry.monitor import DeviceMonitor
from smartann.types import (
    AnnConfig,
    Constraints,
    ControllerState,
    PerfHistory,
    QueryRecord,
)

__all__ = ["RollingWindow", "run_queries"]


class RollingWindow:
    """Recent latencies and energies, summarised for the controller.

    A deque rather than a full history: the controller is meant to react to the
    device's *current* state, and a mean over the whole run would keep reporting
    a quiet-phase p95 long after contention started. The window length is the
    controller's reaction-time constant and belongs in config.
    """

    def __init__(self, size: int = 50, ewma_alpha: float = 0.2) -> None:
        self.size = size
        self.alpha = ewma_alpha
        self.latencies: deque[float] = deque(maxlen=size)
        self.arrivals: deque[float] = deque(maxlen=size)
        self.ewma_latency = 0.0
        self.ewma_energy = 0.0
        self.violations = 0
        self.total = 0

    def observe(self, latency_ms: float, energy_mj: float, t_wall: float, target_ms: float) -> None:
        self.latencies.append(latency_ms)
        self.arrivals.append(t_wall)
        self.ewma_latency = (
            latency_ms
            if self.total == 0
            else self.alpha * latency_ms + (1 - self.alpha) * self.ewma_latency
        )
        self.ewma_energy = (
            energy_mj
            if self.total == 0
            else self.alpha * energy_mj + (1 - self.alpha) * self.ewma_energy
        )
        self.total += 1
        if latency_ms > target_ms:
            self.violations += 1

    def snapshot(self) -> PerfHistory:
        if not self.latencies:
            return PerfHistory(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        arr = np.asarray(self.latencies, dtype=np.float64)
        gaps = np.diff(np.asarray(self.arrivals, dtype=np.float64))
        return PerfHistory(
            ewma_latency_ms=self.ewma_latency,
            p50_latency_ms=float(np.percentile(arr, 50)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
            ewma_energy_mj=self.ewma_energy,
            slo_violation_rate=self.violations / self.total if self.total else 0.0,
            queries_in_window=len(arr),
            interarrival_ewma_s=float(gaps.mean()) if gaps.size else 0.0,
        )


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    index_variant: str
    index_config_hash: str
    cache_state: str = "warm"


def run_queries(
    index: EcoVectorIndex,
    queries: np.ndarray,
    policy: Policy,
    constraints: Constraints,
    monitor: DeviceMonitor,
    writer: RunWriter,
    spec: RunSpec,
    *,
    gt_ids: np.ndarray | None = None,
    gt_dists: np.ndarray | None = None,
    duration_s: float | None = None,
    max_queries: int | None = None,
    window: RollingWindow | None = None,
) -> int:
    """Run queries until *duration_s* elapses or *max_queries* are served.

    Duration rather than a fixed count, because the comparison is against
    wall-clock: under contention an adaptive policy answers *more* queries in
    the same window precisely because it cut its effort, and a fixed query count
    would hide that by giving the two arms different amounts of time.
    """
    from smartann.bench.metrics import recall_at_k_batch

    if duration_s is None and max_queries is None:
        raise ValueError("give run_queries a duration_s or a max_queries bound")

    # NOTHING ELSE MAY RUN WHILE THIS LOOP IS TIMING. Three separate runs in
    # this project were silently corrupted by concurrent work -- twice by another
    # experiment, once by this project's own figure generation, which perturbed
    # the latencies it was plotting. Analysis and experiments are sequenced,
    # never overlapped; scripts/phase2_acceptance.py enforces the same rule with
    # assert_machine_quiet().
    window = window or RollingWindow()
    # The widest probe any decision could use, so one probe serves every choice.
    probe_ann = _widest_ann(policy, index)
    started = time.time()
    n = 0
    qi = 0

    while True:
        if duration_s is not None and time.time() - started >= duration_s:
            break
        if max_queries is not None and n >= max_queries:
            break

        q = queries[qi % queries.shape[0]]
        device = monitor.latest()

        # Probe at the widest setting the policy could ask for, ONCE. The probe
        # only searches the resident centroid graph and costs ~0.08 ms
        # (ADR-0025), so probing wide and then choosing how many of those
        # clusters to actually open is cheaper than deciding blind -- and it
        # lets the policy see this query's features rather than the previous
        # query's. Deciding before the probe would either need lagged features
        # or a second decide() call, and calling decide() twice would advance
        # the AIMD state twice per query.
        t_probe0 = time.perf_counter()
        probe = index.probe(q, probe_ann)
        t_probe_ms = (time.perf_counter() - t_probe0) * 1000.0

        state = ControllerState(device=device, query=probe.features, history=window.snapshot())
        decision = policy.decide_timed(state, constraints)

        # Visit only the nearest `nprobe` of the clusters the probe returned.
        want = min(decision.config.ann.nprobe, probe.cluster_ids.size)
        narrowed = ProbeResult(
            cluster_ids=probe.cluster_ids[:want],
            centroid_dists=probe.centroid_dists[:want],
            features=probe.features,
            t_centroid_ms=probe.t_centroid_ms,
            n_dist_modeled=probe.n_dist_modeled,
        )
        result = index.finish(q, narrowed, decision.config.ann)
        t_wall = time.time()

        recall = recall_idx = None
        if gt_ids is not None and gt_dists is not None and result.ids.size:
            row = qi % queries.shape[0]
            k = decision.config.ann.k
            ids_pad = np.full((1, k), -1, dtype=np.int64)
            d_pad = np.full((1, k), np.inf, dtype=np.float64)
            m = min(result.ids.size, k)
            ids_pad[0, :m], d_pad[0, :m] = result.ids[:m], result.dists[:m]
            se, ov = recall_at_k_batch(
                ids_pad, d_pad, gt_ids[row : row + 1], gt_dists[row : row + 1], k=k
            )
            recall, recall_idx = float(se[0]), float(ov[0])

        total_ms = result.t_total + t_probe_ms
        window.observe(total_ms, result_energy(result), t_wall, constraints.latency_target_ms)

        record = QueryRecord(
            query_id=f"q{qi % queries.shape[0]}",
            run_id=writer.run_id,
            t_wall=t_wall,
            dataset=spec.dataset,
            index_variant=spec.index_variant,
            index_config_hash=spec.index_config_hash,
            cache_state=spec.cache_state,
            config=decision.config,
            policy_name=policy.name,
            controller_overhead_us=decision.overhead_us,
            retrieved_ids=result.ids.tolist(),
            retrieved_dists=result.dists.tolist(),
            recall_at_k=recall,
            recall_at_k_index=recall_idx,
            recall_proxy=0.0,
            t_embed=0.0,
            t_centroid=result.t_centroid,
            t_cluster_load=result.t_cluster_load,
            t_cluster_search=result.t_cluster_search,
            t_merge=result.t_merge,
            t_scr=0.0,
            t_generate=0.0,
            t_total=total_ms,
            n_distance_computations=result.n_distance_computations,
            n_dist_exact=result.n_dist_exact,
            n_dist_modeled=result.n_dist_modeled,
            n_flat_clusters_scanned=result.n_flat_clusters_scanned,
            n_cluster_loads=result.n_cluster_loads,
            bytes_read=result.bytes_read,
            energy_mj_model=0.0,
            # No energy sensor on this platform (ADR-0008). None, never a model
            # value, so nothing downstream can read a prediction as a reading.
            energy_mj_measured=None,
            peak_rss_mb=result.peak_rss_mb,
            anon_rss_mb=result.anon_rss_mb,
            device=device,
            context_tokens=None,
            answer=None,
            em=None,
            f1=None,
        )
        writer.append(record)
        policy.update(record)
        n += 1
        qi += 1

    return n


def result_energy(_result: object) -> float:
    """Placeholder energy for the rolling window.

    Returns 0.0 rather than a modelled value: Phase 3 specifies the energy
    model and Phase 3.5 fits it against the phone. A number here would flow into
    `ewma_energy_mj` and look like a measurement in the parquet.
    """
    return 0.0


def _widest_ann(policy: Policy, index: EcoVectorIndex) -> AnnConfig:
    """The broadest probe any decision from this policy could need.

    Read from the policy's own bounds where it has them, so the probe is never
    narrower than a decision that follows it -- which would silently cap the
    controller's reach and make an adaptive run look like it chose less effort
    than it did.
    """
    upper = getattr(getattr(policy, "cfg", None), "nprobe_max", None)
    if upper is None:  # static policy: its one config is its own upper bound
        upper = getattr(getattr(policy, "config", None), "ann", None)
        upper = upper.nprobe if upper is not None else 64
    live = int(np.count_nonzero(index.sizes > 0))
    n = min(int(upper), live)
    return AnnConfig(nprobe=n, ef_centroid=max(128, n), ef_list=64, k=10, per_cluster_k=32)
