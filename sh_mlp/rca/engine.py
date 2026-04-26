"""
Root Cause Analysis Engine — Module 2
DAG-traversal + PELT change-point detection + attribution scoring.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sh_mlp.contracts import (
    FailureEvent, RCAResult, CauseRecord, ChangePointRecord, Severity
)


# ── PELT change-point detection (pure NumPy — no ruptures dependency) ──

def _pelt_changepoints(series: List[float], min_size: int = 3,
                       penalty: float = 5.0) -> List[int]:
    """
    Pruned Exact Linear Time (PELT) change-point detection.
    Minimises sum of squared residuals with BIC-inspired penalty.
    Returns list of change-point indices (exclusive right boundary).
    """
    n = len(series)
    if n < min_size * 2:
        return []

    y = np.array(series, dtype=float)

    # Cumulative sums for O(1) segment cost computation
    cumsum  = np.concatenate([[0], np.cumsum(y)])
    cumsum2 = np.concatenate([[0], np.cumsum(y**2)])

    def segment_cost(start: int, end: int) -> float:
        """Negative log-likelihood cost for segment [start, end)."""
        n_seg = end - start
        if n_seg < 1:
            return 0.0
        s1 = cumsum[end]  - cumsum[start]
        s2 = cumsum2[end] - cumsum2[start]
        var = s2/n_seg - (s1/n_seg)**2
        return n_seg * np.log(max(var, 1e-10))

    # Dynamic programming
    F    = np.full(n + 1, np.inf)
    F[0] = -penalty
    prev = [-1] * (n + 1)
    candidates = [0]

    for t in range(min_size, n + 1):
        costs = [F[s] + segment_cost(s, t) + penalty for s in candidates
                 if t - s >= min_size]
        if not costs:
            F[t] = np.inf
            continue
        best_idx = np.argmin(costs)
        F[t] = costs[best_idx]
        prev[t] = candidates[best_idx]

        # Prune candidates where F[s] + min_future_cost > F[t]
        candidates = [s for s in candidates
                      if F[s] + segment_cost(s, t) <= F[t]]
        candidates.append(t)

    # Backtrack
    cps = []
    t = n
    while prev[t] > 0:
        cps.append(prev[t])
        t = prev[t]
    return sorted(cps)


# ── Signal congruence table ───────────────────────────────────────

CONGRUENCE_TABLE: Dict[Tuple[str, str], float] = {
    # (upstream_signal_type, downstream_failure_type): score
    ("DL-1", "DL-1"): 1.0, ("DL-1", "ML-1"): 1.0, ("DL-1", "ML-2"): 0.5,
    ("DL-1", "ML-4"): 0.5, ("DL-1", "IL-1"): 0.0,
    ("DL-2", "DL-1"): 1.0, ("DL-2", "ML-1"): 1.0, ("DL-2", "ML-2"): 0.0,
    ("DL-2", "ML-4"): 0.0,
    ("IL-4", "DL-1"): 1.0, ("IL-4", "ML-1"): 1.0, ("IL-4", "ML-2"): 1.0,
    ("IL-4", "ML-4"): 1.0, ("IL-4", "IL-1"): 0.5,
    ("IL-1", "DL-1"): 1.0, ("IL-1", "ML-1"): 1.0, ("IL-1", "ML-2"): 0.5,
    ("IL-1", "ML-4"): 0.5, ("IL-1", "IL-1"): 1.0,
    ("ML-3", "ML-1"): 1.0, ("ML-3", "ML-2"): 0.5, ("ML-3", "ML-4"): 1.0,
}


def _congruence_score(upstream_type: str, downstream_type: str) -> float:
    return CONGRUENCE_TABLE.get((upstream_type, downstream_type), 0.3)


# ── Pipeline DAG ──────────────────────────────────────────────────

class PipelineDAG:
    """Directed acyclic graph of pipeline stages (data flows downstream)."""

    def __init__(self):
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}

    def add_edge(self, parent: str, child: str):
        self._parents.setdefault(child, []).append(parent)
        self._children.setdefault(parent, []).append(child)
        self._parents.setdefault(parent, [])
        self._children.setdefault(child, [])

    def add_stage(self, stage_id: str):
        self._parents.setdefault(stage_id, [])
        self._children.setdefault(stage_id, [])

    def get_parent_stages(self, stage_id: str) -> List[str]:
        return self._parents.get(stage_id, [])

    def all_stages(self) -> List[str]:
        return list(set(list(self._parents.keys()) + list(self._children.keys())))

    @classmethod
    def linear(cls, stage_ids: List[str]) -> "PipelineDAG":
        """Convenience: build a linear DAG from ordered stage list."""
        dag = cls()
        for i, sid in enumerate(stage_ids):
            dag.add_stage(sid)
            if i > 0:
                dag.add_edge(stage_ids[i-1], sid)
        return dag


# ── Signal History ────────────────────────────────────────────────

class SignalHistory:
    """Stores per-stage time series of statistical signals for PELT."""

    LOOKBACK = 30

    def __init__(self):
        self._history: Dict[str, deque] = {}

    def record(self, stage_id: str, signal_name: str, value: float,
               timestamp: datetime = None):
        key = f"{stage_id}:{signal_name}"
        self._history.setdefault(key, deque(maxlen=self.LOOKBACK))
        self._history[key].append((timestamp or datetime.utcnow(), value))

    def get_series(self, stage_id: str, signal_name: str) -> List[Tuple]:
        key = f"{stage_id}:{signal_name}"
        return list(self._history.get(key, []))

    def record_observation(self, obs):
        """Auto-record key signals from a PipelineObservation."""
        from sh_mlp.contracts import PipelineObservation
        sid = obs.stage_id
        ts  = obs.timestamp
        self.record(sid, "exec_time_ms",  obs.exec_time_ms, ts)
        self.record(sid, "memory_rss_mb", obs.memory_rss_mb, ts)
        self.record(sid, "row_count_out", float(obs.row_count_out), ts)
        for fname, fstats in obs.feature_statistics.items():
            self.record(sid, f"feature_mean:{fname}", fstats.mean, ts)


# ── RCA Engine ────────────────────────────────────────────────────

class RCAEngine:
    """
    Root Cause Analysis Engine (Module 2).
    BFS DAG traversal + PELT change-point detection.
    """

    MAX_DEPTH = 5
    W1, W2, W3 = 0.5, 0.3, 0.2    # attribution weights

    def __init__(self, dag: PipelineDAG, history: SignalHistory):
        self.dag     = dag
        self.history = history

    def diagnose(self, event: FailureEvent) -> RCAResult:
        """
        Main diagnosis entrypoint.
        Returns RCAResult with ranked causes and narrative.
        """
        failing_stage = event.stage_id
        T_failure     = event.timestamp
        candidates: Dict[str, float] = {}
        best_cp: Optional[ChangePointRecord] = None
        depth = 0

        # BFS backwards through DAG
        queue   = deque([failing_stage])
        visited = {failing_stage}

        while queue and depth < self.MAX_DEPTH:
            current = queue.popleft()
            parents = self.dag.get_parent_stages(current)
            depth  += 1

            for parent in parents:
                if parent in visited:
                    continue
                visited.add(parent)
                queue.append(parent)

                # Run PELT on multiple signal streams for this parent
                cp, score = self._score_stage(parent, event, T_failure)
                if score > 0:
                    candidates[parent] = score
                    if cp and (best_cp is None or
                               cp.magnitude > best_cp.magnitude):
                        best_cp = cp

        # No candidates found
        if not candidates:
            return self._unknown_result(event, depth)

        # Rank and compute confidence
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_stage, top_score = ranked[0]
        total = sum(candidates.values()) + 1e-8
        confidence = min(top_score / total, 0.99)

        # Build cause records
        primary = CauseRecord(
            stage_id=top_stage,
            attribution_score=top_score,
            signal_type=event.failure_type,
            change_magnitude=best_cp.magnitude if best_cp else 0.0,
            temporal_delta_cycles=self._estimate_delta_cycles(top_stage, T_failure),
        )
        alternatives = [
            CauseRecord(
                stage_id=s, attribution_score=sc,
                signal_type=event.failure_type,
                change_magnitude=0.0, temporal_delta_cycles=0
            )
            for s, sc in ranked[1:3]
        ]

        narrative = self._build_narrative(primary, event, best_cp, confidence, alternatives)

        return RCAResult(
            event_ref=event.event_id,
            primary_cause=primary,
            confidence=confidence,
            narrative=narrative,
            alternatives=alternatives,
            change_point=best_cp,
            dag_traversal_depth=depth,
            evidence_scores=dict(ranked),
        )

    # ── Scoring ───────────────────────────────────────────────────

    def _score_stage(self, stage_id: str, event: FailureEvent,
                     T_failure: datetime) -> Tuple[Optional[ChangePointRecord], float]:
        """Score a candidate stage. Returns (best_change_point, attribution_score)."""
        best_cp    = None
        best_score = 0.0
        signal_names = ["exec_time_ms", "memory_rss_mb", "row_count_out"]

        # Add feature-level signals
        all_keys = [k for k in self.history._history.keys()
                    if k.startswith(f"{stage_id}:feature_mean")]
        signal_names += [k.split(":", 1)[1] for k in all_keys[:5]]

        for sig_name in signal_names:
            series_data = self.history.get_series(stage_id, sig_name)
            if len(series_data) < 6:
                continue

            timestamps = [t for t, _ in series_data]
            values     = [v for _, v in series_data]
            cps = _pelt_changepoints(values, min_size=3, penalty=3.0)

            if not cps:
                continue

            # Most recent change-point
            cp_idx = max(cps)
            if cp_idx < 1 or cp_idx >= len(values):
                continue

            before_vals = values[:cp_idx]
            after_vals  = values[cp_idx:]
            before_mean = np.mean(before_vals)
            after_mean  = np.mean(after_vals)
            magnitude   = abs(after_mean - before_mean) / (abs(before_mean) + 1e-8)

            cp_ts = timestamps[cp_idx]

            # Temporal precedence score
            delta_sec = (T_failure - cp_ts).total_seconds()
            temporal_score = 1.0 / (1.0 + max(0, delta_sec) / 60.0)

            # Signal congruence
            congruence = _congruence_score(sig_name.split(":")[0]
                                           if ":" in sig_name else "DL-1",
                                           event.failure_type)

            # Magnitude score (normalised)
            magnitude_score = min(magnitude, 1.0)

            score = (self.W1 * temporal_score +
                     self.W2 * congruence +
                     self.W3 * magnitude_score)

            if score > best_score:
                best_score = score
                best_cp = ChangePointRecord(
                    stage_id=stage_id,
                    timestamp=cp_ts,
                    signal_name=sig_name,
                    before_value=round(before_mean, 4),
                    after_value=round(after_mean, 4),
                    magnitude=round(magnitude, 4),
                    pelt_penalty=3.0,
                )

        return best_cp, best_score

    def _estimate_delta_cycles(self, stage_id: str,
                                T_failure: datetime) -> int:
        series = self.history.get_series(stage_id, "row_count_out")
        if not series:
            return 0
        cps = _pelt_changepoints([v for _, v in series])
        if not cps:
            return 0
        cp_ts = series[max(cps)][0]
        delta = (T_failure - cp_ts).total_seconds()
        return max(0, int(delta / 60))

    def _unknown_result(self, event: FailureEvent, depth: int) -> RCAResult:
        cause = CauseRecord(
            stage_id=event.stage_id,
            attribution_score=0.0,
            signal_type="UNKNOWN",
            change_magnitude=0.0,
            temporal_delta_cycles=0,
        )
        return RCAResult(
            event_ref=event.event_id,
            primary_cause=cause,
            confidence=0.0,
            narrative=(
                f"No upstream change-points detected within search depth {depth}. "
                f"Failure at stage '{event.stage_id}' may be caused by an "
                f"external factor not captured in monitored signals. "
                f"Manual investigation recommended."
            ),
            dag_traversal_depth=depth,
        )

    def _build_narrative(self, cause: CauseRecord, event: FailureEvent,
                          cp: Optional[ChangePointRecord], confidence: float,
                          alternatives: List[CauseRecord]) -> str:
        alt_str = ", ".join(f"'{a.stage_id}'" for a in alternatives) or "none"
        cp_str = (f"Signal '{cp.signal_name}' changed from "
                  f"{cp.before_value} to {cp.after_value} "
                  f"(magnitude={cp.magnitude:.3f}) at {cp.timestamp.strftime('%H:%M:%S')}"
                  if cp else "No precise change-point identified")

        return (
            f"The {event.severity.value} severity {event.failure_type} failure "
            f"at stage '{event.stage_id}' was most likely caused by upstream "
            f"stage '{cause.stage_id}'. {cp_str}, approximately "
            f"{cause.temporal_delta_cycles} pipeline cycles before the downstream "
            f"failure was observed. Attribution confidence: {confidence:.1%}. "
            f"Alternative causes considered: {alt_str}."
        )
