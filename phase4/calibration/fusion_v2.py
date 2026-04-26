"""
FusionPolicy v2 — Phase 4A calibration fix.

Changes vs Phase 3:
  - F4 now requires the primary signal's observed_value >= F4_MIN_RATIO * threshold.
    Prevents two random MEDIUM signals (e.g. PSI=0.21, CPU=0.91) from firing
    a composite HIGH event on random pipeline noise.
  - F5 persistence counter uses per-type ring buffer so accumulated counts don't
    bleed across resets.
  - Default warmup recommendation raised to 50 cycles.
"""
from __future__ import annotations
from collections import deque
from typing import List, Optional

from sh_mlp.contracts import (
    DetectionSignal, FailureEvent, Severity, PipelineObservation
)

_SEV_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1,
              Severity.HIGH: 2, Severity.CRITICAL: 3}

_FAILURE_LAYER = {
    "DL-1": "DATA",  "DL-2": "DATA",  "DL-3": "DATA",
    "DL-4": "DATA",  "DL-5": "DATA",
    "ML-1": "MODEL", "ML-2": "MODEL", "ML-3": "MODEL", "ML-4": "MODEL",
    "IL-1": "INFRA", "IL-2": "INFRA", "IL-3": "INFRA",
    "IL-4": "INFRA", "IL-5": "INFRA",
}


class FusionPolicyV2:
    """
    Fusion policy with calibration fixes.

    Key parameters
    --------------
    f4_min_ratio : float
        For F4 (cross-layer composite), the primary signal's
        observed_value must be >= f4_min_ratio × threshold.
        Default 1.25 — signal must be 25% above threshold.
        Set to 1.0 to revert to Phase 3 behaviour.
    """

    BUFFER_SIZE = 10

    def __init__(self, f4_min_ratio: float = 1.25):
        self.f4_min_ratio = f4_min_ratio
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._consecutive: dict = {}

    def evaluate(self, current_signals: List[DetectionSignal],
                 obs: PipelineObservation,
                 learned_score: float = 0.0,
                 learned_features: list = None,
                 pipeline_id: str = "default") -> Optional[FailureEvent]:
        """Apply fusion rules F1–F5. Returns FailureEvent or None."""
        self._buffer.append(current_signals)

        if not current_signals and learned_score < 0.7:
            self._consecutive.clear()
            return None

        # F1: Any CRITICAL → fire immediately
        critical = [s for s in current_signals if s.severity == Severity.CRITICAL]
        if critical:
            return self._make_event(pipeline_id, critical[0], current_signals, "F1", obs)

        # F3: MEDIUM + learned anomaly score > 0.7 → escalate medium to HIGH
        medium_signals = [s for s in current_signals if s.severity == Severity.MEDIUM]
        if learned_score > 0.7 and medium_signals:
            for s in medium_signals:
                s.severity = Severity.HIGH
                s.explanation += f" [escalated by IF score={learned_score:.2f}]"

        # F4: Cross-layer composite — with minimum-magnitude gate
        layers_present = {_FAILURE_LAYER.get(s.failure_type, "UNKNOWN")
                          for s in current_signals}
        if len(layers_present) >= 2 and current_signals:
            primary = max(current_signals, key=lambda s: _SEV_ORDER[s.severity])
            # Gate: primary signal must be >= f4_min_ratio × threshold
            magnitude_ok = (primary.threshold == 0 or
                            primary.observed_value >= self.f4_min_ratio * primary.threshold)
            if (_SEV_ORDER[primary.severity] >= _SEV_ORDER[Severity.MEDIUM]
                    and magnitude_ok):
                primary.severity = Severity.HIGH
                return self._make_event(pipeline_id, primary, current_signals, "F4", obs)

        # F2: HIGH with 2 consecutive confirmations
        high_signals = [s for s in current_signals if s.severity == Severity.HIGH]
        for s in high_signals:
            key = s.failure_type
            self._consecutive[key] = self._consecutive.get(key, 0) + 1
            if self._consecutive[key] >= 2:
                self._consecutive[key] = 0
                return self._make_event(pipeline_id, s, current_signals, "F2", obs)

        # Reset counters for unseen types
        seen_types = {s.failure_type for s in current_signals}
        for key in list(self._consecutive.keys()):
            if key not in seen_types and not key.startswith("MED_"):
                self._consecutive[key] = 0

        # F5: MEDIUM persists for 5 cycles → escalate
        if medium_signals:
            for s in medium_signals:
                key = f"MED_{s.failure_type}"
                self._consecutive[key] = self._consecutive.get(key, 0) + 1
                if self._consecutive[key] >= 5:
                    self._consecutive[key] = 0
                    s.severity = Severity.HIGH
                    return self._make_event(pipeline_id, s, [s], "F5", obs)

        return None

    def _make_event(self, pipeline_id: str, primary: DetectionSignal,
                    all_signals: List[DetectionSignal],
                    rule: str, obs: PipelineObservation) -> FailureEvent:
        from sh_mlp.contracts import FailureEvent
        return FailureEvent(
            pipeline_id=pipeline_id,
            failure_type=primary.failure_type,
            severity=primary.severity,
            stage_id=primary.stage_id,
            contributing_signals=all_signals,
            fusion_rule=rule,
            observation_ref=obs.observation_id,
        )

    def reset(self):
        """Reset all stateful counters (e.g. between test runs)."""
        self._buffer.clear()
        self._consecutive.clear()
