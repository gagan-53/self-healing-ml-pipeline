"""
Signal Fusion Policy
Implements the 5-rule fusion policy from Phase 2 Section 3.2.
Converts raw DetectionSignals into FailureEvents.
"""
from __future__ import annotations
from collections import deque
from typing import List, Optional

from sh_mlp.contracts import (
    DetectionSignal, FailureEvent, Severity, PipelineObservation
)

# Severity ordering
_SEV_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1,
              Severity.HIGH: 2, Severity.CRITICAL: 3}

# Layer mapping for cross-layer detection (F4)
_FAILURE_LAYER = {
    "DL-1": "DATA",  "DL-2": "DATA",  "DL-3": "DATA",
    "DL-4": "DATA",  "DL-5": "DATA",
    "ML-1": "MODEL", "ML-2": "MODEL", "ML-3": "MODEL", "ML-4": "MODEL",
    "IL-1": "INFRA", "IL-2": "INFRA", "IL-3": "INFRA",
    "IL-4": "INFRA", "IL-5": "INFRA",
}


class FusionPolicy:
    """
    Stateful fusion policy.
    Maintains a ring buffer of the last 10 signal sets.
    """

    BUFFER_SIZE = 10

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        # Track consecutive High signals per failure_type
        self._consecutive: dict = {}

    def evaluate(self, current_signals: List[DetectionSignal],
                 obs: PipelineObservation,
                 learned_score: float = 0.0,
                 learned_features: list = None,
                 pipeline_id: str = "default") -> Optional[FailureEvent]:
        """
        Apply fusion rules F1-F5.
        Returns a FailureEvent if a rule fires, else None.
        """
        self._buffer.append(current_signals)

        if not current_signals and learned_score < 0.7:
            self._consecutive.clear()
            return None

        # F1: Any CRITICAL signal → immediate event
        critical = [s for s in current_signals if s.severity == Severity.CRITICAL]
        if critical:
            return self._make_event(pipeline_id, critical[0], current_signals, "F1", obs)

        # Augment medium signals with learned layer (F3)
        medium_signals = [s for s in current_signals if s.severity == Severity.MEDIUM]
        if learned_score > 0.7 and medium_signals:
            # Escalate medium → high
            for s in medium_signals:
                s.severity = Severity.HIGH
                s.explanation += f" [escalated by learned anomaly score={learned_score:.2f}]"

        # F4: Signals from ≥2 different layers → composite High
        layers = {_FAILURE_LAYER.get(s.failure_type, "UNKNOWN")
                  for s in current_signals}
        if len(layers) >= 2 and current_signals:
            primary = max(current_signals, key=lambda s: _SEV_ORDER[s.severity])
            if _SEV_ORDER[primary.severity] >= _SEV_ORDER[Severity.MEDIUM]:
                primary.severity = Severity.HIGH
                return self._make_event(pipeline_id, primary, current_signals, "F4", obs)

        # F2: Any HIGH signal with 2 consecutive confirmations
        high_signals = [s for s in current_signals if s.severity == Severity.HIGH]
        for s in high_signals:
            key = s.failure_type
            self._consecutive[key] = self._consecutive.get(key, 0) + 1
            if self._consecutive[key] >= 2:
                self._consecutive[key] = 0
                return self._make_event(pipeline_id, s, current_signals, "F2", obs)

        # Reset consecutive counters for types not seen this cycle
        seen_types = {s.failure_type for s in current_signals}
        for key in list(self._consecutive.keys()):
            if key not in seen_types:
                self._consecutive[key] = 0

        # F5: Single MEDIUM signal — accumulate; escalate after 5 cycles
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
        return FailureEvent(
            pipeline_id=pipeline_id,
            failure_type=primary.failure_type,
            severity=primary.severity,
            stage_id=primary.stage_id,
            contributing_signals=all_signals,
            fusion_rule=rule,
            observation_ref=obs.observation_id,
        )
