"""
Pipeline Observer — System Entry Point
Wraps ML pipeline stages; collects observations; coordinates all modules.
"""
from __future__ import annotations
import time
import hashlib
import json
import numpy as np
from typing import Callable, Optional, List, Any
from datetime import datetime
from contextlib import contextmanager

from sh_mlp.contracts import (
    PipelineObservation, FeatureStats, InfraMetrics, FailureEvent
)
from sh_mlp.detection.engine import DetectionEngine
from sh_mlp.rca.engine import RCAEngine, SignalHistory
from sh_mlp.recovery.planner import RecoveryPlanner


class PipelineObserver:
    """
    Attaches to a pipeline and orchestrates all four modules.

    Usage:
        observer = PipelineObserver(pipeline_id="adult_income")
        with observer.monitor_stage("preprocessing", data_in) as ctx:
            data_out = preprocess(data_in)
            ctx.set_output(data_out)
    """

    def __init__(self, pipeline_id: str,
                 dag,
                 detection_engine: DetectionEngine,
                 rca_engine: RCAEngine,
                 recovery_planner: RecoveryPlanner,
                 signal_history: SignalHistory,
                 auto_recover: bool = True):

        self.pipeline_id   = pipeline_id
        self.dag           = dag
        self._detector     = detection_engine
        self._rca          = rca_engine
        self._planner      = recovery_planner
        self._history      = signal_history
        self.auto_recover  = auto_recover
        self._cycle        = 0
        self._event_count  = 0
        self._all_events:  List[FailureEvent] = []
        self._all_outcomes: list = []

    # ── Stage monitoring context manager ─────────────────────────

    @contextmanager
    def monitor_stage(self, stage_id: str, data_in: Any = None,
                      feature_cols: list = None):
        """
        Context manager that instruments a single pipeline stage.
        Collects timing, memory, row counts, and feature statistics.
        """
        import psutil, os
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / 1024 / 1024
        start_ms   = time.time() * 1000
        row_count_in = len(data_in) if hasattr(data_in, '__len__') else 0

        ctx = _StageContext(stage_id, data_in, feature_cols)
        try:
            yield ctx
            exit_code = 0
        except Exception as e:
            ctx._error = str(e)
            exit_code = 1
            raise
        finally:
            elapsed_ms   = time.time() * 1000 - start_ms
            mem_after    = proc.memory_info().rss / 1024 / 1024
            row_count_out = len(ctx._output) if hasattr(ctx._output, '__len__') else 0

            obs = self._build_observation(
                stage_id=stage_id,
                exec_time_ms=elapsed_ms,
                memory_rss_mb=mem_after,
                row_count_in=row_count_in,
                row_count_out=row_count_out,
                exit_code=exit_code,
                data=ctx._output if ctx._output is not None else ctx._input,
                feature_cols=feature_cols or ctx._feature_cols,
                cpu_pct=psutil.cpu_percent(interval=None),
            )

            # Record in signal history for RCA
            self._history.record_observation(obs)

            # Run detection
            if self._detector.is_warmed_up():
                event = self._detector.observe(obs)
            else:
                self._detector.observe(obs)
                event = None

            # Handle event → RCA → Recovery
            if event is not None:
                self._event_count += 1
                self._all_events.append(event)
                if self.auto_recover:
                    self._handle_event(event)

    def _handle_event(self, event: FailureEvent):
        """RCA + Recovery pipeline for a detected failure."""
        rca     = self._rca.diagnose(event)
        outcome = self._planner.plan_and_execute(rca)
        self._all_outcomes.append(outcome)

    # ── Manual observation injection (for testing) ────────────────

    def inject_observation(self, obs: PipelineObservation) -> Optional[FailureEvent]:
        """Directly inject a pre-built observation. Used in tests."""
        self._history.record_observation(obs)
        if self._detector.is_warmed_up():
            return self._detector.observe(obs)
        self._detector.observe(obs)
        return None

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "pipeline_id":    self.pipeline_id,
            "cycles":         self._cycle,
            "events_detected": self._event_count,
            "recoveries":     len(self._all_outcomes),
            "is_warmed_up":   self._detector.is_warmed_up(),
        }

    # ── Observation builder ───────────────────────────────────────

    def _build_observation(self, stage_id: str, exec_time_ms: float,
                           memory_rss_mb: float, row_count_in: int,
                           row_count_out: int, exit_code: int,
                           data: Any, feature_cols: list,
                           cpu_pct: float) -> PipelineObservation:
        self._cycle += 1
        fstats = {}
        schema_hash = ""

        if data is not None and hasattr(data, '__len__') and len(data) > 0:
            fstats, schema_hash = self._extract_feature_stats(data, feature_cols)

        return PipelineObservation(
            pipeline_id=self.pipeline_id,
            stage_id=stage_id,
            cycle_number=self._cycle,
            exec_time_ms=exec_time_ms,
            memory_rss_mb=memory_rss_mb,
            row_count_in=row_count_in,
            row_count_out=row_count_out,
            exit_code=exit_code,
            feature_statistics=fstats,
            schema_hash=schema_hash,
            infra_metrics=InfraMetrics(cpu_pct=cpu_pct, memory_rss_mb=memory_rss_mb),
        )

    def _extract_feature_stats(self, data: Any, feature_cols: list):
        """Extract FeatureStats from array-like data."""
        fstats = {}
        schema_parts = []

        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                cols = feature_cols or list(data.columns)
                for col in cols[:50]:  # cap at 50 features for overhead
                    arr = data[col].values
                    fstats[str(col)] = FeatureStats.from_array(arr)
                    schema_parts.append(f"{col}:{str(arr.dtype)}")
            elif isinstance(data, np.ndarray):
                if data.ndim == 2:
                    for i in range(min(data.shape[1], 50)):
                        name = feature_cols[i] if feature_cols and i < len(feature_cols) else f"f{i}"
                        fstats[name] = FeatureStats.from_array(data[:, i])
                        schema_parts.append(f"{name}:float")
        except Exception:
            pass

        schema_hash = hashlib.md5(json.dumps(sorted(schema_parts)).encode()).hexdigest()[:16]
        return fstats, schema_hash


class _StageContext:
    """Context object yielded by monitor_stage."""
    def __init__(self, stage_id, data_in, feature_cols):
        self.stage_id     = stage_id
        self._input       = data_in
        self._output      = None
        self._feature_cols = feature_cols or []
        self._error       = None

    def set_output(self, data):
        self._output = data
