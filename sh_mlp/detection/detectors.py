"""
Rule-Based Detectors — all 14 failure types from the Phase 2 taxonomy.
Each detector is a stateless callable: (obs, baseline) -> DetectionSignal | None
"""
from __future__ import annotations
import numpy as np
from scipy import stats as scipy_stats
from typing import Optional

from sh_mlp.contracts import (
    PipelineObservation, BaselineProfile, DetectionSignal, Severity, DetectorMode
)


# ── Helper utilities ──────────────────────────────────────────────

def _clip_proportions(arr: np.ndarray, min_val: float = 1e-4) -> np.ndarray:
    """Clip zero proportions to avoid log(0) in PSI calculation."""
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, min_val, None)
    return arr / arr.sum()


def _psi(baseline_hist: list, observed_hist: list) -> float:
    """Population Stability Index: Σ(P-Q)*ln(P/Q)."""
    P = _clip_proportions(np.array(baseline_hist))
    Q = _clip_proportions(np.array(observed_hist))
    return float(np.sum((P - Q) * np.log(P / Q)))


def _signal(failure_type: str, severity: Severity, stage_id: str,
            test: str, value: float, threshold: float, explanation: str,
            feature: Optional[str] = None) -> DetectionSignal:
    return DetectionSignal(
        failure_type=failure_type, severity=severity, stage_id=stage_id,
        test_statistic=test, observed_value=round(value, 6),
        threshold=threshold, explanation=explanation,
        feature_name=feature, detector_mode=DetectorMode.RULE_BASED,
    )


# ── DL-1: Feature Distribution Drift ─────────────────────────────

class FeatureDriftDetector:
    """PSI on each feature histogram. O(k) in number of features."""

    THRESHOLD_MEDIUM = 0.20
    THRESHOLD_HIGH   = 0.25

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        max_psi = 0.0
        worst   = None

        for fname, fstats in obs.feature_statistics.items():
            bl = baseline.feature_baselines.get(fname)
            if bl is None or not bl.histogram:
                continue
            psi = _psi(bl.histogram, fstats.histogram)
            if psi > max_psi:
                max_psi = psi
                worst   = fname

        if worst is None or max_psi < self.THRESHOLD_MEDIUM:
            return None

        sev = Severity.HIGH if max_psi > self.THRESHOLD_HIGH else Severity.MEDIUM
        return _signal(
            "DL-1", sev, obs.stage_id, "PSI", max_psi,
            self.THRESHOLD_HIGH if sev == Severity.HIGH else self.THRESHOLD_MEDIUM,
            f"Feature '{worst}' PSI={max_psi:.4f} exceeds {sev.value} threshold",
            worst
        )


# ── DL-2: Schema Violation ────────────────────────────────────────

class SchemaViolationDetector:
    """Hash-based schema diff. O(k) schema comparison."""

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        expected = baseline.schema_hash.get(obs.stage_id)
        if expected is None or obs.schema_hash == "" or obs.schema_hash == expected:
            return None
        return _signal(
            "DL-2", Severity.CRITICAL, obs.stage_id, "SCHEMA_DIFF",
            1.0, 0.0,
            f"Schema hash mismatch at stage '{obs.stage_id}': "
            f"expected={expected[:8]} got={obs.schema_hash[:8]}"
        )


# ── DL-3: Label Distribution Shift ───────────────────────────────

class LabelShiftDetector:
    """Jensen-Shannon divergence on predicted class distribution."""

    THRESHOLD_JS = 0.10

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        if obs.prediction_stats is None:
            return None
        pred = obs.prediction_stats.predicted_class_dist
        if not pred:
            return None

        # We use uniform as a proxy if no baseline class distribution stored
        n_classes = len(pred)
        if n_classes < 2:
            return None

        P = np.array(list(pred.values()), dtype=float)
        P = _clip_proportions(P)
        Q = np.ones(n_classes) / n_classes   # uniform baseline

        M = 0.5 * (P + Q)
        js = float(0.5 * np.sum(P * np.log(P / M)) + 0.5 * np.sum(Q * np.log(Q / M)))

        if js < self.THRESHOLD_JS:
            return None
        sev = Severity.HIGH if js > 0.20 else Severity.MEDIUM
        return _signal(
            "DL-3", sev, obs.stage_id, "JS_DIVERGENCE", js, self.THRESHOLD_JS,
            f"Label distribution JS divergence={js:.4f} exceeds threshold {self.THRESHOLD_JS}"
        )


# ── DL-4: Data Ingestion Failure ──────────────────────────────────

class IngestionFailureDetector:
    """Row count deviation and exit code check. O(1)."""

    ROW_RATIO_THRESHOLD = 0.90

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        # Hard failure — non-zero exit code
        if obs.exit_code != 0:
            return _signal(
                "DL-4", Severity.HIGH, obs.stage_id, "EXIT_CODE",
                float(obs.exit_code), 0.0,
                f"Stage '{obs.stage_id}' exited with code {obs.exit_code}"
            )

        # Soft failure — row count drop
        expected_rows = baseline.row_count_mean.get(obs.stage_id, 0)
        if expected_rows < 10:
            return None

        ratio = obs.row_count_out / expected_rows
        if ratio < self.ROW_RATIO_THRESHOLD:
            sev = Severity.CRITICAL if ratio < 0.5 else Severity.HIGH
            return _signal(
                "DL-4", sev, obs.stage_id, "ROW_COUNT_RATIO", ratio,
                self.ROW_RATIO_THRESHOLD,
                f"Row count ratio={ratio:.3f} (got {obs.row_count_out}, "
                f"expected ~{int(expected_rows)})"
            )
        return None


# ── DL-5: Annotation Quality Degradation ─────────────────────────

class AnnotationQualityDetector:
    """Proxy: high missing rate on label column as annotation quality signal."""

    NOISE_THRESHOLD = 0.15

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        label_features = [f for f in obs.feature_statistics if "label" in f.lower()
                          or "target" in f.lower() or "y" == f.lower()]
        if not label_features:
            return None

        for fname in label_features:
            fstats = obs.feature_statistics[fname]
            bl = baseline.feature_baselines.get(fname)
            if bl is None:
                continue
            delta_missing = fstats.pct_missing - bl.pct_missing
            if delta_missing > self.NOISE_THRESHOLD:
                return _signal(
                    "DL-5", Severity.MEDIUM, obs.stage_id, "LABEL_NOISE_PROXY",
                    delta_missing, self.NOISE_THRESHOLD,
                    f"Label column '{fname}' missing rate increased by "
                    f"{delta_missing:.1%} above baseline",
                    fname
                )
        return None


# ── ML-1: Accuracy Degradation ────────────────────────────────────

class AccuracyDegradationDetector:
    """Rolling confidence score drop as accuracy proxy. O(1)."""

    DROP_THRESHOLD = 0.10   # 10% relative drop

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        if obs.prediction_stats is None:
            return None
        current = obs.prediction_stats.mean_confidence
        # Use a fixed reference of 0.75 if no baseline prediction available
        reference = 0.75
        if current <= 0 or reference <= 0:
            return None

        drop = (reference - current) / reference
        if drop > self.DROP_THRESHOLD:
            sev = Severity.HIGH if drop > 0.20 else Severity.MEDIUM
            return _signal(
                "ML-1", sev, obs.stage_id, "CONFIDENCE_DROP", drop,
                self.DROP_THRESHOLD,
                f"Mean prediction confidence dropped {drop:.1%} relative to reference"
            )
        return None


# ── ML-2: Concept Drift (ADWIN proxy) ────────────────────────────

class ConceptDriftDetector:
    """
    Lightweight ADWIN-inspired detector using a two-window mean comparison.
    Full ADWIN requires ruptures library (not available offline);
    this implements the core two-window statistical test.
    """

    Z_THRESHOLD = 2.5   # z-score equivalent threshold

    def __init__(self):
        self._short_window: list = []   # last 5 entropy values
        self._long_window:  list = []   # last 20 entropy values
        self.SHORT_N = 5
        self.LONG_N  = 20

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        if obs.prediction_stats is None:
            return None
        entropy = obs.prediction_stats.entropy_mean

        self._long_window.append(entropy)
        self._short_window.append(entropy)
        if len(self._long_window) > self.LONG_N:
            self._long_window.pop(0)
        if len(self._short_window) > self.SHORT_N:
            self._short_window.pop(0)

        if len(self._long_window) < self.LONG_N:
            return None

        mu_long  = np.mean(self._long_window)
        mu_short = np.mean(self._short_window)
        std_long = np.std(self._long_window) + 1e-8

        z = abs(mu_short - mu_long) / std_long
        if z > self.Z_THRESHOLD:
            return _signal(
                "ML-2", Severity.HIGH, obs.stage_id, "ADWIN_PROXY_Z", z,
                self.Z_THRESHOLD,
                f"Concept drift detected: prediction entropy z-score={z:.2f} "
                f"(short_mean={mu_short:.3f}, long_mean={mu_long:.3f})"
            )
        return None


# ── ML-3: Training Instability ────────────────────────────────────

class TrainingInstabilityDetector:
    """Detects training loss divergence and gradient norm spikes. O(1)."""

    GRAD_NORM_THRESHOLD = 1e4
    LOSS_STAGNATION_EPOCHS = 5

    def __init__(self):
        self._loss_history: list = []

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        # Check for gradient norm in infra metrics (injected as custom metric)
        grad_norm = obs.infra_metrics.__dict__.get("grad_norm", None)
        if grad_norm and grad_norm > self.GRAD_NORM_THRESHOLD:
            return _signal(
                "ML-3", Severity.HIGH, obs.stage_id, "GRAD_NORM",
                grad_norm, self.GRAD_NORM_THRESHOLD,
                f"Gradient norm {grad_norm:.2e} exceeds threshold {self.GRAD_NORM_THRESHOLD:.0e}"
            )

        # Training loss from feature stats (injected as "train_loss" feature)
        if "train_loss" in obs.feature_statistics:
            loss = obs.feature_statistics["train_loss"].mean
            self._loss_history.append(loss)
            if len(self._loss_history) > self.LOSS_STAGNATION_EPOCHS + 1:
                self._loss_history.pop(0)
            if len(self._loss_history) >= self.LOSS_STAGNATION_EPOCHS:
                recent = self._loss_history[-self.LOSS_STAGNATION_EPOCHS:]
                if recent[-1] >= recent[0]:  # non-decreasing loss
                    return _signal(
                        "ML-3", Severity.HIGH, obs.stage_id, "LOSS_STAGNATION",
                        recent[-1], recent[0],
                        f"Training loss non-decreasing over {self.LOSS_STAGNATION_EPOCHS} "
                        f"epochs: {recent[0]:.4f} -> {recent[-1]:.4f}"
                    )
        return None


# ── ML-4: Prediction Confidence Collapse ─────────────────────────

class ConfidenceCollapseDetector:
    """ECE threshold and entropy collapse detection. O(n) predictions."""

    ECE_THRESHOLD = 0.15

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        if obs.prediction_stats is None:
            return None
        ece = obs.prediction_stats.ece
        if ece > self.ECE_THRESHOLD:
            return _signal(
                "ML-4", Severity.MEDIUM, obs.stage_id, "ECE",
                ece, self.ECE_THRESHOLD,
                f"Expected Calibration Error ECE={ece:.4f} exceeds threshold {self.ECE_THRESHOLD}"
            )
        return None


# ── IL-1: Memory Exhaustion ───────────────────────────────────────

class MemoryExhaustionDetector:
    """RSS and GPU memory utilisation. O(1)."""

    RSS_PCT_THRESHOLD  = 0.90
    GPU_PCT_THRESHOLD  = 0.95
    MAX_RSS_MB_DEFAULT = 8192.0   # assume 8GB unless configured

    def __init__(self, max_rss_mb: float = MAX_RSS_MB_DEFAULT,
                 max_gpu_mb: float = 8192.0):
        self.max_rss_mb = max_rss_mb
        self.max_gpu_mb = max_gpu_mb

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        rss_pct = obs.memory_rss_mb / self.max_rss_mb
        if rss_pct > self.RSS_PCT_THRESHOLD:
            sev = Severity.CRITICAL if rss_pct > 0.98 else Severity.HIGH
            return _signal(
                "IL-1", sev, obs.stage_id, "RSS_UTILISATION", rss_pct,
                self.RSS_PCT_THRESHOLD,
                f"RSS memory utilisation={rss_pct:.1%} "
                f"({obs.memory_rss_mb:.0f}MB / {self.max_rss_mb:.0f}MB)"
            )

        if obs.memory_gpu_mb is not None:
            gpu_pct = obs.memory_gpu_mb / self.max_gpu_mb
            if gpu_pct > self.GPU_PCT_THRESHOLD:
                return _signal(
                    "IL-1", Severity.HIGH, obs.stage_id, "GPU_MEMORY", gpu_pct,
                    self.GPU_PCT_THRESHOLD,
                    f"GPU memory utilisation={gpu_pct:.1%}"
                )
        return None


# ── IL-2: Execution Timeout ───────────────────────────────────────

class ExecutionTimeoutDetector:
    """Wall-clock time vs. historical mean + 3σ. O(1)."""

    SIGMA_THRESHOLD = 3.0

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        mu  = baseline.exec_time_mean.get(obs.stage_id, 0)
        sig = baseline.exec_time_std.get(obs.stage_id, 1)
        if mu <= 0:
            return None

        z = (obs.exec_time_ms - mu) / sig
        if z > self.SIGMA_THRESHOLD:
            sev = Severity.HIGH if z > 5.0 else Severity.MEDIUM
            return _signal(
                "IL-2", sev, obs.stage_id, "EXEC_TIME_Z", z, self.SIGMA_THRESHOLD,
                f"Execution time {obs.exec_time_ms:.0f}ms is {z:.1f}σ above mean "
                f"({mu:.0f}ms ± {sig:.0f}ms)"
            )
        return None


# ── IL-3: Dependency Conflict ─────────────────────────────────────

class DependencyConflictDetector:
    """Checks for exit code 1 with ImportError signature. O(1)."""

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        # Dependency conflicts manifest as exit code 1 at import time
        # In production: parse stage stderr for ImportError / AttributeError
        dep_flag = obs.infra_metrics.__dict__.get("dependency_conflict", False)
        if dep_flag:
            return _signal(
                "IL-3", Severity.HIGH, obs.stage_id, "DEP_CONFLICT", 1.0, 0.0,
                f"Dependency version conflict detected at stage '{obs.stage_id}'"
            )
        return None


# ── IL-4: Silent DAG Propagation Failure ─────────────────────────

class SilentDAGFailureDetector:
    """Cross-stage volume ratio. Most critical failure type. O(k)."""

    RATIO_THRESHOLD = 0.85

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        if obs.row_count_in <= 0:
            return None

        ratio = obs.row_count_out / obs.row_count_in
        if ratio < self.RATIO_THRESHOLD and obs.exit_code == 0:
            # Completed successfully but with unexpected data reduction
            sev = Severity.CRITICAL if ratio < 0.5 else Severity.HIGH
            return _signal(
                "IL-4", sev, obs.stage_id, "CROSS_STAGE_VOLUME", ratio,
                self.RATIO_THRESHOLD,
                f"Silent data loss: {obs.row_count_in} rows in → {obs.row_count_out} "
                f"out (ratio={ratio:.3f}) at stage '{obs.stage_id}' with exit_code=0"
            )
        return None


# ── IL-5: Resource Contention ─────────────────────────────────────

class ResourceContentionDetector:
    """Cluster CPU utilisation sustained above 90%. O(1)."""

    CPU_THRESHOLD = 0.90

    def detect(self, obs: PipelineObservation,
               baseline: BaselineProfile) -> Optional[DetectionSignal]:
        cpu = obs.infra_metrics.cluster_cpu_pct / 100.0
        if cpu > self.CPU_THRESHOLD:
            return _signal(
                "IL-5", Severity.MEDIUM, obs.stage_id, "CLUSTER_CPU", cpu,
                self.CPU_THRESHOLD,
                f"Cluster CPU utilisation={cpu:.1%} exceeds threshold {self.CPU_THRESHOLD:.0%}"
            )
        return None


# ── Registry ──────────────────────────────────────────────────────

class DetectorRegistry:
    """Instantiates and holds all 14 rule-based detectors."""

    def __init__(self, max_rss_mb: float = 8192.0, max_gpu_mb: float = 8192.0):
        self.detectors = [
            FeatureDriftDetector(),
            SchemaViolationDetector(),
            LabelShiftDetector(),
            IngestionFailureDetector(),
            AnnotationQualityDetector(),
            AccuracyDegradationDetector(),
            ConceptDriftDetector(),
            TrainingInstabilityDetector(),
            ConfidenceCollapseDetector(),
            MemoryExhaustionDetector(max_rss_mb, max_gpu_mb),
            ExecutionTimeoutDetector(),
            DependencyConflictDetector(),
            SilentDAGFailureDetector(),
            ResourceContentionDetector(),
        ]

    def run_all(self, obs: PipelineObservation,
                baseline: BaselineProfile) -> list:
        """Run all detectors; return list of non-None signals."""
        signals = []
        for d in self.detectors:
            try:
                sig = d.detect(obs, baseline)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                # Detectors must never crash the pipeline
                pass
        return signals
