"""
SH-MLP Interface Contracts
All 8 typed data structures from Phase 2 Section 7.
These are the ONLY communication channel between modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


# ── Enumerations ──────────────────────────────────────────────────

class Severity(Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    def ordinal(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[self.value]


class DecisionMode(Enum):
    AUTONOMOUS      = "AUTONOMOUS"
    SUPERVISED      = "SUPERVISED"
    HITL            = "HITL"
    ESCALATE        = "ESCALATE"
    IMMEDIATE_HALT  = "IMMEDIATE_HALT"
    DEFERRED        = "DEFERRED"


class DetectorMode(Enum):
    RULE_BASED = "RULE_BASED"
    LEARNED    = "LEARNED"


# ── Sub-structures ────────────────────────────────────────────────

@dataclass
class FeatureStats:
    """Statistical profile of a single feature in one observation."""
    mean:          float
    std:           float
    min_val:       float
    max_val:       float
    pct_missing:   float
    histogram:     List[float]   # 10-bin normalised histogram for PSI
    dtype:         str

    @classmethod
    def from_array(cls, arr, bins: int = 10) -> "FeatureStats":
        import numpy as np
        a = np.asarray(arr, dtype=float)
        valid = a[~np.isnan(a)]
        if len(valid) == 0:
            return cls(0, 0, 0, 0, 1.0, [0.1]*bins, "float")
        hist, _ = np.histogram(valid, bins=bins)
        hist = hist / hist.sum() if hist.sum() > 0 else hist.astype(float)
        return cls(
            mean=float(np.mean(valid)),
            std=float(np.std(valid)),
            min_val=float(np.min(valid)),
            max_val=float(np.max(valid)),
            pct_missing=float(np.isnan(a).mean()),
            histogram=hist.tolist(),
            dtype=str(arr.dtype) if hasattr(arr, "dtype") else "float"
        )


@dataclass
class PredictionStats:
    """Statistics over model output predictions."""
    mean_confidence:    float
    entropy_mean:       float
    predicted_class_dist: Dict[str, float]  # class_label -> proportion
    ece:               float = 0.0          # Expected Calibration Error


@dataclass
class InfraMetrics:
    """Cluster-level infrastructure metrics snapshot."""
    cpu_pct:        float
    memory_rss_mb:  float
    gpu_memory_mb:  Optional[float] = None
    io_wait_pct:    float = 0.0
    cluster_cpu_pct: float = 0.0


# ── Contract 1: PipelineObservation ──────────────────────────────

@dataclass
class PipelineObservation:
    """
    Collected by PipelineObserver at stage exit.
    Primary input to the Detection Engine.
    """
    pipeline_id:        str
    stage_id:           str
    observation_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:          datetime = field(default_factory=datetime.utcnow)
    cycle_number:       int = 0
    exec_time_ms:       float = 0.0
    memory_rss_mb:      float = 0.0
    memory_gpu_mb:      Optional[float] = None
    row_count_in:       int = 0
    row_count_out:      int = 0
    exit_code:          int = 0
    feature_statistics: Dict[str, FeatureStats] = field(default_factory=dict)
    schema_hash:        str = ""
    prediction_stats:   Optional[PredictionStats] = None
    infra_metrics:      InfraMetrics = field(default_factory=lambda: InfraMetrics(0,0))


# ── Contract 2: DetectionSignal ───────────────────────────────────

@dataclass
class DetectionSignal:
    """
    Produced by a single detector (rule-based or learned).
    Multiple signals are fused by FusionPolicy into a FailureEvent.
    """
    failure_type:   str           # taxonomy ID e.g. "DL-1"
    severity:       Severity
    stage_id:       str
    test_statistic: str           # e.g. "PSI", "KS", "ADWIN"
    observed_value: float
    threshold:      float
    explanation:    str
    signal_id:      str = field(default_factory=lambda: str(uuid.uuid4()))
    detector_mode:  DetectorMode = DetectorMode.RULE_BASED
    feature_name:   Optional[str] = None
    confidence:     float = 1.0
    timestamp:      datetime = field(default_factory=datetime.utcnow)


# ── Contract 3: FailureEvent ──────────────────────────────────────

@dataclass
class FailureEvent:
    """
    Emitted by FusionPolicy when enough signals warrant a response.
    Primary input to the RCA Engine.
    """
    pipeline_id:          str
    failure_type:         str
    severity:             Severity
    stage_id:             str
    contributing_signals: List[DetectionSignal]
    fusion_rule:          str       # "F1" .. "F5"
    baseline_metric:      float = 0.0
    event_id:             str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:            datetime = field(default_factory=datetime.utcnow)
    observation_ref:      str = ""


# ── Contract 4: RCAResult ─────────────────────────────────────────

@dataclass
class CauseRecord:
    stage_id:             str
    attribution_score:    float
    signal_type:          str
    change_magnitude:     float
    temporal_delta_cycles: int


@dataclass
class ChangePointRecord:
    stage_id:     str
    timestamp:    datetime
    signal_name:  str
    before_value: float
    after_value:  float
    magnitude:    float
    pelt_penalty: float = 0.0


@dataclass
class RCAResult:
    """
    Produced by RCA Engine. Primary input to Recovery Planner.
    """
    event_ref:            str       # FailureEvent.event_id
    primary_cause:        CauseRecord
    confidence:           float     # [0, 1]
    narrative:            str
    alternatives:         List[CauseRecord] = field(default_factory=list)
    change_point:         Optional[ChangePointRecord] = None
    dag_traversal_depth:  int = 0
    evidence_scores:      Dict[str, float] = field(default_factory=dict)
    rca_id:               str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:            datetime = field(default_factory=datetime.utcnow)


# ── Contract 5: RecoveryAction ────────────────────────────────────

@dataclass
class ActionStep:
    step_id:     int
    operation:   str
    parameters:  Dict[str, Any] = field(default_factory=dict)
    rollback_op: Optional[str] = None


@dataclass
class RecoveryAction:
    """
    Selected by Recovery Planner. Passed to PipelineObserver for execution.
    """
    strategy_id:           str
    failure_type:          str
    rca_ref:               str
    decision_mode:         DecisionMode
    steps:                 List[ActionStep]
    reversible:            bool
    estimated_duration_sec: int
    verify_criterion:      str
    action_id:             str = field(default_factory=lambda: str(uuid.uuid4()))
    approved_by:           Optional[str] = None
    approved_at:           Optional[datetime] = None


# ── Contract 6: RecoveryOutcome ───────────────────────────────────

@dataclass
class RecoveryOutcome:
    """
    Produced by Recovery Planner after execution + verification.
    Primary input to Feedback Store.
    """
    action_ref:        str
    failure_type:      str
    strategy_id:       str
    context_vector:    List[float]   # 12-dimensional
    success:           bool
    success_float:     float         # 1.0 | 0.5 | 0.0
    time_to_recovery:  int           # seconds
    performance_delta: float         # post-recovery metric - pre-failure baseline
    decision_mode:     str
    outcome_id:        str = field(default_factory=lambda: str(uuid.uuid4()))
    notes:             Optional[str] = None
    timestamp:         datetime = field(default_factory=datetime.utcnow)


# ── Contract 7: StrategyWeights ───────────────────────────────────

@dataclass
class ArmParameters:
    """LinUCB parameters for a single strategy arm."""
    strategy_id:    str
    A_matrix:       List[List[float]]  # 12×12 regularised design matrix
    b_vector:       List[float]        # 12-dimensional reward accumulator
    theta_vector:   List[float]        # 12-dimensional estimated parameter
    n_observations: int = 0
    total_reward:   float = 0.0
    success_count:  int = 0


@dataclass
class StrategyWeights:
    """Read by Recovery Planner; written by Feedback Store."""
    version:    int
    updated_at: datetime
    arms:       Dict[str, ArmParameters] = field(default_factory=dict)


# ── Contract 8: BaselineProfile ───────────────────────────────────

@dataclass
class FeatureBaseline:
    feature_name: str
    mean:         float
    std:          float
    histogram:    List[float]   # 10-bin normalised histogram
    pct_missing:  float
    dtype:        str


@dataclass
class BaselineProfile:
    """
    Computed during warm-up; updated by EMA.
    Read by Detection Engine.
    """
    pipeline_id:       str
    warmup_cycles:     int
    ema_alpha:         float = 0.05
    feature_baselines: Dict[str, FeatureBaseline] = field(default_factory=dict)
    exec_time_mean:    Dict[str, float] = field(default_factory=dict)
    exec_time_std:     Dict[str, float] = field(default_factory=dict)
    row_count_mean:    Dict[str, float] = field(default_factory=dict)
    schema_hash:       Dict[str, str] = field(default_factory=dict)
    created_at:        datetime = field(default_factory=datetime.utcnow)
    updated_at:        datetime = field(default_factory=datetime.utcnow)
    is_ready:          bool = False   # True after warmup_cycles completed
