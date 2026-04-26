"""
Recovery Strategy Library
22 strategies across 14 failure types.
Each strategy is fully specified per Phase 2 Table 5.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Strategy:
    strategy_id:            str
    failure_type:           str
    priority:               int         # lower = higher priority (tried first)
    description:            str
    steps:                  List[Dict]
    reversible:             bool
    estimated_duration_sec: int
    max_acceptable_ttr_sec: int
    verify_criterion:       str
    precondition:           str = "always"


# ── Strategy Library ──────────────────────────────────────────────

STRATEGY_LIBRARY: List[Strategy] = [

    # DL-1: Feature distribution drift
    Strategy("S-DL1-A", "DL-1", 1,
        "Trigger retraining on 30-day rolling window",
        [{"operation": "trigger_retraining", "parameters": {"window_days": 30}}],
        True, 3600, 7200,
        "Model performance metric recovers to within 5% of pre-drift baseline"),

    Strategy("S-DL1-B", "DL-1", 2,
        "Recalibrate preprocessing scalers using recent data",
        [{"operation": "recalibrate_scalers", "parameters": {"recent_n": 1000}}],
        True, 180, 600,
        "PSI drops below 0.15 on next observation cycle"),

    Strategy("S-DL1-C", "DL-1", 3,
        "Activate fallback model trained on distribution-robust features",
        [{"operation": "activate_fallback_model", "parameters": {"model_tag": "robust"}}],
        True, 30, 120,
        "Prediction distribution stabilises within 2 observation cycles"),

    # DL-2: Schema violation
    Strategy("S-DL2-A", "DL-2", 1,
        "Halt ingestion; quarantine batch; alert upstream data owner",
        [{"operation": "halt_ingestion", "parameters": {}},
         {"operation": "quarantine_batch", "parameters": {"reason": "schema_mismatch"}},
         {"operation": "alert_upstream_owner", "parameters": {}}],
        True, 30, 300,
        "Schema matches expected schema on retry ingestion"),

    Strategy("S-DL2-B", "DL-2", 2,
        "Attempt safe schema coercion (widening casts only)",
        [{"operation": "coerce_schema", "parameters": {"mode": "widening_only"}}],
        True, 90, 300,
        "Schema diff is empty after coercion"),

    Strategy("S-DL2-C", "DL-2", 3,
        "Route to last known-good data snapshot",
        [{"operation": "restore_snapshot", "parameters": {"snapshot": "last_clean"}}],
        True, 300, 900,
        "Downstream feature statistics match snapshot baseline"),

    # DL-3: Label distribution shift
    Strategy("S-DL3-A", "DL-3", 1,
        "Retrain with class-reweighted samples",
        [{"operation": "trigger_retraining",
          "parameters": {"class_weighting": "balanced", "window_days": 14}}],
        True, 2700, 5400,
        "Class F1 scores within 5% of pre-shift baseline"),

    Strategy("S-DL3-B", "DL-3", 2,
        "Adjust decision threshold for new class prevalence",
        [{"operation": "adjust_threshold",
          "parameters": {"method": "prevalence_calibration"}}],
        True, 30, 120,
        "Precision-recall balance restored to within 0.05"),

    # DL-4: Data ingestion failure
    Strategy("S-DL4-A", "DL-4", 1,
        "Retry ingestion with exponential backoff (3 attempts)",
        [{"operation": "retry_ingestion",
          "parameters": {"max_attempts": 3, "backoff_base_sec": 5}}],
        True, 300, 900,
        "Row count within 95% of expected baseline"),

    Strategy("S-DL4-B", "DL-4", 2,
        "Fall back to last successfully ingested snapshot",
        [{"operation": "restore_snapshot", "parameters": {"snapshot": "last_ingested"}}],
        True, 120, 600,
        "Pipeline resumes processing from snapshot data"),

    # DL-5: Annotation quality
    Strategy("S-DL5-A", "DL-5", 1,
        "Quarantine affected annotation batch; trigger label cleaning",
        [{"operation": "quarantine_batch", "parameters": {"reason": "label_noise"}},
         {"operation": "trigger_label_cleaning", "parameters": {}}],
        True, 1800, 3600,
        "Noise rate estimate drops below 10% post-cleaning"),

    # ML-1: Accuracy degradation
    Strategy("S-ML1-A", "ML-1", 1,
        "Full retraining on expanded recent data window",
        [{"operation": "trigger_retraining", "parameters": {"window_days": 60}}],
        True, 5400, 10800,
        "Rolling performance metric recovers above threshold"),

    Strategy("S-ML1-B", "ML-1", 2,
        "Roll back to last model version passing performance threshold",
        [{"operation": "rollback_model", "parameters": {"to_version": "last_passing"}}],
        True, 60, 300,
        "Performance metric above configured threshold after rollback"),

    # ML-2: Concept drift
    Strategy("S-ML2-A", "ML-2", 1,
        "Incremental fine-tuning on recent confirmed labels",
        [{"operation": "incremental_finetune",
          "parameters": {"recent_n": 2000, "epochs": 5}}],
        True, 1800, 3600,
        "Concept drift detector no longer firing after fine-tuning"),

    Strategy("S-ML2-B", "ML-2", 2,
        "Full retraining with recency-weighted samples",
        [{"operation": "trigger_retraining",
          "parameters": {"recency_weight": 2.0, "window_days": 30}}],
        True, 3600, 7200,
        "ADWIN detector quiescent for 10 consecutive cycles"),

    # ML-3: Training instability
    Strategy("S-ML3-A", "ML-3", 1,
        "Retry training with halved learning rate and gradient clipping",
        [{"operation": "trigger_retraining",
          "parameters": {"lr_multiplier": 0.5, "grad_clip": 1.0}}],
        True, 3600, 7200,
        "Training loss decreasing for 10 consecutive epochs"),

    Strategy("S-ML3-B", "ML-3", 2,
        "Reduce training data window to remove outlier batches",
        [{"operation": "trigger_retraining",
          "parameters": {"window_days": 7, "drop_outlier_batches": True}}],
        True, 1200, 3600,
        "Gradient norm below 1e4 for 5 consecutive epochs"),

    # ML-4: Confidence collapse
    Strategy("S-ML4-A", "ML-4", 1,
        "Apply temperature scaling calibration on validation set",
        [{"operation": "calibrate_temperature",
          "parameters": {"val_fraction": 0.1}}],
        True, 120, 600,
        "ECE drops below 0.12 after calibration"),

    # IL-1: Memory exhaustion
    Strategy("S-IL1-A", "IL-1", 1,
        "Retry with binary-searched reduced batch size",
        [{"operation": "retry_reduced_batch",
          "parameters": {"reduction_factor": 0.5}}],
        True, 600, 1800,
        "Stage completes without OOM error"),

    Strategy("S-IL1-B", "IL-1", 2,
        "Activate gradient checkpointing and mixed-precision mode",
        [{"operation": "enable_gradient_checkpointing", "parameters": {}},
         {"operation": "enable_mixed_precision", "parameters": {}}],
        True, 60, 300,
        "Memory utilisation below 85% of allocated limit"),

    # IL-2: Execution timeout
    Strategy("S-IL2-A", "IL-2", 1,
        "Terminate stage; retry with reduced data subset",
        [{"operation": "terminate_stage", "parameters": {}},
         {"operation": "retry_with_subset",
          "parameters": {"subset_fraction": 0.5}}],
        True, 300, 900,
        "Stage completes within 2× historical mean execution time"),

    # IL-3: Dependency conflict
    Strategy("S-IL3-A", "IL-3", 1,
        "Restore pinned dependency versions; retry in isolated venv",
        [{"operation": "restore_pinned_deps", "parameters": {}},
         {"operation": "retry_in_venv", "parameters": {}}],
        True, 600, 1800,
        "Stage completes without ImportError or AttributeError"),

    # IL-4: Silent DAG failure — HITL mandatory (irreversible audit)
    Strategy("S-IL4-A", "IL-4", 1,
        "HALT pipeline; perform DAG lineage audit from last clean checkpoint",
        [{"operation": "halt_pipeline", "parameters": {}},
         {"operation": "trigger_lineage_audit",
          "parameters": {"from_checkpoint": "last_clean"}}],
        False, 3600, 14400,  # NOT reversible — triggers HITL automatically
        "Lineage audit confirms clean data origin; no corrupt outputs remain"),

    # IL-5: Resource contention
    Strategy("S-IL5-A", "IL-5", 1,
        "Defer non-critical pipelines; prioritise production inference",
        [{"operation": "defer_non_critical_pipelines", "parameters": {}},
         {"operation": "set_priority", "parameters": {"level": "production"}}],
        True, 30, 300,
        "Production pipeline execution time normalises within 2× historical mean"),
]


# ── Index for fast lookup ─────────────────────────────────────────

STRATEGIES_BY_TYPE: Dict[str, List[Strategy]] = {}
STRATEGY_BY_ID: Dict[str, Strategy] = {}

for _s in STRATEGY_LIBRARY:
    STRATEGIES_BY_TYPE.setdefault(_s.failure_type, []).append(_s)
    STRATEGY_BY_ID[_s.strategy_id] = _s

# Sort by priority within each failure type
for _ftype in STRATEGIES_BY_TYPE:
    STRATEGIES_BY_TYPE[_ftype].sort(key=lambda x: x.priority)


def get_strategies(failure_type: str) -> List[Strategy]:
    return STRATEGIES_BY_TYPE.get(failure_type, [])


def get_strategy(strategy_id: str) -> Optional[Strategy]:
    return STRATEGY_BY_ID.get(strategy_id)
