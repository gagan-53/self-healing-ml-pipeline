"""
Phase 4A Calibration Runner
============================
Runs the full calibration suite using the optimal parameters found by ThresholdTuner.
Steps:
  1. Load or compute optimal (warmup_cycles, f4_min_ratio)
  2. Inject all 14 fault types with FaultInjectorV2 (1.25× protocol)
  3. Multi-cycle injection for stateful detectors
  4. 50-cycle clean run to verify FP rate
  5. Produce structured calibration report

Usage:
    cd /home/claude && python phase4/calibration/calibration_runner.py
"""
from __future__ import annotations
import sys, os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'phase4'))

import json, time, copy
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

import sh_mlp
from sh_mlp.contracts.data_structures import (
    PipelineObservation, FeatureStats, InfraMetrics, PredictionStats
)
from sh_mlp.detection.baseline import BaselineManager
from sh_mlp.detection.detectors import DetectorRegistry
from sh_mlp.detection.engine import DetectionEngine
from sh_mlp.detection.fusion import FusionPolicy
from calibration.fusion_v2 import FusionPolicyV2
from calibration.fault_injector_v2 import FaultInjectorV2, ALL_FAULT_TYPES

STAGE_IDS = ["ingestion", "preprocessing", "feature_eng", "training", "evaluation"]
SEED = 42

# Stage mapping per fault category
_FAULT_STAGE = {
    "DL-1": "ingestion",    "DL-2": "ingestion",   "DL-3": "training",
    "DL-4": "ingestion",    "DL-5": "ingestion",
    "ML-1": "training",     "ML-2": "training",    "ML-3": "training",
    "ML-4": "training",
    "IL-1": "preprocessing","IL-2": "preprocessing","IL-3": "preprocessing",
    "IL-4": "preprocessing","IL-5": "preprocessing",
}


@dataclass
class FaultResult:
    fault_type:       str
    stage:            str
    detected:         bool
    detected_as:      str
    correct_type:     bool
    n_cycles_needed:  int
    fusion_rule:      Optional[str]
    severity:         Optional[str]
    rca_confidence:   Optional[float]
    rca_primary_cause:Optional[str]
    recovery_strategy:Optional[str]
    recovery_success: Optional[bool]
    decision_mode:    Optional[str]
    latency_ms:       float
    note:             str = ""


def make_obs(stage_id: str, rng: np.random.Generator,
             rows_in: int = 1000, rows_out: int = 1000,
             exec_ms: float = None, rss_mb: float = 512.0,
             n_samples: int = 300) -> PipelineObservation:
    # n_samples=300: reduces PSI variance from random noise (Phase 4A fix)
    fstats = {}
    for i in range(6):
        arr = rng.normal(0, 1, n_samples)
        fstats[f"f{i}"] = FeatureStats.from_array(arr)
    fstats["target"] = FeatureStats(
        mean=0.3, std=0.46, min_val=0, max_val=1,
        pct_missing=0.01, histogram=[0.7] + [0.0] * 8 + [0.3], dtype="int"
    )
    pred = PredictionStats(
        mean_confidence=0.78 + rng.normal(0, 0.01),
        entropy_mean=0.45 + rng.normal(0, 0.005),
        predicted_class_dist={"class_0": 0.65, "class_1": 0.35},
        ece=0.06
    )
    if exec_ms is None:
        exec_ms = 400.0 + rng.normal(0, 28)
    return PipelineObservation(
        pipeline_id="calibration",
        stage_id=stage_id,
        row_count_in=rows_in,
        row_count_out=rows_out,
        exec_time_ms=max(200, exec_ms),
        memory_rss_mb=max(100, rss_mb),
        feature_statistics=fstats,
        schema_hash="schema_v2_calibrated",
        prediction_stats=pred,
        infra_metrics=InfraMetrics(
            cpu_pct=35.0 + rng.normal(0, 3),
            memory_rss_mb=rss_mb,
            cluster_cpu_pct=40.0 + rng.normal(0, 5)
        ),
    )


def _make_patched_pipeline(warmup_cycles: int, f4_min_ratio: float, auto_approve: bool = True):
    """
    Build a sh_mlp pipeline with the FusionPolicyV2 patch applied.
    The patch replaces the FusionPolicy instance inside the DetectionEngine.
    """
    observer = sh_mlp.create_pipeline(
        pipeline_id="calibration",
        stage_ids=STAGE_IDS,
        warmup_cycles=warmup_cycles,
        auto_approve=auto_approve,
        auto_recover=True,
    )
    # Patch fusion policy with calibrated version
    observer._detector._fusion = FusionPolicyV2(f4_min_ratio=f4_min_ratio)
    return observer


def run_calibration(warmup_cycles: int = 50,
                    f4_min_ratio: float = 1.25,
                    clean_run_cycles: int = 50,
                    seed: int = SEED,
                    verbose: bool = True) -> Dict:
    """
    Full calibration run with optimal parameters.
    Returns structured calibration report.
    """
    rng = np.random.default_rng(seed)
    injector = FaultInjectorV2(seed=seed)

    observer = _make_patched_pipeline(warmup_cycles, f4_min_ratio)

    if verbose:
        print("=" * 65)
        print("Phase 4A — Calibration Runner")
        print(f"Config: warmup={warmup_cycles}  f4_ratio={f4_min_ratio}  seed={seed}")
        print("=" * 65)

    # ── Step 1: Warmup ────────────────────────────────────────────
    if verbose:
        print(f"\n[1] Warmup ({warmup_cycles} cycles)...", end="", flush=True)
    t0 = time.perf_counter()
    for cycle in range(warmup_cycles + 5):
        for stage in STAGE_IDS:
            obs = make_obs(stage, rng)
            observer.inject_observation(obs)
    warmup_time = time.perf_counter() - t0
    if verbose:
        print(f" done ({warmup_time:.1f}s)  warmed={observer._detector.is_warmed_up()}")

    # ── Step 2: Clean run (FP measurement) ────────────────────────
    if verbose:
        print(f"\n[2] Clean run ({clean_run_cycles} cycles)...", end="", flush=True)
    fp_events = []
    t0 = time.perf_counter()
    for _ in range(clean_run_cycles):
        for stage in STAGE_IDS:
            event = observer.inject_observation(make_obs(stage, rng))
            if event is not None:
                fp_events.append({"stage": stage, "type": event.failure_type,
                                  "rule": event.fusion_rule})
    clean_time = time.perf_counter() - t0
    fp_count = len(fp_events)
    fp_rate = fp_count / (clean_run_cycles * len(STAGE_IDS))
    if verbose:
        marker = "✓" if fp_count <= 3 else "✗"
        print(f" {marker} FP={fp_count} ({fp_rate:.1%})  [{clean_time:.1f}s]")

    # ── Step 3: Fault injection (all 14 types) ────────────────────
    if verbose:
        print(f"\n[3] Fault injection — all {len(ALL_FAULT_TYPES)} types")
    fault_results: List[FaultResult] = []

    for fault_type in ALL_FAULT_TYPES:
        stage = _FAULT_STAGE[fault_type]
        n_cycles = injector.required_cycles(fault_type)

        obs_template = make_obs(stage, rng, rows_in=1000, rows_out=1000)

        t_start = time.perf_counter()
        detected_event = None
        for cycle_i in range(n_cycles):
            obs_faulty = injector.inject(obs_template, fault_type)
            event = observer.inject_observation(obs_faulty)
            if event is not None:
                detected_event = event
                break
        latency_ms = (time.perf_counter() - t_start) * 1000

        detected = detected_event is not None
        detected_as = detected_event.failure_type if detected else "NONE"
        correct = detected_as == fault_type

        rca_conf = rca_cause = strat = success = decision = None
        if detected:
            try:
                rca = observer._rca.diagnose(detected_event)
                recovery = observer._planner.plan_and_execute(rca)
                rca_conf   = round(rca.confidence, 3)
                rca_cause  = rca.primary_cause.stage_id
                strat      = recovery.strategy_id
                success    = recovery.success
                decision   = recovery.decision_mode
            except Exception as e:
                rca_cause = f"ERROR:{e}"

        fr = FaultResult(
            fault_type=fault_type, stage=stage,
            detected=detected, detected_as=detected_as, correct_type=correct,
            n_cycles_needed=n_cycles,
            fusion_rule=detected_event.fusion_rule if detected else None,
            severity=detected_event.severity.value if detected else None,
            rca_confidence=rca_conf, rca_primary_cause=rca_cause,
            recovery_strategy=strat, recovery_success=success,
            decision_mode=decision,
            latency_ms=round(latency_ms, 2),
            note="multi-cycle" if n_cycles > 1 else ""
        )
        fault_results.append(fr)

        # 3 clean cooldown cycles
        for _ in range(3):
            observer.inject_observation(make_obs(stage, rng))

        if verbose:
            marker = "✓" if detected else "✗"
            type_marker = "✓" if correct else "~"
            print(f"  {fault_type:<6} {marker} detected={detected_as:<6}  "
                  f"type={type_marker}  cycles={n_cycles}  {latency_ms:.0f}ms")

    # ── Step 4: Summary ───────────────────────────────────────────
    n_detected = sum(1 for r in fault_results if r.detected)
    n_correct  = sum(1 for r in fault_results if r.correct_type)
    n_recovered = sum(1 for r in fault_results if r.recovery_success)

    report = {
        "phase": "4A_calibration",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "warmup_cycles": warmup_cycles,
            "f4_min_ratio": f4_min_ratio,
            "clean_run_cycles": clean_run_cycles,
            "magnitude_factor": injector.magnitude_factor,
            "seed": seed,
        },
        "clean_run": {
            "cycles": clean_run_cycles,
            "fp_count": fp_count,
            "fp_rate": round(fp_rate, 4),
            "meets_target": fp_count <= int(clean_run_cycles * 0.05),
            "fp_events": fp_events[:10],  # first 10 for logging
        },
        "fault_injection": {
            "total": len(ALL_FAULT_TYPES),
            "detected": n_detected,
            "detection_rate": round(n_detected / len(ALL_FAULT_TYPES), 3),
            "correct_type": n_correct,
            "type_precision": round(n_correct / n_detected, 3) if n_detected else 0,
            "recovered": n_recovered,
            "recovery_rate": round(n_recovered / n_detected, 3) if n_detected else 0,
        },
        "phase4a_acceptance": {
            "detection_rate_ge_90pct": (n_detected / len(ALL_FAULT_TYPES)) >= 0.90,
            "fp_rate_le_5pct": fp_rate <= 0.05,
            "all_14_types_attempted": len(fault_results) == 14,
        },
        "per_fault_results": [asdict(r) for r in fault_results],
    }

    if verbose:
        print("\n" + "=" * 65)
        print("CALIBRATION SUMMARY")
        print("=" * 65)
        print(f"  Detection rate : {n_detected}/{len(ALL_FAULT_TYPES)} "
              f"({n_detected/len(ALL_FAULT_TYPES):.1%})")
        print(f"  Type precision : {n_correct}/{n_detected} "
              f"({n_correct/n_detected:.1%})" if n_detected else "  Type precision : n/a")
        print(f"  Recovery rate  : {n_recovered}/{n_detected} "
              f"({n_recovered/n_detected:.1%})" if n_detected else "  Recovery rate  : n/a")
        print(f"  False positives: {fp_count} in {clean_run_cycles} cycles "
              f"({fp_rate:.1%})")
        print()
        for k, v in report["phase4a_acceptance"].items():
            m = "✓" if v else "✗"
            print(f"  {m} {k}: {v}")

    return report


if __name__ == "__main__":
    # Load tuner results if available, else use defaults
    tuner_path = os.path.join(_REPO_ROOT, "phase4/results/tuner_results.json")
    warmup = 50
    f4_ratio = 1.25

    if os.path.exists(tuner_path):
        with open(tuner_path) as f:
            tuner_data = json.load(f)
        optimal = tuner_data.get("optimal", {})
        warmup   = optimal.get("warmup_cycles", warmup)
        f4_ratio = optimal.get("f4_min_ratio", f4_ratio)
        print(f"Loaded optimal params from tuner: warmup={warmup}, f4_ratio={f4_ratio}")

    report = run_calibration(
        warmup_cycles=warmup,
        f4_min_ratio=f4_ratio,
        clean_run_cycles=50,
        seed=SEED,
        verbose=True
    )

    out = os.path.join(_REPO_ROOT, "phase4/results/calibration_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCalibration report saved: {out}")
