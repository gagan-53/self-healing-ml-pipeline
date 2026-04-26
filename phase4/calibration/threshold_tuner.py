"""
Threshold Tuner — Phase 4A calibration tool.

Empirically determines:
  1. Minimum warmup cycles for EMA baseline to converge (FP reduction)
  2. Optimal f4_min_ratio for the F4 cross-layer fusion rule (FP vs detection tradeoff)

Runs a grid search over (warmup_cycles, f4_min_ratio) and reports the Pareto-optimal
configuration that maximises detection rate subject to FP_rate < 0.05.

Usage:
    python threshold_tuner.py
    # Or import:
    from phase4.calibration.threshold_tuner import run_tuner, TunerConfig
"""
from __future__ import annotations
import sys, os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'phase4'))

import json, time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict
import numpy as np

import sh_mlp
from sh_mlp.contracts.data_structures import (
    PipelineObservation, FeatureStats, InfraMetrics, PredictionStats
)
from sh_mlp.detection.engine import DetectionEngine
from sh_mlp.detection.baseline import BaselineManager
from sh_mlp.detection.detectors import DetectorRegistry
from calibration.fusion_v2 import FusionPolicyV2
from calibration.fault_injector_v2 import FaultInjectorV2

STAGE_IDS = ["ingestion", "preprocessing", "feature_eng", "training", "evaluation"]
ALL_FAULT_TYPES = [
    "DL-1", "DL-2", "DL-3", "DL-4", "DL-5",
    "ML-1", "ML-2", "ML-3", "ML-4",
    "IL-1", "IL-2", "IL-3", "IL-4", "IL-5"
]


@dataclass
class TunerConfig:
    warmup_options:   List[int]   = field(default_factory=lambda: [30, 40, 50, 60])
    f4_ratio_options: List[float] = field(default_factory=lambda: [1.0, 1.1, 1.25, 1.5, 2.0])
    clean_run_cycles: int         = 50
    seed:             int         = 42
    target_fp_rate:   float       = 0.05   # FP / clean cycles ≤ 5%


@dataclass
class TunerResult:
    warmup_cycles:    int
    f4_min_ratio:     float
    fp_count:         int
    fp_rate:          float
    detection_count:  int
    detection_rate:   float
    meets_fp_target:  bool


def _make_clean_obs(stage_id: str, rng: np.random.Generator) -> PipelineObservation:
    """Build a clean synthetic observation with small natural variation."""
    fstats = {}
    for i in range(6):
        arr = rng.normal(0, 1, 100)
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
    exec_ms = 400.0 + rng.normal(0, 28)
    rss_mb  = 512.0 + rng.normal(0, 15)
    return PipelineObservation(
        pipeline_id="tuner",
        stage_id=stage_id,
        row_count_in=1000,
        row_count_out=1000,
        exec_time_ms=max(200, exec_ms),
        memory_rss_mb=max(100, rss_mb),
        feature_statistics=fstats,
        schema_hash="schema_stable_v1",
        prediction_stats=pred,
        infra_metrics=InfraMetrics(
            cpu_pct=35.0 + rng.normal(0, 3),
            memory_rss_mb=rss_mb,
            cluster_cpu_pct=40.0 + rng.normal(0, 5)
        ),
    )


class _MinimalDetectionEngine:
    """Minimal engine wiring for tuner — uses FusionPolicyV2 with tunable params."""

    def __init__(self, warmup_cycles: int, f4_min_ratio: float):
        self._baseline = BaselineManager("tuner", warmup_cycles, ema_alpha=0.05)
        self._detectors = DetectorRegistry(max_rss_mb=8192.0)
        self._fusion = FusionPolicyV2(f4_min_ratio=f4_min_ratio)
        self._ready = False

    def observe(self, obs):
        ready = self._baseline.update(obs)
        if not ready:
            return None
        self._ready = True
        baseline = self._baseline.get()
        signals = self._detectors.run_all(obs, baseline)
        return self._fusion.evaluate(signals, obs, pipeline_id="tuner")

    def is_warmed_up(self):
        return self._ready


def _evaluate_config(warmup: int, f4_ratio: float,
                     clean_cycles: int, seed: int,
                     injector: FaultInjectorV2) -> TunerResult:
    rng = np.random.default_rng(seed)
    engine = _MinimalDetectionEngine(warmup, f4_ratio)

    # Warm up
    for _ in range(warmup + 5):
        for stage in STAGE_IDS:
            engine.observe(_make_clean_obs(stage, rng))

    # Clean run — count false positives
    fp_count = 0
    for _ in range(clean_cycles):
        for stage in STAGE_IDS:
            event = engine.observe(_make_clean_obs(stage, rng))
            if event is not None:
                fp_count += 1

    fp_rate = fp_count / (clean_cycles * len(STAGE_IDS))

    # Fault injection — count detections
    detection_count = 0
    stage_map = {
        "DL": "ingestion", "ML": "training", "IL": "preprocessing"
    }
    for fault_type in ALL_FAULT_TYPES:
        prefix = fault_type[:2]
        stage = stage_map.get(prefix, "ingestion")
        obs_clean = _make_clean_obs(stage, rng)
        obs_clean.row_count_in = 1000
        obs_clean.row_count_out = 1000

        n_cycles = injector.required_cycles(fault_type)
        detected = False
        for _ in range(n_cycles):
            obs_faulty = injector.inject(obs_clean, fault_type)
            event = engine.observe(obs_faulty)
            if event is not None:
                detected = True
                break
        if detected:
            detection_count += 1

        # 3 clean cycles cooldown
        for _ in range(3):
            engine.observe(_make_clean_obs(stage, rng))

    detection_rate = detection_count / len(ALL_FAULT_TYPES)

    return TunerResult(
        warmup_cycles=warmup,
        f4_min_ratio=f4_ratio,
        fp_count=fp_count,
        fp_rate=round(fp_rate, 4),
        detection_count=detection_count,
        detection_rate=round(detection_rate, 4),
        meets_fp_target=(fp_rate <= 0.05),
    )


def run_tuner(config: TunerConfig = None, verbose: bool = True) -> List[TunerResult]:
    """
    Run grid search over (warmup_cycles, f4_min_ratio).
    Returns all results sorted by (meets_fp_target DESC, detection_rate DESC).
    """
    if config is None:
        config = TunerConfig()

    injector = FaultInjectorV2(seed=config.seed)
    results = []

    total = len(config.warmup_options) * len(config.f4_ratio_options)
    i = 0
    for warmup in config.warmup_options:
        for f4_ratio in config.f4_ratio_options:
            i += 1
            if verbose:
                print(f"  [{i:2d}/{total}] warmup={warmup:3d}  f4_ratio={f4_ratio:.2f} ... ",
                      end="", flush=True)
            t0 = time.perf_counter()
            result = _evaluate_config(
                warmup, f4_ratio,
                config.clean_run_cycles, config.seed, injector
            )
            elapsed = time.perf_counter() - t0
            results.append(result)
            if verbose:
                marker = "✓" if result.meets_fp_target else "✗"
                print(f"{marker} FP={result.fp_count:3d} ({result.fp_rate:.1%})  "
                      f"Det={result.detection_count}/{len(ALL_FAULT_TYPES)} "
                      f"({result.detection_rate:.1%})  [{elapsed:.1f}s]")

    # Sort: meets FP target first, then by detection rate
    results.sort(key=lambda r: (not r.meets_fp_target, -r.detection_rate))
    return results


def find_optimal(results: List[TunerResult]) -> TunerResult:
    """Return the best configuration (highest detection rate that meets FP target)."""
    passing = [r for r in results if r.meets_fp_target]
    if not passing:
        # Fall back to lowest FP if none meet target
        return min(results, key=lambda r: (r.fp_rate, -r.detection_rate))
    return passing[0]


def save_tuner_results(results: List[TunerResult], path: str):
    data = {
        "grid_results": [asdict(r) for r in results],
        "optimal": asdict(find_optimal(results)),
        "all_fault_types": ALL_FAULT_TYPES,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4A — Threshold Tuner")
    print("Grid search: warmup_cycles × f4_min_ratio")
    print("=" * 60)
    config = TunerConfig(
        warmup_options=[30, 40, 50, 60],
        f4_ratio_options=[1.0, 1.1, 1.25, 1.5, 2.0],
        clean_run_cycles=50,
        seed=42
    )
    results = run_tuner(config, verbose=True)
    optimal = find_optimal(results)

    print("\n" + "=" * 60)
    print("OPTIMAL CONFIGURATION")
    print("=" * 60)
    print(f"  warmup_cycles : {optimal.warmup_cycles}")
    print(f"  f4_min_ratio  : {optimal.f4_min_ratio}")
    print(f"  FP rate       : {optimal.fp_rate:.1%} (target ≤5%)")
    print(f"  Detection rate: {optimal.detection_rate:.1%} "
          f"({optimal.detection_count}/{len(ALL_FAULT_TYPES)})")

    out = os.path.join(_REPO_ROOT, "phase4/results/tuner_results.json")
    save_tuner_results(results, out)
    print(f"\nResults saved: {out}")
