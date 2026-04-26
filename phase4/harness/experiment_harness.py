"""
Phase 4B Experiment Harness — skeleton built in Phase 4A.
Will be fully populated in Phase 4B (formal experiments).

This module provides:
  - ExperimentProtocol: pre-registration of experimental design
  - ExperimentRunner: runs one RQ × dataset × seed combination
  - ResultCollector: aggregates results and computes 95% CIs (Wilson score)
  - BaselineComparator: Evidently AI / NannyML wrappers (stubs in 4A)
  - AblationRunner: runs the 4 ablation variants

All configuration is specified at the top of each class so results are
traceable to a specific pre-registered protocol.
"""
from __future__ import annotations
import json, time, os, copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

# ── Pre-registration constants ────────────────────────────────────────────────
# These values must not be changed after Phase 4B begins.

PREREGISTERED_PROTOCOL = {
    "version": "4B.1",
    "registered_at": "2026-03-24",
    "rq1_detection_target":   0.90,   # ≥90% detection accuracy
    "rq1_fp_target":          0.10,   # <10% false positive rate
    "rq2_rca_target":         0.80,   # ≥80% RCA accuracy
    "rq3_recovery_target":    0.80,   # ≥80% recovery success rate
    "rq4_improvement_target": 0.10,   # ≥10% improvement after 20 events
    "rq4_ttr_improvement":    0.20,   # ≥20% TTR reduction
    "seeds": [42, 123, 999],          # 3 seeds for stability analysis
    "datasets": [
        "adult_income",               # DL-1, DL-3
        "covertype",                  # ML-2
        "credit_card_fraud",          # ML-1, DL-3
        "synthetic",                  # All 14 types (always included)
    ],
    "fault_types": [
        "DL-1","DL-2","DL-3","DL-4","DL-5",
        "ML-1","ML-2","ML-3","ML-4",
        "IL-1","IL-2","IL-3","IL-4","IL-5"
    ],
    "n_recovery_events_per_type": 50,  # for RQ3/RQ4 convergence
    "confidence_level": 0.95,          # Wilson score CIs
}

ABLATION_VARIANTS = {
    "full":            "Full SH-MLP (M1+M2+M3+M4)",
    "no_feedback":     "SH-MLP without feedback loop (static strategy)",
    "no_rca":          "SH-MLP without RCA (random strategy selection)",
    "data_model_only": "SH-MLP with data+model detection only (no infra layer)",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SingleRunResult:
    """Result from one (fault_type, dataset, seed) experimental run."""
    run_id:          str
    rq:              str           # "RQ1", "RQ2", "RQ3", "RQ4"
    fault_type:      str
    dataset:         str
    seed:            int
    variant:         str           # ablation variant or "full"
    detected:        bool
    detected_as:     str
    correct_type:    bool
    rca_correct:     bool
    rca_confidence:  float
    recovery_success:bool
    time_to_recovery_sec: float
    performance_delta: float
    fp_count:        int
    latency_ms:      float
    n_prior_events:  int           # for learning curve (RQ4)
    timestamp:       str           = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class WilsonCI:
    """95% Wilson score confidence interval for a proportion."""
    n_success: int
    n_total:   int
    point:     float
    lower:     float
    upper:     float

    @classmethod
    def compute(cls, n_success: int, n_total: int,
                confidence: float = 0.95) -> "WilsonCI":
        if n_total == 0:
            return cls(0, 0, 0.0, 0.0, 0.0)
        from scipy import stats as scipy_stats
        z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
        p = n_success / n_total
        denom = 1 + z**2 / n_total
        centre = (p + z**2 / (2*n_total)) / denom
        spread = z * np.sqrt(p*(1-p)/n_total + z**2/(4*n_total**2)) / denom
        return cls(
            n_success=n_success, n_total=n_total,
            point=round(p, 4),
            lower=round(max(0, centre - spread), 4),
            upper=round(min(1, centre + spread), 4),
        )


@dataclass
class AggregatedRQResult:
    """Aggregated result for one research question."""
    rq:              str
    variant:         str
    n_trials:        int
    point_estimate:  float
    ci_lower:        float
    ci_upper:        float
    meets_target:    bool
    target:          float
    std_across_seeds:float
    breakdown:       Dict           = field(default_factory=dict)


# ── Result Collector ──────────────────────────────────────────────────────────

class ResultCollector:
    """
    Accumulates SingleRunResults and computes aggregate statistics.
    All statistics use Wilson score CIs (correct for proportions near 0/1).
    """

    def __init__(self):
        self._results: List[SingleRunResult] = []

    def add(self, result: SingleRunResult):
        self._results.append(result)

    def compute_rq1(self, variant: str = "full") -> AggregatedRQResult:
        """RQ1: detection accuracy and FP rate."""
        runs = [r for r in self._results if r.rq == "RQ1" and r.variant == variant]
        if not runs:
            return AggregatedRQResult("RQ1", variant, 0, 0.0, 0.0, 0.0, False,
                                      PREREGISTERED_PROTOCOL["rq1_detection_target"], 0.0)
        n_total   = len(runs)
        n_success = sum(1 for r in runs if r.detected)
        ci = WilsonCI.compute(n_success, n_total)
        seeds = list(set(r.seed for r in runs))
        per_seed = [sum(1 for r in runs if r.seed == s and r.detected)
                    / max(1, sum(1 for r in runs if r.seed == s))
                    for s in seeds]
        return AggregatedRQResult(
            rq="RQ1", variant=variant, n_trials=n_total,
            point_estimate=ci.point, ci_lower=ci.lower, ci_upper=ci.upper,
            meets_target=ci.lower >= PREREGISTERED_PROTOCOL["rq1_detection_target"],
            target=PREREGISTERED_PROTOCOL["rq1_detection_target"],
            std_across_seeds=round(float(np.std(per_seed)), 4),
            breakdown={ft: {"detected": sum(1 for r in runs if r.fault_type == ft and r.detected),
                            "total": sum(1 for r in runs if r.fault_type == ft)}
                       for ft in PREREGISTERED_PROTOCOL["fault_types"]}
        )

    def compute_rq2(self, variant: str = "full") -> AggregatedRQResult:
        """RQ2: RCA accuracy."""
        runs = [r for r in self._results if r.rq == "RQ2" and r.variant == variant
                and r.detected]
        if not runs:
            return AggregatedRQResult("RQ2", variant, 0, 0.0, 0.0, 0.0, False,
                                      PREREGISTERED_PROTOCOL["rq2_rca_target"], 0.0)
        n_total   = len(runs)
        n_success = sum(1 for r in runs if r.rca_correct)
        ci = WilsonCI.compute(n_success, n_total)
        seeds = list(set(r.seed for r in runs))
        per_seed = [sum(1 for r in runs if r.seed == s and r.rca_correct)
                    / max(1, sum(1 for r in runs if r.seed == s))
                    for s in seeds]
        return AggregatedRQResult(
            rq="RQ2", variant=variant, n_trials=n_total,
            point_estimate=ci.point, ci_lower=ci.lower, ci_upper=ci.upper,
            meets_target=ci.lower >= PREREGISTERED_PROTOCOL["rq2_rca_target"],
            target=PREREGISTERED_PROTOCOL["rq2_rca_target"],
            std_across_seeds=round(float(np.std(per_seed)), 4),
        )

    def compute_rq3(self, variant: str = "full") -> AggregatedRQResult:
        """RQ3: recovery success rate."""
        runs = [r for r in self._results if r.rq == "RQ3" and r.variant == variant
                and r.detected]
        if not runs:
            return AggregatedRQResult("RQ3", variant, 0, 0.0, 0.0, 0.0, False,
                                      PREREGISTERED_PROTOCOL["rq3_recovery_target"], 0.0)
        n_total   = len(runs)
        n_success = sum(1 for r in runs if r.recovery_success)
        ci = WilsonCI.compute(n_success, n_total)
        seeds = list(set(r.seed for r in runs))
        per_seed = [sum(1 for r in runs if r.seed == s and r.recovery_success)
                    / max(1, sum(1 for r in runs if r.seed == s))
                    for s in seeds]
        return AggregatedRQResult(
            rq="RQ3", variant=variant, n_trials=n_total,
            point_estimate=ci.point, ci_lower=ci.lower, ci_upper=ci.upper,
            meets_target=ci.lower >= PREREGISTERED_PROTOCOL["rq3_recovery_target"],
            target=PREREGISTERED_PROTOCOL["rq3_recovery_target"],
            std_across_seeds=round(float(np.std(per_seed)), 4),
        )

    def compute_rq4_learning_curve(self, fault_type: str,
                                   variant_a: str = "full",
                                   variant_b: str = "no_feedback") -> Dict:
        """
        RQ4: learning curve — recovery success rate as function of n_prior_events.
        Compares full SH-MLP vs static (no_feedback) variant.
        Returns dict with event_counts, rates_a, rates_b.
        """
        runs_a = sorted(
            [r for r in self._results if r.fault_type == fault_type
             and r.variant == variant_a and r.detected],
            key=lambda r: r.n_prior_events
        )
        runs_b = sorted(
            [r for r in self._results if r.fault_type == fault_type
             and r.variant == variant_b and r.detected],
            key=lambda r: r.n_prior_events
        )
        if not runs_a:
            return {"fault_type": fault_type, "insufficient_data": True}

        # Rolling success rate at each event count
        window = 5
        event_counts, rates_a, rates_b = [], [], []
        for i in range(window, len(runs_a) + 1):
            window_runs = runs_a[max(0, i-window):i]
            rate = sum(1 for r in window_runs if r.recovery_success) / len(window_runs)
            event_counts.append(i)
            rates_a.append(round(rate, 4))
            if i <= len(runs_b):
                window_b = runs_b[max(0, i-window):i]
                rates_b.append(round(
                    sum(1 for r in window_b if r.recovery_success) / len(window_b), 4
                ))
            else:
                rates_b.append(None)

        # Compute improvement at 20-event mark
        at20_a = np.mean([r.recovery_success for r in runs_a if r.n_prior_events <= 20])
        at20_b = np.mean([r.recovery_success for r in runs_b if r.n_prior_events <= 20]) \
            if runs_b else None
        delta = float(at20_a - at20_b) if at20_b is not None else None

        return {
            "fault_type":     fault_type,
            "event_counts":   event_counts,
            "rates_full":     rates_a,
            "rates_static":   rates_b,
            "at_20_events": {
                "full":   round(float(at20_a), 4),
                "static": round(float(at20_b), 4) if at20_b is not None else None,
                "delta":  round(delta, 4) if delta is not None else None,
                "meets_target": (delta >= PREREGISTERED_PROTOCOL["rq4_improvement_target"]
                                 if delta is not None else None),
            }
        }

    def save(self, path: str):
        data = {
            "protocol": PREREGISTERED_PROTOCOL,
            "n_results": len(self._results),
            "results": [asdict(r) for r in self._results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        from dataclasses import fields
        field_names = {f.name for f in fields(SingleRunResult)}
        for r in data.get("results", []):
            filtered = {k: v for k, v in r.items() if k in field_names}
            self._results.append(SingleRunResult(**filtered))


# ── Baseline Comparator (stubs, fully implemented in Phase 4B) ───────────────

class BaselineComparator:
    """
    Wraps comparison baselines.
    Phase 4A: stubs only. Phase 4B: full Evidently AI and NannyML wrappers.
    """

    def __init__(self, method: str = "vanilla"):
        """
        method: 'vanilla' | 'evidently' | 'nannynml'
        """
        self.method = method

    def detect(self, obs, baseline_stats: dict) -> bool:
        """
        Returns True if the baseline method detects an anomaly.
        STUB — returns False for all baselines until Phase 4B implementation.
        """
        # Phase 4B: replace stubs with real Evidently/NannyML calls
        if self.method == "vanilla":
            return False  # no monitoring
        elif self.method == "evidently":
            return self._evidently_detect(obs, baseline_stats)
        elif self.method == "nannynml":
            return self._nannynml_detect(obs, baseline_stats)
        return False

    def _evidently_detect(self, obs, baseline_stats: dict) -> bool:
        """Phase 4B: integrate with evidently.metrics.DataDriftPreset."""
        # TODO Phase 4B: implement
        return False

    def _nannynml_detect(self, obs, baseline_stats: dict) -> bool:
        """Phase 4B: integrate with nannyml.drift.UnivariateDriftCalculator."""
        # TODO Phase 4B: implement
        return False


# ── Ablation Runner (skeleton) ────────────────────────────────────────────────

class AblationRunner:
    """
    Runs the four ablation variants for the paper's ablation table.
    Phase 4A: interface defined; execution in Phase 4B.
    """

    VARIANTS = list(ABLATION_VARIANTS.keys())

    def run_variant(self, variant: str, fault_type: str,
                    obs_sequence, warmup_cycles: int = 50) -> SingleRunResult:
        """
        Run one experimental trial for the given ablation variant.
        Phase 4B: implement each variant's pipeline configuration.
        """
        # TODO Phase 4B: implement variant configurations
        raise NotImplementedError(f"AblationRunner.run_variant() is Phase 4B scope")

    def build_ablation_table(self, results: List[AggregatedRQResult]) -> Dict:
        """Build the paper's ablation table from aggregated results."""
        table = {}
        for result in results:
            variant = result.variant
            rq = result.rq
            if variant not in table:
                table[variant] = {"description": ABLATION_VARIANTS.get(variant, variant)}
            table[variant][rq] = {
                "point": result.point_estimate,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "meets_target": result.meets_target,
            }
        return table


# ── Experiment Runner (Phase 4B scope, interface defined in 4A) ───────────────

class ExperimentRunner:
    """
    Orchestrates the full Phase 4B experimental protocol.
    Interface defined in Phase 4A; implementation in Phase 4B.
    """

    def __init__(self, protocol: dict = None, warmup_cycles: int = 50,
                 f4_min_ratio: float = 1.25):
        self.protocol      = protocol or PREREGISTERED_PROTOCOL
        self.warmup_cycles = warmup_cycles
        self.f4_min_ratio  = f4_min_ratio
        self.collector     = ResultCollector()

    def run_rq1(self, dataset: str, seed: int) -> List[SingleRunResult]:
        """RQ1: detection accuracy and FP rate on one dataset × seed."""
        raise NotImplementedError("Phase 4B scope")

    def run_rq2(self, dataset: str, seed: int) -> List[SingleRunResult]:
        """RQ2: RCA accuracy on one dataset × seed."""
        raise NotImplementedError("Phase 4B scope")

    def run_rq3(self, dataset: str, seed: int) -> List[SingleRunResult]:
        """RQ3: recovery success rate on one dataset × seed."""
        raise NotImplementedError("Phase 4B scope")

    def run_rq4(self, fault_type: str, seed: int) -> List[SingleRunResult]:
        """RQ4: learning curve for one fault type × seed."""
        raise NotImplementedError("Phase 4B scope")

    def run_full_protocol(self) -> ResultCollector:
        """
        Execute the full pre-registered experimental protocol.
        Phase 4B: runs all RQ × dataset × seed combinations.
        """
        raise NotImplementedError("Phase 4B scope")

    def save_pre_registration(self, path: str):
        """Save the pre-registered protocol for transparency."""
        data = {
            "preregistered_at": datetime.utcnow().isoformat(),
            "protocol": self.protocol,
            "ablation_variants": ABLATION_VARIANTS,
            "warmup_cycles_calibrated": self.warmup_cycles,
            "f4_min_ratio_calibrated": self.f4_min_ratio,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Pre-registration saved: {path}")
