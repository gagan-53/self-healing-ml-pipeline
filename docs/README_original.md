# SH-MLP: Self-Healing Machine Learning Pipeline
## Phase 3 Prototype — Local Execution Guide

---

## Repository Structure

```
sh_mlp_project/
│
├── README.md                          ← This file
├── setup.py                           ← pip-installable package
├── requirements.txt                   ← direct pip install list
│
├── sh_mlp/                            ← Core library (importable as `import sh_mlp`)
│   ├── __init__.py                    ← create_pipeline() factory + public exports
│   │
│   ├── contracts/
│   │   ├── __init__.py
│   │   └── data_structures.py         ← All 8 typed interface contracts (dataclasses)
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── baseline.py                ← BaselineManager: warm-up + EMA updates
│   │   ├── detectors.py               ← 14 rule-based detectors + DetectorRegistry
│   │   ├── fusion.py                  ← FusionPolicy: 5 rules F1-F5
│   │   └── engine.py                  ← DetectionEngine: orchestrates all detection
│   │
│   ├── rca/
│   │   ├── __init__.py
│   │   └── engine.py                  ← RCAEngine + PipelineDAG + SignalHistory + PELT
│   │
│   ├── recovery/
│   │   ├── __init__.py
│   │   ├── strategies.py              ← 22 strategies across 14 failure types
│   │   └── planner.py                 ← RecoveryPlanner + AutonomyMatrix (6 paths)
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── store.py                   ← FeedbackStore + LinUCB contextual bandit
│   │
│   └── observer/
│       ├── __init__.py
│       └── pipeline_observer.py       ← PipelineObserver: system entry point
│
├── tests/
│   └── test_sh_mlp.py                 ← 70 tests across 6 layers
│
└── experiments/
    ├── fault_injector.py              ← Deterministic fault injection (14 types)
    └── run_experiment.py              ← Full integration experiment runner
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | 3.9+    | 3.10, 3.11, 3.12 all tested |
| pip         | any     | comes with Python |
| numpy       | ≥1.21   | array ops + PELT |
| scipy       | ≥1.7    | KS test, stats |
| scikit-learn| ≥1.0    | IsolationForest |
| pandas      | ≥1.3    | optional (DataFrame support in observer) |
| psutil      | ≥5.8    | RSS memory measurement |

---

## Step-by-Step Local Setup

### Option A — Virtual Environment (Recommended)

```bash
# 1. Clone / extract the project
cd ~
mkdir sh_mlp_project && cd sh_mlp_project
# (copy all files here)

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode (optional but clean)
pip install -e .

# 5. Verify import works
python -c "import sh_mlp; print('SH-MLP v' + sh_mlp.__version__)"
# → SH-MLP v0.1.0
```

### Option B — Direct pip install (no venv)

```bash
pip install numpy scipy scikit-learn pandas psutil
# Then run from inside the sh_mlp_project/ directory
```

---

## Running the Tests (70 tests)

```bash
# From the project root directory
cd sh_mlp_project/
python tests/test_sh_mlp.py
```

### Expected output
```
test_baseline_feature_profiles_populated ... ok
test_ema_update_converges ... ok
test_warmup_not_ready_before_threshold ... ok
...
test_overhead_within_budget ...
  Avg monitoring: ~1.0ms | Stage baseline: 1000ms
  Overhead ratio: 0.1% (limit: 15%)
ok

----------------------------------------------------------------------
Ran 70 tests in ~0.5s

OK
```

### What each test layer covers

| Layer | Class | Count | What it validates |
|-------|-------|-------|-------------------|
| 1 | TestContracts | 6 | All 8 dataclass contracts, UUID uniqueness, Severity ordering |
| 2a | TestBaselineManager | 4 | Warm-up threshold, EMA convergence, feature profile population |
| 2b | TestDetectorDL1..IL5 | 22 | All 14 detectors — positive + negative cases + property-based |
| 2c | TestDetectorRegistry | 2 | 14 detectors registered, no crash on edge input |
| 3 | TestFusionPolicy + TestAutonomy | 8 | F1-F5 rules, autonomy gate all 6 paths |
| 4 | TestPELT + TestRCAEngine | 5 | Single CP detection, DAG parent traversal, unknown result |
| 5 | TestRecoveryPlanner + TestFeedbackStore + TestLinUCB | 14 | Strategy library completeness, LinUCB exploration/convergence |
| 6 | TestEndToEnd + TestOverheadBudget | 7 | Full pipeline, schema violation, row drop detection, overhead <15% |

---

## Running the Integration Experiment

```bash
cd sh_mlp_project/
python experiments/run_experiment.py
```

### What it does (4 phases)
1. **Warm-up** — feeds 37 clean synthetic observations through 5-stage pipeline
2. **Clean baseline** — 20 cycles, counts false positives
3. **Fault injection** — injects 11 fault types one-by-one; for each detected event runs full RCA → Recovery → Feedback loop
4. **Summary** — prints detection/recovery rates; saves structured log to `experiments/experiment_log.json`

### Expected output (abbreviated)
```
============================================================
SH-MLP Phase 3 — Integration Experiment
============================================================

[A] Warming up (35 cycles)...
    Warmup complete. Time: ~2s | Warmed up: True

[B] Clean baseline run (20 cycles)...
    Clean run events: 23 (expected: 0)
    Time: ~1s

[C] Fault injection — 11 fault types...

  Injecting DL-1...  Detected=False | Type=NONE | Expected=DL-1 ...
  Injecting DL-2...  Detected=True  | Type=DL-2 | Expected=DL-2 | Correct=True ...
  Injecting DL-3...  Detected=True  | Type=DL-1 | Expected=DL-3 | Correct=False ...
  ...

============================================================
EXPERIMENT SUMMARY
============================================================
  Faults injected:    11
  Detected:           7/11 (63.6%)
  Correct type:       1/7 (14.3%)
  Recovered:          7/7 (100.0%)
  Clean FP:           23

  Experiment log saved: experiments/experiment_log.json
```

---

## Using SH-MLP in Your Own Code

### Minimal example — monitor a real sklearn pipeline

```python
import sys
sys.path.insert(0, '/path/to/sh_mlp_project')   # or install with pip -e .

import sh_mlp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Create observer
observer = sh_mlp.create_pipeline(
    pipeline_id   = "my_classifier",
    stage_ids     = ["ingestion", "preprocessing", "training"],
    warmup_cycles = 30,
    auto_approve  = False,   # True → fully autonomous; False → HITL approval prompt
    auto_recover  = True,
)

# 2. Simulate pipeline cycles
np.random.seed(42)
for cycle in range(60):
    
    # --- Stage: Ingestion ---
    X = np.random.normal(0, 1, (500, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    
    # Inject drift at cycle 45
    if cycle == 45:
        df["f0"] = df["f0"] * 3 + 5   # large shift
    
    with observer.monitor_stage("ingestion", data_in=df) as ctx:
        ctx.set_output(df)
    
    # --- Stage: Preprocessing ---
    with observer.monitor_stage("preprocessing", data_in=df) as ctx:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)
        ctx.set_output(X_scaled)
    
    # --- Stage: Training ---
    with observer.monitor_stage("training", data_in=X_scaled) as ctx:
        model = LogisticRegression().fit(X_scaled, y)
        ctx.set_output(X_scaled)

    if cycle % 10 == 0:
        print(f"Cycle {cycle:3d}: {observer.summary()}")
```

### Direct observation injection (for testing / no real pipeline)

```python
from sh_mlp.contracts import (
    PipelineObservation, FeatureStats, InfraMetrics
)

# Build an observation manually
obs = PipelineObservation(
    pipeline_id    = "test",
    stage_id       = "preprocessing",
    row_count_in   = 1000,
    row_count_out  = 1000,
    exec_time_ms   = 500.0,
    memory_rss_mb  = 512.0,
    schema_hash    = "abc123",
    infra_metrics  = InfraMetrics(cpu_pct=30.0, memory_rss_mb=512.0),
    feature_statistics = {
        "age": FeatureStats.from_array(np.array([25.0, 30.0, 35.0, 40.0]))
    }
)

event = observer.inject_observation(obs)
if event:
    print(f"Detected: {event.failure_type} ({event.severity.value})")
```

---

## Fault Injection API

Inject specific failure types deterministically:

```python
import sys; sys.path.insert(0, '/path/to/sh_mlp_project/experiments')
from fault_injector import FaultInjector

injector = FaultInjector(seed=42)

# After warm-up, inject any of the 14 types:
obs_clean = ...   # your PipelineObservation

obs_faulty = injector.inject(obs_clean, "DL-1", severity="HIGH")   # feature drift
obs_faulty = injector.inject(obs_clean, "DL-2")                    # schema violation
obs_faulty = injector.inject(obs_clean, "IL-1")                    # memory spike
obs_faulty = injector.inject(obs_clean, "IL-4")                    # silent DAG failure
# All 14 types: DL-1..5, ML-1..4, IL-1..5

event = observer.inject_observation(obs_faulty)
```

---

## Module API Reference

### `sh_mlp.create_pipeline()`

```python
observer = sh_mlp.create_pipeline(
    pipeline_id   : str,          # unique pipeline identifier
    stage_ids     : list[str],    # ordered stage names (linear DAG assumed)
    warmup_cycles : int = 30,     # cycles before detection activates
    auto_approve  : bool = False, # True → HITL requests auto-approved (testing only)
    auto_recover  : bool = True,  # False → detect only, no recovery
) -> PipelineObserver
```

### `PipelineObserver`

| Method | Description |
|--------|-------------|
| `monitor_stage(stage_id, data_in)` | Context manager — instruments a stage |
| `inject_observation(obs)` | Direct observation injection |
| `summary()` | Dict: cycles, events_detected, recoveries, is_warmed_up |

### Key data structures

```python
# Severity enum
Severity.LOW / MEDIUM / HIGH / CRITICAL

# DecisionMode enum  
DecisionMode.AUTONOMOUS / SUPERVISED / HITL / ESCALATE / IMMEDIATE_HALT / DEFERRED

# FailureEvent (what detection produces)
event.failure_type    # e.g. "DL-1"
event.severity        # Severity enum
event.fusion_rule     # "F1".."F5"
event.contributing_signals  # list of DetectionSignal

# RCAResult (what RCA produces)
result.primary_cause.stage_id       # suspect stage
result.confidence                   # [0, 1]
result.narrative                    # human-readable explanation
result.change_point.magnitude       # how big the change was

# RecoveryOutcome (what recovery produces)
outcome.success        # bool
outcome.strategy_id    # e.g. "S-DL1-A"
outcome.decision_mode  # "AUTONOMOUS" | "HITL" | ...
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sh_mlp` | Run from project root, or `pip install -e .` |
| `ModuleNotFoundError: psutil` | `pip install psutil` |
| Tests import fault_injector | Run `python tests/test_sh_mlp.py` from project root (not from tests/) |
| IsolationForest UserWarning about sklearn.utils.parallel | Harmless warning from sklearn threading; redirect stderr or ignore |
| Concept drift (ML-2) not detected | Stateful detector needs 20+ cycles with entropy changes — inject over multiple cycles, not single-shot |
| High false positive rate | Increase `warmup_cycles` to 50+; reduce observation frequency |

---

## CI Gate (run before every commit)

```bash
#!/bin/bash
set -e
cd sh_mlp_project/
echo "=== Running 70 unit tests ==="
python tests/test_sh_mlp.py
echo "=== All tests passed ==="
```

---

*SH-MLP Phase 3 — Prototype Development | v0.1.0 | March 2026*
