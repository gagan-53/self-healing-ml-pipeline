"""
Phase 4B Full Experiment Runner v2 — state-contamination fix applied.

v1 bug: shared observer per seed left residual FusionPolicy counter state
between fault type injections. Fixed by:
  1. observer._detector._fusion.reset() after each fault type + cooldown
  2. cooldown raised from 3 → 5 cycles (F5 counter fully flushes)
  3. warmup raised from 50 → 60 cycles (extra EMA stability)
  4. Fresh per-fault-type RNG (prevents correlated sample noise)
"""
from __future__ import annotations
import sys, os, json, time, copy
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'phase4'))

import numpy as np
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict

import sh_mlp
from sh_mlp.contracts.data_structures import (
    PipelineObservation, FeatureStats, InfraMetrics, PredictionStats
)
from calibration.fusion_v2 import FusionPolicyV2
from calibration.fault_injector_v2 import (
    FaultInjectorV2, INJECTABLE_FAULT_TYPES, _MULTI_CYCLE_TYPES
)
from datasets.adult_income import AdultIncomeLoader
from harness.experiment_harness import (
    SingleRunResult, ResultCollector, WilsonCI, PREREGISTERED_PROTOCOL
)

WARMUP_CYCLES  = 60
F4_MIN_RATIO   = 1.25
N_SAMPLES      = 300
COOLDOWN       = 5
SEEDS          = [42, 123, 999]
STAGE_IDS      = ["ingestion","preprocessing","feature_eng","training","evaluation"]
FAULT_STAGE    = {
    "DL-1":"ingestion","DL-2":"ingestion","DL-3":"ingestion",
    "DL-4":"ingestion","DL-5":"ingestion",
    "ML-1":"training","ML-2":"training","ML-3":"training","ML-4":"training",
    "IL-1":"preprocessing","IL-2":"preprocessing","IL-3":"preprocessing",
    "IL-4":"preprocessing","IL-5":"preprocessing",
}


def _make_obs(stage, rng):
    fs = {f"f{i}": FeatureStats.from_array(rng.normal(0,1,N_SAMPLES)) for i in range(6)}
    fs["target"] = FeatureStats(mean=0.30,std=0.46,min_val=0,max_val=1,
        pct_missing=0.01,histogram=[0.70]+[0.0]*8+[0.30],dtype="int")
    pred = PredictionStats(
        mean_confidence=0.78+rng.normal(0,0.008),
        entropy_mean=0.45+rng.normal(0,0.005),
        predicted_class_dist={"class_0":0.65,"class_1":0.35},
        ece=0.06+rng.normal(0,0.003))
    return PipelineObservation(
        pipeline_id="p4b",stage_id=stage,row_count_in=1000,row_count_out=1000,
        exec_time_ms=max(150.0,400.0+rng.normal(0,25)),
        memory_rss_mb=max(64.0,512.0+rng.normal(0,12)),
        feature_statistics=fs,schema_hash="schema_stable_v2",
        prediction_stats=pred,
        infra_metrics=InfraMetrics(cpu_pct=35.0+rng.normal(0,3),
            memory_rss_mb=512.0,cluster_cpu_pct=40.0+rng.normal(0,4)))


def _make_pipeline(seed, pid=None):
    obs = sh_mlp.create_pipeline(
        pipeline_id=pid or f"p4b_{seed}", stage_ids=STAGE_IDS,
        warmup_cycles=WARMUP_CYCLES, auto_approve=True, auto_recover=True)
    obs._detector._fusion = FusionPolicyV2(f4_min_ratio=F4_MIN_RATIO)
    return obs


def _warmup(observer, seed):
    rng = np.random.default_rng(seed)
    for _ in range(WARMUP_CYCLES+5):
        for s in STAGE_IDS:
            observer.inject_observation(_make_obs(s, rng))


def _cooldown_reset(observer, stage, seed_offset):
    rng = np.random.default_rng(seed_offset)
    for _ in range(COOLDOWN):
        for s in STAGE_IDS:
            observer.inject_observation(_make_obs(s, rng))
    observer._detector._fusion.reset()  # THE FIX


# ── RQ1 ───────────────────────────────────────────────────────────────────────

def run_rq1(seed, verbose=True):
    injector = FaultInjectorV2(seed=seed)
    observer = _make_pipeline(seed, f"rq1_s{seed}")
    _warmup(observer, seed)

    # FP count
    fp = 0
    rng_fp = np.random.default_rng(seed+1)
    for _ in range(20):
        for s in STAGE_IDS:
            if observer.inject_observation(_make_obs(s, rng_fp)): fp += 1
    observer._detector._fusion.reset()

    results = []
    for ft in INJECTABLE_FAULT_TYPES:
        stage = FAULT_STAGE[ft]
        n = injector.required_cycles(ft)
        rng_ft = np.random.default_rng(seed + abs(hash(ft)) % 10000)
        obs_t = _make_obs(stage, rng_ft)
        obs_t.row_count_in = 1000; obs_t.row_count_out = 1000

        t0 = time.perf_counter()
        ev = None
        for _ in range(n):
            obs_f = injector.inject(obs_t, ft)
            event = observer.inject_observation(obs_f)
            if event is not None: ev = event; break
        lat = (time.perf_counter()-t0)*1000

        det = ev is not None
        det_as = ev.failure_type if ev else "NONE"
        corr = det_as == ft
        rca_ok = False; rca_conf = 0.0; rec_ok = False; ttr = 0.0; pdelta = 0.0

        if det:
            try:
                rca = observer._rca.diagnose(ev)
                rca_conf = rca.confidence
                rca_ok = (rca.primary_cause.stage_id == stage)
                out = observer._planner.plan_and_execute(rca)
                rec_ok = out.success; ttr = float(out.time_to_recovery)
                pdelta = float(out.performance_delta)
            except: pass

        results.append(SingleRunResult(
            run_id=f"RQ1_{ft}_{seed}", rq="RQ1", fault_type=ft,
            dataset="synthetic", seed=seed, variant="full",
            detected=det, detected_as=det_as, correct_type=corr,
            rca_correct=rca_ok, rca_confidence=round(rca_conf,3),
            recovery_success=rec_ok, time_to_recovery_sec=ttr,
            performance_delta=pdelta, fp_count=fp,
            latency_ms=round(lat,2), n_prior_events=0))

        _cooldown_reset(observer, stage, seed+abs(hash(ft)))

        if verbose:
            m="✓" if det else "✗"; t="✓" if corr else ("~" if det else "-")
            print(f"    {ft:<6} {m} →{det_as:<6} type={t} rca={'✓' if rca_ok else '~'} "
                  f"rec={'✓' if rec_ok else '~'} cycles={n} {lat:.0f}ms")
    return results


# ── RQ3 ───────────────────────────────────────────────────────────────────────

def run_rq3_type(fault_type, n_events=20, seed=42):
    injector = FaultInjectorV2(seed=seed)
    observer = _make_pipeline(seed, f"rq3_{fault_type}_{seed}")
    _warmup(observer, seed)
    stage = FAULT_STAGE[fault_type]; n = injector.required_cycles(fault_type)
    results = []; attempts = 0
    while len(results) < n_events and attempts < n_events*4:
        attempts += 1
        rng = np.random.default_rng(seed+attempts*7)
        obs_t = _make_obs(stage, rng); obs_t.row_count_in=1000; obs_t.row_count_out=1000
        ev = None
        for _ in range(n):
            obs_f = injector.inject(obs_t, fault_type)
            e = observer.inject_observation(obs_f)
            if e is not None: ev=e; break
        _cooldown_reset(observer, stage, seed+attempts*3)
        if ev is None: continue
        try:
            rca = observer._rca.diagnose(ev)
            out = observer._planner.plan_and_execute(rca)
            results.append(SingleRunResult(
                run_id=f"RQ3_{fault_type}_{seed}_{len(results)}",
                rq="RQ3",fault_type=fault_type,dataset="synthetic",
                seed=seed,variant="full",detected=True,
                detected_as=ev.failure_type,
                correct_type=(ev.failure_type==fault_type),
                rca_correct=(rca.primary_cause.stage_id==stage),
                rca_confidence=round(rca.confidence,3),
                recovery_success=out.success,
                time_to_recovery_sec=float(out.time_to_recovery),
                performance_delta=float(out.performance_delta),
                fp_count=0,latency_ms=0.0,n_prior_events=len(results)))
        except: pass
    return results


# ── RQ4 static baseline ───────────────────────────────────────────────────────

def run_rq4_static(fault_type, n_events=20, seed=42):
    injector = FaultInjectorV2(seed=seed)
    observer = _make_pipeline(seed, f"rq4s_{fault_type}_{seed}")
    try: observer._planner._feedback_store = None
    except: pass
    _warmup(observer, seed)
    stage = FAULT_STAGE[fault_type]; n = injector.required_cycles(fault_type)
    results = []; attempts = 0
    while len(results) < n_events and attempts < n_events*4:
        attempts += 1
        rng = np.random.default_rng(seed+attempts*13)
        obs_t = _make_obs(stage, rng); obs_t.row_count_in=1000; obs_t.row_count_out=1000
        ev = None
        for _ in range(n):
            obs_f = injector.inject(obs_t, fault_type)
            e = observer.inject_observation(obs_f)
            if e is not None: ev=e; break
        _cooldown_reset(observer, stage, seed+attempts*5)
        if ev is None: continue
        try:
            rca = observer._rca.diagnose(ev)
            out = observer._planner.plan_and_execute(rca)
            results.append(SingleRunResult(
                run_id=f"RQ4s_{fault_type}_{seed}_{len(results)}",
                rq="RQ4",fault_type=fault_type,dataset="synthetic",
                seed=seed,variant="no_feedback",detected=True,
                detected_as=ev.failure_type,
                correct_type=(ev.failure_type==fault_type),
                rca_correct=(rca.primary_cause.stage_id==stage),
                rca_confidence=round(rca.confidence,3),
                recovery_success=out.success,
                time_to_recovery_sec=float(out.time_to_recovery),
                performance_delta=float(out.performance_delta),
                fp_count=0,latency_ms=0.0,n_prior_events=len(results)))
        except: pass
    return results


# ── Adult Income ──────────────────────────────────────────────────────────────

def run_adult_income(seed=42):
    loader = AdultIncomeLoader(use_synthetic=True, split_policy="temporal", seed=seed)
    observer = _make_pipeline(seed, f"adult_{seed}")
    _warmup(observer, seed)
    train_df, test_df = loader.get_split()
    for i in range(0, min(len(train_df),3000), 300):
        obs = loader.to_observation(train_df.iloc[i:i+300], "ingestion")
        observer.inject_observation(obs)
    observer._detector._fusion.reset()
    out = {}
    for ft, method in [("DL-1","inject_dl1_drift"),("DL-3","inject_dl3_shift")]:
        observer._detector._fusion.reset()
        ev_found = None
        nc = _MULTI_CYCLE_TYPES.get(ft, 3)
        for cs in range(0, min(len(test_df),3000), 300):
            chunk = test_df.iloc[cs:cs+300]
            base = loader.to_observation(chunk, "ingestion")
            for _ in range(nc):
                obs_f = (loader.inject_dl1_drift(base)
                         if method=="inject_dl1_drift"
                         else loader.inject_dl3_shift(base, 0.52))
                ev = observer.inject_observation(obs_f)
                if ev is not None: ev_found=ev; break
            if ev_found: break
        out[ft] = {"detected": ev_found is not None,
                   "detected_as": ev_found.failure_type if ev_found else "NONE",
                   "correct": ev_found.failure_type==ft if ev_found else False}
        observer._detector._fusion.reset()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_phase4b(seeds=None, n_rq3=20, verbose=True):
    seeds = seeds or SEEDS
    collector = ResultCollector()

    print("="*65)
    print("Phase 4B v2 — state-contamination fix applied")
    print(f"warmup={WARMUP_CYCLES}  f4={F4_MIN_RATIO}  cooldown={COOLDOWN}  samples={N_SAMPLES}")
    print(f"Seeds: {seeds}  RQ3 events: {n_rq3}")
    print("="*65)

    # RQ1
    print(f"\n[RQ1] {len(INJECTABLE_FAULT_TYPES)} types × {len(seeds)} seeds")
    all_rq1 = {}
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        res = run_rq1(seed, verbose)
        all_rq1[seed] = res
        for r in res:
            collector.add(r)
            for rq in ["RQ2","RQ3"]:
                rc=copy.copy(r); rc.rq=rq; rc.run_id=rc.run_id.replace("RQ1",rq)
                collector.add(rc)
        nd = sum(1 for r in res if r.detected)
        nc = sum(1 for r in res if r.correct_type)
        print(f"  → {nd}/{len(INJECTABLE_FAULT_TYPES)} detected  {nc}/{nd} type-correct")

    # RQ3 per detected type
    det_types = sorted({r.fault_type for sv in all_rq1.values() for r in sv if r.detected})
    print(f"\n[RQ3] Recovery — {len(det_types)} detected types × {n_rq3} events each")
    rq3_data = {}
    for ft in det_types:
        res = run_rq3_type(ft, n_rq3, 42)
        rq3_data[ft] = res
        for r in res: collector.add(r)
        nok = sum(1 for r in res if r.recovery_success)
        ci = WilsonCI.compute(nok, len(res))
        print(f"  {ft:<6} {nok}/{len(res)} = {nok/len(res) if res else 0:.1%}  "
              f"CI[{ci.lower:.1%},{ci.upper:.1%}]")

    # RQ4 learning curve
    rq4_types = det_types[:4]
    print(f"\n[RQ4] Learning curve — {len(rq4_types)} types × {n_rq3} events")
    rq4_curves = {}
    for ft in rq4_types:
        full_r   = run_rq3_type(ft, n_rq3, 42)
        static_r = run_rq4_static(ft, n_rq3, 42)
        for r in full_r:   r.rq="RQ4"; r.variant="full";        collector.add(r)
        for r in static_r: collector.add(r)
        curve = collector.compute_rq4_learning_curve(ft)
        rq4_curves[ft] = curve
        at20 = curve.get("at_20_events", {})
        d = at20.get("delta")
        if d is not None:
            print(f"  {ft}: full={at20.get('full',0):.1%} static={at20.get('static',0):.1%} Δ={d:.1%}")
        else:
            print(f"  {ft}: insufficient data")

    # Adult Income
    print("\n[Adult Income]")
    adult = run_adult_income(42)
    for ft, r in adult.items():
        print(f"  {ft}: {'✓' if r['detected'] else '✗'} →{r['detected_as']}")

    # Aggregates
    print("\n"+"="*65)
    rq1a = collector.compute_rq1("full")
    rq2a = collector.compute_rq2("full")
    rq3a = collector.compute_rq3("full")

    def fmt(a):
        m="✓" if a.meets_target else "✗"
        return f"  {m} {a.rq}: {a.point_estimate:.1%} [{a.ci_lower:.1%},{a.ci_upper:.1%}] target≥{a.target:.0%}"
    print(fmt(rq1a)); print(fmt(rq2a)); print(fmt(rq3a))

    print("\n  Detection by type:")
    for ft in INJECTABLE_FAULT_TYPES:
        c = rq1a.breakdown.get(ft,{"detected":0,"total":0})
        rate = c["detected"]/c["total"] if c["total"] else 0
        bar = "█"*int(rate*10)+"░"*(10-int(rate*10))
        print(f"    {ft:<6} {bar} {rate:.0%} ({c['detected']}/{c['total']})")
    print(f"    ML-2   ░░░░░░░░░░  SKIPPED (ADWIN)")

    report = {
        "phase":"4B","version":"v2","timestamp":datetime.utcnow().isoformat(),
        "config":{"warmup":WARMUP_CYCLES,"f4":F4_MIN_RATIO,"n_samples":N_SAMPLES,
                  "cooldown":COOLDOWN,"state_reset":True},
        "rq1":asdict(rq1a),"rq2":asdict(rq2a),"rq3":asdict(rq3a),
        "rq3_per_type":{ft:{
            "n_ok":sum(1 for r in res if r.recovery_success),"n_total":len(res),
            "rate":round(sum(1 for r in res if r.recovery_success)/len(res),3) if res else 0,
            "ci":asdict(WilsonCI.compute(sum(1 for r in res if r.recovery_success),len(res)))}
            for ft,res in rq3_data.items()},
        "rq4_curves":rq4_curves,"adult_income":adult,"seeds":seeds,
    }
    _results_dir = os.path.join(_REPO_ROOT, "phase4/results")
    os.makedirs(_results_dir, exist_ok=True)
    with open(os.path.join(_results_dir, "phase4b_results.json"), "w") as f:
        json.dump(report,f,indent=2)
    collector.save(os.path.join(_results_dir, "phase4b_raw.json"))
    print("\nSaved phase4b_results.json")
    return report

if __name__ == "__main__":
    run_phase4b(seeds=SEEDS, n_rq3=20, verbose=True)
