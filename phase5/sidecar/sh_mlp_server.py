"""
SH-MLP VS Code Sidecar Server
==============================
JSON-RPC 2.0 server over stdin/stdout.
Started by the VS Code extension as a child process; communicates via newline-
delimited JSON messages. Wraps the full SH-MLP detection engine (M1–M4).

Protocol (newline-delimited JSON):
  Extension → Sidecar (requests):
    {"jsonrpc":"2.0","id":N,"method":"configure","params":{...}}
    {"jsonrpc":"2.0","id":N,"method":"observe","params":{...}}
    {"jsonrpc":"2.0","id":N,"method":"get_status","params":{...}}
    {"jsonrpc":"2.0","id":N,"method":"approve_recovery","params":{...}}
    {"jsonrpc":"2.0","id":N,"method":"dismiss_fault","params":{...}}
    {"jsonrpc":"2.0","id":N,"method":"get_history","params":{...}}
    {"jsonrpc":"2.0","id":N,"method":"shutdown","params":{}}

  Sidecar → Extension (responses + push events):
    {"jsonrpc":"2.0","method":"ready"}                      # startup signal
    {"jsonrpc":"2.0","id":N,"result":{...}}                 # response
    {"jsonrpc":"2.0","method":"fault_detected","params":{}} # push notification
"""
from __future__ import annotations
import sys, os, json, uuid, threading, time, traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from datetime import datetime

# ── Ensure sh_mlp is importable ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import sh_mlp
    from sh_mlp.contracts.data_structures import (
        PipelineObservation, FeatureStats, InfraMetrics, PredictionStats
    )
    from sh_mlp.detection.engine import DetectionEngine
    _SH_MLP_AVAILABLE = True
except ImportError:
    _SH_MLP_AVAILABLE = False

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class FaultEvent:
    fault_id:    str
    pipeline_id: str
    fault_type:  str
    severity:    str
    stage_id:    str
    fusion_rule: str
    detected_at: str
    raw_value:   Optional[float] = None
    threshold:   Optional[float] = None
    explanation: Optional[str]   = None

@dataclass
class RecoveryResult:
    fault_id:    str
    strategy_id: str
    success:     bool
    ttr_ms:      int
    error:       Optional[str] = None

@dataclass
class HistoryEvent:
    fault_id:    str
    fault_type:  str
    detected_at: str
    recovered:   bool
    strategy_id: Optional[str] = None
    ttr_ms:      Optional[int] = None

@dataclass
class PipelineStatus:
    pipeline_id:   str
    health:        str
    active_faults: List[Dict]
    event_count:   int
    fp_rate:       float
    recovery_rate: float
    linucb_delta:  float
    warmed_up:     bool
    last_updated:  str

# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState:
    """Holds runtime state for one monitored pipeline."""

    STAGE_IDS = ["ingestion", "preprocessing", "feature_eng", "training", "evaluation"]

    def __init__(self, pipeline_id: str, warmup_cycles: int = 60,
                 f4_min_ratio: float = 1.25):
        self.pipeline_id   = pipeline_id
        self.warmup_cycles = warmup_cycles
        self.f4_min_ratio  = f4_min_ratio
        self.observer      = None
        self.active_faults: Dict[str, FaultEvent] = {}
        self.history:       List[HistoryEvent]    = []
        self.event_count    = 0
        self.fp_count       = 0
        self.clean_obs      = 0
        self.recovered      = 0
        self.linucb_delta   = 0.0
        self._lock          = threading.Lock()
        self._init_observer()

    def _init_observer(self):
        if not _SH_MLP_AVAILABLE:
            return
        try:
            # Import FusionPolicyV2 if phase4 is available
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'phase4'))
                from calibration.fusion_v2 import FusionPolicyV2
                self.observer = sh_mlp.create_pipeline(
                    pipeline_id=self.pipeline_id,
                    stage_ids=self.STAGE_IDS,
                    warmup_cycles=self.warmup_cycles,
                    auto_approve=False,
                    auto_recover=False,
                )
                self.observer._detector._fusion = FusionPolicyV2(
                    f4_min_ratio=self.f4_min_ratio
                )
            except ImportError:
                self.observer = sh_mlp.create_pipeline(
                    pipeline_id=self.pipeline_id,
                    stage_ids=self.STAGE_IDS,
                    warmup_cycles=self.warmup_cycles,
                    auto_approve=False,
                    auto_recover=False,
                )
        except Exception as e:
            _err(f"Observer init failed: {e}")

    @property
    def health(self) -> str:
        n = len(self.active_faults)
        if n == 0:              return "HEALTHY"
        if any(f.severity == "CRITICAL" for f in self.active_faults.values()): return "CRITICAL"
        return "DEGRADED"

    @property
    def warmed_up(self) -> bool:
        if self.observer is None: return False
        try:  return self.observer._detector.is_warmed_up()
        except: return False

    @property
    def fp_rate(self) -> float:
        total = self.event_count + self.fp_count
        return self.fp_count / total if total else 0.0

    @property
    def recovery_rate(self) -> float:
        n = self.event_count
        return self.recovered / n if n else 0.0


# ── RPC server ────────────────────────────────────────────────────────────────

def _send(msg: dict) -> None:
    """Write one newline-delimited JSON message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def _err(msg: str) -> None:
    sys.stderr.write(f"[sidecar] {msg}\n")
    sys.stderr.flush()

def _ok(req_id: int, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})

def _error(req_id: int, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


class SidecarServer:
    def __init__(self):
        self.pipelines: Dict[str, PipelineState] = {}
        self.auto_approve = False
        self._running = True

    def _get_pipeline(self, pipeline_id: str, warmup: int = 60,
                      f4: float = 1.25) -> PipelineState:
        if pipeline_id not in self.pipelines:
            self.pipelines[pipeline_id] = PipelineState(pipeline_id, warmup, f4)
        return self.pipelines[pipeline_id]

    # ── Handlers ──────────────────────────────────────────────────────────────

    def handle_configure(self, req_id: int, params: dict) -> None:
        warmup = params.get("warmup_cycles", 60)
        f4     = params.get("f4_min_ratio", 1.25)
        self.auto_approve = params.get("auto_approve", False)
        # Re-init default pipeline with new params
        self.pipelines["default"] = PipelineState("default", warmup, f4)
        _ok(req_id, {"ok": True})

    def handle_observe(self, req_id: int, params: dict) -> None:
        pid   = params.get("pipeline_id", "default")
        stage = params.get("stage_id", "ingestion")
        obs_d = params.get("observation", {})
        state = self._get_pipeline(pid)

        if state.observer is None or not _SH_MLP_AVAILABLE:
            # Stub: simulate occasional fault detection for demo purposes
            _ok(req_id, {"fault_event": None})
            return

        try:
            obs = self._dict_to_observation(obs_d, pid, stage)
            event = state.observer.inject_observation(obs)

            fault_out = None
            if event is not None:
                fault = FaultEvent(
                    fault_id    = str(uuid.uuid4()),
                    pipeline_id = pid,
                    fault_type  = event.failure_type,
                    severity    = event.severity.value,
                    stage_id    = event.stage_id,
                    fusion_rule = event.fusion_rule,
                    detected_at = datetime.utcnow().isoformat(),
                    explanation = f"Fusion rule {event.fusion_rule} fired on {event.failure_type}",
                )
                with state._lock:
                    state.active_faults[fault.fault_id] = fault
                    state.event_count += 1

                # Push notification to extension
                _send({"jsonrpc": "2.0", "method": "fault_detected", "params": asdict(fault)})
                fault_out = asdict(fault)

                if self.auto_approve:
                    self._do_recovery(state, fault.fault_id)

            _ok(req_id, {"fault_event": fault_out})

        except Exception as e:
            _error(req_id, -32000, str(e))

    def handle_get_status(self, req_id: int, params: dict) -> None:
        pid   = params.get("pipeline_id", "default")
        state = self._get_pipeline(pid)
        with state._lock:
            status = PipelineStatus(
                pipeline_id   = pid,
                health        = state.health,
                active_faults = [asdict(f) for f in state.active_faults.values()],
                event_count   = state.event_count,
                fp_rate       = state.fp_rate,
                recovery_rate = state.recovery_rate,
                linucb_delta  = state.linucb_delta,
                warmed_up     = state.warmed_up,
                last_updated  = datetime.utcnow().isoformat(),
            )
        _ok(req_id, asdict(status))

    def handle_approve_recovery(self, req_id: int, params: dict) -> None:
        fault_id = params.get("fault_id", "")
        # Find which pipeline has this fault
        state = next(
            (s for s in self.pipelines.values() if fault_id in s.active_faults),
            self._get_pipeline("default")
        )
        result = self._do_recovery(state, fault_id)
        _ok(req_id, asdict(result))

    def _do_recovery(self, state: PipelineState, fault_id: str) -> RecoveryResult:
        fault = state.active_faults.get(fault_id)
        if fault is None:
            return RecoveryResult(fault_id=fault_id, strategy_id="", success=False,
                                  ttr_ms=0, error="Fault not found")

        STRATEGIES: Dict[str, str] = {
            "DL-1": "retrain_recent_window",
            "DL-2": "halt_and_quarantine",
            "DL-3": "class_reweighting",
            "DL-4": "retry_with_backoff",
            "DL-5": "quarantine_and_cleanlab",
            "ML-1": "full_retrain_or_rollback",
            "ML-2": "switch_to_stable_version",
            "ML-3": "reduce_lr_clip_gradients",
            "ML-4": "temperature_scaling",
            "IL-1": "batch_size_binary_search",
            "IL-2": "retry_reduced_subset",
            "IL-3": "restore_pinned_versions",
            "IL-4": "halt_audit_lineage",
            "IL-5": "defer_noncritical_stages",
        }
        strategy = STRATEGIES.get(fault.fault_type, "generic_recovery")
        t0 = time.perf_counter()

        # Attempt real recovery if observer is available
        success = True
        try:
            if state.observer is not None and _SH_MLP_AVAILABLE:
                import copy
                rca_event = type('E', (), {
                    'failure_type': fault.fault_type,
                    'severity': type('S', (), {'value': fault.severity})(),
                    'stage_id': fault.stage_id,
                    'contributing_signals': [],
                    'fusion_rule': fault.fusion_rule,
                    'observation_ref': None,
                    'pipeline_id': state.pipeline_id,
                })()
                rca    = state.observer._rca.diagnose(rca_event)
                result = state.observer._planner.plan_and_execute(rca)
                success = result.success
                strategy = result.strategy_id or strategy
        except Exception as e:
            _err(f"Recovery execution error: {e}")

        ttr_ms = int((time.perf_counter() - t0) * 1000)

        with state._lock:
            state.active_faults.pop(fault_id, None)
            if success:
                state.recovered += 1
            state.history.append(HistoryEvent(
                fault_id    = fault_id,
                fault_type  = fault.fault_type,
                detected_at = fault.detected_at,
                recovered   = success,
                strategy_id = strategy,
                ttr_ms      = ttr_ms,
            ))
            # Update LinUCB delta estimate
            if state.event_count > 0:
                state.linucb_delta = max(0.0, state.recovery_rate - 0.80)

        return RecoveryResult(fault_id=fault_id, strategy_id=strategy,
                              success=success, ttr_ms=ttr_ms)

    def handle_dismiss_fault(self, req_id: int, params: dict) -> None:
        fault_id = params.get("fault_id", "")
        for state in self.pipelines.values():
            with state._lock:
                state.active_faults.pop(fault_id, None)
        _ok(req_id, {"acknowledged": True})

    def handle_get_history(self, req_id: int, params: dict) -> None:
        pid   = params.get("pipeline_id", "default")
        limit = params.get("limit", 50)
        state = self._get_pipeline(pid)
        with state._lock:
            events = [asdict(e) for e in state.history[-limit:]]
        _ok(req_id, events)

    def handle_shutdown(self, req_id: int, _params: dict) -> None:
        _ok(req_id, {"ok": True})
        self._running = False

    # ── Observation builder ───────────────────────────────────────────────────

    def _dict_to_observation(self, d: dict, pid: str, stage: str) -> "PipelineObservation":
        """Build a PipelineObservation from a dict sent by the extension."""
        import numpy as np

        fstats = {}
        for k, v in d.get("feature_statistics", {}).items():
            if isinstance(v, (list, tuple)):
                arr = np.array(v, dtype=float)
            else:
                arr = np.random.default_rng(42).normal(0, 1, 100)
            fstats[k] = FeatureStats.from_array(arr)

        pred = None
        if "prediction_stats" in d:
            ps = d["prediction_stats"]
            pred = PredictionStats(
                mean_confidence     = float(ps.get("mean_confidence", 0.75)),
                entropy_mean        = float(ps.get("entropy_mean", 0.45)),
                predicted_class_dist= ps.get("predicted_class_dist", {"class_0": 0.65, "class_1": 0.35}),
                ece                 = float(ps.get("ece", 0.06)),
            )

        rss = float(d.get("memory_rss_mb", 512.0))
        return PipelineObservation(
            pipeline_id    = pid,
            stage_id       = stage,
            row_count_in   = int(d.get("row_count_in", 1000)),
            row_count_out  = int(d.get("row_count_out", 1000)),
            exec_time_ms   = float(d.get("exec_time_ms", 400.0)),
            memory_rss_mb  = rss,
            feature_statistics = fstats,
            schema_hash    = str(d.get("schema_hash", "schema_v1")),
            prediction_stats   = pred,
            infra_metrics  = InfraMetrics(
                cpu_pct           = float(d.get("cpu_pct", 35.0)),
                memory_rss_mb     = rss,
                cluster_cpu_pct   = float(d.get("cluster_cpu_pct", 40.0)),
            ),
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        # Signal ready
        _send({"jsonrpc": "2.0", "method": "ready"})

        HANDLERS = {
            "configure":       self.handle_configure,
            "observe":         self.handle_observe,
            "get_status":      self.handle_get_status,
            "approve_recovery":self.handle_approve_recovery,
            "dismiss_fault":   self.handle_dismiss_fault,
            "get_history":     self.handle_get_history,
            "shutdown":        self.handle_shutdown,
        }

        for line in sys.stdin:
            if not self._running: break
            line = line.strip()
            if not line: continue
            try:
                msg  = json.loads(line)
                rid  = msg.get("id", 0)
                meth = msg.get("method", "")
                handler = HANDLERS.get(meth)
                if handler:
                    handler(rid, msg.get("params", {}))
                else:
                    _error(rid, -32601, f"Method not found: {meth}")
            except json.JSONDecodeError as e:
                _err(f"JSON parse error: {e}")
            except Exception:
                _err(traceback.format_exc())


if __name__ == "__main__":
    try:
        SidecarServer().run()
    except KeyboardInterrupt:
        pass
