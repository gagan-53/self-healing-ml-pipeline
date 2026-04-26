"""
Recovery Planner — Module 3
Autonomy decision matrix + strategy selection + execution + verification.
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Optional, List

from sh_mlp.contracts import (
    RCAResult, RecoveryAction, RecoveryOutcome, ActionStep,
    Severity, DecisionMode, StrategyWeights
)
from sh_mlp.recovery.strategies import get_strategies, get_strategy, Strategy


# ── Autonomy Decision Matrix ──────────────────────────────────────

class AutonomyMatrix:
    """
    Implements the 6-path autonomy decision matrix from Phase 2 Table 6.
    ALL four conditions must hold for autonomous execution.
    """

    AUTONOMOUS_SEVERITY  = {Severity.LOW, Severity.MEDIUM}
    CONFIDENCE_THRESHOLD = 0.85
    MIN_PRIOR_SUCCESSES  = 2

    @classmethod
    def decide(cls, severity: Severity, confidence: float,
               strategy: Strategy, prior_successes: int) -> DecisionMode:

        # CRITICAL → immediate halt
        if severity == Severity.CRITICAL:
            return DecisionMode.IMMEDIATE_HALT

        # Low confidence → escalate
        if confidence < 0.70:
            return DecisionMode.ESCALATE

        # HIGH severity → always HITL
        if severity == Severity.HIGH:
            return DecisionMode.HITL

        # IL-4 is never autonomous (irreversible)
        if not strategy.reversible:
            return DecisionMode.HITL

        # Medium/Low but low confidence → HITL
        if confidence < cls.CONFIDENCE_THRESHOLD:
            return DecisionMode.HITL

        # Novel strategy (no prior successes) → supervised
        if prior_successes < cls.MIN_PRIOR_SUCCESSES:
            return DecisionMode.SUPERVISED

        # All conditions met → autonomous
        return DecisionMode.AUTONOMOUS


# ── Recovery Planner ──────────────────────────────────────────────

class RecoveryPlanner:
    """
    Module 3: Selects and executes recovery strategies.
    Uses LinUCB weights from Feedback Store to rank strategies.
    Applies autonomy decision matrix before execution.
    """

    VERIFY_WAIT_SEC = 5     # wait after recovery before verifying
    HITL_TIMEOUT_SEC = 300  # 5 minutes (shortened for prototype; 30 min in production)

    def __init__(self, feedback_store=None, auto_approve: bool = False):
        """
        feedback_store: FeedbackStore instance (M4)
        auto_approve: if True, HITL requests are auto-approved (for testing)
        """
        self._feedback_store = feedback_store
        self._auto_approve   = auto_approve
        self._outcome_log: List[RecoveryOutcome] = []

    def plan_and_execute(self, rca: RCAResult,
                         pipeline_context: dict = None) -> RecoveryOutcome:
        """
        Main entry point: receives RCAResult, returns RecoveryOutcome.
        """
        failure_type = rca.primary_cause.signal_type
        if failure_type == "UNKNOWN":
            failure_type = self._infer_type_from_event(rca)

        severity   = self._infer_severity(rca)
        confidence = rca.confidence

        # Get strategies ordered by LinUCB score
        candidates = get_strategies(failure_type)
        if not candidates:
            # Fallback for unmapped failure type
            return self._deferred_outcome(rca, failure_type)

        ctx_vec = self._build_context_vector(rca, failure_type, severity, pipeline_context)
        ordered = self._rank_strategies(candidates, ctx_vec, failure_type)
        selected = ordered[0]

        # Count prior successes for this strategy
        prior_successes = (self._feedback_store.count_successes(failure_type, selected.strategy_id)
                          if self._feedback_store else 0)

        # Apply autonomy matrix
        decision = AutonomyMatrix.decide(severity, confidence, selected, prior_successes)

        # Build RecoveryAction
        action = RecoveryAction(
            strategy_id=selected.strategy_id,
            failure_type=failure_type,
            rca_ref=rca.rca_id,
            decision_mode=decision,
            steps=[ActionStep(i, s["operation"], s.get("parameters", {}))
                   for i, s in enumerate(selected.steps)],
            reversible=selected.reversible,
            estimated_duration_sec=selected.estimated_duration_sec,
            verify_criterion=selected.verify_criterion,
        )

        # Execute
        if decision == DecisionMode.IMMEDIATE_HALT:
            return self._halt_outcome(action, ctx_vec)

        if decision in (DecisionMode.HITL, DecisionMode.SUPERVISED, DecisionMode.ESCALATE):
            approved = self._request_approval(action, rca, ordered[1:3])
            if not approved:
                return self._deferred_outcome(rca, failure_type)

        # Simulate execution (in production: call PipelineObserver.execute())
        start_time = time.time()
        exec_success = self._simulate_execution(selected, failure_type)
        elapsed = int(time.time() - start_time)

        # Verify
        time.sleep(self.VERIFY_WAIT_SEC * 0.01)  # shortened for prototype
        verified = exec_success  # in production: check verify_criterion

        success_float = 1.0 if verified else (0.5 if exec_success else 0.0)
        perf_delta    = 0.05 if verified else -0.05

        outcome = RecoveryOutcome(
            action_ref=action.action_id,
            failure_type=failure_type,
            strategy_id=selected.strategy_id,
            context_vector=ctx_vec,
            success=verified,
            success_float=success_float,
            time_to_recovery=max(elapsed, 1),
            performance_delta=perf_delta,
            decision_mode=decision.value,
        )

        self._outcome_log.append(outcome)

        # Record in Feedback Store
        if self._feedback_store:
            self._feedback_store.record_and_update(outcome)

        return outcome

    # ── Private helpers ───────────────────────────────────────────

    def _rank_strategies(self, candidates: List[Strategy],
                          ctx_vec: list, failure_type: str) -> List[Strategy]:
        """Rank by LinUCB score if available, else by priority."""
        if self._feedback_store is None:
            return candidates
        scored = []
        for s in candidates:
            score = self._feedback_store.ucb_score(s.strategy_id, ctx_vec)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    def _build_context_vector(self, rca: RCAResult, failure_type: str,
                              severity: Severity, context: dict) -> list:
        """Build 12-dim context vector for LinUCB."""
        ctx = context or {}
        type_hash = abs(hash(failure_type)) % 14 / 14.0
        sev_enc   = severity.ordinal() / 3.0
        import math, time as _t
        hour = datetime.utcnow().hour
        return [
            type_hash,                                          # x1: failure type
            sev_enc,                                            # x2: severity
            rca.confidence,                                     # x3: rca confidence
            min(rca.primary_cause.change_magnitude, 1.0),      # x4: signal magnitude
            ctx.get("time_since_last_failure_h", 24.0) / 168.0, # x5: hours since last
            ctx.get("pipeline_age_days", 30.0) / 365.0,        # x6: pipeline age
            min(ctx.get("data_volume_ratio", 1.0), 3.0) / 3.0, # x7: volume ratio
            float(ctx.get("upstream_change", False)),           # x8: upstream change flag
            ctx.get("prior_success_rate", 0.5),                 # x9: historical success
            math.sin(2 * math.pi * hour / 24),                  # x10a: time of day sin
            math.cos(2 * math.pi * hour / 24),                  # x10b: time of day cos
            ctx.get("infra_pressure", 0.3),                     # x11: infra pressure
        ]

    def _simulate_execution(self, strategy: Strategy, failure_type: str) -> bool:
        """
        Prototype execution simulator.
        In production: call PipelineObserver.execute(strategy.steps).
        Returns True on simulated success (80% base rate).
        """
        import random
        success_probs = {
            "DL-1": 0.85, "DL-2": 0.90, "DL-3": 0.80, "DL-4": 0.95,
            "DL-5": 0.75, "ML-1": 0.82, "ML-2": 0.78, "ML-3": 0.70,
            "ML-4": 0.88, "IL-1": 0.85, "IL-2": 0.90, "IL-3": 0.92,
            "IL-4": 0.60, "IL-5": 0.95,
        }
        return random.random() < success_probs.get(failure_type, 0.80)

    def _request_approval(self, action: RecoveryAction, rca: RCAResult,
                          alternatives: list) -> bool:
        """HITL approval request. Auto-approves in test mode."""
        if self._auto_approve:
            action.approved_by = "AUTO_APPROVE"
            action.approved_at = datetime.utcnow()
            return True
        # In production: show rca.narrative + action details via HITL interface
        print(f"\n[HITL REQUEST] {action.decision_mode.value}")
        print(f"  Failure: {action.failure_type} | Strategy: {action.strategy_id}")
        print(f"  Narrative: {rca.narrative[:120]}...")
        try:
            resp = input("  Approve? (y/n, default=y): ").strip().lower()
            approved = resp in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            approved = True  # default approve in non-interactive mode
        if approved:
            action.approved_by = "OPERATOR"
            action.approved_at = datetime.utcnow()
        return approved

    def _halt_outcome(self, action: RecoveryAction, ctx: list) -> RecoveryOutcome:
        return RecoveryOutcome(
            action_ref=action.action_id,
            failure_type=action.failure_type,
            strategy_id="HALT",
            context_vector=ctx,
            success=False, success_float=0.0,
            time_to_recovery=0, performance_delta=0.0,
            decision_mode=DecisionMode.IMMEDIATE_HALT.value,
            notes="Pipeline halted. Manual intervention required.",
        )

    def _deferred_outcome(self, rca: RCAResult, failure_type: str) -> RecoveryOutcome:
        return RecoveryOutcome(
            action_ref="DEFERRED",
            failure_type=failure_type,
            strategy_id="DEFERRED",
            context_vector=[0.0] * 12,
            success=False, success_float=0.0,
            time_to_recovery=0, performance_delta=0.0,
            decision_mode=DecisionMode.DEFERRED.value,
        )

    def _infer_type_from_event(self, rca: RCAResult) -> str:
        return rca.event_ref.split("_")[0] if "_" in rca.event_ref else "DL-1"

    def _infer_severity(self, rca: RCAResult) -> Severity:
        if rca.confidence > 0.85:
            return Severity.MEDIUM
        return Severity.HIGH

    def get_outcome_log(self) -> list:
        return list(self._outcome_log)
