"""
LinUCB Contextual Bandit — Feedback Store Module 4
Implements strategy selection and weight updates per Phase 2 Section 6.2.
"""
from __future__ import annotations
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from sh_mlp.contracts import (
    RecoveryOutcome, StrategyWeights, ArmParameters
)
from sh_mlp.recovery.strategies import STRATEGY_LIBRARY

CONTEXT_DIM = 12    # d = 12-dimensional context vector
ALPHA       = 0.5   # exploration constant


class LinUCB:
    """
    Contextual LinUCB bandit for recovery strategy selection.

    For each arm a (strategy):
      A_a  = X_a^T X_a + I_d   (regularised design matrix, d×d)
      b_a  = X_a^T r_a          (reward-weighted context accumulator, d)
      θ_a  = A_a^{-1} b_a       (estimated reward parameter, d)

    UCB score: x^T θ_a + α √(x^T A_a^{-1} x)
    """

    def __init__(self, strategy_ids: List[str], alpha: float = ALPHA,
                 d: int = CONTEXT_DIM):
        self.alpha = alpha
        self.d     = d

        # Initialise per-arm parameters
        self._A:     Dict[str, np.ndarray] = {}
        self._b:     Dict[str, np.ndarray] = {}
        self._theta: Dict[str, np.ndarray] = {}
        self._A_inv: Dict[str, np.ndarray] = {}
        self._n_obs: Dict[str, int]   = {}
        self._total_r: Dict[str, float] = {}
        self._success: Dict[str, int]   = {}

        for sid in strategy_ids:
            self._init_arm(sid)

    def _init_arm(self, sid: str):
        self._A[sid]     = np.eye(self.d)
        self._b[sid]     = np.zeros(self.d)
        self._theta[sid] = np.zeros(self.d)
        self._A_inv[sid] = np.eye(self.d)
        self._n_obs[sid]   = 0
        self._total_r[sid] = 0.0
        self._success[sid] = 0

    def ucb_score(self, strategy_id: str, context: List[float]) -> float:
        """
        Compute UCB score for a strategy given current context.
        Higher score → more likely to be selected.
        """
        if strategy_id not in self._A:
            self._init_arm(strategy_id)

        x       = np.array(context[:self.d], dtype=float)
        theta   = self._theta[strategy_id]
        A_inv   = self._A_inv[strategy_id]

        exploit = float(x @ theta)
        explore = float(self.alpha * np.sqrt(x @ A_inv @ x))
        return exploit + explore

    def update(self, strategy_id: str, context: List[float], reward: float):
        """
        Update arm parameters after observing reward.
        O(d^2) matrix update; A^{-1} computed lazily.
        """
        if strategy_id not in self._A:
            self._init_arm(strategy_id)

        x = np.array(context[:self.d], dtype=float)

        self._A[strategy_id]   += np.outer(x, x)
        self._b[strategy_id]   += reward * x
        self._A_inv[strategy_id] = np.linalg.inv(self._A[strategy_id])
        self._theta[strategy_id] = self._A_inv[strategy_id] @ self._b[strategy_id]
        self._n_obs[strategy_id]   += 1
        self._total_r[strategy_id] += reward

    def count_successes(self, strategy_id: str) -> int:
        return self._success.get(strategy_id, 0)

    def record_success(self, strategy_id: str):
        self._success[strategy_id] = self._success.get(strategy_id, 0) + 1

    def to_strategy_weights(self) -> StrategyWeights:
        arms = {}
        for sid in self._A:
            arms[sid] = ArmParameters(
                strategy_id=sid,
                A_matrix=self._A[sid].tolist(),
                b_vector=self._b[sid].tolist(),
                theta_vector=self._theta[sid].tolist(),
                n_observations=self._n_obs[sid],
                total_reward=self._total_r[sid],
                success_count=self._success.get(sid, 0),
            )
        return StrategyWeights(
            version=sum(self._n_obs.values()),
            updated_at=datetime.utcnow(),
            arms=arms,
        )


class FeedbackStore:
    """
    Feedback Store (Module 4).
    Records RecoveryOutcomes and updates LinUCB strategy weights.
    """

    # Reward weights (from Phase 2 Section 6.4)
    W_SUCCESS = 0.6
    W_SPEED   = 0.2
    W_PERF    = 0.2

    def __init__(self):
        strategy_ids = [s.strategy_id for s in STRATEGY_LIBRARY]
        self._bandit = LinUCB(strategy_ids)
        self._log: List[dict] = []
        self._success_by_type: Dict[str, Dict[str, int]] = {}

    # ── Public API ────────────────────────────────────────────────

    def record_and_update(self, outcome: RecoveryOutcome):
        """Record outcome and update LinUCB for selected arm."""
        reward = self._compute_reward(outcome)

        self._log.append({
            "timestamp":    outcome.timestamp.isoformat(),
            "failure_type": outcome.failure_type,
            "strategy_id":  outcome.strategy_id,
            "context":      outcome.context_vector,
            "reward":       round(reward, 4),
            "success":      outcome.success,
            "ttr_sec":      outcome.time_to_recovery,
        })

        self._bandit.update(outcome.strategy_id, outcome.context_vector, reward)

        if outcome.success:
            self._bandit.record_success(outcome.strategy_id)
            self._success_by_type \
                .setdefault(outcome.failure_type, {}) \
                .setdefault(outcome.strategy_id, 0)
            self._success_by_type[outcome.failure_type][outcome.strategy_id] += 1

    def ucb_score(self, strategy_id: str, context: List[float]) -> float:
        return self._bandit.ucb_score(strategy_id, context)

    def count_successes(self, failure_type: str, strategy_id: str) -> int:
        return (self._success_by_type
                .get(failure_type, {})
                .get(strategy_id, 0))

    def get_weights(self) -> StrategyWeights:
        return self._bandit.to_strategy_weights()

    def get_log(self) -> List[dict]:
        return list(self._log)

    def summary(self) -> dict:
        total = len(self._log)
        successes = sum(1 for e in self._log if e["success"])
        return {
            "total_events":   total,
            "success_count":  successes,
            "success_rate":   round(successes / total, 3) if total > 0 else 0.0,
            "avg_reward":     round(sum(e["reward"] for e in self._log) / total, 3)
                              if total > 0 else 0.0,
        }

    # ── Reward computation ────────────────────────────────────────

    def _compute_reward(self, outcome: RecoveryOutcome) -> float:
        """
        Composite reward: 0.6×success + 0.2×speed + 0.2×performance.
        All components in [0, 1]; total reward in [0, 1].
        """
        # Success component
        success_r = outcome.success_float

        # Speed component: normalise by max acceptable TTR for this strategy
        from sh_mlp.recovery.strategies import get_strategy
        strat = get_strategy(outcome.strategy_id)
        max_ttr = strat.max_acceptable_ttr_sec if strat else 3600
        speed_r = max(0.0, 1.0 - outcome.time_to_recovery / max_ttr)

        # Performance component: delta in [-1, 0] → [0, 1]
        perf_r = max(0.0, 1.0 + outcome.performance_delta)

        return (self.W_SUCCESS * success_r +
                self.W_SPEED   * speed_r   +
                self.W_PERF    * perf_r)
