"""
Detection Engine — Module 1
Coordinates: DetectorRegistry + FusionPolicy + BaselineManager + async IsolationForest
"""
from __future__ import annotations
import threading
import time
from typing import Optional, List
import numpy as np
from sklearn.ensemble import IsolationForest

from sh_mlp.contracts import (
    PipelineObservation, FailureEvent, BaselineProfile
)
from sh_mlp.detection.baseline import BaselineManager
from sh_mlp.detection.detectors import DetectorRegistry
from sh_mlp.detection.fusion import FusionPolicy


class DetectionEngine:
    """
    Main Detection Engine (Module 1).

    Usage:
        engine = DetectionEngine(pipeline_id="my_pipeline")
        event = engine.observe(observation)  # returns FailureEvent | None
    """

    def __init__(self, pipeline_id: str, warmup_cycles: int = 30,
                 ema_alpha: float = 0.05, observation_freq: int = 5,
                 if_n_estimators: int = 100, max_rss_mb: float = 8192.0):

        self.pipeline_id = pipeline_id
        self.observation_freq = observation_freq
        self._cycle = 0

        # Sub-components
        self._baseline_mgr = BaselineManager(pipeline_id, warmup_cycles, ema_alpha)
        self._detectors    = DetectorRegistry(max_rss_mb)
        self._fusion       = FusionPolicy()

        # Async IsolationForest
        self._if_model: Optional[IsolationForest] = None
        self._if_buffer: List[list] = []
        self._if_lock = threading.Lock()
        self._last_if_score = 0.0
        self._last_if_features: list = []
        self._if_n_estimators = if_n_estimators
        self._if_ready = False
        self._observation_vectors: List[list] = []

        # Structured event log
        self._event_log: list = []
        self._observation_log: list = []

    # ── Public API ────────────────────────────────────────────────

    def observe(self, obs: PipelineObservation) -> Optional[FailureEvent]:
        """
        Process one PipelineObservation.
        Returns FailureEvent if fusion fires, else None.
        Runtime: O(k) inline; async IsolationForest in background.
        """
        self._cycle += 1
        self._observation_log.append(obs)

        # Update baseline (blocks until warmup complete)
        ready = self._baseline_mgr.update(obs)
        if not ready:
            return None

        baseline = self._baseline_mgr.get()

        # Run rule-based detectors (inline, O(k))
        signals = self._detectors.run_all(obs, baseline)

        # Collect observation vector for IsolationForest
        vec = self._build_feature_vector(obs, baseline)
        self._observation_vectors.append(vec)

        # Train / score IsolationForest asynchronously
        if self._cycle % self.observation_freq == 0:
            threading.Thread(
                target=self._async_if_update,
                args=(list(self._observation_vectors),),
                daemon=True
            ).start()

        # Retrieve last async IF score (from previous cycle)
        with self._if_lock:
            if_score    = self._last_if_score
            if_features = self._last_if_features

        # Apply fusion policy
        event = self._fusion.evaluate(
            current_signals=signals,
            obs=obs,
            learned_score=if_score,
            learned_features=if_features,
            pipeline_id=self.pipeline_id,
        )

        if event is not None:
            self._event_log.append(event)

        return event

    def get_baseline(self) -> Optional[BaselineProfile]:
        return self._baseline_mgr.get() if self._baseline_mgr.is_ready() else None

    def get_event_log(self) -> list:
        return list(self._event_log)

    def is_warmed_up(self) -> bool:
        return self._baseline_mgr.is_ready()

    # ── Async IsolationForest ─────────────────────────────────────

    def _async_if_update(self, vectors: list):
        """Train IF on accumulated vectors; score latest. Non-blocking."""
        if len(vectors) < 20:
            return
        X = np.array(vectors, dtype=float)
        try:
            if not self._if_ready or len(vectors) % 50 == 0:
                model = IsolationForest(
                    n_estimators=self._if_n_estimators,
                    contamination=0.05,
                    random_state=42,
                    n_jobs=1
                )
                model.fit(X)
                with self._if_lock:
                    self._if_model = model
                    self._if_ready = True

            if self._if_model is not None:
                latest = X[-1:, :]
                raw_score = self._if_model.score_samples(latest)[0]
                # Convert to [0,1] anomaly score (higher = more anomalous)
                score = float(1 - (raw_score + 0.5))
                score = max(0.0, min(1.0, score))

                # Feature attribution by ablation (top-3)
                top_features = self._ablation_attribution(latest[0])

                with self._if_lock:
                    self._last_if_score    = score
                    self._last_if_features = top_features
        except Exception:
            pass

    def _ablation_attribution(self, vec: np.ndarray) -> list:
        """Identify top-3 features contributing to anomaly by ablation."""
        if self._if_model is None:
            return []
        baseline_score = self._if_model.score_samples(vec.reshape(1, -1))[0]
        contributions = []
        for i in range(len(vec)):
            ablated = vec.copy()
            ablated[i] = 0.0
            ablated_score = self._if_model.score_samples(ablated.reshape(1, -1))[0]
            contributions.append((i, abs(ablated_score - baseline_score)))
        contributions.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in contributions[:3]]

    # ── Feature vector for IsolationForest ───────────────────────

    def _build_feature_vector(self, obs: PipelineObservation,
                              baseline: BaselineProfile) -> list:
        """Build a fixed-length feature vector from observation. O(k)."""
        vec = []
        # Feature means (normalised by baseline mean)
        for fname in sorted(baseline.feature_baselines.keys())[:20]:
            bl_mean = baseline.feature_baselines[fname].mean
            obs_mean = obs.feature_statistics.get(fname,
                       type("X", (), {"mean": bl_mean})()).mean
            vec.append(obs_mean / (bl_mean + 1e-8))

        # Pad to 20 dimensions
        vec = (vec + [1.0] * 20)[:20]

        # Infrastructure ratios
        sid = obs.stage_id
        mu_t = baseline.exec_time_mean.get(sid, obs.exec_time_ms + 1)
        mu_r = baseline.row_count_mean.get(sid, obs.row_count_out + 1)
        vec += [
            obs.exec_time_ms / (mu_t + 1e-8),
            obs.memory_rss_mb / 8192.0,
            obs.row_count_out / (mu_r + 1e-8),
            float(obs.exit_code != 0),
            obs.infra_metrics.cpu_pct / 100.0,
        ]
        return vec[:25]
