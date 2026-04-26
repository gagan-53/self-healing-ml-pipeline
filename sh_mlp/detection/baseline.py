"""
Baseline Profile Manager
Manages the statistical baseline used by all rule-based detectors.
Updates via Exponential Moving Average (alpha=0.05).
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from sh_mlp.contracts import (
    BaselineProfile, FeatureBaseline, PipelineObservation, FeatureStats
)


class BaselineManager:
    """
    Manages one BaselineProfile per pipeline.
    Initialised during warm-up; updated via EMA on each observation.
    """

    def __init__(self, pipeline_id: str, warmup_cycles: int = 30,
                 ema_alpha: float = 0.05):
        self.profile = BaselineProfile(
            pipeline_id=pipeline_id,
            warmup_cycles=warmup_cycles,
            ema_alpha=ema_alpha,
        )
        self._warmup_buffer: list = []
        self._cycles_seen: int = 0

    # ── Public interface ──────────────────────────────────────────

    def update(self, obs: PipelineObservation) -> bool:
        """
        Ingest one observation.
        Returns True once the baseline is ready (warmup complete).
        """
        self._cycles_seen += 1

        if not self.profile.is_ready:
            self._warmup_buffer.append(obs)
            if self._cycles_seen >= self.profile.warmup_cycles:
                self._initialise_from_warmup()
            return self.profile.is_ready

        self._ema_update(obs)
        return True

    def get(self) -> BaselineProfile:
        return self.profile

    def is_ready(self) -> bool:
        return self.profile.is_ready

    # ── Warm-up initialisation ────────────────────────────────────

    def _initialise_from_warmup(self):
        """Compute initial baseline from accumulated warm-up observations."""
        buf = self._warmup_buffer

        # Feature baselines — aggregate across all warmup observations
        all_features: Dict[str, list] = {}
        for obs in buf:
            for fname, fstats in obs.feature_statistics.items():
                all_features.setdefault(fname, []).append(fstats)

        for fname, stats_list in all_features.items():
            means = [s.mean for s in stats_list]
            stds  = [s.std  for s in stats_list]
            # Average histogram across warmup observations
            hists = np.array([s.histogram for s in stats_list])
            avg_hist = hists.mean(axis=0).tolist()
            self.profile.feature_baselines[fname] = FeatureBaseline(
                feature_name=fname,
                mean=float(np.mean(means)),
                std=float(np.mean(stds)),
                histogram=avg_hist,
                pct_missing=float(np.mean([s.pct_missing for s in stats_list])),
                dtype=stats_list[0].dtype,
            )

        # Infrastructure baselines — per stage
        stages: Dict[str, list] = {}
        for obs in buf:
            stages.setdefault(obs.stage_id, []).append(obs)

        for sid, obs_list in stages.items():
            times = [o.exec_time_ms for o in obs_list]
            rows  = [o.row_count_out for o in obs_list]
            self.profile.exec_time_mean[sid] = float(np.mean(times))
            self.profile.exec_time_std[sid]  = float(np.std(times)) + 1e-6
            self.profile.row_count_mean[sid]  = float(np.mean(rows))
            if obs_list:
                self.profile.schema_hash[sid] = obs_list[-1].schema_hash

        self.profile.is_ready = True
        self.profile.updated_at = datetime.utcnow()
        self._warmup_buffer.clear()

    # ── EMA update ────────────────────────────────────────────────

    def _ema_update(self, obs: PipelineObservation):
        """Apply EMA update after warm-up. O(k) in number of features."""
        a = self.profile.ema_alpha

        # Feature statistics EMA
        for fname, fstats in obs.feature_statistics.items():
            if fname in self.profile.feature_baselines:
                bl = self.profile.feature_baselines[fname]
                bl.mean = (1 - a) * bl.mean + a * fstats.mean
                bl.std  = (1 - a) * bl.std  + a * fstats.std
                bl.pct_missing = (1 - a) * bl.pct_missing + a * fstats.pct_missing
                # EMA on histogram bins
                new_hist = np.array(bl.histogram)
                obs_hist = np.array(fstats.histogram)
                bl.histogram = ((1 - a) * new_hist + a * obs_hist).tolist()

        # Infrastructure EMA
        sid = obs.stage_id
        if sid in self.profile.exec_time_mean:
            prev_mean = self.profile.exec_time_mean[sid]
            prev_std  = self.profile.exec_time_std[sid]
            self.profile.exec_time_mean[sid] = (1-a)*prev_mean + a*obs.exec_time_ms
            delta = obs.exec_time_ms - prev_mean
            self.profile.exec_time_std[sid]  = (1-a)*prev_std + a*abs(delta)
            self.profile.row_count_mean[sid]  = (
                (1-a)*self.profile.row_count_mean.get(sid, obs.row_count_out)
                + a * obs.row_count_out
            )

        self.profile.updated_at = datetime.utcnow()


def compute_schema_hash(feature_names: list, dtypes: list) -> str:
    """Stable schema hash from feature names and dtypes."""
    schema_str = json.dumps(sorted(zip(feature_names, dtypes)))
    return hashlib.md5(schema_str.encode()).hexdigest()[:16]
