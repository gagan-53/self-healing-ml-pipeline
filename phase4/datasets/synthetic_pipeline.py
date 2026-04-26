"""
Synthetic Pipeline Dataset — Phase 4B primary validation environment.
Generates controlled observations for all 14 fault types.
Serves as the universal fallback when network datasets unavailable.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from typing import Iterator, Optional
from sh_mlp.contracts.data_structures import (
    PipelineObservation, FeatureStats, InfraMetrics, PredictionStats
)

STAGE_IDS = ["ingestion", "preprocessing", "feature_eng", "training", "evaluation"]

_FAULT_STAGE = {
    "DL-1": "ingestion",    "DL-2": "ingestion",   "DL-3": "training",
    "DL-4": "ingestion",    "DL-5": "ingestion",
    "ML-1": "training",     "ML-2": "training",    "ML-3": "training",
    "ML-4": "training",
    "IL-1": "preprocessing","IL-2": "preprocessing","IL-3": "preprocessing",
    "IL-4": "preprocessing","IL-5": "preprocessing",
}


class SyntheticPipelineDataset:
    """
    Generates SH-MLP PipelineObservation objects for a synthetic 5-stage ML pipeline.
    
    Design parameters match Phase 3 integration test but with:
    - 300-sample feature arrays (Phase 4A calibration fix)
    - Stable schema hash (consistent across cycles)
    - Realistic prediction stats
    """

    def __init__(self, seed: int = 42, n_features: int = 6):
        self.seed = seed
        self.n_features = n_features
        self._rng = np.random.default_rng(seed)
        self._schema_hash = "synthetic_v2_stable"

    def make_obs(self, stage_id: str,
                 rows_in: int = 1000, rows_out: int = 1000,
                 exec_ms: float = None, rss_mb: float = 512.0,
                 n_samples: int = 300) -> PipelineObservation:
        fstats = {}
        for i in range(self.n_features):
            arr = self._rng.normal(0, 1, n_samples)
            fstats[f"f{i}"] = FeatureStats.from_array(arr)
        fstats["target"] = FeatureStats(
            mean=0.30, std=0.46, min_val=0.0, max_val=1.0,
            pct_missing=0.01,
            histogram=[0.70] + [0.0] * 8 + [0.30],
            dtype="int"
        )
        pred = PredictionStats(
            mean_confidence=0.78 + self._rng.normal(0, 0.008),
            entropy_mean=0.45 + self._rng.normal(0, 0.005),
            predicted_class_dist={"class_0": 0.65, "class_1": 0.35},
            ece=0.06 + self._rng.normal(0, 0.003),
        )
        if exec_ms is None:
            exec_ms = 400.0 + self._rng.normal(0, 25)
        return PipelineObservation(
            pipeline_id="synthetic_v2",
            stage_id=stage_id,
            row_count_in=rows_in,
            row_count_out=rows_out,
            exec_time_ms=max(150.0, exec_ms),
            memory_rss_mb=max(64.0, rss_mb + self._rng.normal(0, 12)),
            feature_statistics=fstats,
            schema_hash=self._schema_hash,
            prediction_stats=pred,
            infra_metrics=InfraMetrics(
                cpu_pct=35.0 + self._rng.normal(0, 3),
                memory_rss_mb=rss_mb,
                cluster_cpu_pct=40.0 + self._rng.normal(0, 4),
            ),
        )

    def stream(self, n_cycles: int, stage_id: str = None) -> Iterator[PipelineObservation]:
        stages = [stage_id] if stage_id else STAGE_IDS
        for _ in range(n_cycles):
            for s in stages:
                yield self.make_obs(s)

    def get_fault_stage(self, fault_type: str) -> str:
        return _FAULT_STAGE.get(fault_type, "ingestion")
