"""
Adult Income Dataset Connector — Phase 4A
Loads UCI Adult Income dataset and wraps it as SH-MLP pipeline observations.

Dataset: https://archive.ics.uci.edu/ml/datasets/Adult
License: CC0 Public Domain
Task:    Binary classification (income >50K vs ≤50K)
Rows:    48,842 (train + test combined)
Features: 14 (8 categorical, 6 continuous)

Drift injection support:
  DL-1: Temporal split by age > 50 vs ≤ 50 to simulate covariate shift
  DL-3: Class prevalence change (filter to high-income subset)
"""
from __future__ import annotations
import os, io
import numpy as np
import pandas as pd
from typing import Tuple, Iterator, Optional

from sh_mlp.contracts.data_structures import (
    PipelineObservation, FeatureStats, InfraMetrics, PredictionStats
)

# Download URL (fallback to synthetic if no network)
_ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
_ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

_NUMERIC_COLS = ["age", "fnlwgt", "education_num", "capital_gain",
                 "capital_loss", "hours_per_week"]
_CATEGORICAL_COLS = ["workclass", "education", "marital_status", "occupation",
                     "relationship", "race", "sex", "native_country"]


def _generate_synthetic_adult(n_rows: int = 48842,
                               seed: int = 42,
                               high_income_rate: float = 0.24) -> pd.DataFrame:
    """
    Generate synthetic Adult Income-like data when the real dataset is unavailable.
    Preserves approximate statistical properties.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age":           rng.integers(17, 90, n_rows),
        "fnlwgt":        rng.integers(12285, 1490400, n_rows),
        "education_num": rng.integers(1, 16, n_rows),
        "capital_gain":  np.where(rng.random(n_rows) < 0.08,
                                  rng.integers(1, 100000, n_rows), 0),
        "capital_loss":  np.where(rng.random(n_rows) < 0.05,
                                  rng.integers(1, 4000, n_rows), 0),
        "hours_per_week": rng.integers(1, 99, n_rows),
        "workclass":     rng.choice(
            ["Private", "Self-emp", "Gov", "Without-pay"], n_rows,
            p=[0.70, 0.12, 0.16, 0.02]
        ),
        "education":     rng.choice(
            ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate", "Other"],
            n_rows, p=[0.32, 0.22, 0.16, 0.06, 0.02, 0.22]
        ),
        "marital_status":rng.choice(
            ["Married", "Never-married", "Divorced", "Other"], n_rows,
            p=[0.46, 0.33, 0.14, 0.07]
        ),
        "occupation":    rng.choice(
            ["Prof-specialty", "Craft-repair", "Exec-managerial",
             "Adm-clerical", "Sales", "Other"], n_rows,
            p=[0.13, 0.12, 0.12, 0.11, 0.10, 0.42]
        ),
        "sex":           rng.choice(["Male", "Female"], n_rows, p=[0.67, 0.33]),
        "income":        rng.choice([0, 1], n_rows,
                                    p=[1-high_income_rate, high_income_rate]),
    })
    return df


class AdultIncomeLoader:
    """
    Loads and serves Adult Income data as SH-MLP PipelineObservation objects.

    Parameters
    ----------
    use_synthetic : bool
        If True, always use synthetic data (no network required).
    split_policy : str
        'temporal' — split by age (young vs old) to induce covariate shift
        'random'   — random 80/20 train/test split
    chunk_size : int
        Rows per PipelineObservation
    seed : int
    """

    def __init__(self, use_synthetic: bool = True,
                 split_policy: str = "temporal",
                 chunk_size: int = 500,
                 seed: int = 42):
        self.use_synthetic = use_synthetic
        self.split_policy  = split_policy
        self.chunk_size    = chunk_size
        self.seed          = seed
        self._rng          = np.random.default_rng(seed)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load dataset (synthetic or real)."""
        if self._df is not None:
            return self._df
        if self.use_synthetic:
            self._df = _generate_synthetic_adult(seed=self.seed)
        else:
            try:
                self._df = pd.read_csv(
                    _ADULT_URL, header=None, names=_COLUMNS,
                    na_values=" ?", skipinitialspace=True
                )
                self._df["income"] = (self._df["income"]
                                      .str.strip().str.replace(".", "")
                                      .map({">50K": 1, "<=50K": 0}))
                self._df = self._df.dropna()
            except Exception:
                print("  [AdultIncome] Network unavailable — using synthetic data")
                self._df = _generate_synthetic_adult(seed=self.seed)
        return self._df

    def get_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (train_df, test_df) split.
        Temporal split: train=age≤40, test=age>40 (simulates time-based drift).
        """
        df = self.load()
        if self.split_policy == "temporal":
            train = df[df["age"] <= 40].copy()
            test  = df[df["age"] >  40].copy()
        else:
            mask  = self._rng.random(len(df)) < 0.8
            train = df[mask].copy()
            test  = df[~mask].copy()
        return train, test

    def get_baseline_profile(self) -> Dict:
        """Compute statistical baseline for the training split."""
        train, _ = self.get_split()
        profile = {}
        for col in _NUMERIC_COLS:
            if col in train.columns:
                profile[col] = {
                    "mean": float(train[col].mean()),
                    "std":  float(train[col].std()),
                    "min":  float(train[col].min()),
                    "max":  float(train[col].max()),
                }
        profile["income_rate"] = float(train["income"].mean())
        profile["n_rows"]      = len(train)
        return profile

    def to_observation(self, chunk: pd.DataFrame, stage_id: str,
                       pipeline_id: str = "adult_income",
                       schema_hash: str = "adult_v1") -> PipelineObservation:
        """Convert a DataFrame chunk to a PipelineObservation."""
        fstats = {}
        for col in _NUMERIC_COLS:
            if col in chunk.columns:
                arr = chunk[col].dropna().values.astype(float)
                if len(arr) > 0:
                    fstats[col] = FeatureStats.from_array(arr)

        # Add target column stats
        if "income" in chunk.columns:
            income_rate = float(chunk["income"].mean())
            fstats["target"] = FeatureStats(
                mean=income_rate, std=float(chunk["income"].std()),
                min_val=0.0, max_val=1.0,
                pct_missing=float(chunk["income"].isna().mean()),
                histogram=[1-income_rate, 0, 0, 0, 0, 0, 0, 0, 0, income_rate],
                dtype="int"
            )

        # Simulated prediction stats (as if model ran)
        pred = PredictionStats(
            mean_confidence=0.74 + self._rng.normal(0, 0.01),
            entropy_mean=0.52 + self._rng.normal(0, 0.005),
            predicted_class_dist={
                "class_0": 1 - float(chunk["income"].mean()),
                "class_1": float(chunk["income"].mean())
            } if "income" in chunk.columns else {"class_0": 0.76, "class_1": 0.24},
            ece=0.07
        )

        return PipelineObservation(
            pipeline_id=pipeline_id,
            stage_id=stage_id,
            row_count_in=len(chunk),
            row_count_out=len(chunk),
            exec_time_ms=120.0 + self._rng.normal(0, 15),
            memory_rss_mb=380.0 + self._rng.normal(0, 20),
            feature_statistics=fstats,
            schema_hash=schema_hash,
            prediction_stats=pred,
            infra_metrics=InfraMetrics(
                cpu_pct=42.0 + self._rng.normal(0, 4),
                memory_rss_mb=380.0,
                cluster_cpu_pct=45.0 + self._rng.normal(0, 5)
            ),
        )

    def stream_chunks(self, split: str = "train",
                      shuffle: bool = True) -> Iterator[PipelineObservation]:
        """
        Yield PipelineObservation objects one chunk at a time.
        split: 'train' or 'test'
        """
        train_df, test_df = self.get_split()
        df = train_df if split == "train" else test_df
        if shuffle:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        n_chunks = max(1, len(df) // self.chunk_size)
        for i in range(n_chunks):
            chunk = df.iloc[i*self.chunk_size:(i+1)*self.chunk_size]
            yield self.to_observation(chunk, stage_id="ingestion")

    def inject_dl1_drift(self, obs: PipelineObservation) -> PipelineObservation:
        """
        Apply DL-1 drift to an Adult Income observation.
        Simulates covariate shift by shifting age distribution and capital gain.
        """
        import copy
        obs = copy.deepcopy(obs)
        if "age" in obs.feature_statistics:
            fs = obs.feature_statistics["age"]
            fs.mean = fs.mean * 1.35  # older population
            hist = np.array(fs.histogram)
            shifted = np.roll(hist, 3)
            shifted = np.clip(shifted, 1e-6, None)
            shifted /= shifted.sum()
            fs.histogram = shifted.tolist()
        if "capital_gain" in obs.feature_statistics:
            fs = obs.feature_statistics["capital_gain"]
            fs.mean = fs.mean * 2.5
        return obs

    def inject_dl3_shift(self, obs: PipelineObservation,
                         new_income_rate: float = 0.52) -> PipelineObservation:
        """
        Apply DL-3 label shift to an Adult Income observation.
        Increases high-income proportion to simulate label distribution shift.
        """
        import copy
        obs = copy.deepcopy(obs)
        if obs.prediction_stats:
            obs.prediction_stats.predicted_class_dist = {
                "class_0": 1 - new_income_rate,
                "class_1": new_income_rate
            }
        if "target" in obs.feature_statistics:
            obs.feature_statistics["target"].mean = new_income_rate
        return obs


# Alias for convenience
from typing import Dict
