"""
SH-MLP: Self-Healing Machine Learning Pipeline
Phase 3 Prototype — Complete Implementation

Author: Solo Research Project
Phase: 3 — Prototype Development (Weeks 9–14)
"""

__version__ = "0.1.0"

from sh_mlp.contracts import *
from sh_mlp.detection import DetectionEngine
from sh_mlp.rca import RCAEngine, PipelineDAG, SignalHistory
from sh_mlp.recovery import RecoveryPlanner
from sh_mlp.feedback import FeedbackStore
from sh_mlp.observer.pipeline_observer import PipelineObserver


def create_pipeline(pipeline_id: str, stage_ids: list,
                    warmup_cycles: int = 30,
                    auto_approve: bool = False,
                    auto_recover: bool = True) -> PipelineObserver:
    """
    Factory function: creates a fully wired PipelineObserver.

    Args:
        pipeline_id:   Unique identifier for this pipeline
        stage_ids:     Ordered list of stage names (linear DAG assumed)
        warmup_cycles: Cycles before detection activates (default: 30)
        auto_approve:  Auto-approve HITL requests (testing only)
        auto_recover:  Automatically trigger RCA+Recovery on events

    Returns:
        PipelineObserver ready for use
    """
    dag          = PipelineDAG.linear(stage_ids)
    history      = SignalHistory()
    detector     = DetectionEngine(pipeline_id, warmup_cycles=warmup_cycles)
    rca_engine   = RCAEngine(dag, history)
    feedback     = FeedbackStore()
    planner      = RecoveryPlanner(feedback_store=feedback,
                                   auto_approve=auto_approve)

    return PipelineObserver(
        pipeline_id=pipeline_id,
        dag=dag,
        detection_engine=detector,
        rca_engine=rca_engine,
        recovery_planner=planner,
        signal_history=history,
        auto_recover=auto_recover,
    )
