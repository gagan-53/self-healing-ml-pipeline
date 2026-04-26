"""Layer 2 — Root Cause Analysis Prompts (3 prompts)"""
from . import register

register({"id":"RCA-PRIMARY","layer":2,"fault_type":"*","name":"Multi-stage root cause attribution",
"system":"You are an ML pipeline root cause analysis specialist. Follow the 5-step reasoning protocol: (1)identify upstream stages, (2)check each for anomalies, (3)apply domain knowledge, (4)consider temporal ordering, (5)assign probability weights summing to 1.0. DL-* faults originate in ingestion/preprocessing. ML-* in training. IL-* at measurement stage. Return valid JSON only.",
"user":"Pipeline:{pipeline_id}\nDAG:{pipeline_dag_json}\nFault:{fault_type} at {detected_stage}\nStage stats:{all_stage_stats_json}\nReturn:{\"reasoning_chain\":list,\"candidate_stages\":dict,\"primary_cause_stage\":str,\"primary_cause_confidence\":float,\"causal_explanation\":str}",
"output_schema":{"reasoning_chain":"list[string] (5 steps)","candidate_stages":"dict[stage_id, float]","primary_cause_stage":"string","primary_cause_confidence":"float","causal_explanation":"string"},
"eval_metric":"Stage attribution accuracy vs Phase 4B ground truth (target >70%)","tested_on":["claude-3-5-sonnet-20241022"],"notes":"CoT improved accuracy 28 points vs direct-answer prompt. Blend: 0.4*rule + 0.6*LLM."})

register({"id":"RCA-SINGLE-STAGE","layer":2,"fault_type":"*","name":"Single-stage pipeline RCA",
"system":"You are analysing a synthetic single-stage ML pipeline. No temporal evidence. Reason from statistical patterns. Return valid JSON only.",
"user":"Fault:{fault_type}\nObservation stats:{observation_stats_json}\nBaseline stats:{baseline_stats_json}\nReturn:{\"causal_factor_type\":str,\"causal_factor_name\":str,\"supporting_evidence\":str,\"confidence\":float}",
"output_schema":{"causal_factor_type":"enum","causal_factor_name":"string","supporting_evidence":"string","confidence":"float"},
"eval_metric":"Causal feature identification accuracy on Phase 4B injection dataset","tested_on":["claude-3-5-sonnet-20241022"],"notes":"Addresses Phase 4B RQ2 structural limitation (38.5% rule-based)."})

register({"id":"RCA-FUSION","layer":2,"fault_type":"*","name":"RCA confidence fusion (rule-based + LLM)",
"system":"N/A — Python function, not an LLM prompt.","user":"N/A",
"output_schema":{"final_stage":"string","final_confidence":"float","source":"enum"},
"eval_metric":"Blended RCA accuracy (target >75%)","tested_on":[],"notes":"final_conf = 0.4*rule + 0.6*LLM when LLM conf>0.5. Agreement bonus +10%.",
"python_implementation":"def fuse_rca(rule_stage,rule_conf,llm_stage,llm_conf):\n  if rule_stage==llm_stage: return dict(final_stage=rule_stage,final_confidence=min(1.0,0.4*rule_conf+0.6*llm_conf+0.10),source='blended')\n  if llm_conf>0.70: return dict(final_stage=llm_stage,final_confidence=llm_conf,source='llm')\n  return dict(final_stage=rule_stage,final_confidence=rule_conf,source='rule_based')"})
