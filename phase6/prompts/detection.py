"""Layer 1 — Detection Reasoning Prompts (6 prompts: DET-DL-1/2/3, DET-ML-1, DET-IL-1, DET-EXPLAIN)"""
from . import register

register({"id":"DET-DL-1","layer":1,"fault_type":"DL-1","name":"Feature distribution drift detection",
"system":"You are an ML pipeline monitoring specialist. Reason only from the statistics provided. Return valid JSON only.",
"user":"Pipeline:{pipeline_id} Stage:{stage_id}\nPSI values:{psi_values_json}\nMax PSI:{max_psi:.4f} Threshold:{psi_threshold:.4f}\nReturn:{\"is_drift\":bool,\"confidence\":float,\"affected_features\":list,\"recommended_action\":str,\"explanation\":str}",
"output_schema":{"is_drift":"boolean","confidence":"float","affected_features":"list","recommended_action":"enum","explanation":"string"},
"eval_metric":"Agreement rate with rule-based DL-1 detector (target >85%)","tested_on":["claude-3-5-sonnet-20241022"],"notes":"recommended_action maps to VS Code notification urgency."})

register({"id":"DET-DL-2","layer":1,"fault_type":"DL-2","name":"Schema violation classification",
"system":"You are an ML data pipeline expert. A schema hash change always indicates a structural change. Return valid JSON only.",
"user":"Pipeline:{pipeline_id}\nExpected hash:{expected_hash} Observed:{observed_hash}\nExpected cols:{expected_columns_json}\nObserved cols:{observed_columns_json}\nModel features:{model_feature_columns_json}\nReturn:{\"violation_type\":str,\"severity\":str,\"affected_model_features\":list,\"safe_to_continue\":bool,\"suggested_fix\":str}",
"output_schema":{"violation_type":"enum","severity":"enum","affected_model_features":"list","safe_to_continue":"boolean","suggested_fix":"string"},
"eval_metric":"Violation type classification accuracy","tested_on":["claude-3-5-sonnet-20241022"],"notes":"safe_to_continue gates the pipeline halt decision."})

register({"id":"DET-DL-3","layer":1,"fault_type":"DL-3","name":"Label distribution shift analysis",
"system":"You are an ML monitoring specialist. JS divergence >0.10 = detectable shift. Return valid JSON only.",
"user":"Pipeline:{pipeline_id}\nJS divergence:{js_divergence:.4f} Threshold:{js_threshold:.4f}\nBaseline dist:{baseline_class_dist_json}\nCurrent dist:{current_class_dist_json}\nReturn:{\"is_label_shift\":bool,\"shift_type\":str,\"confidence\":float,\"recommended_action\":str}",
"output_schema":{"is_label_shift":"boolean","shift_type":"enum","confidence":"float","recommended_action":"enum"},
"eval_metric":"Shift type classification vs Adult Income temporal split","tested_on":["claude-3-5-sonnet-20241022"],"notes":"cycles_until_model_impact feeds recovery planner urgency."})

register({"id":"DET-ML-1","layer":1,"fault_type":"ML-1","name":"Model accuracy degradation triage",
"system":"You are an MLOps engineer. Distinguish: distribution shift, training instability, deployment event, genuine decay. Return valid JSON only.",
"user":"Pipeline:{pipeline_id}\nConf drop:{confidence_drop:.4f} ({confidence_drop_pct:.1f}%)\nRecent faults:{recent_fault_history_json}\nReturn:{\"is_accuracy_degradation\":bool,\"most_likely_cause\":str,\"is_upstream_fault\":bool,\"urgency\":str}",
"output_schema":{"is_accuracy_degradation":"boolean","most_likely_cause":"enum","is_upstream_fault":"boolean","urgency":"enum"},
"eval_metric":"Cause attribution accuracy vs Phase 4B injection ground truth","tested_on":["claude-3-5-sonnet-20241022"],"notes":"is_upstream_fault prevents double-recovery."})

register({"id":"DET-IL-1","layer":1,"fault_type":"IL-1","name":"Memory exhaustion risk assessment",
"system":"You are an ML infrastructure engineer. RSS >90% = HIGH. Rising trend over 3+ cycles is more alarming than a spike. Return valid JSON only.",
"user":"Pipeline:{pipeline_id}\nRSS:{current_rss_mb:.1f}MB / {max_rss_mb:.1f}MB ({rss_pct:.1%})\nTrend:{rss_trend_json}\nReturn:{\"will_oom\":bool,\"estimated_cycles_to_oom\":int,\"severity\":str,\"recommended_batch_size\":int}",
"output_schema":{"will_oom":"boolean","estimated_cycles_to_oom":"integer|null","severity":"enum","recommended_batch_size":"integer|null"},
"eval_metric":"OOM prediction accuracy on memory stress test suite","tested_on":["claude-3-5-sonnet-20241022"],"notes":"recommended_batch_size seeds the binary search recovery."})

register({"id":"DET-EXPLAIN","layer":1,"fault_type":"*","name":"Detection natural-language explainer",
"system":"You are a friendly ML pipeline monitoring assistant in VS Code. Max 4 sentences. No jargon unless defined. Return valid JSON only.",
"user":"Fault:{fault_type} Severity:{severity} Stage:{stage_id}\nStrategy:{strategy_id}\nReturn:{\"headline\":str,\"detail\":str,\"action_label\":str,\"severity_emoji\":str,\"post_recovery_tip\":str}",
"output_schema":{"headline":"string (<=12 words)","detail":"string","action_label":"string (<=5 words)","severity_emoji":"string","post_recovery_tip":"string|null"},
"eval_metric":"Developer satisfaction rating (target >4.0/5)","tested_on":["claude-3-5-sonnet-20241022","claude-3-haiku-20240307"],"notes":"headline feeds into VS Code showWarningMessage()."})
