"""Layer 3 — Recovery Justification Prompts (4 prompts)"""
from . import register

register({"id":"REC-VALIDATE","layer":3,"fault_type":"*","name":"Recovery strategy pre-validation",
"system":"You are an ML pipeline recovery planner. Approve, reject, or modify the selected strategy. Never reject without an alternative. Return valid JSON only.",
"user":"Fault:{fault_type} Severity:{severity}\nStrategy:{selected_strategy} (hist. success:{historical_success_rate:.1%})\nRecent failures of this strategy:{recent_strategy_failures}\nReturn:{\"approved\":bool,\"decision\":str,\"concern\":str,\"alternative_strategy\":str,\"rationale\":str}",
"output_schema":{"approved":"boolean","decision":"enum[approve,reject,modify]","concern":"string|null","alternative_strategy":"string|null","rationale":"string"},
"eval_metric":"False rejection rate (target <5%)","tested_on":["claude-3-5-sonnet-20241022"],"notes":"recent_strategy_failures prevents approving a repeatedly-failing strategy."})

register({"id":"REC-OUTCOME","layer":3,"fault_type":"*","name":"Recovery outcome explanation",
"system":"You are reporting the outcome of an autonomous recovery action. Be honest about partial successes. End with a monitoring tip. Max 5 sentences. Return valid JSON only.",
"user":"Fault:{fault_type} Strategy:{strategy_id} Success:{success} TTR:{ttr_ms}ms\nPost-stats:{post_recovery_stats_json}\nReturn:{\"headline\":str,\"detail\":str,\"success_level\":str,\"monitoring_tip\":str,\"manual_action_needed\":bool}",
"output_schema":{"headline":"string","detail":"string","success_level":"enum[full,partial,failed]","monitoring_tip":"string","manual_action_needed":"boolean"},
"eval_metric":"Developer comprehension rating","tested_on":["claude-3-5-sonnet-20241022","claude-3-haiku-20240307"],"notes":"Use Haiku for cost efficiency — summarisation, not reasoning."})

register({"id":"REC-DL-3-REWEIGHT","layer":3,"fault_type":"DL-3","name":"Class reweighting parameter advisor",
"system":"Compute inverse-frequency class weights. Clip to [0.1,10.0]. Flag if GT labels unavailable. Return valid JSON only.",
"user":"Baseline dist:{baseline_class_dist_json}\nCurrent dist:{current_class_dist_json}\nGT labels available:{gt_labels_available}\nReturn:{\"class_weights\":dict,\"weights_are_approximate\":bool,\"warning\":str}",
"output_schema":{"class_weights":"dict[class,float]","weights_are_approximate":"boolean","warning":"string|null"},
"eval_metric":"Class balance restoration accuracy on Adult Income","tested_on":["claude-3-5-sonnet-20241022"],"notes":"Addresses Phase 4B DL-3 33% recovery failure (GT labels unavailable in synthetic)."})

register({"id":"REC-IL-1-BATCH","layer":3,"fault_type":"IL-1","name":"Batch size binary search advisor",
"system":"Compute optimal batch size reduction. RSS scales linearly with batch size. Start at 50% if RSS>95%, 70% if 90-95%. Min batch=8. Return valid JSON only.",
"user":"RSS:{current_rss_mb:.1f}MB/{max_rss_mb:.1f}MB ({rss_pct:.1%})\nCurrent batch:{current_batch_size}\nReturn:{\"recommended_batch_size\":int,\"binary_search_range\":dict,\"expected_rss_mb\":float,\"convergence_risk\":str}",
"output_schema":{"recommended_batch_size":"integer","binary_search_range":"dict","expected_rss_mb":"float","convergence_risk":"enum"},
"eval_metric":"Memory target achieved after one adjustment (target >90%)","tested_on":["claude-3-5-sonnet-20241022"],"notes":"IL-1 already 100% recovery. This makes binary search more efficient."})
