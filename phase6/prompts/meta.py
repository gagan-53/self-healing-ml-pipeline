"""Layer 4 — Meta Prompts / VS Code Interface (4 prompts)"""
from . import register

register({"id":"META-HEALTH-QA","layer":4,"fault_type":"*","name":"Pipeline health Q&A assistant",
"system":"You are the SH-MLP pipeline health assistant in VS Code. Answer concisely using only the provided monitoring state. Max 4 sentences. Return valid JSON only.",
"user":"Monitoring state:{monitoring_state_json}\nConversation:{conversation_history_json}\nQuestion:\"{question}\"\nReturn:{\"answer\":str,\"confidence\":float,\"follow_up_suggestion\":str,\"suggested_command\":str}",
"output_schema":{"answer":"string","confidence":"float","follow_up_suggestion":"string|null","suggested_command":"string|null"},
"eval_metric":"Answer accuracy rate vs monitoring state ground truth (target >90%)","tested_on":["claude-3-5-sonnet-20241022"],"notes":"Use Haiku for latency-sensitive chat, Sonnet for multi-fault questions."})

register({"id":"META-TRIAGE","layer":4,"fault_type":"*","name":"Multi-fault triage prioritiser",
"system":"Rank active faults by business impact. CRITICAL first, then model-correctness faults, then trending infra faults. Address upstream faults before downstream. Return valid JSON only.",
"user":"Active faults:{active_faults_json}\nDAG:{pipeline_dag_json}\nReturn:{\"priority_order\":list,\"triage_rationale\":dict,\"causal_dependencies\":list,\"can_auto_resolve\":bool}",
"output_schema":{"priority_order":"list[fault_id]","triage_rationale":"dict[fault_id,string]","causal_dependencies":"list","can_auto_resolve":"boolean"},
"eval_metric":"Priority agreement with expert-annotated triage decisions on 30 scenarios","tested_on":["claude-3-5-sonnet-20241022"],"notes":"causal_dependencies prevents conflicting simultaneous recoveries."})

register({"id":"META-REPORT","layer":4,"fault_type":"*","name":"Monitoring session report generator",
"system":"Generate a clear GitHub-flavoured Markdown summary of a monitoring session. Max 400 words. Tables for fault summaries. Precise times and metrics.",
"user":"Pipeline:{pipeline_id} Duration:{duration_minutes}min\nEvents:{all_events_json}\nRecoveries:{recovery_outcomes_json}\nOutput raw markdown only (no JSON wrapper).",
"output_schema":"raw markdown string",
"eval_metric":"Report completeness score","tested_on":["claude-3-5-sonnet-20241022"],"notes":"Only Layer 4 prompt returning raw text. Extension renders as VS Code markdown preview."})

register({"id":"META-ONBOARD","layer":4,"fault_type":"*","name":"First-run developer onboarding",
"system":"Explain what SH-MLP is doing in 3-4 sentences tailored to the developer's specific pipeline code. Reference actual function names or patterns visible in the snippet.",
"user":"Pipeline snippet:{pipeline_file_snippet}\nDetected patterns:{detected_patterns_json}\nWarmup:{warmup_pct:.0f}% ({cycles_complete}/{warmup_cycles} cycles)\nReturn:{\"welcome_message\":str,\"what_is_being_monitored\":list,\"estimated_ready_in\":str,\"tip\":str}",
"output_schema":{"welcome_message":"string","what_is_being_monitored":"list[string]","estimated_ready_in":"string","tip":"string"},
"eval_metric":"First-run satisfaction (target >4.2/5)","tested_on":["claude-3-5-sonnet-20241022"],"notes":"Limit snippet to 60 lines for cost. Pattern detection done client-side in TypeScript."})
