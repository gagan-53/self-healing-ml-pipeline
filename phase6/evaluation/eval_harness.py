"""
Phase 6 Prompt Evaluation Harness
Usage: python evaluation/eval_harness.py --layer 1
Requires: ANTHROPIC_API_KEY env variable
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prompt_runner import run_prompt, list_prompts

def _score(parsed, ground_truth, prompt_id):
    if parsed is None: return 0.0
    if prompt_id.startswith('DET-'):
        s = 0.5 if parsed.get('is_drift') == ground_truth.get('is_drift') else 0.0
        s += 0.5 if parsed.get('recommended_action') in ground_truth.get('allowed_actions',[]) else 0.0
        return s
    if prompt_id.startswith('RCA-'):
        return 1.0 if parsed.get('primary_cause_stage') == ground_truth.get('stage') else 0.0
    if prompt_id.startswith('REC-'):
        s = 0.5 if parsed.get('approved') == ground_truth.get('approved') else 0.0
        s += 0.5 if parsed.get('decision') in ground_truth.get('allowed_decisions',[]) else 0.0
        return s
    return 1.0 if parsed else 0.0

TEST_CASES = [
    {'prompt_id':'DET-DL-1','variables':{'pipeline_id':'adult_income','stage_id':'ingestion',
     'psi_values_json':'{"age":0.31}','max_psi':0.31,'psi_threshold':0.25,
     'n_features_above':1,'warmup_cycles':60,'observation_window':'1 cycle',
     'feature_stats_json':'{"age":{"mean":54.2,"std":12.1}}',
     'baseline_profile_json':'{"age":{"mean":35.8,"std":10.4}}'},
     'ground_truth':{'is_drift':True,'allowed_actions':['escalate','halt']}},
    {'prompt_id':'DET-DL-1','variables':{'pipeline_id':'synthetic','stage_id':'ingestion',
     'psi_values_json':'{"f0":0.04}','max_psi':0.04,'psi_threshold':0.25,
     'n_features_above':0,'warmup_cycles':60,'observation_window':'1 cycle',
     'feature_stats_json':'{"f0":{"mean":0.02,"std":1.01}}',
     'baseline_profile_json':'{"f0":{"mean":0.0,"std":1.0}}'},
     'ground_truth':{'is_drift':False,'allowed_actions':['ignore','monitor']}},
    {'prompt_id':'REC-VALIDATE','variables':{'fault_type':'DL-1','severity':'HIGH',
     'stage_id':'ingestion','rca_stage':'ingestion','rca_confidence':0.82,
     'selected_strategy':'retrain_recent_window','historical_success_rate':0.90,
     'strategy_description':'Refit on recent observations','current_row_count':8500,
     'memory_pct':0.62,'minutes_since_last_recovery':45,'recent_strategy_failures':0},
     'ground_truth':{'approved':True,'allowed_decisions':['approve']}},
]

def run_evaluation(cases, model='claude-sonnet-4-20250514', verbose=False):
    results = []
    for case in cases:
        print(f"  {case['prompt_id']}...", end='', flush=True)
        try:
            out = run_prompt(case['prompt_id'], case['variables'], model=model)
            score = _score(out['parsed'], case['ground_truth'], case['prompt_id'])
            marker = '✓' if score >= 0.8 else ('~' if score >= 0.5 else '✗')
            print(f" {marker} {score:.2f} ({out['latency_ms']}ms)")
            results.append({'prompt_id':case['prompt_id'],'score':score,
                           'latency_ms':out['latency_ms'],'error':out['error']})
        except Exception as e:
            print(f' ✗ ERROR: {e}')
            results.append({'prompt_id':case['prompt_id'],'score':0.0,'error':str(e)})
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, choices=[1,2,3,4])
    parser.add_argument('--model', default='claude-sonnet-4-20250514')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    cases = [c for c in TEST_CASES if args.layer is None or c['prompt_id'].startswith(['DET-','RCA-','REC-','META-'][[1,2,3,4].index(args.layer)])]
    print(f'Evaluating {len(cases)} cases with {args.model}')
    results = run_evaluation(cases, model=args.model, verbose=args.verbose)
    scores = [r['score'] for r in results]
    print(f'\nMean score: {np.mean(scores):.3f}  Pass rate: {sum(s>=0.8 for s in scores)/len(scores):.1%}')
