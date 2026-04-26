"""
SH-MLP Prompt Runner — executes Phase 6 prompts against the Anthropic API.
Usage:
    from phase6.prompt_runner import run_prompt, list_prompts
    result = run_prompt('DET-DL-1', {'pipeline_id':'p','max_psi':0.31,...})
"""
from __future__ import annotations
import json, time, os, re
from typing import Any, Optional

from prompts import PROMPT_CATALOGUE
from prompts import detection, rca, recovery, meta  # noqa: side-effect imports

_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        import anthropic
        _CLIENT = anthropic.Anthropic()
    return _CLIENT

def _fill(template, variables):
    result = template
    for k, v in variables.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, indent=2)
        result = result.replace('{' + k + '}', str(v))
    return result

def run_prompt(prompt_id, variables, model='claude-sonnet-4-20250514',
               max_tokens=1024, temperature=0.1, verbose=False):
    defn = next((p for p in PROMPT_CATALOGUE if p['id'] == prompt_id), None)
    if defn is None:
        raise ValueError(f"Prompt '{prompt_id}' not found. Available: {[p['id'] for p in PROMPT_CATALOGUE]}")
    user_prompt = _fill(defn['user'], variables)
    t0 = time.perf_counter()
    response = _get_client().messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=defn['system'], messages=[{'role':'user','content':user_prompt}])
    latency_ms = int((time.perf_counter() - t0) * 1000)
    raw_text = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    parsed = None; error = None
    if defn.get('output_schema') != 'raw markdown string':
        try:
            clean = re.sub(r'```(?:json)?\s*', '', raw_text).strip()
            m = re.search(r'\{[\s\S]+\}', clean)
            parsed = json.loads(m.group() if m else clean)
        except json.JSONDecodeError as e:
            error = str(e)
    return {'prompt_id':prompt_id,'raw_response':raw_text,
            'parsed':parsed or (raw_text if defn.get('output_schema')=='raw markdown string' else None),
            'latency_ms':latency_ms,'tokens_used':tokens,'error':error}

def list_prompts(layer=None):
    prompts = PROMPT_CATALOGUE
    if layer is not None:
        prompts = [p for p in prompts if p['layer'] == layer]
    return [{'id':p['id'],'name':p['name'],'layer':p['layer'],'fault_type':p['fault_type']} for p in prompts]

if __name__ == '__main__':
    layers = {1:'Detection',2:'RCA',3:'Recovery',4:'Meta'}
    print('SH-MLP Prompt Library Catalogue')
    print('='*50)
    for layer in [1,2,3,4]:
        ps = list_prompts(layer=layer)
        print(f'\nLayer {layer} — {layers[layer]} ({len(ps)} prompts):')
        for p in ps:
            print(f"  {p['id']:<22} {p['name']}")
    print(f'\nTotal: {len(PROMPT_CATALOGUE)} prompts')
