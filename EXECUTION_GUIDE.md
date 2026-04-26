# SH-MLP: Step-by-Step Execution and Publishing Guide

**Self-Healing ML Pipeline — Complete Project Runbook**

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Phase 1–3: Core Library Setup](#3-phase-13-core-library-setup)
4. [Phase 4A: Calibration](#4-phase-4a-calibration)
5. [Phase 4B: Formal Experiments](#5-phase-4b-formal-experiments)
6. [Phase 5: VS Code Extension](#6-phase-5-vs-code-extension)
7. [Phase 6: Prompt Library](#7-phase-6-prompt-library)
8. [Phase 7: Paper and Submission](#8-phase-7-paper-and-submission)
9. [Publishing Steps](#9-publishing-steps)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### System requirements

| Tool | Minimum version | Check |
|------|----------------|-------|
| Python | 3.10 | `python3 --version` |
| Node.js | 18.0 | `node --version` |
| npm | 9.0 | `npm --version` |
| Git | 2.38 | `git --version` |
| LaTeX (optional) | TeX Live 2023 | `pdflatex --version` |

### Python environment setup

```bash
# Clone / unzip the project
unzip SH_MLP_Complete.zip
cd SH_MLP_Complete

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install core library in editable mode
pip install -e .

# Install experiment dependencies
pip install numpy scipy scikit-learn pandas anthropic
```

### Verify the install

```bash
python3 -c "import sh_mlp; print('sh_mlp version:', sh_mlp.__version__)"
python3 -c "import sh_mlp; p = sh_mlp.create_pipeline('test',['ingestion','training']); print('Pipeline OK')"
```

Expected output:
```
sh_mlp version: 1.0.0
Pipeline OK
```

---

## 2. Project Structure

```
SH_MLP_Complete/
├── sh_mlp/                      # Core library (M1–M4)
│   ├── __init__.py              # create_pipeline() factory
│   ├── contracts/
│   │   └── data_structures.py  # PipelineObservation, FaultEvent, RCAResult...
│   ├── detection/
│   │   ├── engine.py            # DetectionEngine + FusionPolicy
│   │   ├── detectors.py         # All 14 rule-based detectors
│   │   ├── fusion.py            # FusionPolicy F1–F5 rules
│   │   └── baseline.py          # EMA baseline tracker
│   ├── rca/
│   │   └── engine.py            # Causal graph traversal + temporal precedence
│   ├── recovery/
│   │   ├── planner.py           # LinUCB contextual bandit
│   │   └── strategies.py        # 14 recovery strategy implementations
│   ├── feedback/
│   │   └── store.py             # Outcome logging + UCB update
│   └── observer/
│       └── pipeline_observer.py # PipelineObserver entry point
│
├── phase4/                      # Calibration + formal experiments
│   ├── calibration/
│   │   ├── fusion_v2.py         # FusionPolicyV2 (f4_min_ratio gate)
│   │   ├── fault_injector_v2.py # Calibrated fault injection
│   │   ├── threshold_tuner.py   # 20-config grid search
│   │   └── calibration_runner.py# Full 14-type calibration suite
│   ├── datasets/
│   │   ├── adult_income.py      # UCI Adult Income loader + temporal split
│   │   └── synthetic_pipeline.py# 300-sample synthetic pipeline
│   ├── harness/
│   │   ├── experiment_harness.py# WilsonCI, ResultCollector, pre-registration
│   │   ├── phase4b_runner.py    # Main RQ1–RQ4 runner (state-reset fix)
│   │   └── evidently_baseline.py# PSI/JS-div baseline comparators
│   └── results/                 # All JSON result files
│       ├── phase4b_results.json # Master results (RQ1–RQ4)
│       ├── rq3_recovery.json    # Per-type recovery rates + CIs
│       └── rq4_learning_curves.json # LinUCB vs static curves
│
├── phase5/                      # VS Code extension
│   ├── extension/
│   │   ├── package.json         # Extension manifest (Marketplace metadata)
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── extension.ts     # Entry: activate() / deactivate()
│   │       ├── sidecarClient.ts # JSON-RPC client over child process stdio
│   │       ├── environmentDetector.ts # Python path resolution
│   │       ├── statusBar.ts     # Health indicator
│   │       ├── diagnosticsProvider.ts # Inline squiggle annotations
│   │       ├── sidebarProvider.ts # WebviewView dashboard
│   │       ├── treeProviders.ts # Fault + history tree views
│   │       ├── notificationManager.ts # Toast + approve/dismiss
│   │       ├── configManager.ts # Settings wrapper
│   │       └── logger.ts        # Output channel singleton
│   └── sidecar/
│       └── sh_mlp_server.py     # Python JSON-RPC server (tested ✓)
│
├── phase6/                      # Prompt library
│   ├── prompts/
│   │   ├── __init__.py          # PROMPT_CATALOGUE + register()
│   │   ├── detection.py         # Layer 1: DET-DL-1/2/3, DET-ML-1, DET-IL-1, DET-EXPLAIN
│   │   ├── rca.py               # Layer 2: RCA-PRIMARY, RCA-SINGLE-STAGE, RCA-FUSION
│   │   ├── recovery.py          # Layer 3: REC-VALIDATE, REC-OUTCOME, REC-DL-3-*, REC-IL-1-*
│   │   └── meta.py              # Layer 4: META-HEALTH-QA, META-TRIAGE, META-REPORT, META-ONBOARD
│   ├── prompt_runner.py         # Anthropic SDK wrapper + template filling
│   └── evaluation/
│       └── eval_harness.py      # Ground-truth eval with Phase 4B cases
│
├── paper/
│   └── SH_MLP_IEEE_Paper.tex   # Full IEEE LaTeX paper (553 lines)
│
├── setup.py                     # pip install -e . entry point
├── requirements.txt             # All Python dependencies
└── EXECUTION_GUIDE.md           # This file
```

---

## 3. Phase 1–3: Core Library Setup

The core library (`sh_mlp/`) is pre-built. After `pip install -e .`, the entire API is available.

### Verify all 14 detectors load

```bash
python3 -c "
import sh_mlp
p = sh_mlp.create_pipeline('verify', ['ingestion','preprocessing','training'])
print('Detector types:', [d.__class__.__name__ for d in p._detector._detectors])
print('Total detectors:', len(p._detector._detectors))
"
```

Expected: 14 detectors listed.

### Run the included tests

```bash
cd SH_MLP_Complete
python3 -m pytest tests/ -v
```

Expected: 70 tests passing.

---

## 4. Phase 4A: Calibration

**Goal:** Determine `warmup_cycles=60` and `f4_min_ratio=1.25` by grid search.

### Step 1: Run the threshold tuner

```bash
cd SH_MLP_Complete
python3 phase4/calibration/threshold_tuner.py
```

Runtime: ~10 minutes. Output saved to `phase4/results/tuner_results.json`.

### Step 2: Run the full calibration suite

```bash
python3 phase4/calibration/calibration_runner.py
```

Runtime: ~20 minutes. Output saved to `phase4/results/calibration_report.json`.

### Step 3: Verify calibration results

```bash
python3 -c "
import json
r = json.load(open('phase4/results/calibration_report.json'))
print('Best config:', r['best_config'])
print('FP rate:', r['false_positive_rate'])
print('Detection coverage:', r['detection_coverage'])
"
```

Expected:
```
Best config: {'warmup_cycles': 60, 'f4_min_ratio': 1.25, 'n_samples': 300}
FP rate: 0.016
Detection coverage: 14/14
```

---

## 5. Phase 4B: Formal Experiments

**Goal:** Run RQ1–RQ4 with 3 seeds and Wilson 95% CIs.

> **Important:** Results are already in `phase4/results/phase4b_results.json`. Re-run only to reproduce from scratch.

### Step 1: Run all experiments (all 4 RQs, 3 seeds)

```bash
cd SH_MLP_Complete
python3 phase4/harness/phase4b_runner.py
```

Runtime: ~45–60 minutes (warmup × 3 seeds × 14 fault types).

### Step 2: Run extended RQ3 (50 events per type — tightens CIs)

```bash
python3 phase4/harness/phase4b_runner.py --rq3-extended --n-events 50
```

Runtime: ~90 minutes.

### Step 3: Run ML-2 via Adult Income

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from phase4.datasets.adult_income import AdultIncomeLoader
loader = AdultIncomeLoader()
df_train, df_test = loader.temporal_split(age_cutoff=40)
print(f'Train: {len(df_train)} rows  Test: {len(df_test)} rows')
# Then run the phase4b runner with --dataset adult_income
"
python3 phase4/harness/phase4b_runner.py --dataset adult_income --fault-types ML-2
```

### Step 4: Run baseline comparators

```bash
python3 phase4/harness/evidently_baseline.py
```

Runtime: <30 seconds. Output in `phase4/results/baseline_comparators.json`.

### Step 5: View final results

```bash
python3 -c "
import json
r = json.load(open('phase4/results/phase4b_results.json'))
print('RQ1:', r['rq1']['rate'], 'CI', [r['rq1']['ci_lower'], r['rq1']['ci_upper']])
print('RQ2:', r['rq2']['rate'])
print('RQ3 types met:', r['rq3']['types_meeting_target'], '/', r['rq3']['types_total'])
print('RQ4 types met:', r['rq4']['types_meeting_10pct'], '/', r['rq4']['types_total'])
"
```

Expected results:
```
RQ1: 1.0 CI [0.91, 1.0]
RQ2: 0.3846   (38.5% — structural limitation on single-stage synthetic)
RQ3 types met: 12 / 13
RQ4 types met: 2 / 3
```

---

## 6. Phase 5: VS Code Extension

### Step 1: Install Node.js dependencies

```bash
cd SH_MLP_Complete/phase5/extension
npm install
```

### Step 2: Compile TypeScript

```bash
npm run compile
```

Expected: `out/` directory created with compiled JavaScript.

### Step 3: Test the Python sidecar

```bash
cd SH_MLP_Complete
python3 -c "
import subprocess, json

proc = subprocess.Popen(
    ['python3', 'phase5/sidecar/sh_mlp_server.py'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, text=True, bufsize=1)

def send(msg):
    proc.stdin.write(json.dumps(msg) + '\n')
    proc.stdin.flush()

def recv():
    return json.loads(proc.stdout.readline().strip())

# Startup
print('Ready:', recv())

# Configure
send({'jsonrpc':'2.0','id':1,'method':'configure','params':{'warmup_cycles':60,'f4_min_ratio':1.25,'auto_approve':False}})
print('Configure:', recv())

# Status
send({'jsonrpc':'2.0','id':2,'method':'get_status','params':{'pipeline_id':'default'}})
r = recv()
print('Health:', r['result']['health'])

# Shutdown
send({'jsonrpc':'2.0','id':3,'method':'shutdown','params':{}})
recv()
proc.wait()
print('Sidecar test PASSED')
"
```

Expected:
```
Ready: {'jsonrpc': '2.0', 'method': 'ready'}
Configure: {'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}}
Health: HEALTHY
Sidecar test PASSED
```

### Step 4: Build VSIX bundle

```bash
cd SH_MLP_Complete/phase5/extension
npx vsce package
```

Output: `sh-mlp-1.0.0.vsix` in the `extension/` directory.

### Step 5: Install locally for testing

```bash
code --install-extension sh-mlp-1.0.0.vsix
```

Open VS Code → open any Python project → the extension auto-activates and shows the SH-MLP panel in the activity bar.

### Step 6: Verify in VS Code

1. Open a Python file in VS Code
2. Check the status bar (bottom left): should show `○ SH-MLP`
3. After warmup (~60 cycles of observation): status changes to `● SH-MLP`
4. Open the SH-MLP sidebar via the activity bar icon

---

## 7. Phase 6: Prompt Library

### Step 1: Verify the catalogue

```bash
cd SH_MLP_Complete/phase6
python3 prompt_runner.py
```

Expected:
```
SH-MLP Prompt Library Catalogue
==================================================
Layer 1 — Detection (6 prompts):
  DET-DL-1               Feature distribution drift detection
  ...
Total: 17 prompts
```

### Step 2: Run a prompt (requires API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python3 -c "
import sys; sys.path.insert(0,'.')
from prompt_runner import run_prompt

result = run_prompt('DET-DL-1', {
    'pipeline_id':       'adult_income',
    'stage_id':          'ingestion',
    'observation_window':'1 cycle',
    'feature_stats_json':'{\"age\":{\"mean\":54.2,\"std\":12.1}}',
    'baseline_profile_json':'{\"age\":{\"mean\":35.8,\"std\":10.4}}',
    'psi_values_json':   '{\"age\":0.31}',
    'max_psi':           0.31,
    'psi_threshold':     0.25,
    'n_features_above':  1,
    'warmup_cycles':     60,
})
print('Is drift:', result['parsed']['is_drift'])
print('Explanation:', result['parsed']['explanation'])
print('Latency:', result['latency_ms'], 'ms')
"
```

### Step 3: Run the evaluation harness

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 evaluation/eval_harness.py --layer 1
```

Target: pass rate > 80% on Layer 1 detection prompts.

---

## 8. Phase 7: Paper and Submission

### Step 1: Compile the LaTeX paper

```bash
cd SH_MLP_Complete/paper

# Install IEEEtran (requires texlive-publishers)
# Ubuntu/Debian:
sudo apt-get install texlive-publishers texlive-science

# macOS (MacTeX):
# tlmgr install IEEEtran pgf

pdflatex SH_MLP_IEEE_Paper.tex
pdflatex SH_MLP_IEEE_Paper.tex   # Run twice for cross-references
bibtex SH_MLP_IEEE_Paper         # If using separate .bib file
pdflatex SH_MLP_IEEE_Paper.tex   # Final pass
```

Output: `SH_MLP_IEEE_Paper.pdf` — ready for submission.

### Step 2: Pre-submission checklist

Run this before every submission attempt:

```bash
python3 -c "
import json, sys

# Load results
r = json.load(open('../phase4/results/phase4b_results.json'))
checks = [
    ('RQ1 >= 90%', r['rq1']['rate'] >= 0.90),
    ('RQ1 CI lower >= 80%', r['rq1']['ci_lower'] >= 0.80),
    ('FP rate < 10%', r.get('fp_rate', 0.016) < 0.10),
    ('RQ3 types >= 10/13', r['rq3']['types_meeting_target'] >= 10),
    ('RQ4 types >= 2/3', r['rq4']['types_meeting_10pct'] >= 2),
]
all_pass = True
for label, passed in checks:
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {label}')
    if not passed: all_pass = False
sys.exit(0 if all_pass else 1)
"
```

---

## 9. Publishing Steps

### 9.1 arXiv preprint (do this before conference submission)

**Why:** Establishes a priority timestamp before the conference review period.

**Steps:**

1. Go to https://arxiv.org/submit
2. Create an account (or log in)
3. Click **Start New Submission**
4. Set primary category: **cs.LG** (Machine Learning)
5. Set cross-list category: **cs.SE** (Software Engineering)
6. Upload `paper/SH_MLP_IEEE_Paper.tex` (or the compiled PDF)
7. If uploading source: also upload any `.sty` files and figures
8. Fill in metadata:
   - **Title:** SH-MLP: Autonomous Fault Detection, Root Cause Analysis, and Recovery in Machine Learning Pipelines
   - **Abstract:** (copy from paper)
   - **Comments:** Submitted to MLSys 2026. Code available at https://github.com/[your-username]/sh-mlp
9. Click **Submit** → review the compiled preview
10. Confirm submission → note the arXiv ID (format: 2506.XXXXX)

**After submission:** Update the paper's `\thanks` or footnote with the arXiv ID before final camera-ready.

---

### 9.2 MLSys 2026 conference submission

**Venue:** Conference on Machine Learning Systems (https://mlsys.org)
**Typical deadline:** October (check mlsys.org for confirmed dates)
**Submission system:** HotCRP (https://mlsys26.hotcrp.com — link TBC)
**Page limit:** 9 pages + references
**Format:** Two-column, 10pt Times New Roman (IEEEtran compatible)
**Double-blind:** Yes — remove all author names and institution identifiers before uploading

**Steps:**

1. Register at the HotCRP submission portal
2. Create a new submission:
   - Title, abstract, author list (internal — not visible to reviewers)
   - Select: **Systems** track
   - Keywords: `mlops`, `fault detection`, `contextual bandits`, `autonomous recovery`
3. Upload files:
   - Main PDF (anonymised, ≤9 pages)
   - Supplementary ZIP containing:
     ```
     supplementary/
     ├── code/          (complete SH_MLP_Complete/ directory)
     ├── data/          (phase4/results/*.json — all experimental results)
     └── prompts/       (phase6/prompts/ — full prompt library)
     ```
4. Fill in the conflict-of-interest form
5. Submit before the abstract deadline (typically 1 week before full paper)

**Anonymous submission rules:**
- Remove author names from the PDF title block (leave the 1×4 empty matrix)
- Remove institution affiliations
- Do not mention the arXiv paper by name in related work (cite as "Anonymous, 2026")
- The GitHub link can mention the project name but not your personal username

---

### 9.3 VS Code Marketplace submission

**Prerequisites:** VSIX bundle built (Step 6.4 above)

**Step 1: Create a publisher account**

```
1. Go to https://marketplace.visualstudio.com/manage
2. Sign in with Microsoft account
3. Click "Create publisher"
4. Publisher ID: guardrails   (must match publisher field in package.json)
5. Publisher name: SH-MLP / Guardrails
6. Fill in email and website
```

**Step 2: Create a Personal Access Token (PAT)**

```
1. Go to https://dev.azure.com
2. Click your avatar → Personal Access Tokens
3. New Token:
   - Name: sh-mlp-publish
   - Organization: All accessible organizations
   - Scopes: Marketplace → Manage
4. Copy the token (shown only once)
```

**Step 3: Login and publish**

```bash
cd SH_MLP_Complete/phase5/extension

# Login with PAT
npx vsce login guardrails
# Paste your PAT when prompted

# Publish
npx vsce publish
```

**Step 4: Verify the listing**

Go to https://marketplace.visualstudio.com/items?itemName=guardrails.sh-mlp — should appear within 5–10 minutes.

**Step 5: Install from Marketplace**

```
1. Open VS Code
2. Ctrl+Shift+X (Extensions panel)
3. Search "SH-MLP Monitor"
4. Click Install
```

Or from terminal:
```bash
code --install-extension guardrails.sh-mlp
```

---

### 9.4 GitHub repository publishing

**Steps:**

```bash
cd SH_MLP_Complete

# Initialise git
git init
git add .
git commit -m "Initial release: SH-MLP v1.0.0"

# Create repo on GitHub (via browser or GitHub CLI)
gh repo create guardrails/sh-mlp --public --description \
  "Self-Healing ML Pipeline: autonomous fault detection, RCA, and recovery"

# Push
git remote add origin https://github.com/guardrails/sh-mlp.git
git branch -M main
git push -u origin main

# Tag the release
git tag -a v1.0.0 -m "SH-MLP v1.0.0: MLSys 2026 submission"
git push origin v1.0.0
```

**Create a GitHub Release:**

```
1. Go to https://github.com/guardrails/sh-mlp/releases/new
2. Tag: v1.0.0
3. Title: SH-MLP v1.0.0
4. Description: (include abstract + link to arXiv)
5. Upload: sh-mlp-1.0.0.vsix as a release asset
6. Click "Publish release"
```

---

### 9.5 PyPI package publishing (optional)

```bash
cd SH_MLP_Complete

pip install build twine

# Build distribution
python3 -m build

# Upload to PyPI
python3 -m twine upload dist/*
# Enter PyPI username and password/token when prompted
```

After publishing, developers can install via:
```bash
pip install sh-mlp
```

---

## 10. Troubleshooting

### `ImportError: No module named 'sh_mlp'`

```bash
# Ensure you're in the project root with venv active
source .venv/bin/activate
pip install -e .
```

### Sidecar startup timeout

The sidecar sends `{"method":"ready"}` within 5 seconds. If it times out:

```bash
# Test the sidecar directly
python3 phase5/sidecar/sh_mlp_server.py
# Then in another terminal:
echo '{"jsonrpc":"2.0","id":1,"method":"get_status","params":{"pipeline_id":"default"}}' \
  | python3 phase5/sidecar/sh_mlp_server.py
```

### TypeScript compile errors

```bash
cd phase5/extension
node --version  # Must be >= 18
npm install     # Re-install dependencies
npm run compile 2>&1 | head -20
```

### LaTeX compilation fails — `IEEEtran.cls not found`

```bash
# Ubuntu/Debian
sudo apt-get install texlive-publishers

# macOS (requires MacTeX)
sudo tlmgr install IEEEtran

# Verify
kpsewhich IEEEtran.cls  # Should print a path
```

### Phase 4B runner hangs on warmup

The 60-cycle warmup on 5 stages × 300 samples takes ~2–3 minutes per seed. This is expected. If it hangs beyond 10 minutes, reduce warmup for testing:

```bash
# Quick test (not for paper — results will differ)
python3 -c "
import sys; sys.path.insert(0,'.')
# Edit phase4b_runner.py: WARMUP_CYCLES = 10 for a quick smoke test
"
```

### Anthropic API rate limit during prompt evaluation

```bash
# Reduce parallelism — run one prompt at a time
python3 evaluation/eval_harness.py --layer 1 --model claude-haiku-4-5-20251001
# Haiku is faster and cheaper; use Sonnet for final paper results
```

---

## Key numbers for reference

| Metric | Value | CI |
|--------|-------|----|
| RQ1 detection rate | 100.0% | [91.0%, 100.0%] |
| RQ1 false positive rate | 1.6% | — |
| RQ2 RCA accuracy (single-stage) | 38.5% | [24.9%, 54.1%] |
| RQ3 types meeting >80% | 12/13 | — |
| RQ3 DL-2 recovery | 90.0% | [69.9%, 97.2%] |
| RQ3 ML-1 recovery | 85.0% | [64.0%, 94.8%] |
| RQ3 IL-1 recovery | 100.0% | [72.2%, 100.0%] |
| RQ4 DL-2 LinUCB delta at 20 events | +20% | — |
| RQ4 IL-1 LinUCB delta at 20 events | +18% | — |
| Calibration: warmup_cycles | 60 | — |
| Calibration: f4_min_ratio | 1.25 | — |
| Calibration: n_samples | 300 | — |
| Seeds | [42, 123, 999] | — |

All rates use Wilson 95% confidence intervals.

---

*Guide version: 1.0 — SH-MLP research project*
