# SH-MLP: Self-Healing Machine Learning Pipeline

Autonomous fault detection, root cause analysis, and recovery for ML pipelines.

[![Release](https://img.shields.io/github/v/release/gagan-53/self-healing-ml-pipeline)](https://github.com/gagan-53/self-healing-ml-pipeline/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

SH-MLP is a four-layer system that observes ML pipelines in real time, detects 14 categories of faults across data, model, and infrastructure layers, performs causal-graph-based root cause analysis, and applies recovery actions chosen by a LinUCB contextual bandit. The library ships with a Python core, a VS Code extension, and a 17-prompt LLM library for natural-language explanations.

## Key Results (Phase 4B)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Detection rate (RQ1) | 100.0% | [91.0%, 100.0%] |
| False positive rate | 1.6% | — |
| Recovery types meeting >80% (RQ3) | 12 / 13 | — |
| LinUCB delta @ 20 events (RQ4) | up to +20% | — |

All rates use Wilson 95% confidence intervals over 3 seeds (`[42, 123, 999]`).

## Quick Start

```bash
git clone https://github.com/gagan-53/self-healing-ml-pipeline.git
cd self-healing-ml-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

python3 -c "import sh_mlp; p = sh_mlp.create_pipeline('demo', ['ingestion','training']); print('OK')"
```

## Repository Layout

```
sh_mlp/        # Core library: detection, RCA, recovery, observer
phase4/        # Calibration + RQ1–RQ4 experiment harness
phase5/        # VS Code extension (TypeScript) + Python sidecar
phase6/        # 17-prompt LLM library (detection / RCA / recovery / meta)
paper/         # IEEE paper (LaTeX + PDF)
```

## VS Code Extension

A pre-built VSIX is attached to each [release](https://github.com/gagan-53/self-healing-ml-pipeline/releases). To install:

```bash
code --install-extension sh-mlp-1.0.0.vsix
```

To build from source:

```bash
cd phase5/extension
npm install && npm run compile && npx vsce package
```

## Reproducing the Experiments

See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for the full runbook (Phase 4A calibration, Phase 4B RQ1–RQ4, baselines, and the pre-submission checklist).

## Citation

```
@misc{shmlp2026,
  title  = {SH-MLP: Autonomous Fault Detection, Root Cause Analysis, and Recovery in Machine Learning Pipelines},
  year   = {2026},
  url    = {https://github.com/gagan-53/self-healing-ml-pipeline}
}
```

## License

MIT — see [LICENSE](LICENSE).
