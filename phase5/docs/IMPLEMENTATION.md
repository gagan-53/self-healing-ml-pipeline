# SH-MLP VS Code Extension — Implementation Guide

## Overview

Phase 5 delivers the SH-MLP VS Code extension: a developer-facing monitoring tool
that surfaces fault detection, RCA, and recovery from inside the editor. Developers
worldwide install it from the VS Code Marketplace in under 90 seconds with zero
configuration required.

---

## Architecture

```
┌─────────────────────────────────────────┐    JSON-RPC     ┌──────────────────────────┐
│       VS Code (TypeScript layer)         │   over stdio    │   Python sidecar          │
│                                          │ ◄──────────── ► │   sh_mlp_server.py        │
│  extension.ts        ← main entry       │                  │                           │
│  sidecarClient.ts    ← RPC client       │                  │  SidecarServer            │
│  statusBar.ts        ← status indicator │                  │  PipelineState            │
│  diagnosticsProvider.ts ← squiggles     │                  │  M1 DetectionEngine       │
│  sidebarProvider.ts  ← webview panel    │                  │  M2 RCA Engine            │
│  treeProviders.ts    ← fault/history    │                  │  M3 Recovery Planner      │
│  notificationManager.ts ← toasts       │                  │  M4 Feedback Store        │
│  environmentDetector.ts ← Python path  │                  │                           │
│  configManager.ts    ← settings         │                  │                           │
│  logger.ts           ← output channel   │                  │                           │
└─────────────────────────────────────────┘                  └──────────────────────────┘
```

### Key design decisions

**Sidecar pattern over cloud API**: The Python sidecar runs locally alongside VS Code.
No internet required. Privacy preserved. Works in air-gapped environments.

**JSON-RPC 2.0 over stdio**: VS Code child processes communicate via stdin/stdout.
The protocol is minimal (6 request types, 2 push event types) and versioned.

**TypeScript never touches ML logic**: The extension only calls the 6 RPC methods.
The entire SH-MLP algorithm lives in Python.

---

## Directory Structure

```
phase5/
├── extension/                    # VS Code extension (TypeScript)
│   ├── package.json              # Extension manifest, contributes, dependencies
│   ├── tsconfig.json             # TypeScript config
│   └── src/
│       ├── extension.ts          # Entry point: activate() / deactivate()
│       ├── sidecarClient.ts      # JSON-RPC client over child process stdio
│       ├── environmentDetector.ts# Python path resolution (venv/conda/system)
│       ├── statusBar.ts          # Always-visible health indicator
│       ├── diagnosticsProvider.ts# Inline fault squiggle annotations
│       ├── sidebarProvider.ts    # WebviewView health dashboard
│       ├── treeProviders.ts      # Fault and history tree views
│       ├── notificationManager.ts# Toast notifications + approve/dismiss
│       ├── configManager.ts      # Typed wrapper for workspace settings
│       └── logger.ts             # Output channel singleton
├── sidecar/
│   └── sh_mlp_server.py          # Python JSON-RPC server (the sidecar)
└── docs/
    ├── IMPLEMENTATION.md          # This file
    └── Phase5_Documentation.docx  # Full phase documentation
```

---

## JSON-RPC Message Schema

All messages are newline-delimited JSON (one message per line).

### Requests (Extension → Sidecar)

```jsonc
// configure — set calibration parameters
{"jsonrpc":"2.0","id":1,"method":"configure",
 "params":{"warmup_cycles":60,"f4_min_ratio":1.25,"auto_approve":false}}

// observe — submit one pipeline observation for analysis
{"jsonrpc":"2.0","id":2,"method":"observe",
 "params":{"pipeline_id":"my_pipeline","stage_id":"ingestion","observation":{...}}}

// get_status — current health, active faults, stats
{"jsonrpc":"2.0","id":3,"method":"get_status","params":{"pipeline_id":"default"}}

// approve_recovery — trigger recovery for a specific fault
{"jsonrpc":"2.0","id":4,"method":"approve_recovery","params":{"fault_id":"uuid"}}

// dismiss_fault — acknowledge and clear without recovery
{"jsonrpc":"2.0","id":5,"method":"dismiss_fault","params":{"fault_id":"uuid"}}

// get_history — last N recovery events
{"jsonrpc":"2.0","id":6,"method":"get_history","params":{"pipeline_id":"default","limit":50}}

// shutdown — graceful exit
{"jsonrpc":"2.0","id":7,"method":"shutdown","params":{}}
```

### Responses (Sidecar → Extension)

```jsonc
// ready — sent once at startup before any requests
{"jsonrpc":"2.0","method":"ready"}

// fault_detected — push notification (no request id)
{"jsonrpc":"2.0","method":"fault_detected",
 "params":{"fault_id":"uuid","fault_type":"DL-1","severity":"HIGH","stage_id":"ingestion",...}}

// normal response
{"jsonrpc":"2.0","id":N,"result":{...}}

// error response
{"jsonrpc":"2.0","id":N,"error":{"code":-32000,"message":"description"}}
```

---

## Configuration Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `shmlp.pythonPath` | string | `""` | Python executable path (auto-detected if empty) |
| `shmlp.warmupCycles` | number | `60` | EMA baseline warmup cycles |
| `shmlp.f4MinRatio` | number | `1.25` | F4 cross-layer minimum magnitude gate |
| `shmlp.autoApprove` | boolean | `false` | Auto-approve all recovery actions |
| `shmlp.showInlineDiag` | boolean | `true` | Show inline fault annotations |
| `shmlp.notifyOnFault` | boolean | `true` | Show notification toast on fault |
| `shmlp.pipelineGlobs` | string[] | `["**/*pipeline*.py",...]` | Files to watch |

---

## Installation

### From VS Code Marketplace (primary path)
```
1. Open VS Code
2. Ctrl+Shift+X → search "SH-MLP Monitor"
3. Click Install
4. Open any Python file → monitoring starts automatically
```

### From VSIX bundle (enterprise/offline)
```bash
code --install-extension sh-mlp-1.0.0.vsix
```

### Build from source
```bash
cd phase5/extension
npm install
npm run compile
npx vsce package        # produces sh-mlp-1.0.0.vsix
```

---

## Python Integration

### Automatic observation (recommended)
The extension watches pipeline files and auto-submits observations when files change.

### Manual observation via Python SDK
```python
import sh_mlp_vscode

# Submit an observation directly from Python code
client = sh_mlp_vscode.ExtensionClient()
client.observe(
    pipeline_id="my_pipeline",
    stage_id="ingestion",
    observation={
        "row_count_in": 10000,
        "row_count_out": 9850,
        "exec_time_ms": 450.0,
        "memory_rss_mb": 512.0,
        "feature_statistics": { ... },
        "schema_hash": "schema_v3",
    }
)
```

### Direct sidecar usage (CLI mode)
```bash
# Run the sidecar standalone for testing
python sidecar/sh_mlp_server.py

# Send a test observation
echo '{"jsonrpc":"2.0","id":1,"method":"get_status","params":{"pipeline_id":"default"}}' \
  | python sidecar/sh_mlp_server.py
```

---

## Sprint Plan

### Sprint 1 (Weeks 19–20) — Python sidecar + JSON-RPC protocol
- [x] `sh_mlp_server.py` — complete JSON-RPC server
- [x] All 6 request handlers implemented
- [x] Push notification for `fault_detected`
- [x] Integration with SH-MLP M1–M4 modules
- [x] Cross-platform Python environment detection

### Sprint 2 (Weeks 21–22) — Extension UI
- [x] `extension.ts` — activation, lifecycle, poll loop
- [x] `sidecarClient.ts` — full RPC client with event emitter
- [x] `statusBar.ts` — always-visible health indicator
- [x] `diagnosticsProvider.ts` — inline squiggle annotations
- [x] `sidebarProvider.ts` — webview health dashboard
- [x] `treeProviders.ts` — fault and history tree views
- [x] `notificationManager.ts` — toast + approve/dismiss

### Sprint 3 (Weeks 23–24) — Polish + distribution
- [x] `package.json` — full extension manifest with Marketplace metadata
- [x] `configManager.ts` — typed settings wrapper
- [x] Zero-config activation via `activationEvents`
- [ ] VSIX bundle (requires `npm run package`)
- [ ] Marketplace submission
- [ ] End-to-end integration test suite

---

## Testing the Sidecar Manually

```bash
cd /home/claude/phase5/sidecar
python sh_mlp_server.py &
SIDECAR_PID=$!

# Test get_status
echo '{"jsonrpc":"2.0","id":1,"method":"get_status","params":{"pipeline_id":"default"}}' \
  >&${SIDECAR_PID}

# Shutdown
echo '{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}' >&${SIDECAR_PID}
```

---

## Fault Type Reference

| Code | Layer | Description | Recovery |
|------|-------|-------------|----------|
| DL-1 | Data  | Feature distribution drift (PSI) | Retrain on recent window |
| DL-2 | Data  | Schema violation | Halt + quarantine batch |
| DL-3 | Data  | Label distribution shift | Class reweighting |
| DL-4 | Data  | Ingestion failure | Retry with backoff |
| DL-5 | Data  | Annotation quality degradation | Quarantine + cleanlab |
| ML-1 | Model | Accuracy degradation | Full retrain / rollback |
| ML-2 | Model | Concept drift (ADWIN) | Switch to stable version |
| ML-3 | Model | Training instability | Reduce LR + clip gradients |
| ML-4 | Model | Confidence collapse (ECE) | Temperature scaling |
| IL-1 | Infra | Memory exhaustion | Batch-size binary search |
| IL-2 | Infra | Execution timeout | Retry with reduced subset |
| IL-3 | Infra | Dependency conflict | Restore pinned versions |
| IL-4 | Infra | Silent DAG failure | Halt + audit DAG lineage |
| IL-5 | Infra | Resource contention | Defer non-critical stages |
