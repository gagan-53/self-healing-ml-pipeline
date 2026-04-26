/**
 * SidebarProvider — WebviewView rendering the SH-MLP health dashboard.
 *
 * The webview is a self-contained HTML page that:
 *   - Shows pipeline health badge, active fault cards, LinUCB stats
 *   - Sends approve/dismiss commands back to the extension via postMessage
 *   - Auto-refreshes every 3 seconds via the poll loop
 */
import * as vscode from 'vscode';
import { SidecarClient, PipelineStatus, FaultEvent } from './sidecarClient';

export class SidebarProvider implements vscode.WebviewViewProvider {
    private view?: vscode.WebviewView;

    constructor(
        private readonly context: vscode.ExtensionContext,
        private readonly sidecar: SidecarClient,
    ) {}

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ): void {
        this.view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this.buildHtml(webviewView.webview);

        // Message handler — approve / dismiss from the webview
        webviewView.webview.onDidReceiveMessage(async (msg) => {
            if (msg.command === 'approve') {
                await vscode.commands.executeCommand('shmlp.approveRecovery', msg.faultId);
                await this.update();
            } else if (msg.command === 'dismiss') {
                await this.sidecar.dismissFault(msg.faultId);
                await this.update();
            } else if (msg.command === 'refresh') {
                await this.update();
            }
        });

        // Initial update
        this.update().catch(() => {});
    }

    async update(): Promise<void> {
        if (!this.view) { return; }
        try {
            const status = await this.sidecar.getStatus();
            this.view.webview.postMessage({ command: 'update', status });
        } catch { /* sidecar not ready */ }
    }

    private buildHtml(webview: vscode.Webview): string {
        const nonce = Math.random().toString(36).slice(2);
        return /* html */`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';">
<style nonce="${nonce}">
  :root { font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); color: var(--vscode-foreground); }
  body { margin: 0; padding: 8px; background: transparent; }
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge.healthy  { background: var(--vscode-testing-iconPassed); color: #fff; }
  .badge.degraded { background: var(--vscode-editorWarning-foreground); color: #000; }
  .badge.critical { background: var(--vscode-editorError-foreground); color: #fff; }
  .badge.unknown  { background: var(--vscode-disabledForeground); color: #fff; }
  .section-title { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--vscode-descriptionForeground); margin: 12px 0 4px; }
  .fault-card { border: 1px solid var(--vscode-editorWarning-foreground); border-radius: 6px; padding: 8px; margin-bottom: 6px; background: var(--vscode-editor-background); }
  .fault-title { font-weight: 600; font-size: 12px; margin-bottom: 2px; }
  .fault-meta { font-size: 11px; color: var(--vscode-descriptionForeground); margin-bottom: 6px; }
  .actions { display: flex; gap: 6px; }
  .btn { padding: 3px 10px; border-radius: 4px; border: 1px solid var(--vscode-button-border); cursor: pointer; font-size: 11px; }
  .btn.primary { background: var(--vscode-button-background); color: var(--vscode-button-foreground); border-color: transparent; }
  .btn.secondary { background: transparent; color: var(--vscode-foreground); }
  .btn:hover.primary  { background: var(--vscode-button-hoverBackground); }
  .btn:hover.secondary { background: var(--vscode-list-hoverBackground); }
  .stat-row { display: flex; justify-content: space-between; font-size: 11px; margin: 3px 0; }
  .stat-label { color: var(--vscode-descriptionForeground); }
  .stat-val { font-weight: 600; }
  .history-item { font-size: 11px; padding: 3px 0; border-bottom: 1px solid var(--vscode-editorWidget-border); display: flex; justify-content: space-between; }
  .recovered { color: var(--vscode-testing-iconPassed); }
  .not-recovered { color: var(--vscode-editorError-foreground); }
  #empty { font-size: 12px; color: var(--vscode-descriptionForeground); text-align: center; margin-top: 24px; }
  .spinner { animation: spin 1s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div id="root"><div id="empty"><span class="spinner">↻</span> Connecting...</div></div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
const root   = document.getElementById('root');

function healthClass(h) {
  return {HEALTHY:'healthy', DEGRADED:'degraded', CRITICAL:'critical', UNKNOWN:'unknown'}[h] ?? 'unknown';
}

function render(s) {
  const faults   = s.active_faults ?? [];
  const rec      = (s.recovery_rate * 100).toFixed(0);
  const fp       = (s.fp_rate * 100).toFixed(1);
  const delta    = (s.linucb_delta * 100).toFixed(0);
  const warmed   = s.warmed_up ? 'Yes' : 'Warming up…';

  let faultHTML = '';
  if (faults.length === 0) {
    faultHTML = '<div style="font-size:11px;color:var(--vscode-descriptionForeground);padding:4px 0">No active faults</div>';
  } else {
    faults.forEach(f => {
      faultHTML += '<div class="fault-card">'
        + '<div class="fault-title">' + f.fault_type + ' — ' + f.severity + '</div>'
        + '<div class="fault-meta">Stage: ' + f.stage_id + ' · ' + (f.detected_at ?? '') + '</div>'
        + (f.explanation ? '<div class="fault-meta">' + f.explanation + '</div>' : '')
        + '<div class="actions">'
        + '<button class="btn primary" onclick="approve(\\''+f.fault_id+'\\')">Accept recovery ✓</button>'
        + '<button class="btn secondary" onclick="dismiss(\\''+f.fault_id+'\\')">Dismiss</button>'
        + '</div></div>';
    });
  }

  root.innerHTML =
    '<div class="badge ' + healthClass(s.health) + '">' + s.health + '</div>'
    + '<div class="section-title">Active faults (' + faults.length + ')</div>'
    + faultHTML
    + '<div class="section-title">Statistics</div>'
    + '<div class="stat-row"><span class="stat-label">Events seen</span><span class="stat-val">' + s.event_count + '</span></div>'
    + '<div class="stat-row"><span class="stat-label">Recovery rate</span><span class="stat-val">' + rec + '%</span></div>'
    + '<div class="stat-row"><span class="stat-label">FP rate</span><span class="stat-val">' + fp + '%</span></div>'
    + '<div class="stat-row"><span class="stat-label">LinUCB Δ vs static</span><span class="stat-val">+' + delta + '%</span></div>'
    + '<div class="stat-row"><span class="stat-label">Warmed up</span><span class="stat-val">' + warmed + '</span></div>'
    + '<div style="margin-top:12px"><button class="btn secondary" onclick="refresh()">↻ Refresh</button></div>';
}

function approve(id) { vscode.postMessage({ command: 'approve', faultId: id }); }
function dismiss(id) { vscode.postMessage({ command: 'dismiss', faultId: id }); }
function refresh()   { vscode.postMessage({ command: 'refresh' }); }

window.addEventListener('message', e => {
  const msg = e.data;
  if (msg.command === 'update') { render(msg.status); }
});
</script>
</body>
</html>`;
    }
}
