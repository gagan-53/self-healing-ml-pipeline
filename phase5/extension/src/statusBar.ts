/**
 * StatusBarManager — always-visible health indicator in VS Code status bar.
 *
 * States:  ● MONITORING (green) | ⚠ FAULT (amber) | ✕ CRITICAL (red) | ○ IDLE (gray)
 * Updates every 3 seconds via the poll loop in extension.ts.
 */
import * as vscode from 'vscode';
import { SidecarClient, PipelineStatus } from './sidecarClient';

const ICONS: Record<string, string> = {
    HEALTHY:  '$(pass-filled)',
    DEGRADED: '$(warning)',
    CRITICAL: '$(error)',
    UNKNOWN:  '$(circle-outline)',
};

export class StatusBarManager {
    private item: vscode.StatusBarItem;

    constructor(private readonly sidecar: SidecarClient) {
        this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
        this.item.command = 'shmlp.showPanel';
        this.item.tooltip  = 'SH-MLP Monitor — click to open panel';
        this.setIdle();
        this.item.show();
    }

    setIdle(): void {
        this.item.text    = '$(circle-outline) SH-MLP';
        this.item.color   = new vscode.ThemeColor('statusBarItem.foreground');
        this.item.tooltip = 'SH-MLP: idle';
    }

    async update(): Promise<void> {
        if (!this.sidecar.isRunning) { this.setIdle(); return; }
        try {
            const status = await this.sidecar.getStatus();
            this.applyStatus(status);
        } catch {
            this.item.text  = '$(sync~spin) SH-MLP';
            this.item.color = undefined;
        }
    }

    private applyStatus(s: PipelineStatus): void {
        const icon   = ICONS[s.health] ?? '$(circle-outline)';
        const faults = s.active_faults.length;
        const rec    = (s.recovery_rate * 100).toFixed(0);

        this.item.text = faults > 0
            ? `${icon} SH-MLP ${faults} fault${faults > 1 ? 's' : ''}`
            : `${icon} SH-MLP ${rec}% recovery`;

        this.item.color = s.health === 'CRITICAL' ? new vscode.ThemeColor('statusBarItem.errorBackground')
                        : s.health === 'DEGRADED'  ? new vscode.ThemeColor('statusBarItem.warningBackground')
                        : undefined;

        this.item.tooltip =
            `SH-MLP Monitor\n` +
            `Health: ${s.health}\n` +
            `Events: ${s.event_count}  FP rate: ${(s.fp_rate * 100).toFixed(1)}%\n` +
            `Recovery: ${rec}%  LinUCB Δ: +${(s.linucb_delta * 100).toFixed(0)}%\n` +
            `Click to open panel`;
    }

    dispose(): void { this.item.dispose(); }
}
