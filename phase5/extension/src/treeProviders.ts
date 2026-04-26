/**
 * Tree providers for the SH-MLP sidebar tree views.
 * FaultTreeProvider  — shows active faults with approve/dismiss actions
 * HistoryTreeProvider — shows last N recovery events
 */
import * as vscode from 'vscode';
import { SidecarClient, FaultEvent, HistoryEvent } from './sidecarClient';

// ── Fault tree ────────────────────────────────────────────────────────────────

class FaultItem extends vscode.TreeItem {
    constructor(public readonly fault: FaultEvent) {
        super(`${fault.fault_type} — ${fault.severity}`, vscode.TreeItemCollapsibleState.Collapsed);
        this.description   = fault.stage_id;
        this.tooltip       = `${fault.fault_type}\nStage: ${fault.stage_id}\nFusion: ${fault.fusion_rule}\nDetected: ${fault.detected_at}`;
        this.iconPath      = fault.severity === 'CRITICAL'
            ? new vscode.ThemeIcon('error',   new vscode.ThemeColor('editorError.foreground'))
            : new vscode.ThemeIcon('warning', new vscode.ThemeColor('editorWarning.foreground'));
        this.contextValue  = 'faultItem';
        this.command = {
            command: 'shmlp.approveRecovery',
            title:   'Approve recovery',
            arguments: [fault.fault_id],
        };
    }
}

class FaultDetailItem extends vscode.TreeItem {
    constructor(label: string, detail: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = detail;
    }
}

export class FaultTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    private _onChange = new vscode.EventEmitter<vscode.TreeItem | undefined | null | void>();
    readonly onDidChangeTreeData = this._onChange.event;
    private faults: FaultEvent[] = [];

    constructor(private readonly sidecar: SidecarClient) {
        sidecar.on('fault', () => this.refresh());
    }

    async refresh(): Promise<void> {
        try {
            const status = await this.sidecar.getStatus();
            this.faults  = status.active_faults;
            this._onChange.fire();
        } catch { /* sidecar not ready */ }
    }

    getTreeItem(el: vscode.TreeItem): vscode.TreeItem { return el; }

    getChildren(el?: vscode.TreeItem): vscode.TreeItem[] {
        if (!el) {
            if (this.faults.length === 0) {
                const empty = new vscode.TreeItem('No active faults', vscode.TreeItemCollapsibleState.None);
                empty.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
                return [empty];
            }
            return this.faults.map(f => new FaultItem(f));
        }
        if (el instanceof FaultItem) {
            const f = el.fault;
            return [
                new FaultDetailItem('Stage',        f.stage_id),
                new FaultDetailItem('Fusion rule',  f.fusion_rule),
                new FaultDetailItem('Detected',     f.detected_at),
                ...(f.raw_value != null ? [new FaultDetailItem('Value', `${f.raw_value.toFixed(3)} (thr=${f.threshold?.toFixed(3)})`)] : []),
                ...(f.explanation ? [new FaultDetailItem('Explanation', f.explanation!)] : []),
            ];
        }
        return [];
    }
}

// ── History tree ──────────────────────────────────────────────────────────────

class HistoryItem extends vscode.TreeItem {
    constructor(ev: HistoryEvent) {
        const label = `${ev.fault_type} — ${ev.recovered ? 'recovered' : 'unresolved'}`;
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = ev.detected_at.split('T')[1]?.slice(0, 8) ?? ev.detected_at;
        this.tooltip     = ev.strategy_id
            ? `Strategy: ${ev.strategy_id}  TTR: ${ev.ttr_ms}ms`
            : 'No recovery applied';
        this.iconPath    = ev.recovered
            ? new vscode.ThemeIcon('check',   new vscode.ThemeColor('testing.iconPassed'))
            : new vscode.ThemeIcon('circle-outline', new vscode.ThemeColor('descriptionForeground'));
    }
}

export class HistoryTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    private _onChange = new vscode.EventEmitter<vscode.TreeItem | undefined | null | void>();
    readonly onDidChangeTreeData = this._onChange.event;
    private events: HistoryEvent[] = [];

    constructor(private readonly sidecar: SidecarClient) {}

    async refresh(): Promise<void> {
        try {
            this.events = await this.sidecar.getHistory('default', 20);
            this._onChange.fire();
        } catch { /* sidecar not ready */ }
    }

    getTreeItem(el: vscode.TreeItem): vscode.TreeItem { return el; }

    getChildren(): vscode.TreeItem[] {
        if (this.events.length === 0) {
            const empty = new vscode.TreeItem('No events yet', vscode.TreeItemCollapsibleState.None);
            return [empty];
        }
        return this.events.map(e => new HistoryItem(e));
    }
}
