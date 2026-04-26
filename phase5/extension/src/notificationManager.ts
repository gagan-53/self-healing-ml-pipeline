/**
 * NotificationManager — shows toast notifications on fault detection.
 *
 * Respects shmlp.notifyOnFault and shmlp.autoApprove settings.
 * Deduplicates notifications — one toast per fault_id.
 */
import * as vscode from 'vscode';
import { SidecarClient, FaultEvent } from './sidecarClient';
import { ConfigManager } from './configManager';
import { FaultDiagnosticsProvider } from './diagnosticsProvider';

export class NotificationManager {
    private notified = new Set<string>();

    constructor(
        private readonly sidecar: SidecarClient,
        private readonly config:  ConfigManager,
        private readonly diag:    FaultDiagnosticsProvider,
    ) {
        sidecar.on('fault', (fault: FaultEvent) => this.onFault(fault));
    }

    async checkAndNotify(): Promise<void> {
        try {
            const status = await this.sidecar.getStatus();
            for (const fault of status.active_faults) {
                if (!this.notified.has(fault.fault_id)) { await this.onFault(fault); }
            }
        } catch { /* sidecar not ready */ }
    }

    private async onFault(fault: FaultEvent): Promise<void> {
        if (this.notified.has(fault.fault_id)) { return; }
        this.notified.add(fault.fault_id);

        if (this.config.autoApprove) {
            await this.sidecar.approveRecovery(fault.fault_id);
            this.diag.removeFault(fault.fault_id);
            return;
        }

        if (!this.config.notifyOnFault) { return; }

        const RECOVERY_LABELS: Record<string, string> = {
            'DL-1': 'Retrain on recent window',
            'DL-2': 'Halt + quarantine batch',
            'DL-3': 'Apply class reweighting',
            'DL-4': 'Retry ingestion with backoff',
            'DL-5': 'Quarantine + run cleanlab',
            'ML-1': 'Full retrain / model rollback',
            'ML-2': 'Switch to stable model version',
            'ML-3': 'Reduce LR + clip gradients',
            'ML-4': 'Apply temperature scaling',
            'IL-1': 'Binary-search optimal batch size',
            'IL-2': 'Retry with reduced dataset subset',
            'IL-3': 'Restore pinned dependency versions',
            'IL-4': 'Halt + audit DAG lineage',
            'IL-5': 'Defer non-critical pipeline stages',
        };
        const action = RECOVERY_LABELS[fault.fault_type] ?? 'Apply recovery';
        const msg    = `SH-MLP: ${fault.fault_type} detected at ${fault.stage_id} stage (${fault.severity}).`;

        const choice = await vscode.window.showWarningMessage(
            msg,
            { modal: false },
            action,
            'View Details',
            'Dismiss',
        );

        if (choice === action) {
            const result = await this.sidecar.approveRecovery(fault.fault_id);
            this.diag.removeFault(fault.fault_id);
            const outcome = result.success
                ? `Recovery succeeded: ${result.strategy_id} (${result.ttr_ms}ms)`
                : `Recovery failed: ${result.error ?? 'unknown error'}`;
            vscode.window.showInformationMessage(`SH-MLP: ${outcome}`);
        } else if (choice === 'View Details') {
            vscode.commands.executeCommand('shmlp.showPanel');
        } else if (choice === 'Dismiss') {
            await this.sidecar.dismissFault(fault.fault_id);
            this.diag.removeFault(fault.fault_id);
        }
    }
}
