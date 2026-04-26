/**
 * FaultDiagnosticsProvider
 *
 * Creates VS Code diagnostic markers (squiggle underlines) on the line where
 * a fault was detected. Clicking the squiggle shows the fault details in a
 * hover tooltip. Diagnostics are cleared when a fault is dismissed or recovered.
 */
import * as vscode from 'vscode';
import { SidecarClient, FaultEvent } from './sidecarClient';
import { ConfigManager } from './configManager';

const SEVERITY_MAP: Record<string, vscode.DiagnosticSeverity> = {
    CRITICAL: vscode.DiagnosticSeverity.Error,
    HIGH:     vscode.DiagnosticSeverity.Warning,
    MEDIUM:   vscode.DiagnosticSeverity.Information,
    LOW:      vscode.DiagnosticSeverity.Hint,
};

const FAULT_DESCRIPTIONS: Record<string, string> = {
    'DL-1': 'Feature distribution drift (PSI above threshold)',
    'DL-2': 'Schema violation — unexpected column added/removed/renamed',
    'DL-3': 'Label distribution shift — class imbalance changed significantly',
    'DL-4': 'Data ingestion failure — row count dropped or exit code non-zero',
    'DL-5': 'Annotation quality degradation — missing rate spiked on label column',
    'ML-1': 'Model accuracy degradation — confidence dropped below threshold',
    'ML-2': 'Concept drift detected — ADWIN entropy window divergence',
    'ML-3': 'Training instability — gradient norm spike or stagnating loss',
    'ML-4': 'Confidence collapse — Expected Calibration Error above threshold',
    'IL-1': 'Memory exhaustion — RSS exceeded 90% of available memory',
    'IL-2': 'Execution timeout — step duration exceeded 3σ above baseline',
    'IL-3': 'Dependency conflict — incompatible package versions detected',
    'IL-4': 'Silent DAG failure — rows lost without non-zero exit code',
    'IL-5': 'Resource contention — cluster CPU above 90% saturation',
};

export class FaultDiagnosticsProvider {
    private collection: vscode.DiagnosticCollection;
    private activeFaults: Map<string, FaultEvent> = new Map();

    constructor(
        context: vscode.ExtensionContext,
        private readonly sidecar: SidecarClient,
        private readonly config: ConfigManager,
    ) {
        this.collection = vscode.languages.createDiagnosticCollection('sh-mlp');
        context.subscriptions.push(this.collection);

        // Listen for real-time fault events from the sidecar
        this.sidecar.on('fault', (fault: FaultEvent) => {
            if (this.config.showInlineDiag) { this.addFault(fault); }
        });
    }

    addFault(fault: FaultEvent): void {
        this.activeFaults.set(fault.fault_id, fault);
        this.refreshDiagnostics();
    }

    removeFault(faultId: string): void {
        this.activeFaults.delete(faultId);
        this.refreshDiagnostics();
    }

    clearAll(): void {
        this.activeFaults.clear();
        this.collection.clear();
    }

    refresh(): void {
        this.refreshDiagnostics();
    }

    onFileChange(_uri: vscode.Uri): void {
        // Future: trigger live observation on file save
    }

    private refreshDiagnostics(): void {
        const byFile = new Map<string, vscode.Diagnostic[]>();

        for (const fault of this.activeFaults.values()) {
            // Map stage_id to a file heuristic — show on the currently active editor
            const editors = vscode.window.visibleTextEditors.filter(e =>
                e.document.languageId === 'python'
            );
            const targetUri = editors[0]?.document.uri;
            if (!targetUri) { continue; }

            const key  = targetUri.toString();
            const diag = this.makeDiagnostic(fault, editors[0].document);
            const list = byFile.get(key) ?? [];
            list.push(diag);
            byFile.set(key, list);
        }

        this.collection.clear();
        for (const [uriStr, diags] of byFile.entries()) {
            this.collection.set(vscode.Uri.parse(uriStr), diags);
        }
    }

    private makeDiagnostic(fault: FaultEvent, doc: vscode.TextDocument): vscode.Diagnostic {
        // Find the first line mentioning the stage name, or default to line 0
        let targetLine = 0;
        const stagePattern = new RegExp(fault.stage_id.replace(/_/g, '[_-]?'), 'i');
        for (let i = 0; i < Math.min(doc.lineCount, 200); i++) {
            if (stagePattern.test(doc.lineAt(i).text)) { targetLine = i; break; }
        }

        const lineText = doc.lineAt(targetLine);
        const range    = new vscode.Range(
            targetLine, lineText.firstNonWhitespaceCharacterIndex,
            targetLine, lineText.text.length
        );

        const desc     = FAULT_DESCRIPTIONS[fault.fault_type] ?? fault.fault_type;
        const severity = SEVERITY_MAP[fault.severity] ?? vscode.DiagnosticSeverity.Warning;

        const message  = fault.raw_value != null
            ? `SH-MLP [${fault.fault_type}] ${desc} — observed=${fault.raw_value.toFixed(3)}, threshold=${fault.threshold?.toFixed(3) ?? '?'}`
            : `SH-MLP [${fault.fault_type}] ${desc}`;

        const diag         = new vscode.Diagnostic(range, message, severity);
        diag.source        = 'SH-MLP';
        diag.code          = { value: fault.fault_type, target: vscode.Uri.parse(`https://github.com/guardrails/sh-mlp/wiki/${fault.fault_type}`) };
        diag.relatedInformation = [
            new vscode.DiagnosticRelatedInformation(
                new vscode.Location(doc.uri, range),
                `Stage: ${fault.stage_id} | Fusion rule: ${fault.fusion_rule} | Detected: ${fault.detected_at}`
            )
        ];
        return diag;
    }

    dispose(): void { this.collection.dispose(); }
}
