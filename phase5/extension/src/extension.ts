/**
 * SH-MLP VS Code Extension — Main Entry Point
 * 
 * Activation: auto-activates on Python files and pipeline globs.
 * Lifecycle:  starts Python sidecar, registers commands/providers, tears down on deactivate.
 */
import * as vscode from 'vscode';
import { SidecarClient } from './sidecarClient';
import { StatusBarManager } from './statusBar';
import { FaultDiagnosticsProvider } from './diagnosticsProvider';
import { SidebarProvider } from './sidebarProvider';
import { FaultTreeProvider, HistoryTreeProvider } from './treeProviders';
import { NotificationManager } from './notificationManager';
import { EnvironmentDetector } from './environmentDetector';
import { ConfigManager } from './configManager';
import { Logger } from './logger';

let sidecar: SidecarClient | undefined;
let statusBar: StatusBarManager | undefined;
let diagnostics: FaultDiagnosticsProvider | undefined;
let notifications: NotificationManager | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    const logger = Logger.getInstance();
    logger.info('SH-MLP extension activating...');

    // ── Config ───────────────────────────────────────────────────────────────
    const config = new ConfigManager();

    // ── Python environment detection ──────────────────────────────────────────
    const detector = new EnvironmentDetector(logger);
    const pythonPath = await detector.resolve(config.pythonPath);
    if (!pythonPath) {
        vscode.window.showErrorMessage(
            'SH-MLP: Could not find Python interpreter. Set shmlp.pythonPath in settings.',
            'Open Settings'
        ).then(sel => sel && vscode.commands.executeCommand('workbench.action.openSettings', 'shmlp.pythonPath'));
        return;
    }
    logger.info(`Python resolved: ${pythonPath}`);

    // ── Sidecar client ────────────────────────────────────────────────────────
    sidecar = new SidecarClient(pythonPath, context.extensionPath, logger);
    await sidecar.start();

    // Configure with user settings
    await sidecar.configure({
        warmup_cycles: config.warmupCycles,
        f4_min_ratio: config.f4MinRatio,
        auto_approve: config.autoApprove,
    });

    // ── UI components ─────────────────────────────────────────────────────────
    statusBar = new StatusBarManager(sidecar);
    diagnostics = new FaultDiagnosticsProvider(context, sidecar, config);
    notifications = new NotificationManager(sidecar, config, diagnostics);

    const faultTree   = new FaultTreeProvider(sidecar);
    const historyTree = new HistoryTreeProvider(sidecar);
    const sidebar     = new SidebarProvider(context, sidecar);

    // Register tree views
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider('shmlp.faultsView',  faultTree),
        vscode.window.registerTreeDataProvider('shmlp.historyView', historyTree),
        vscode.window.registerWebviewViewProvider('shmlp.healthView', sidebar),
    );

    // ── Commands ──────────────────────────────────────────────────────────────
    context.subscriptions.push(
        vscode.commands.registerCommand('shmlp.startMonitor', async () => {
            await sidecar!.start();
            statusBar!.update();
            vscode.window.showInformationMessage('SH-MLP monitoring started.');
        }),

        vscode.commands.registerCommand('shmlp.stopMonitor', async () => {
            await sidecar!.stop();
            statusBar!.setIdle();
            vscode.window.showInformationMessage('SH-MLP monitoring stopped.');
        }),

        vscode.commands.registerCommand('shmlp.showPanel', () => {
            vscode.commands.executeCommand('workbench.view.extension.shmlp-sidebar');
        }),

        vscode.commands.registerCommand('shmlp.approveRecovery', async (faultId: string) => {
            const result = await sidecar!.approveRecovery(faultId);
            const msg = result.success
                ? `Recovery applied: ${result.strategy_id} (${result.ttr_ms}ms)`
                : `Recovery failed: ${result.error ?? 'unknown'}`;
            vscode.window.showInformationMessage(`SH-MLP: ${msg}`);
            diagnostics!.refresh();
            faultTree.refresh();
        }),

        vscode.commands.registerCommand('shmlp.viewHistory', () => {
            vscode.commands.executeCommand('shmlp.historyView.focus');
        }),

        vscode.commands.registerCommand('shmlp.configure', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'shmlp');
        }),
    );

    // ── File watcher ──────────────────────────────────────────────────────────
    const watcher = vscode.workspace.createFileSystemWatcher(
        `{${config.pipelineGlobs.join(',')}}`
    );
    watcher.onDidChange(uri => diagnostics?.onFileChange(uri));
    context.subscriptions.push(watcher);

    // ── Poll loop ─────────────────────────────────────────────────────────────
    const pollInterval = setInterval(async () => {
        try {
            await statusBar!.update();
            await faultTree.refresh();
            await historyTree.refresh();
            await notifications!.checkAndNotify();
        } catch { /* sidecar may be restarting */ }
    }, 3000);

    context.subscriptions.push({
        dispose: () => {
            clearInterval(pollInterval);
            sidecar?.stop();
            statusBar?.dispose();
            diagnostics?.dispose();
        }
    });

    logger.info('SH-MLP extension activated successfully.');
    vscode.window.showInformationMessage('SH-MLP Monitor active — 14 fault detectors running.');
}

export function deactivate(): void {
    sidecar?.stop();
}
