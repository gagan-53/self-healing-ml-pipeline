/**
 * ConfigManager — typed wrapper around VS Code workspace configuration.
 */
import * as vscode from 'vscode';

export class ConfigManager {
    private get cfg() { return vscode.workspace.getConfiguration('shmlp'); }

    get pythonPath():     string   { return this.cfg.get<string>('pythonPath', ''); }
    get warmupCycles():   number   { return this.cfg.get<number>('warmupCycles', 60); }
    get f4MinRatio():     number   { return this.cfg.get<number>('f4MinRatio', 1.25); }
    get autoApprove():    boolean  { return this.cfg.get<boolean>('autoApprove', false); }
    get showInlineDiag(): boolean  { return this.cfg.get<boolean>('showInlineDiag', true); }
    get notifyOnFault():  boolean  { return this.cfg.get<boolean>('notifyOnFault', true); }
    get pipelineGlobs():  string[] {
        return this.cfg.get<string[]>('pipelineGlobs', [
            '**/*pipeline*.py', '**/*train*.py', '**/*ingest*.py'
        ]);
    }
}
