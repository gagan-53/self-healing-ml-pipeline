/**
 * Logger — singleton wrapper around VS Code output channel.
 */
import * as vscode from 'vscode';

export class Logger {
    private static instance: Logger;
    private channel: vscode.OutputChannel;

    private constructor() {
        this.channel = vscode.window.createOutputChannel('SH-MLP');
    }

    static getInstance(): Logger {
        if (!Logger.instance) { Logger.instance = new Logger(); }
        return Logger.instance;
    }

    info(msg: string):  void { this.channel.appendLine(`[INFO]  ${new Date().toISOString()} ${msg}`); }
    warn(msg: string):  void { this.channel.appendLine(`[WARN]  ${new Date().toISOString()} ${msg}`); }
    debug(msg: string): void { this.channel.appendLine(`[DEBUG] ${new Date().toISOString()} ${msg}`); }
    error(msg: string): void { this.channel.appendLine(`[ERROR] ${new Date().toISOString()} ${msg}`); }
    show(): void { this.channel.show(); }
}
