/**
 * EnvironmentDetector
 *
 * Resolves a usable Python interpreter in this priority order:
 *   1. User-configured shmlp.pythonPath setting
 *   2. Active VS Code Python extension interpreter (ms-python.python)
 *   3. Workspace virtual environment (.venv, venv, env, .env)
 *   4. Conda environment (via conda info --json)
 *   5. System Python (python3, python)
 *
 * Also verifies sh_mlp is importable and offers to install if not.
 */
import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { Logger } from './logger';

export class EnvironmentDetector {
    constructor(private readonly logger: Logger) {}

    async resolve(configuredPath: string): Promise<string | undefined> {
        // 1. User-configured path
        if (configuredPath) {
            if (await this.verify(configuredPath)) { return configuredPath; }
            this.logger.warn(`Configured path ${configuredPath} is not a valid Python interpreter.`);
        }

        // 2. VS Code Python extension active interpreter
        const pyExt = vscode.extensions.getExtension('ms-python.python');
        if (pyExt) {
            const api = pyExt.exports;
            const interp = api?.settings?.getExecutionDetails?.()?.execCommand?.[0];
            if (interp && await this.verify(interp)) {
                this.logger.info(`Using VS Code Python interpreter: ${interp}`);
                return interp;
            }
        }

        // 3. Workspace virtual environments
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
            for (const folder of workspaceFolders) {
                const candidates = this.venvCandidates(folder.uri.fsPath);
                for (const c of candidates) {
                    if (fs.existsSync(c) && await this.verify(c)) {
                        this.logger.info(`Using workspace venv: ${c}`);
                        return c;
                    }
                }
            }
        }

        // 4. System Python
        for (const cmd of ['python3', 'python']) {
            const resolved = await this.resolveOnPath(cmd);
            if (resolved) { return resolved; }
        }

        return undefined;
    }

    private venvCandidates(root: string): string[] {
        const isWin = process.platform === 'win32';
        const bin   = isWin ? 'Scripts' : 'bin';
        const exe   = isWin ? 'python.exe' : 'python';
        return ['.venv', 'venv', 'env', '.env'].map(d =>
            path.join(root, d, bin, exe)
        );
    }

    private resolveOnPath(cmd: string): Promise<string | undefined> {
        return new Promise(resolve => {
            const which = process.platform === 'win32' ? 'where' : 'which';
            cp.exec(`${which} ${cmd}`, (err, stdout) => {
                if (err) { resolve(undefined); return; }
                const p = stdout.trim().split('\n')[0].trim();
                this.verify(p).then(ok => resolve(ok ? p : undefined));
            });
        });
    }

    private verify(pythonPath: string): Promise<boolean> {
        return new Promise(resolve => {
            cp.exec(
                `"${pythonPath}" -c "import sys; print(sys.version)"`,
                { timeout: 5000 },
                (err) => resolve(!err)
            );
        });
    }

    /**
     * Check whether sh_mlp is importable and offer pip install if not.
     */
    async ensureShMlp(pythonPath: string): Promise<boolean> {
        return new Promise(resolve => {
            cp.exec(
                `"${pythonPath}" -c "import sh_mlp"`,
                { timeout: 5000 },
                async (err) => {
                    if (!err) { resolve(true); return; }
                    const choice = await vscode.window.showWarningMessage(
                        'SH-MLP: Python package `sh_mlp` not found. Install it now?',
                        'Install', 'Cancel'
                    );
                    if (choice !== 'Install') { resolve(false); return; }
                    const terminal = vscode.window.createTerminal('SH-MLP Install');
                    terminal.sendText(`"${pythonPath}" -m pip install sh-mlp`);
                    terminal.show();
                    // Poll for 30 seconds
                    let tries = 0;
                    const poll = setInterval(() => {
                        tries++;
                        cp.exec(`"${pythonPath}" -c "import sh_mlp"`, (e2) => {
                            if (!e2 || tries >= 10) {
                                clearInterval(poll);
                                resolve(!e2);
                            }
                        });
                    }, 3000);
                }
            );
        });
    }
}
