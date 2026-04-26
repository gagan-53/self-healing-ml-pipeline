/**
 * SidecarClient — TypeScript client for the Python SH-MLP sidecar process.
 *
 * Protocol: newline-delimited JSON over stdin/stdout (JSON-RPC 2.0 subset).
 * The sidecar process is started as a child_process, communicated via stdio,
 * and gracefully shut down on extension deactivation.
 */
import * as cp from 'child_process';
import * as path from 'path';
import * as readline from 'readline';
import { EventEmitter } from 'events';
import { Logger } from './logger';

// ── Message types ────────────────────────────────────────────────────────────

export interface FaultEvent {
    fault_id:      string;
    pipeline_id:   string;
    fault_type:    string;
    severity:      'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
    stage_id:      string;
    fusion_rule:   string;
    detected_at:   string;
    raw_value?:    number;
    threshold?:    number;
    explanation?:  string;
}

export interface RecoveryResult {
    fault_id:    string;
    strategy_id: string;
    success:     boolean;
    ttr_ms:      number;
    error?:      string;
}

export interface PipelineStatus {
    pipeline_id:     string;
    health:          'HEALTHY' | 'DEGRADED' | 'CRITICAL' | 'UNKNOWN';
    active_faults:   FaultEvent[];
    event_count:     number;
    fp_rate:         number;
    recovery_rate:   number;
    linucb_delta:    number;
    warmed_up:       boolean;
    last_updated:    string;
}

export interface HistoryEvent {
    fault_id:      string;
    fault_type:    string;
    detected_at:   string;
    recovered:     boolean;
    strategy_id?:  string;
    ttr_ms?:       number;
}

export interface ConfigureParams {
    warmup_cycles: number;
    f4_min_ratio:  number;
    auto_approve:  boolean;
}

// ── RPC message shapes ────────────────────────────────────────────────────────

interface RpcRequest {
    jsonrpc: '2.0';
    id:      number;
    method:  string;
    params:  unknown;
}

interface RpcResponse {
    jsonrpc: '2.0';
    id:      number;
    result?: unknown;
    error?:  { code: number; message: string };
}

// ── Client ────────────────────────────────────────────────────────────────────

export class SidecarClient extends EventEmitter {
    private proc:       cp.ChildProcess | undefined;
    private msgId:      number = 1;
    private pending:    Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }> = new Map();
    private rl:         readline.Interface | undefined;
    private _running:   boolean = false;
    private sidecarPath: string;

    constructor(
        private readonly pythonPath: string,
        extensionPath: string,
        private readonly logger: Logger
    ) {
        super();
        this.sidecarPath = path.join(extensionPath, '..', 'sidecar', 'sh_mlp_server.py');
    }

    get isRunning(): boolean { return this._running; }

    async start(): Promise<void> {
        if (this._running) { return; }

        this.logger.info(`Starting sidecar: ${this.pythonPath} ${this.sidecarPath}`);

        this.proc = cp.spawn(this.pythonPath, [this.sidecarPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
        });

        this.proc.stderr?.on('data', (d: Buffer) => {
            const msg = d.toString().trim();
            if (msg) { this.logger.debug(`[sidecar stderr] ${msg}`); }
        });

        this.rl = readline.createInterface({ input: this.proc.stdout! });
        this.rl.on('line', (line: string) => this.handleLine(line));

        this.proc.on('exit', (code) => {
            this._running = false;
            this.logger.warn(`Sidecar exited with code ${code}`);
            this.emit('disconnected');
        });

        // Wait for ready signal
        await this.waitForReady();
        this._running = true;
        this.logger.info('Sidecar ready.');
        this.emit('connected');
    }

    private waitForReady(): Promise<void> {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Sidecar startup timeout')), 10000);
            this.once('ready', () => { clearTimeout(timeout); resolve(); });
            this.proc!.on('error', (e) => { clearTimeout(timeout); reject(e); });
        });
    }

    private handleLine(line: string): void {
        if (!line.trim()) { return; }
        try {
            const msg = JSON.parse(line) as RpcResponse & { method?: string; params?: unknown };
            if (msg.method === 'ready') {
                this.emit('ready');
                return;
            }
            if (msg.method === 'fault_detected') {
                this.emit('fault', msg.params as FaultEvent);
                return;
            }
            const pending = this.pending.get(msg.id);
            if (!pending) { return; }
            this.pending.delete(msg.id);
            if (msg.error) {
                pending.reject(new Error(msg.error.message));
            } else {
                pending.resolve(msg.result);
            }
        } catch (e) {
            this.logger.warn(`Bad sidecar line: ${line}`);
        }
    }

    private call<T>(method: string, params: unknown = {}): Promise<T> {
        return new Promise((resolve, reject) => {
            if (!this._running || !this.proc?.stdin) {
                reject(new Error('Sidecar not running'));
                return;
            }
            const id = this.msgId++;
            this.pending.set(id, {
                resolve: (v) => resolve(v as T),
                reject,
            });
            const req: RpcRequest = { jsonrpc: '2.0', id, method, params };
            this.proc.stdin!.write(JSON.stringify(req) + '\n');
        });
    }

    // ── Public API ────────────────────────────────────────────────────────────

    async configure(params: ConfigureParams): Promise<void> {
        await this.call('configure', params);
    }

    async getStatus(pipelineId: string = 'default'): Promise<PipelineStatus> {
        return this.call<PipelineStatus>('get_status', { pipeline_id: pipelineId });
    }

    async observe(pipelineId: string, stageId: string, observation: Record<string, unknown>): Promise<{ fault_event?: FaultEvent }> {
        return this.call('observe', { pipeline_id: pipelineId, stage_id: stageId, observation });
    }

    async approveRecovery(faultId: string): Promise<RecoveryResult> {
        return this.call<RecoveryResult>('approve_recovery', { fault_id: faultId });
    }

    async dismissFault(faultId: string): Promise<void> {
        await this.call('dismiss_fault', { fault_id: faultId });
    }

    async getHistory(pipelineId: string = 'default', limit: number = 50): Promise<HistoryEvent[]> {
        return this.call<HistoryEvent[]>('get_history', { pipeline_id: pipelineId, limit });
    }

    async stop(): Promise<void> {
        if (!this._running) { return; }
        try {
            await this.call('shutdown', {});
        } catch { /* process may have already exited */ }
        this.proc?.kill();
        this._running = false;
    }
}
