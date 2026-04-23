/**
 * python_bridge.js — Persistent Python Daemon Manager
 * ====================================================
 * Spawns python_daemon.py once at startup.
 * Routes match requests via stdin/stdout JSON pipe.
 * Auto-restarts if daemon crashes.
 * 
 * Usage (in routes):
 *   import { pythonBridge } from '../python_bridge.js';
 *   const result = await pythonBridge.match(imagePath);
 *   const done = await pythonBridge.regenerate();
 *   const status = await pythonBridge.status();
 */

import { spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { createInterface } from 'readline';
import { randomBytes } from 'crypto';

class PythonBridge {
  constructor() {
    this._process = null;
    this._readline = null;
    this._pending = new Map();   // requestId → { resolve, reject, timer }
    this._ready = false;
    this._starting = false;
    this._restartCount = 0;
    this._maxRestarts = 5;
    this._requestTimeout = 90000;  // 90s per request (model load on cold start)
  }

  _getPythonCmd() {
    if (existsSync('venv/Scripts/python.exe')) return 'venv/Scripts/python.exe';
    if (existsSync('venv/bin/python3'))        return 'venv/bin/python3';
    if (existsSync('venv/bin/python'))         return 'venv/bin/python';
    return 'python';
  }

  start() {
    // Skip local daemon if using remote AI service OR running on Vercel
    if (process.env.AI_SERVICE_URL || process.env.VERCEL) {
      this._ready = !!process.env.AI_SERVICE_URL;
      if (process.env.VERCEL) {
        console.log(`[PythonBridge] Running on Vercel — local daemon disabled.`);
      }
      if (process.env.AI_SERVICE_URL) {
        console.log(`[PythonBridge] Using remote AI service: ${process.env.AI_SERVICE_URL}`);
      } else if (process.env.VERCEL) {
        console.warn(`[PythonBridge] WARNING: No AI_SERVICE_URL provided. AI features will be disabled.`);
      }
      return;
    }
    if (this._process || this._starting) return;
    this._starting = true;
    this._ready = false;

    const pythonCmd = this._getPythonCmd();
    console.log(`[PythonBridge] Spawning: ${pythonCmd} python_daemon.py`);

    this._process = spawn(pythonCmd, ['python_daemon.py'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: process.cwd(),
    });

    // Read stdout line by line
    this._readline = createInterface({ input: this._process.stdout });
    this._readline.on('line', (line) => this._onLine(line));

    // Log stderr (Python debug output)
    this._process.stderr.on('data', (data) => {
      process.stdout.write(`[Python] ${data}`);
    });

    // Handle crash / exit
    this._process.on('close', (code) => {
      console.log(`[PythonBridge] Process exited with code ${code}`);
      this._process = null;
      this._readline = null;
      this._ready = false;
      this._starting = false;

      // Reject all pending requests
      for (const [id, pending] of this._pending) {
        clearTimeout(pending.timer);
        pending.reject(new Error(`Python daemon exited (code ${code})`));
      }
      this._pending.clear();

      // Auto-restart unless max restarts hit
      if (this._restartCount < this._maxRestarts) {
        this._restartCount++;
        const delay = Math.min(1000 * this._restartCount, 10000);
        console.log(`[PythonBridge] Restarting in ${delay}ms (attempt ${this._restartCount}/${this._maxRestarts})`);
        setTimeout(() => this.start(), delay);
      } else {
        console.error('[PythonBridge] Max restarts reached — daemon offline');
      }
    });

    this._process.on('error', (err) => {
      console.error('[PythonBridge] Spawn error:', err.message);
      this._starting = false;
    });
  }

  _onLine(line) {
    line = line.trim();
    if (!line) return;

    let msg;
    try {
      msg = JSON.parse(line);
    } catch (e) {
      console.warn('[PythonBridge] Non-JSON from daemon:', line);
      return;
    }

    // First message is the ready signal
    if (msg.ready !== undefined && !this._ready) {
      this._ready = true;
      this._starting = false;
      this._restartCount = 0;  // Reset restart counter on successful boot
      console.log(`[PythonBridge] Daemon ready — models_ok=${msg.models_ok}, cache=${msg.cache_size} persons`);
      return;
    }

    // Route response to pending request
    const id = msg.id;
    if (!id) return;

    const pending = this._pending.get(id);
    if (!pending) return;

    clearTimeout(pending.timer);
    this._pending.delete(id);
    pending.resolve(msg);
  }

  _send(payload) {
    return new Promise((resolve, reject) => {
      if (!this._process || !this._ready) {
        // If not ready yet, queue with a short poll
        let waited = 0;
        const poll = setInterval(() => {
          waited += 200;
          if (this._ready && this._process) {
            clearInterval(poll);
            this._doSend(payload, resolve, reject);
          } else if (waited > 30000) {
            clearInterval(poll);
            reject(new Error('Python daemon not ready after 30s'));
          }
        }, 200);
        return;
      }
      this._doSend(payload, resolve, reject);
    });
  }

  _doSend(payload, resolve, reject) {
    const id = randomBytes(8).toString('hex');
    payload.id = id;

    const timer = setTimeout(() => {
      this._pending.delete(id);
      reject(new Error(`Python request timeout after ${this._requestTimeout}ms`));
    }, this._requestTimeout);

    this._pending.set(id, { resolve, reject, timer });

    try {
      this._process.stdin.write(JSON.stringify(payload) + '\n');
    } catch (e) {
      clearTimeout(timer);
      this._pending.delete(id);
      reject(new Error(`Failed to write to daemon stdin: ${e.message}`));
    }
  }

  // Public API

  // Remote HTTP API (for Vercel/Railway deployment)
  async _sendHTTP(payload) {
    const url = process.env.AI_SERVICE_URL;
    if (!url) throw new Error('AI_SERVICE_URL not set');
    const response = await fetch(`${url}/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(90000)
    });
    return response.json();
  }

  async match(imagePath) {
    if (process.env.AI_SERVICE_URL) {
      // Remote daemon — send image as base64
      const imageBuffer = readFileSync(imagePath);
      const imageB64 = imageBuffer.toString('base64');
      return this._sendHTTP({ image_b64: imageB64 });
    }
    return this._send({ image_path: imagePath });
  }

  async regenerate() {
    return this._send({ command: 'regenerate' });
  }

  async invalidate(filename) {
    return this._send({ command: 'invalidate', filename });
  }

  async ping() {
    return this._send({ command: 'ping' });
  }

  async status() {
    return this._send({ command: 'status' });
  }

  isReady() {
    return this._ready;
  }
}

// Singleton — one daemon for the whole Node.js process
export const pythonBridge = new PythonBridge();

// Start immediately when this module is imported
pythonBridge.start();
