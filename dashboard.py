import os
import re
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

LOG_PATH = r"C:\Users\Saryal\.gemini\antigravity\brain\917df9e9-4a55-4fce-8dee-ec14a5d98261\.system_generated\tasks\task-1740.log"
METRICS_FILE = "training_metrics.json"

def get_gpu_stats():
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(',')]
        return {
            "gpu_util": parts[0] + "%",
            "vram_used": parts[1] + " MB",
            "vram_total": parts[2] + " MB",
            "temp": parts[3] + " °C",
            "power": parts[4] + " W"
        }
    except Exception:
        return {"gpu_util": "N/A", "vram_used": "N/A", "vram_total": "N/A", "temp": "N/A", "power": "N/A"}

def parse_training_log():
    try:
        last_step = "0"
        total_steps = "500"
        percent = "0%"
        eta = "Считается..."
        speed = "N/A"
        recent_logs = []

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in reversed(lines):
                match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([^\]]+)\]', line)
                if match:
                    percent = match.group(1) + "%"
                    last_step = match.group(2)
                    total_steps = match.group(3)
                    eta = match.group(5).strip()
                    speed = match.group(6).strip()
                    break
            for line in lines[-20:]:
                if line.strip():
                    recent_logs.append(line.strip())

        # Читаем точный лосс из JSON
        current_loss = "Ожидание шага 5..."
        loss_history = []
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    current_loss = str(data.get("current_loss", "..."))
                    loss_history = [e["loss"] for e in data.get("history", [])]
            except Exception:
                pass

        return {
            "step": last_step,
            "total_steps": total_steps,
            "percent": percent,
            "eta": eta,
            "speed": speed,
            "latest_loss": current_loss,
            "loss_history": loss_history,
            "logs": recent_logs
        }
    except Exception as e:
        return {"error": str(e), "step": "0", "total_steps": "500", "percent": "0%"}

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tabis1.0 0.8b - Live Fine-Tuning Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0b0f19;
            --card: #151d30;
            --accent: #38bdf8;
            --accent-green: #34d399;
            --text: #f1f5f9;
            --muted: #94a3b8;
            --border: #23314f;
        }
        body {
            margin: 0;
            padding: 24px;
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }
        .title h1 { margin: 0; font-size: 26px; color: var(--accent); }
        .title p { margin: 4px 0 0 0; color: var(--muted); font-size: 14px; }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            background: rgba(52, 211, 153, 0.15);
            color: var(--accent-green);
            font-weight: 600;
            font-size: 13px;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        .pulse {
            width: 8px; height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }
        .card-label { font-size: 13px; color: var(--muted); margin-bottom: 6px; }
        .card-value { font-size: 26px; font-weight: 700; color: var(--text); }
        .progress-box {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .progress-bar-bg {
            background: #1e293b;
            border-radius: 8px;
            height: 20px;
            overflow: hidden;
            margin: 12px 0;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            height: 100%;
            width: 0%;
            transition: width 0.4s ease;
        }
        .progress-details {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: var(--muted);
        }
        .chart-box {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
            height: 220px;
        }
        .console-box {
            background: #050811;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            color: #94a3b8;
            max-height: 180px;
            overflow-y: auto;
        }
        .console-line { margin: 3px 0; white-space: pre-wrap; word-break: break-all; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title">
                <h1>⚡ tabis1.0 0.8b Fine-Tuning Dashboard</h1>
                <p>NVIDIA GeForce RTX 5060 • 500 шагов • Живой график Loss</p>
            </div>
            <div class="status-badge">
                <div class="pulse"></div> ИДЕТ ОБУЧЕНИЕ
            </div>
        </header>

        <div class="progress-box">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-weight: 600; font-size: 16px;">Прогресс обучения</span>
                <span id="percent" style="font-size: 24px; font-weight: 700; color: var(--accent);">0%</span>
            </div>
            <div class="progress-bar-bg">
                <div id="bar" class="progress-bar-fill"></div>
            </div>
            <div class="progress-details">
                <span>Шаг: <strong id="step" style="color:var(--text);">0 / 500</strong></span>
                <span>Скорость: <strong id="speed" style="color:var(--text);">N/A</strong></span>
                <span>Расчетное время (ETA): <strong id="eta" style="color:var(--text);">Считается...</strong></span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-label">🔥 Текущий Loss (Ошибка)</div>
                <div class="card-value" id="loss" style="color: #f59e0b;">Ожидание...</div>
            </div>
            <div class="card">
                <div class="card-label">VRAM (RTX 5060)</div>
                <div class="card-value" id="vram">...</div>
            </div>
            <div class="card">
                <div class="card-label">Загрузка GPU / Темп.</div>
                <div class="card-value" id="gpu_load">...</div>
            </div>
            <div class="card">
                <div class="card-label">Всего шагов</div>
                <div class="card-value">500 шагов (~50 мин)</div>
            </div>
        </div>

        <div class="chart-box">
            <canvas id="lossChart"></canvas>
        </div>

        <div class="console-box">
            <div style="color: #fff; font-weight: 600; margin-bottom: 8px;">🖥️ Лог процесса:</div>
            <div id="console-logs"></div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss (чем ниже, тем умнее модель)',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#f1f5f9' } } }
            }
        });

        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('percent').textContent = data.train.percent;
                document.getElementById('bar').style.width = data.train.percent;
                document.getElementById('step').textContent = `${data.train.step} / ${data.train.total_steps}`;
                document.getElementById('speed').textContent = data.train.speed;
                document.getElementById('eta').textContent = data.train.eta;
                document.getElementById('loss').textContent = data.train.latest_loss;

                document.getElementById('vram').textContent = `${data.gpu.vram_used} / ${data.gpu.vram_total}`;
                document.getElementById('gpu_load').textContent = `${data.gpu.gpu_util} (${data.gpu.temp})`;

                if (data.train.loss_history && data.train.loss_history.length > 0) {
                    chart.data.labels = data.train.loss_history.map((_, i) => `${(i+1)*5}`);
                    chart.data.datasets[0].data = data.train.loss_history;
                    chart.update();
                }

                const logsEl = document.getElementById('console-logs');
                logsEl.innerHTML = '';
                (data.train.logs || []).forEach(line => {
                    const div = document.createElement('div');
                    div.className = 'console-line';
                    div.textContent = line;
                    logsEl.appendChild(div);
                });
            } catch (e) {
                console.error(e);
            }
        }
        setInterval(updateStats, 2000);
        updateStats();
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {
                "gpu": get_gpu_stats(),
                "train": parse_training_log()
            }
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=7860):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Live Dashboard запущен на: http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run(7860)
