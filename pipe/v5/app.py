"""
v5/app.py
=========
FastAPI web server for the v5 end-to-end pipeline.

Serves the web UI and handles API requests for code analysis and training.

Usage:
    pip install fastapi uvicorn
    python3 pipe/v5/app.py
    # Open http://localhost:5000

Endpoints:
    GET  /                → Web UI
    POST /api/analyze     → Quick analysis (LLM only, no training)
    POST /api/run         → Full pipeline (train → analyze → fix → retrain)
    GET  /api/health      → Server + LLM status
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Dict

# ── path setup ────────────────────────────────────────────────────────────────
_V5_DIR   = Path(__file__).resolve().parent
_PIPE_DIR = _V5_DIR.parent
for _p in (str(_PIPE_DIR.parent), str(_PIPE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from v5.config import WEB_HOST, WEB_PORT, LLM_PORT, DEFAULT_EPOCHS
from v5.pipeline import (
    check_llm_server, run_quick_analysis, run_full_pipeline,
)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vision Pipeline v5", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# In-memory job store for SSE progress
_jobs: Dict[str, list] = {}

# ── default model code ───────────────────────────────────────────────────────
DEFAULT_CODE = '''\
class SimpleCNN(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss)
        self.log("accuracy", (preds == y).float().mean())
        return loss
'''


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page web UI."""
    html_path = _V5_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    """Check server and LLM status."""
    llm_ok = check_llm_server()
    return {"status": "ok", "llm_available": llm_ok, "llm_port": LLM_PORT}


@app.get("/api/default-code")
async def default_code():
    """Return the default SimpleCNN code."""
    return {"code": DEFAULT_CODE}


@app.post("/api/analyze")
async def analyze(request: Request):
    """Quick analysis: LLM reviews code statically, no training."""
    body = await request.json()
    user_code = body.get("code", "").strip()
    if not user_code:
        return JSONResponse({"error": "No code provided"}, status_code=400)

    if not check_llm_server():
        return JSONResponse({"error": "LLM server not available on port " + str(LLM_PORT)},
                            status_code=503)

    try:
        result = await asyncio.to_thread(run_quick_analysis, user_code)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/run")
async def run_pipeline(request: Request):
    """Full pipeline: train → analyze → fix → retrain → compare."""
    body = await request.json()
    user_code = body.get("code", "").strip()
    epochs = body.get("epochs", DEFAULT_EPOCHS)

    if not user_code:
        return JSONResponse({"error": "No code provided"}, status_code=400)

    if not check_llm_server():
        return JSONResponse({"error": "LLM server not available"}, status_code=503)

    # Create a job ID for SSE tracking
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = []

    def emit(event_type, message, data):
        _jobs.setdefault(job_id, []).append({
            "type": event_type, "message": message, "data": data,
        })

    try:
        result = await asyncio.to_thread(
            run_full_pipeline, user_code, epochs, emit
        )
        result["job_id"] = job_id
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Clean up old jobs
        if len(_jobs) > 50:
            oldest = list(_jobs.keys())[:25]
            for k in oldest:
                _jobs.pop(k, None)


@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    """SSE endpoint for real-time progress updates."""
    async def event_generator():
        seen = 0
        max_wait = 1800  # 30 min
        waited = 0
        while waited < max_wait:
            events = _jobs.get(job_id, [])
            while seen < len(events):
                ev = events[seen]
                yield f"data: {json.dumps(ev)}\n\n"
                seen += 1
                if ev.get("type") in ("complete", "error"):
                    return
            await asyncio.sleep(1)
            waited += 1
        yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🚀 Vision Pipeline v5 — Web App")
    print(f"  Open http://localhost:{WEB_PORT}")
    print("═" * 60 + "\n")
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="info")
