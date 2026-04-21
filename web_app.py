import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Vision AI Pipeline Web UI")

# Absolute project root — all paths derived from here
_PROJECT_ROOT = Path(__file__).parent.resolve()

# Mount static directory
STATIC_DIR = _PROJECT_ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state for the running pipeline process
pipeline_queue = None
current_process = None

@app.on_event("startup")
async def startup_event():
    global pipeline_queue
    pipeline_queue = asyncio.Queue()

class RunConfig(BaseModel):
    mode: str               # 'full', 'stageA', 'stageB'
    groq_key: str           # API Key
    ollama_port: str        # 11434
    no_vlm: bool            # True/False
    stage_c: bool           # True/False
    user_code: str = ""     # Optional custom model code
    local_llm_format: str = "ollama"  # "ollama" | "openai"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(STATIC_DIR / "index.html", "r") as f:
        return f.read()

@app.post("/api/run")
async def run_pipeline(config: RunConfig):
    global current_process
    
    # Kill entire process group of any still-running pipeline (takes out child processes too)
    if current_process and current_process.returncode is None:
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except Exception:
            try:
                current_process.terminate()
            except Exception:
                pass

    # Clear the queue
    while not pipeline_queue.empty():
        pipeline_queue.get_nowait()

    # Build command — absolute path so it works regardless of working directory.
    # -u forces unbuffered stdout/stderr so output streams to the UI in real time.
    unified_pipeline = str(_PROJECT_ROOT / "unified_pipeline.py")
    cmd = [sys.executable, "-u", unified_pipeline, "--auto-approve"]

    if config.mode == 'stageA':
        cmd.extend(["--only-stage", "a"])
    elif config.mode == 'stageB':
        cmd.extend(["--only-stage", "b"])

    if config.no_vlm:
        cmd.append("--no-vlm")
    if config.stage_c and config.mode != 'stageA':
        cmd.append("--stage-c")

    if config.user_code and config.user_code.strip():
        custom_code_path = _PROJECT_ROOT / "pipe" / "custom_model.py"
        with open(custom_code_path, "w") as f:
            f.write(config.user_code)
        cmd.extend(["--code", str(custom_code_path)])  # absolute path

    # Only forward a valid numeric port
    if config.ollama_port and config.ollama_port.strip().isdigit():
        cmd.extend(["--local-llm-port", config.ollama_port.strip()])

    if config.local_llm_format and config.local_llm_format in ("ollama", "openai"):
        cmd.extend(["--local-llm-format", config.local_llm_format])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # propagate unbuffered mode to all child processes
    if config.groq_key and config.groq_key.strip():
        groq_key = config.groq_key.strip()
        env["GROQ_API_KEY"] = groq_key
        cmd.extend(["--groq-api-key", groq_key])  # explicit flag + env var

    # Notify start
    await pipeline_queue.put({"type": "stdout", "content": f"$ {' '.join(cmd)}\n"})

    # start_new_session=True puts the child in its own process group so that
    # os.killpg() can cleanly terminate the whole subtree.
    current_process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(_PROJECT_ROOT),
        start_new_session=True,
    )
    
    # Fire and forget readers
    asyncio.create_task(read_stream(current_process.stdout, "stdout"))
    asyncio.create_task(read_stream(current_process.stderr, "stderr"))
    asyncio.create_task(wait_for_process(current_process))
    
    return {"status": "started"}

async def read_stream(stream, stream_type):
    while True:
        chunk = await stream.read(1024)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        await pipeline_queue.put({"type": stream_type, "content": text})

async def wait_for_process(process):
    await process.wait()
    await pipeline_queue.put({"type": "end", "code": process.returncode})

@app.get("/api/stream")
async def stream_logs(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                # Wait for data from the queue
                data = await asyncio.wait_for(pipeline_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop yielding if process ende
                if data["type"] == "end":
                    break
            except asyncio.TimeoutError:
                # Send a heartbeat or just continue to check disconnect
                yield ": heartbeat\n\n"
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/reports")
async def get_reports():
    pipeline_report_path = _PROJECT_ROOT / "master_pipeline_report.md"
    analysis_report_path = _PROJECT_ROOT / "ai_reasoning_summary_v3.md"
    
    reports = {
        "pipeline_report": None,
        "analysis_report": None,
    }

    if pipeline_report_path.exists():
        with open(pipeline_report_path, "r", encoding="utf-8") as f:
            reports["pipeline_report"] = f.read()

    if analysis_report_path.exists():
        with open(analysis_report_path, "r", encoding="utf-8") as f:
            reports["analysis_report"] = f.read()

    return reports

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5056)
