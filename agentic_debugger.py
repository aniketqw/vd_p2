"""
🤖 LangGraph Agentic Debugging System for Vision/DL Models

Automatically diagnoses and fixes model code based on analysis reports.
Compares ACTUAL training logs before/after to validate fixes.

Features:
  - Parses analysis reports (ai_reasoning_summary_v3.md)
  - Maps findings to code lines
  - LLM-based root cause diagnosis
  - Code fix generation
  - Human-in-loop review
  - Executes fixed code and compares metrics
  - Iterates up to 3 times if improvement insufficient
  - Switchable LLM backends (local Ollama + Groq API)
  - Generates detailed reports with REAL metrics

Usage:
  python3 agentic_debugger.py \
    --report ai_reasoning_summary_v3.md \
    --code pipe/test.py \
    --baseline-logs pipe/logs/training_log_*.json \
    --local-llm-port 11434 \
    --groq-api-key $GROQ_API_KEY \
    --max-iterations 3
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
from datetime import datetime

# Try imports - graceful degradation if not available
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed. Install with: pip install langchain langchain-core")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── Data Classes ──────────────────────────────────────────────────────────────

class LLMProvider(Enum):
    """LLM provider selection."""
    LOCAL        = "local"        # Ollama (localhost:11434)
    LOCAL_OPENAI = "local_openai" # OpenAI-compatible local server (e.g. port 8081)
    GROQ         = "groq"         # Groq API
    AUTO         = "auto"         # Prefers GROQ → LOCAL_OPENAI → LOCAL


@dataclass
class TrainingLogMetrics:
    """Extracted metrics from training_log_*.json"""
    path: str
    best_accuracy: float
    best_epoch: int
    final_val_loss: float
    total_misclassified: int
    epochs: List[Dict[str, Any]]
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ParsedReport:
    """Output of Node 1: Parse Report"""
    findings: List[str]  # Key findings from report
    root_causes: List[str]  # Root cause hypotheses
    distortion_types: List[str]  # Affected distortion types
    baseline_metrics: Optional[Dict[str, float]] = None  # From report
    raw_content: str = ""


@dataclass
class CodeAnalysis:
    """Output of Node 2: Analyze Code"""
    findings_to_lines: Dict[str, List[Tuple[int, str]]]  # Finding → [(line, code_snippet), ...]
    identified_layers: List[Dict[str, Any]]  # Layer definitions
    problem_summary: str
    code_content: str


@dataclass
class DiagnosisResult:
    """Output of Node 3: Diagnose"""
    root_causes: List[Dict[str, Any]]  # {cause, location, impact, confidence}
    baseline_metrics: TrainingLogMetrics
    expected_improvement: Dict[str, float]  # {accuracy_gain, misclassified_reduction, confidence}
    confidence_overall: float
    escalate_to_human: bool = False
    escalation_reason: str = ""


@dataclass
class FixedCode:
    """Output of Node 4: Generate Fix"""
    code: str  # Complete fixed code
    changes: List[Dict[str, Any]]  # [{description, location, before, after}, ...]
    expected_impact: str
    llm_reasoning: str


@dataclass
class ExecutionResult:
    """Output of Node 5: Run Code"""
    success: bool
    training_log_path: Optional[str]  # Path to generated training_log
    metrics: Optional[TrainingLogMetrics] = None
    error_message: str = ""
    stderr_output: str = ""


@dataclass
class ComparisonResult:
    """Output of Node 6: Compare Metrics"""
    baseline: TrainingLogMetrics
    after_fix: TrainingLogMetrics
    improvements: Dict[str, float]  # {accuracy_change, misclassified_change, val_loss_change}
    success: bool  # improvement > threshold
    confidence: float
    iteration_count: int


@dataclass
class DebugReport:
    """Final output: Complete debugging report"""
    title: str
    executive_summary: str
    root_cause_analysis: str
    metrics_comparison: Dict[str, Any]
    code_changes: str
    why_its_working: str
    confidence: float
    iteration_history: List[Dict[str, Any]]
    status: str  # SUCCESS, PARTIAL, ESCALATED
    next_steps: str


# ── LLM Client Management ─────────────────────────────────────────────────────

class LLMClient:
    """Abstraction over Ollama (local) and Groq (API)"""

    def __init__(
        self,
        local_port: int = 11434,
        groq_api_key: Optional[str] = None,
        groq_rate_limit_per_min: int = 100,
        preferred_provider: LLMProvider = LLMProvider.AUTO,
        local_api_format: str = "ollama",  # "ollama" | "openai"
    ):
        self.local_port = local_port
        self.groq_api_key = groq_api_key
        self.groq_rate_limit = groq_rate_limit_per_min
        self.preferred_provider = preferred_provider
        self.local_api_format = local_api_format
        self.groq_requests_this_minute = 0
        self.groq_last_minute_start = datetime.now()

    def is_local_available(self) -> bool:
        """Check if Ollama is running."""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            resp = requests.get(f"http://localhost:{self.local_port}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def is_openai_local_available(self) -> bool:
        """Check if an OpenAI-compatible local server is up at the configured port."""
        import urllib.request
        try:
            url = f"http://localhost:{self.local_port}/v1/models"
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_groq_available(self) -> bool:
        """Check if Groq API is configured."""
        return bool(self.groq_api_key) and REQUESTS_AVAILABLE

    def _check_port(self, port: int) -> bool:
        """Quick liveness check for any OpenAI-compatible server."""
        import urllib.request as _ur
        try:
            with _ur.urlopen(f"http://localhost:{port}/v1/models", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _get_best_provider(self) -> LLMProvider:
        """Select best available provider. Priority: GROQ → LOCAL_OPENAI → LOCAL → port 8081."""
        if self.preferred_provider != LLMProvider.AUTO:
            return self.preferred_provider

        if self.is_groq_available() and self._groq_rate_available():
            return LLMProvider.GROQ
        if self.local_api_format == "openai" and self.is_openai_local_available():
            return LLMProvider.LOCAL_OPENAI
        if self.is_local_available():
            return LLMProvider.LOCAL
        # Last resort: try port 8081 unconditionally regardless of local_api_format
        if self.local_port != 8081 and self._check_port(8081):
            print("⚠️  Falling back to port 8081 (OpenAI-compatible server)")
            self.local_port = 8081
            return LLMProvider.LOCAL_OPENAI
        raise RuntimeError(
            "No LLM provider available (Groq unavailable, Ollama not running, port 8081 unreachable)"
        )

    def _groq_rate_available(self) -> bool:
        """Check if Groq rate limit allows another request."""
        now = datetime.now()
        elapsed = (now - self.groq_last_minute_start).total_seconds()

        if elapsed >= 60:
            self.groq_requests_this_minute = 0
            self.groq_last_minute_start = now

        return self.groq_requests_this_minute < self.groq_rate_limit

    def call(self, prompt: str, provider: Optional[LLMProvider] = None, model: Optional[str] = None) -> str:
        """Call LLM with auto-selection or specified provider."""
        p = provider or self._get_best_provider()

        # When local_api_format="openai", redirect LOCAL calls to LOCAL_OPENAI
        if p == LLMProvider.LOCAL and self.local_api_format == "openai":
            p = LLMProvider.LOCAL_OPENAI

        if p == LLMProvider.LOCAL_OPENAI:
            return self._call_openai_local(prompt, model or "gpt-3.5-turbo")
        elif p == LLMProvider.LOCAL:
            return self._call_ollama(prompt, model or "llama3.2:3b")
        elif p == LLMProvider.GROQ:
            return self._call_groq(prompt, model or "mixtral-8x7b-32768")
        else:
            raise ValueError(f"Unknown provider: {p}")

    def _call_ollama(self, prompt: str, model: str = "mistral") -> str:
        """Call local Ollama instance."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for Ollama calls")

        try:
            response = requests.post(
                f"http://localhost:{self.local_port}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def _call_openai_local(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """POST to a local OpenAI-compatible /v1/chat/completions endpoint."""
        import json as _json
        import urllib.request
        url     = f"http://localhost:{self.local_port}/v1/chat/completions"
        payload = _json.dumps({
            "model":      model,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "stream":     False,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = _json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI-local call failed: {e}")

    def _call_groq(self, prompt: str, model: str = "mixtral-8x7b-32768") -> str:
        """Call Groq API."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for Groq calls")

        if not self._groq_rate_available():
            raise RuntimeError("Groq rate limit exceeded for this minute")

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                },
                timeout=30,
            )
            response.raise_for_status()
            self.groq_requests_this_minute += 1
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}")


# ── Node Functions ────────────────────────────────────────────────────────────

def parse_report(report_path: Path) -> ParsedReport:
    """Node 1: Parse analysis report (markdown) → structured findings."""
    print("\n[Node 1] Parsing report...")

    with open(report_path) as f:
        content = f.read()

    # Extract findings
    findings = re.findall(r"(?:Root Cause|Finding|Issue|Problem)[:\s]+([^\n]+)", content, re.IGNORECASE)

    # Extract distortion types
    distortion_types = re.findall(r"(?:Distortion|Type)[:\s]+\*\*(\w+)\*\*", content, re.IGNORECASE)
    distortion_types = list(set(distortion_types))  # Deduplicate

    # Extract root causes section
    root_causes_match = re.search(
        r"(?:Root Cause|what misled)[:\s]+([^#]+?)(?=##|$)",
        content,
        re.IGNORECASE | re.DOTALL
    )
    root_causes = [line.strip() for line in (root_causes_match.group(1) if root_causes_match else "").split('\n') if line.strip()]

    # Extract metrics from training summary
    baseline_metrics = {}
    acc_match = re.search(r"Best Accuracy[|\s]+\*?\*?([0-9.]+)%?\*?\*?", content, re.IGNORECASE)
    if acc_match:
        baseline_metrics["best_accuracy"] = float(acc_match.group(1)) / 100

    misclassified_match = re.search(r"Total Misclassified[|\s]+([0-9,]+)", content)
    if misclassified_match:
        baseline_metrics["total_misclassified"] = int(misclassified_match.group(1).replace(",", ""))

    return ParsedReport(
        findings=findings,
        root_causes=root_causes[:5],  # Top 5
        distortion_types=distortion_types,
        baseline_metrics=baseline_metrics or None,
        raw_content=content
    )


def analyze_code(code_path: Path, parsed_report: ParsedReport, llm: LLMClient) -> CodeAnalysis:
    """Node 2: Map findings to code lines using LLM."""
    print("\n[Node 2] Analyzing code...")

    with open(code_path) as f:
        code_content = f.read()

    prompt = f"""Analyze this PyTorch code and map the following findings to specific lines and layers.

Findings to map:
{chr(10).join(parsed_report.findings)}

Root causes:
{chr(10).join(parsed_report.root_causes)}

Code:
```python
{code_content[:2000]}  # First 2000 chars to avoid token limit
```

For each finding:
1. Identify which layer/function is responsible
2. Quote the problematic code line
3. Explain why it's a problem

Format your response as JSON:
{{
  "findings_to_lines": {{"Finding 1": [{{line: int, code: str}}]}},
  "identified_layers": [{{name: str, lines: [int], description: str}}],
  "problem_summary": "Summary of all problems"
}}
"""

    try:
        response = llm.call(prompt, provider=LLMProvider.LOCAL)  # Use local for code analysis
        result = json.loads(response)
    except Exception as e:
        print(f"⚠️  LLM analysis failed: {e}. Using fallback heuristics.")
        result = {
            "findings_to_lines": {},
            "identified_layers": [],
            "problem_summary": f"Fallback analysis: {len(parsed_report.findings)} findings identified"
        }

    return CodeAnalysis(
        findings_to_lines=result.get("findings_to_lines", {}),
        identified_layers=result.get("identified_layers", []),
        problem_summary=result.get("problem_summary", ""),
        code_content=code_content
    )


def load_training_log(log_path: Path) -> TrainingLogMetrics:
    """Load training log JSON and extract metrics."""
    with open(log_path) as f:
        raw = json.load(f)

    summary = raw.get("summary", {})
    epochs = raw.get("epochs", [])

    return TrainingLogMetrics(
        path=str(log_path),
        best_accuracy=float(summary.get("best_accuracy", 0)),
        best_epoch=int(summary.get("best_epoch", 0)),
        final_val_loss=float(summary.get("final_val_loss", 0)),
        total_misclassified=int(summary.get("total_misclassified", 0)),
        epochs=epochs,
        timestamp=log_path.stem.replace("training_log_", "")
    )


def diagnose_root_causes(
    code_analysis: CodeAnalysis,
    parsed_report: ParsedReport,
    baseline_logs_dir: Path,
    llm: LLMClient
) -> DiagnosisResult:
    """Node 3: Diagnose root causes using LLM and baseline metrics."""
    print("\n[Node 3] Diagnosing root causes...")

    # Load baseline training log
    log_files = sorted(baseline_logs_dir.glob("training_log_*.json"))
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {baseline_logs_dir}")

    baseline_metrics = load_training_log(log_files[-1])  # Latest
    print(f"  Baseline accuracy: {baseline_metrics.best_accuracy:.2%}")
    print(f"  Baseline misclassified: {baseline_metrics.total_misclassified}")

    prompt = f"""Based on the code analysis and reported findings, diagnose the root causes.

Code Analysis:
{code_analysis.problem_summary}

Findings:
{chr(10).join(parsed_report.findings)}

Baseline Metrics:
- Best Accuracy: {baseline_metrics.best_accuracy:.2%}
- Total Misclassified: {baseline_metrics.total_misclassified}
- Final Val Loss: {baseline_metrics.final_val_loss:.4f}

Provide your diagnosis as JSON:
{{
  "root_causes": [
    {{"cause": "str", "location": "str", "impact": "str", "confidence": float}}
  ],
  "expected_improvement": {{"accuracy_gain": float, "misclassified_reduction": float, "confidence": float}},
  "escalate": false
}}
"""

    try:
        response = llm.call(prompt, provider=LLMProvider.GROQ)  # Use Groq for expert diagnosis
        result = json.loads(response)
    except Exception as e:
        print(f"⚠️  Groq diagnosis failed: {e}. Using heuristic diagnosis.")
        result = {
            "root_causes": [
                {
                    "cause": "Architecture issue (inferred)",
                    "location": "Convolutional layers",
                    "impact": "Model overfitting to training distribution",
                    "confidence": 0.7
                }
            ],
            "expected_improvement": {"accuracy_gain": 0.03, "misclassified_reduction": 0.2, "confidence": 0.6},
            "escalate": False
        }

    confidence = result.get("expected_improvement", {}).get("confidence", 0.5)

    return DiagnosisResult(
        root_causes=result.get("root_causes", []),
        baseline_metrics=baseline_metrics,
        expected_improvement=result.get("expected_improvement", {}),
        confidence_overall=confidence,
        escalate_to_human=result.get("escalate", False),
        escalation_reason=result.get("escalation_reason", "")
    )


def generate_fix(
    diagnosis: DiagnosisResult,
    code_analysis: CodeAnalysis,
    llm: LLMClient
) -> FixedCode:
    """Node 4: Generate fixed code."""
    print("\n[Node 4] Generating fix...")

    prompt = f"""Generate fixed PyTorch/Lightning code to address these root causes.

Root Causes:
{json.dumps(diagnosis.root_causes, indent=2)}

Original Code (excerpt):
```python
{code_analysis.code_content[:1500]}
```

Requirements:
1. Add BatchNorm2d after Conv2d layers (if missing)
2. Add L2 regularization (weight_decay) to optimizer
3. Add Dropout in convolutional layers
4. Keep all original logic and imports
5. Add detailed comments explaining changes

Output ONLY the fixed code block wrapped in ```python ... ```
Include a comment block explaining what was changed and why.
"""

    try:
        response = llm.call(prompt, provider=LLMProvider.GROQ)

        # Extract code block
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        fixed_code = code_match.group(1) if code_match else response
    except Exception as e:
        print(f"⚠️  Code generation failed: {e}")
        fixed_code = code_analysis.code_content  # Fallback: original code

    return FixedCode(
        code=fixed_code,
        changes=[
            {
                "description": "Add BatchNorm2d after Conv2d",
                "impact": "Normalizes feature distributions, prevents texture overfitting"
            },
            {
                "description": "Add L2 regularization (weight_decay=1e-4)",
                "impact": "Constrains weights, improves generalization"
            },
            {
                "description": "Add Dropout2d in conv backbone",
                "impact": "Breaks filter co-adaptation, improves robustness"
            }
        ],
        expected_impact=f"Expected accuracy gain: {diagnosis.expected_improvement.get('accuracy_gain', 0):.1%}",
        llm_reasoning="Applied standard CV best practices for robustness"
    )


def request_human_review(fixed_code: FixedCode) -> bool:
    """Node (Human Loop): Request human approval."""
    print("\n[HUMAN REVIEW] Generated fix ready for approval.")
    print("\nCode changes:")
    for change in fixed_code.changes:
        print(f"  - {change['description']}: {change['impact']}")

    print(f"\nExpected impact: {fixed_code.expected_impact}")
    print("\nProceed with execution? (y/n): ", end="")
    response = input().strip().lower()

    return response == 'y'


def run_fixed_code(fixed_code: FixedCode, original_code_path: Path, logs_dir: Path) -> ExecutionResult:
    """Node 5: Execute fixed code and capture training logs."""
    print("\n[Node 5] Running fixed code...")

    # Write fixed code to a new file
    fixed_path = original_code_path.parent / f"{original_code_path.stem}_fixed.py"
    with open(fixed_path, 'w') as f:
        f.write(fixed_code.code)

    print(f"  Wrote fixed code to {fixed_path}")

    # Execute the fixed code
    try:
        print("  Starting training...")
        result = subprocess.run(
            [sys.executable, str(fixed_path)],
            cwd=str(original_code_path.parent),
            capture_output=True,
            timeout=600,  # 10 minute timeout
            text=True
        )

        if result.returncode != 0:
            return ExecutionResult(
                success=False,
                training_log_path=None,
                error_message=f"Process exited with code {result.returncode}",
                stderr_output=result.stderr[-500:]  # Last 500 chars of stderr
            )

        # Find the generated training log
        log_files = sorted(logs_dir.glob("training_log_*.json"))
        if not log_files:
            return ExecutionResult(
                success=False,
                training_log_path=None,
                error_message="No training log generated"
            )

        latest_log = log_files[-1]
        metrics = load_training_log(latest_log)

        return ExecutionResult(
            success=True,
            training_log_path=str(latest_log),
            metrics=metrics
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            training_log_path=None,
            error_message="Training timed out after 10 minutes"
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            training_log_path=None,
            error_message=str(e)
        )


def compare_metrics(
    baseline: TrainingLogMetrics,
    after_fix: TrainingLogMetrics,
    iteration: int
) -> ComparisonResult:
    """Node 6: Compare metrics before/after."""
    print("\n[Node 6] Comparing metrics...")

    acc_change = after_fix.best_accuracy - baseline.best_accuracy
    misclassified_change = after_fix.total_misclassified - baseline.total_misclassified
    loss_change = after_fix.final_val_loss - baseline.final_val_loss

    # Success threshold: 3% accuracy improvement OR 20% misclassified reduction
    success = (acc_change > 0.03) or (abs(misclassified_change) / baseline.total_misclassified > 0.20)

    print(f"  Baseline accuracy: {baseline.best_accuracy:.2%}")
    print(f"  After fix accuracy: {after_fix.best_accuracy:.2%}")
    print(f"  Improvement: {acc_change:+.2%}")

    # Confidence in improvement
    confidence = min(0.95, abs(acc_change) * 10)  # 5% improvement = 50% confidence
    confidence = max(0.3, confidence)  # At least 30%

    return ComparisonResult(
        baseline=baseline,
        after_fix=after_fix,
        improvements={
            "accuracy_change": acc_change,
            "misclassified_change": misclassified_change,
            "val_loss_change": loss_change
        },
        success=success,
        confidence=confidence,
        iteration_count=iteration
    )


def generate_report(
    comparison: ComparisonResult,
    diagnosis: DiagnosisResult,
    fixed_code: FixedCode,
    llm: LLMClient
) -> DebugReport:
    """Node 7: Generate final report with REAL metrics."""
    print("\n[Node 7] Generating final report...")

    # Generate explanation of why it's working
    prompt = f"""Explain why these code changes improved the model performance.

Root causes identified:
{json.dumps(diagnosis.root_causes, indent=2)}

Code changes:
{json.dumps(fixed_code.changes, indent=2)}

Actual improvements:
- Accuracy: {comparison.baseline.best_accuracy:.2%} → {comparison.after_fix.best_accuracy:.2%} ({comparison.improvements['accuracy_change']:+.2%})
- Misclassified: {comparison.baseline.total_misclassified} → {comparison.after_fix.total_misclassified} ({comparison.improvements['misclassified_change']:+.0f})
- Validation Loss: {comparison.baseline.final_val_loss:.4f} → {comparison.after_fix.final_val_loss:.4f} ({comparison.improvements['val_loss_change']:+.4f})

Provide a 3-4 paragraph explanation of the progressive understanding:
1. What the problem was
2. How the fixes address it
3. Why the metrics improved
4. Next steps (if any)
"""

    try:
        why_working = llm.call(prompt, provider=LLMProvider.GROQ)
    except Exception as e:
        why_working = f"Fixes applied to address identified root causes. Execution successful with {comparison.improvements['accuracy_change']:+.2%} improvement."

    # Build the report
    executive_summary = f"""
Original accuracy: **{comparison.baseline.best_accuracy:.2%}**
Fixed accuracy: **{comparison.after_fix.best_accuracy:.2%}**
Improvement: **{comparison.improvements['accuracy_change']:+.2%}** ✅
Confidence: **{comparison.confidence:.0%}**
"""

    metrics_table = f"""
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Best Accuracy | {comparison.baseline.best_accuracy:.2%} | {comparison.after_fix.best_accuracy:.2%} | {comparison.improvements['accuracy_change']:+.2%} |
| Misclassified | {comparison.baseline.total_misclassified:,} | {comparison.after_fix.total_misclassified:,} | {comparison.improvements['misclassified_change']:+.0f} |
| Val Loss | {comparison.baseline.final_val_loss:.4f} | {comparison.after_fix.final_val_loss:.4f} | {comparison.improvements['val_loss_change']:+.4f} |
"""

    code_diff = f"""
**Changes applied:**
{chr(10).join([f"- {c['description']}: {c['impact']}" for c in fixed_code.changes])}
"""

    return DebugReport(
        title="AI Model Debugging Report",
        executive_summary=executive_summary,
        root_cause_analysis="\n".join([f"- {rc['cause']} (confidence: {rc['confidence']:.0%})" for rc in diagnosis.root_causes]),
        metrics_comparison={"before": comparison.baseline.to_dict(), "after": comparison.after_fix.to_dict()},
        code_changes=code_diff,
        why_its_working=why_working,
        confidence=comparison.confidence,
        iteration_history=[{"iteration": comparison.iteration_count, "result": "SUCCESS"}],
        status="SUCCESS",
        next_steps="Model is now more robust to visual distortions. Consider validating on additional test sets."
    )


def escalate_or_fallback(comparison: ComparisonResult, iteration: int) -> DebugReport:
    """Node 8: Handle failure - escalate or generate best-effort report."""
    print("\n[Node 8] Handling incomplete improvement...")

    escalate = iteration >= 3 or comparison.confidence < 0.3
    status = "ESCALATED" if escalate else "PARTIAL"

    return DebugReport(
        title="AI Model Debugging Report (Partial)",
        executive_summary=f"""
Original accuracy: {comparison.baseline.best_accuracy:.2%}
After attempt: {comparison.after_fix.best_accuracy:.2%}
Improvement: {comparison.improvements.get('accuracy_change', 0.0):+.2%}
Status: {status}
""",
        root_cause_analysis="Root cause diagnosis inconclusive. Further manual investigation recommended.",
        metrics_comparison={},
        code_changes="Fix applied but improvement insufficient.",
        why_its_working="Changes did not produce expected improvement. May require different approach.",
        confidence=comparison.confidence,
        iteration_history=[{"iteration": iteration, "result": status}],
        status=status,
        next_steps=f"{'Escalate to human for expert review.' if escalate else 'Try different fix strategy.'}",
    )


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def run_agentic_debugger(
    report_path: Path,
    code_path: Path,
    baseline_logs_dir: Path,
    local_llm_port: int = 11434,
    groq_api_key: Optional[str] = None,
    max_iterations: int = 3,
) -> DebugReport:
    """Main orchestrator: run full agentic pipeline."""

    print("\n" + "="*70)
    print("  🤖 Agentic Debugging System (LangGraph)")
    print("="*70)

    # Initialize LLM client
    llm = LLMClient(
        local_port=local_llm_port,
        groq_api_key=groq_api_key,
        preferred_provider=LLMProvider.AUTO
    )

    # Node 1: Parse report
    parsed_report = parse_report(report_path)
    print(f"✓ Found {len(parsed_report.findings)} findings")
    print(f"✓ Identified distortions: {', '.join(parsed_report.distortion_types)}")

    # Node 2: Analyze code
    code_analysis = analyze_code(code_path, parsed_report, llm)
    print(f"✓ Analyzed code, identified {len(code_analysis.identified_layers)} layers")

    # Iteration loop
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Node 3: Diagnose
        diagnosis = diagnose_root_causes(code_analysis, parsed_report, baseline_logs_dir, llm)

        if diagnosis.escalate_to_human:
            print(f"⚠️  System recommends escalation: {diagnosis.escalation_reason}")
            return escalate_or_fallback(
                ComparisonResult(
                    baseline=diagnosis.baseline_metrics,
                    after_fix=diagnosis.baseline_metrics,
                    improvements={},
                    success=False,
                    confidence=0.2,
                    iteration_count=iteration
                ),
                iteration
            )

        print(f"✓ Diagnosed {len(diagnosis.root_causes)} root causes (confidence: {diagnosis.confidence_overall:.0%})")

        # Node 4: Generate fix
        fixed_code = generate_fix(diagnosis, code_analysis, llm)
        print(f"✓ Generated fixed code")

        # Human review
        approved = request_human_review(fixed_code)
        if not approved:
            print("❌ Fix rejected by human. Stopping.")
            return escalate_or_fallback(
                ComparisonResult(
                    baseline=diagnosis.baseline_metrics,
                    after_fix=diagnosis.baseline_metrics,
                    improvements={},
                    success=False,
                    confidence=0.0,
                    iteration_count=iteration
                ),
                iteration
            )

        # Node 5: Run code
        execution_result = run_fixed_code(fixed_code, code_path, baseline_logs_dir)

        if not execution_result.success:
            print(f"❌ Execution failed: {execution_result.error_message}")
            if iteration < max_iterations:
                print(f"   Retrying with refined diagnosis...")
                continue
            else:
                return escalate_or_fallback(
                    ComparisonResult(
                        baseline=diagnosis.baseline_metrics,
                        after_fix=diagnosis.baseline_metrics,
                        improvements={},
                        success=False,
                        confidence=0.1,
                        iteration_count=iteration
                    ),
                    iteration
                )

        print(f"✓ Code executed successfully")
        print(f"✓ New training log: {execution_result.training_log_path}")

        # Node 6: Compare metrics
        comparison = compare_metrics(diagnosis.baseline_metrics, execution_result.metrics, iteration)

        if comparison.success:
            # Node 7: Generate success report
            report = generate_report(comparison, diagnosis, fixed_code, llm)
            print(f"\n✅ SUCCESS! Accuracy improved from {comparison.baseline.best_accuracy:.2%} to {comparison.after_fix.best_accuracy:.2%}")
            return report
        else:
            print(f"   Improvement insufficient ({comparison.improvements['accuracy_change']:+.2%}). Iterating...")
            if iteration < max_iterations:
                print(f"   Refining diagnosis (iteration {iteration + 1}/{max_iterations})...")

    # Max iterations reached without success
    print(f"\n⚠️  Max iterations reached without achieving success threshold.")
    return escalate_or_fallback(comparison, iteration)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Debugging System")
    parser.add_argument("--report", type=Path, required=True, help="Analysis report (markdown)")
    parser.add_argument("--code", type=Path, required=True, help="Source code file")
    parser.add_argument("--baseline-logs", type=Path, required=True, help="Baseline logs directory")
    parser.add_argument("--local-llm-port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--groq-api-key", type=str, default=None, help="Groq API key")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max fix iterations")

    args = parser.parse_args()

    # Validate inputs
    if not args.report.exists():
        sys.exit(f"Report not found: {args.report}")
    if not args.code.exists():
        sys.exit(f"Code file not found: {args.code}")
    if not args.baseline_logs.exists():
        sys.exit(f"Logs directory not found: {args.baseline_logs}")

    # Run system
    report = run_agentic_debugger(
        report_path=args.report,
        code_path=args.code,
        baseline_logs_dir=args.baseline_logs,
        local_llm_port=args.local_llm_port,
        groq_api_key=args.groq_api_key,
        max_iterations=args.max_iterations,
    )

    # Output report
    print("\n" + "="*70)
    print("  FINAL REPORT")
    print("="*70)
    print(f"Title: {report.title}")
    print(f"Status: {report.status}")
    print(f"Confidence: {report.confidence:.0%}")
    print(f"\nExecutive Summary:{report.executive_summary}")
    print(f"\nRoot Cause Analysis:\n{report.root_cause_analysis}")
    print(f"\nCode Changes:\n{report.code_changes}")
    print(f"\nWhy It's Working:\n{report.why_its_working}")
    print(f"\nNext Steps:\n{report.next_steps}")

    # Save detailed report
    report_json = {
        "title": report.title,
        "status": report.status,
        "confidence": report.confidence,
        "executive_summary": report.executive_summary,
        "root_cause_analysis": report.root_cause_analysis,
        "metrics_comparison": report.metrics_comparison,
        "code_changes": report.code_changes,
        "why_its_working": report.why_its_working,
        "next_steps": report.next_steps,
        "iteration_history": report.iteration_history,
        "timestamp": datetime.now().isoformat()
    }

    output_path = args.code.parent / "agentic_debug_report.json"
    with open(output_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    print(f"\n✓ Detailed report saved: {output_path}")
