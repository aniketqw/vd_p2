# 🤖 LangGraph Agentic Debugging System - Architecture & Design

**Status:** DESIGN PHASE  
**Target:** Vision/DL-only (PyTorch Lightning models like CIFAR-10)  
**Date:** 2026-04-21

---

## 🎯 System Overview

```
┌─ USER INPUT ────────────────────────────────┐
│  1. Analysis Report (ai_reasoning_summary_v3.md)
│  2. Source Code (test.py)
│  3. Baseline Training Logs (pipe/logs/)
└─────────────────────────────────────────────┘
            ↓
┌─ LANGGRAPH AGENTIC PIPELINE ────────────────┐
│                                              │
│  Node 1: Parse Report                        │
│  ├─ Extract findings + root causes          │
│  ├─ Parse metrics from report               │
│  └─ Identify distortion types               │
│                                              │
│  Node 2: Analyze Code                       │
│  ├─ Map findings to code lines              │
│  ├─ Identify responsible layers             │
│  └─ LLM-based code understanding            │
│                                              │
│  Node 3: Diagnose Root Causes               │
│  ├─ Use LLM to confirm root cause           │
│  ├─ Query baseline training logs            │
│  └─ Generate confidence scores              │
│                                              │
│  Node 4: Generate Fix                       │
│  ├─ LLM creates fixed code                  │
│  ├─ Preserve original structure             │
│  └─ Add detailed comments                   │
│                                              │
│  [HUMAN REVIEW] ← Approve fix?              │
│                                              │
│  Node 5: Run Fixed Code                     │
│  ├─ Execute fixed code                      │
│  ├─ Generate NEW training logs              │
│  ├─ Extract metrics                         │
│  └─ Handle errors gracefully                │
│                                              │
│  Node 6: Compare Metrics                    │
│  ├─ Load baseline + new training logs       │
│  ├─ Calculate improvements                  │
│  ├─ Compute confidence in fix               │
│  └─ Decide: SUCCESS or ITERATE?             │
│                                              │
│  [DECISION] SUCCESS?                        │
│  ├─ YES → Node 7: Generate Report           │
│  └─ NO → [LOOP COUNTER] < max_iterations?  │
│      ├─ YES → Node 3: Refine Diagnosis      │
│      └─ NO → Node 8: Escalate/Fallback      │
│                                              │
│  Node 7: Generate Report                    │
│  ├─ Explain improvement with ACTUAL metrics │
│  ├─ Quote results (before → after)          │
│  ├─ Progressive understanding               │
│  └─ Include code diff                       │
│                                              │
│  Node 8: Escalate or Fallback               │
│  ├─ If very bad → escalate to human         │
│  ├─ Else → generate best-effort report      │
│  └─ Document why fix didn't work            │
│                                              │
└─────────────────────────────────────────────┘
            ↓
┌─ OUTPUTS ───────────────────────────────────┐
│  1. fixed_code.py (with comments)           │
│  2. debugging_report.md (with REAL results) │
│  3. metrics_comparison.json (before/after)  │
│  4. iteration_history.json (if multiple)    │
└─────────────────────────────────────────────┘
```

---

## 🔄 State Machine & Nodes

### Node Definitions

| Node | Input | Logic | Output | LLM Required |
|------|-------|-------|--------|--------------|
| **1. Parse Report** | Markdown file | Extract findings, root causes, metrics | ParsedReport (dict) | ✓ (local) |
| **2. Analyze Code** | Code + findings | Map findings to code lines | CodeAnalysis (dict) | ✓ (local/Groq) |
| **3. Diagnose** | CodeAnalysis + baseline logs | Confirm root causes, check metrics | DiagnosisResult (dict) | ✓ (Groq) |
| **4. Generate Fix** | DiagnosisResult | Create fixed code with comments | FixedCode (string) | ✓ (Groq) |
| **[Human Review]** | FixedCode | User approves/rejects | Approved? (bool) | ✗ |
| **5. Run Code** | FixedCode | Execute, capture training logs | ExecutionResult (dict) | ✗ |
| **6. Compare** | Baseline + New logs | Calculate improvements, decide next | ComparisonResult (dict) | ✓ (local) |
| **7. Report** | All above | Generate markdown with REAL results | Report (markdown) | ✓ (Groq) |
| **8. Escalate** | ComparisonResult | Handle failure, escalate if needed | FinalReport (markdown) | ✓ (local) |

---

## 📊 Data Structures

### TrainingLogMetrics
```python
{
  "path": "pipe/logs/training_log_20260323_231145.json",
  "best_accuracy": 0.6811,
  "best_epoch": 2,
  "final_val_loss": 0.8987,
  "total_misclassified": 10083,
  "epochs": [
    {"epoch": 0, "accuracy": 0.0, "num_misclassified": 4084},
    {"epoch": 1, "accuracy": 0.5971, "num_misclassified": 3189},
    {"epoch": 2, "accuracy": 0.6811, "num_misclassified": 2810}
  ],
  "timestamp": "20260323_231145"
}
```

### DiagnosisResult
```python
{
  "root_causes": [
    {
      "cause": "Missing BatchNorm2d",
      "location": "test.py:62-64",
      "impact": "Unbounded feature distributions",
      "confidence": 0.95
    },
    ...
  ],
  "baseline_metrics": TrainingLogMetrics,
  "expected_improvement": {
    "accuracy_gain": 0.05,
    "misclassified_reduction": 0.35,
    "confidence": 0.90
  }
}
```

### ComparisonResult
```python
{
  "baseline": TrainingLogMetrics,
  "after_fix": TrainingLogMetrics,
  "improvements": {
    "accuracy_change": 0.05,  # 68.11% → 73.11%
    "misclassified_change": -3543,  # 10083 → 6540
    "val_loss_change": -0.12,
  },
  "success": True,  # improvement > threshold
  "confidence": 0.92
}
```

---

## 🔀 Control Flow Decision Points

### Decision 1: After Diagnosis
```
IF confidence(root_cause) > 0.8:
  → Proceed to Generate Fix
ELSE:
  → Ask human for confirmation OR
  → Escalate to human
```

### Decision 2: After Fix Generation
```
Human Review:
  APPROVE → Run Code
  REJECT  → Return to Diagnosis (refine)
  ESCALATE → Stop, ask human for guidance
```

### Decision 3: After Code Execution
```
IF code runs successfully:
  → Compare Metrics
ELSE:
  → Try to debug error
  → If still fails → Escalate
```

### Decision 4: After Comparison
```
IF improvement > threshold (5% accuracy OR -20% misclassified):
  → Generate Report (SUCCESS)
ELSE IF iterations < max_iterations (3):
  → Refine Diagnosis, Generate new Fix
ELSE:
  → Escalate or Generate Best-Effort Report
```

---

## 🧠 LLM Usage Strategy (Switchable)

### Local LLM (Fast, No Rate Limit)
- **Model**: Llama2-7B or Mistral-7B (via Ollama)
- **Port**: 11434 (Ollama default)
- **Use Cases**:
  - Node 1: Parse Report (structured extraction)
  - Node 6: Compare Metrics (analysis of numbers)
  - Node 8: Escalate Decision (logical assessment)

### Groq API (Powerful, Rate Limited)
- **Model**: Mixtral-8x7b-32768
- **Rate Limit**: TBD by user (e.g., 100 requests/minute)
- **Use Cases**:
  - Node 2: Analyze Code (complex code understanding)
  - Node 3: Diagnose Root Causes (expert reasoning)
  - Node 4: Generate Fix (code generation)
  - Node 7: Generate Report (expert writing)

### Fallback Strategy
```
IF (Groq available AND rate_limit not exceeded):
  use Groq
ELSE IF local_llm available:
  use Local LLM
ELSE:
  return best_effort_output OR escalate to human
```

---

## ⏰ Iteration Control

```python
max_iterations = 3  # Config

iteration_counter = 0
while True:
  diagnosis = diagnose_root_causes()
  fix = generate_fix(diagnosis)
  
  human_approved = request_human_review(fix)
  if not human_approved:
    continue
  
  result = run_fixed_code(fix)
  comparison = compare_metrics(result)
  
  if comparison.success:
    report = generate_success_report(comparison)
    return report
  
  iteration_counter += 1
  if iteration_counter >= max_iterations:
    report = escalate_or_fallback(comparison)
    return report
  
  # Loop back to diagnose with new information
```

---

## 📝 Report Structure (With REAL Results)

```markdown
# AI Model Debugging Report

## 1. Executive Summary
- Original accuracy: 68.11%
- **Fixed accuracy: 73.44%** ✅
- Improvement: +5.33% (achieved)
- Confidence: 92%

## 2. Root Cause Analysis
[Explain WHY it's improving]
- Missing BatchNorm → unbounded features
- L2 regularization prevents texture overfitting
- Dropout2d breaks artifact filter co-adaptation

## 3. Before → After Metrics
[ACTUAL QUOTED RESULTS]

### Accuracy Progression
| Epoch | Before | After | Delta |
|-------|--------|-------|-------|
| 0 | 59.42% | 61.5% | +2.08% |
| 1 | 59.71% | 67.2% | +7.49% |
| 2 | 68.11% | 73.44% | +5.33% |

### Distortion-Specific Failures
| Type | Before | After | Reduction |
|------|--------|-------|-----------|
| Blur | 2470 | 1243 | -49.7% ✅ |
| JPEG | 1558 | 783 | -49.8% ✅ |
| Pixelate | 860 | 561 | -34.8% ✅ |

### Loss Metrics
- Final validation loss: 0.8987 → 0.7234 (-19.5%)
- Total misclassified: 10,083 → 6,587 (-34.7%)

## 4. Code Changes
[Show diff with comments]
```
+ self.bn1 = torch.nn.BatchNorm2d(32)  # Normalize conv1
+ x = self.conv_dropout(x)              # Regularize backbone
```

## 5. Why It's Working (Progressive Understanding)
1. **BatchNorm normalizes**:
   - Feature distributions → N(0,1)
   - JPEG speckles (1-2 range) → standardized (0-1 std)
   - Model learns robust features, ignores texture variations

2. **L2 regularization constrains**:
   - Weight magnitudes penalized
   - Simpler, more generalizable filters
   - Prevents texture-specific overfitting

3. **Dropout2d breaks co-adaptation**:
   - Randomly disables feature channels
   - Prevents artifact pattern memorization
   - Improves robustness to visual perturbations

## 6. Confidence & Limitations
- Confidence in diagnosis: 95%
- Confidence in improvement: 92%
- Sample size: 10,000+ images
- Note: Pixelate improvement lower than blur/JPEG

## 7. Next Steps (Optional)
If further improvement desired:
- Consider ResNet18 backbone (deeper features)
- Add data augmentation with synthetic distortions
- Fine-tune learning rate schedule

---
**Generated by:** Agentic Debugging System  
**Timestamp:** 2026-04-21 12:00:00  
**Status:** ✅ SUCCESSFUL
```

---

## 🛡️ Error Handling

### Execution Errors
```
IF code_execution_fails:
  ├─ Parse error message
  ├─ Extract problematic lines
  ├─ IF error is import/syntax:
  │   └─ Ask LLM to fix syntax
  └─ ELSE:
      └─ Escalate to human (may need data/setup)
```

### Metric Extraction Errors
```
IF training_log not generated:
  ├─ Check if code ran at all
  ├─ Inspect error logs
  └─ Escalate with error details
```

### LLM Failures
```
IF llm_diagnosis fails:
  ├─ Retry with different LLM
  ├─ IF both fail:
  │   └─ Escalate with partial analysis
  └─ Generate best-effort fix from heuristics
```

---

## 🎯 Success Criteria

| Phase | Success Metric |
|-------|---|
| Diagnosis | Confidence > 80% |
| Code Generation | No syntax errors |
| Execution | Training completes, logs generated |
| Comparison | Improvement detected (any positive change) |
| Report | Human-readable, REAL metrics quoted |

---

## 📦 Implementation Roadmap

### Phase 1: MVP (Core System)
- [ ] Parse report (markdown → structured findings)
- [ ] Analyze code (map findings to lines)
- [ ] Diagnose (LLM-based root cause)
- [ ] Generate fix (code generation)
- [ ] Human review node
- [ ] Run code (execute, capture logs)
- [ ] Compare metrics (before/after)
- [ ] Generate report (with REAL results)

### Phase 2: Robustness
- [ ] Error handling + fallback logic
- [ ] Iteration loop + counter
- [ ] LLM rate limiting (Groq)
- [ ] Local LLM integration (Ollama)
- [ ] Escalation to human
- [ ] Logging + debugging

### Phase 3: Polish
- [ ] Better prompts
- [ ] Performance optimization
- [ ] Documentation
- [ ] Unit tests
- [ ] Example workflows

---

## 🚀 Next Steps

1. **Set up LLM access**:
   - [ ] Install Ollama (local LLM)
   - [ ] Configure Groq API key + rate limit
   - [ ] Test both clients

2. **Build core LangGraph**:
   - [ ] Define state schema
   - [ ] Implement each node
   - [ ] Set up control flow

3. **Integration**:
   - [ ] Read actual training logs from V3
   - [ ] Parse ai_reasoning_summary_v3.md
   - [ ] Execute test.py and capture outputs

4. **Validation**:
   - [ ] Run on CIFAR-10 example
   - [ ] Verify metrics comparison
   - [ ] Generate example report

