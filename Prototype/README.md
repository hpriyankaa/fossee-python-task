# PEC (Planner–Executor–Critic) — Minimal, Practical Prototype

A lightweight pipeline that analyzes student-written Python code, generates **Socratic prompts** that probe understanding, and **scores** the prompts against a rubric — using an **open-source model** (e.g., Qwen2.5-Coder) served via **vLLM**.

---

## Installation

> Python 3.10–3.12 recommended. Works on Linux/WSL/macOS. (Windows users: run vLLM in WSL.)

1) Create & activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux/WSL
# .venv\Scripts\activate         # Windows PowerShell (if not using WSL)
```
2) Install client deps for the PEC app:
```bash
pip install -r requirements.txt   # requests>=2.32.0
```
3) (Server) Install vLLM (GPU box or same machine if you have a GPU):
```bash
pip install -U vllm
```

## Running the Prototype

1) Start the model server (vLLM)
>Run in a separate terminal (Linux/WSL/macOS). For an 8GB GPU, these settings are safe:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 \
  --port 8000
```
>Tip: If VRAM is tight, lower --max-model-len to 1024 or add --quantization bitsandbytes.
2) Point PEC to your server
```bash
export VLLM_BASE_URL="http://localhost:8000"
export VLLM_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
```
3) Run PEC over the sample dataset
```bash
python main.py
```


## Output

Clean, human-readable console report per sample (Task → Planner Summary → Socratic Questions → Rubric Scores).

# What Each Component Does
## Planner (Analysis)

__Input:__ task, student_code

__Output (JSON):__

+ concepts[] — Core Python ideas involved (e.g., recursion, control flow, loop invariants)

+ bloom — estimated Bloom’s level (Remember/Understand/Apply/Analyze/Evaluate/Create)

+ misconceptions[] — Likely gaps (off-by-one, parity confusion, missing base case, etc.)

__Purpose:__ Turn raw code into a conceptual diagnosis that drives targeted questioning.

## Executor (Socratic Prompting)

__Input:__ task, student_code, planner_json

__Output (JSON):__

+ questions[] —2–4 Socratic questions (edge cases, “what happens if…”, “why…”)

+ rationale — brief reason these questions probe true understanding

__Purpose:__ Ask guiding questions that uncover reasoning — without revealing solutions.

## Critic (Scoring Against Rubric)

__Input:__ task, student_code, planner_json, executor_json, rubric

__Output (JSON):__

+ scores — {Relevance, Depth, ConceptAccuracy, NonDisclosure, Clarity} (1–5 each)

+ justification — short explanation of the scoring

__Purpose__: Ensure prompts are on-task, deep, accurate, non-disclosive, and clear.

__Overall Score__: Weighted average emphasizing ConceptAccuracy & NonDisclosure.

# Possible Future Research Plan

## Cross-Model Comparisons

Conduct side-by-side evaluations of Qwen2.5-Coder, Qwen3-Coder, and DeepSeek Coder v2 using the same PEC prototype. Track differences in accuracy, interpretability, efficiency, and pedagogical usefulness.

## Instruction-Tuning Customization

Fine-tune Qwen2.5-Coder-Instruct on education-specific datasets (e.g., student submissions with annotated misconceptions) to better align the model with pedagogical needs.

## Prototype Enhancement (PEC 2.0)

Expand the Planner–Executor–Critic pipeline into a fully adaptive feedback loop, where identified misconceptions automatically trigger new rounds of targeted Socratic questioning. Add multi-round evaluation to measure whether students improve after model-driven feedback.
