# PEC (Planner-Executor-Critic) Prototype
This prototype demonstrates a lightweight **Planner–Executor–Critic (PEC)** pipeline that analyzes student-written Python code, generates **Socratic prompts** to probe understanding, and scores those prompts against a rubric. It is designed to explore how open-source models can be adapted for high-level student competence analysis.

The pipeline follows the **orchestrator–evaluator pattern** described in Anthropic’s AI design patterns, ensuring modularity and clear separation of analysis, questioning, and scoring. It supports both **local inference (Ollama)** and **cloud inference (OpenRouter)**. 

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

## Running the Prototype

The PEC prototype supports two primary backends:

### 1. Ollama (Local Inference)
Ollama is a lightweight local model runner that works on CPU or GPU.

1. Install Ollama: [https://ollama.com/download](https://ollama.com/download)  
2. Pull the required model (select the model size, e.g., 1.5B, 7B, based on system resources and availability, this provides scalability):
   ```bash
   ollama pull qwen2.5-coder:1.5b-instruct
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
4. Set Environment Variables:
   ```bash
   export PEC_BACKEND=ollama
   export OLLAMA_MODEL="qwen2.5-coder:1.5b-instruct"
   ```
5. Run the PEC Prototype now:
   On Windows, Use:
   ```bash
   python main.py
   ```
   On Linux/WSL/macOS, Use:
   ```bash
   python3 main.py
   ```

### 2. OpenRouter (Hosted Inference)
OpenRouter provides API access to many open and commercial models without requiring local GPU resources.  
Since models are hosted remotely, you can run any available weight (e.g., 7B, 14B, 32B), and in the case of **Qwen2.5-Coder-Instruct**, multiple weights are offered for free.

1. Sign up at [https://openrouter.ai](https://openrouter.ai) and generate an API key.  
2. Set environment variables:
   ```bash
   export PEC_BACKEND=openrouter
   export OPENROUTER_API_KEY="sk-or-..."
   export OPENROUTER_MODEL="qwen-2.5-coder-32b-instruct:free"
   ```
3. Run the PEC Prototype now:
   On Windows, Use:
   ```bash
   python main.py
   ```
   On Linux/WSL/macOS, Use:
   ```bash
   python3 main.py
   ```

## Work in Progress: Alternative Backends

This prototype currently runs with **Ollama** (local) and **OpenRouter** (hosted), making it portable across CPU and non-GPU environments.  

Looking ahead, it can also be adapted to use **vLLM** for faster local inference on GPU-equipped systems.  
This would allow scaling to larger models with optimized throughput, which could be developed further based on project needs and availability of GPU resources.  


## Output
Clean, human-readable console report per sample (Task → Planner Summary → Socratic Questions → Rubric Scores). 
<br>
- Results are also stored in the **`artifacts/`** folder for later use:  
  - `.jsonl` → raw structured runs (full PEC outputs)  
  - `.csv` → summary table (scores + questions + justification)  
  - `.md` → nicely formatted report, ready for inclusion in papers or appendices
<br>
Output screenshots from sample dataset used in this repo can be found below:
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/img1.jpeg" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 1: Output of Sample Data 1</em></p>
<br>
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/img2.jpeg" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 2: Output of Sample Data 2</em></p>
<br>
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/img3.jpeg" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 3: Output of Sample Data 3</em></p>
<br>
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/img4.png" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 4: Output of Artifacts Directory</em></p>

# What Each Component Does
## Planner (Analysis)

__Input:__ task, student_code

__Output (JSON):__

+ concepts[] - Core Python ideas involved (e.g., recursion, control flow, loop invariants)

+ bloom - estimated Bloom’s level (Remember/Understand/Apply/Analyze/Evaluate/Create)

+ misconceptions[] - Likely gaps (off-by-one, parity confusion, missing base case, etc.)

__Purpose:__ Turn raw code into a conceptual diagnosis that drives targeted questioning.

## Executor (Socratic Prompting)

__Input:__ task, student_code, planner_json

__Output (JSON):__

+ questions[] - 2–4 Socratic questions (edge cases, “what happens if…”, “why…”)

+ rationale - brief reason these questions probe true understanding

__Purpose:__ Ask guiding questions that uncover reasoning - without revealing solutions.

## Critic (Scoring Against Rubric)

__Input:__ task, student_code, planner_json, executor_json, rubric

__Output (JSON):__

+ scores - {Relevance, Depth, ConceptAccuracy, NonDisclosure, Clarity} (1–5 each)

+ justification - short explanation of the scoring

__Purpose__: Ensure prompts are on-task, deep, accurate, non-disclosive, and clear.

__Overall Score__: Weighted average emphasizing ConceptAccuracy & NonDisclosure.

# Future Research Plan

## Cross-Model Comparisons

Conduct side-by-side evaluations of Qwen2.5-Coder, Qwen3-Coder, and DeepSeek Coder v2 using the same PEC prototype. Track differences in accuracy, interpretability, efficiency, and pedagogical usefulness.

## Instruction-Tuning Customization

Fine-tune Qwen2.5-Coder-Instruct on education-specific datasets (e.g., student submissions with annotated misconceptions) to better align the model with pedagogical needs.

## Prototype Enhancement (PEC 2.0)

Expand the Planner–Executor–Critic pipeline into a fully adaptive feedback loop, where identified misconceptions automatically trigger new rounds of targeted Socratic questioning. Add multi-round evaluation to measure whether students improve after model-driven feedback.
