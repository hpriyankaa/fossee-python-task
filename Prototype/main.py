import json, pathlib, csv, datetime, textwrap, time, re
from typing import Dict, Any, List

# Use the existing vLLM client from your codes/ package
from codes.llm import chat

ROOT = pathlib.Path(__file__).parent
read_text = lambda *p: ROOT.joinpath(*p).read_text(encoding="utf-8")

# ---------- Pretty printing helpers (console looks clean) ----------

def _fmt_bullet_list(items, indent="  - "):
    if not items:
        return indent + "None"
    return "\n".join(f"{indent}{it}" for it in items)

def _fmt_rubric_table(scores: dict) -> str:
    order = ["Relevance", "Depth", "ConceptAccuracy", "NonDisclosure", "Clarity"]
    left_w = max(len(k) for k in order)
    lines = []
    lines.append(f"{'Criterion'.ljust(left_w)} | Score")
    lines.append(f"{'-'*left_w}-|------")
    for k in order:
        v = scores.get(k, "")
        lines.append(f"{k.ljust(left_w)} | {v}")
    return "\n".join(lines)

def _fmt_header(title: str, ch="="):
    return f"{title}\n{ch * len(title)}"

def _wrap(txt: str, width=100, indent=""):
    return textwrap.fill(txt or "", width=width, initial_indent=indent, subsequent_indent=indent)

def _as_text(val) -> str:
    # Normalize rationale/fields that may be str | list | dict | None
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return json.dumps(val, ensure_ascii=False)

# ---------- Robust JSON extraction (local to main.py only) ----------

def _strip_code_fences(s: str) -> str:
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()

def _find_balanced_object(s: str) -> str:
    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s[start:i+1]
    return ""

def extract_json_local(s: str) -> dict:
    if not s or not s.strip():
        raise ValueError("Empty model output.")
    s1 = _strip_code_fences(s)
    try:
        obj = json.loads(s1)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    block = _find_balanced_object(s1)
    if block:
        return json.loads(block)
    raise ValueError("No JSON found in model output.")

# ---------- Builders that mirror your existing prompt pattern ----------

FORCE_JSON_NOTE = "\n\nReturn ONLY a single JSON object. No extra text."

def build_planner_prompt(task: str, code: str, tmpl: str) -> str:
    return f"{tmpl}\n\nTASK:\n{task}\n\nSTUDENT_CODE:\n{code}\n\nJSON:"

def build_executor_prompt(task: str, code: str, planner: dict, tmpl: str) -> str:
    return f"""{tmpl}

TASK:
{task}

STUDENT_CODE:
{code}

PLANNER_JSON:
{json.dumps(planner, ensure_ascii=False, indent=2)}

JSON:"""

def build_critic_prompt(task: str, code: str, planner: dict, executor: dict, rubric: dict, tmpl: str) -> str:
    return f"""{tmpl}

TASK:
{task}

STUDENT_CODE:
{code}

PLANNER_JSON:
{json.dumps(planner, ensure_ascii=False)}

EXECUTOR_JSON:
{json.dumps(executor, ensure_ascii=False)}

RUBRIC:
{json.dumps(rubric, ensure_ascii=False)}

JSON:"""

# ---------- Safe call wrapper (single retry if JSON fails) ----------

def ask_json(prompt: str, temp_first=0.2, temp_second=0.0, max_new=400) -> dict:
    out = chat(prompt, temperature=temp_first, max_new_tokens=max_new)
    try:
        return extract_json_local(out)
    except Exception:
        out2 = chat(prompt + FORCE_JSON_NOTE, temperature=temp_second, max_new_tokens=max_new)
        return extract_json_local(out2)

# ---------- Run one sample (no edits to codes/* or prompt files needed) ----------

def run_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    task, code = sample["task"], sample["code"]
    planner_tmpl = read_text("prompts", "planner.txt")
    executor_tmpl = read_text("prompts", "executor.txt")
    critic_tmpl = read_text("prompts", "critic.txt")
    rubric = json.loads(read_text("rubrics", "pec_rubric.json"))

    # Planner
    p_prompt = build_planner_prompt(task, code, planner_tmpl)
    planner = ask_json(p_prompt, temp_first=0.1, temp_second=0.0, max_new=400)

    # Executor
    e_prompt = build_executor_prompt(task, code, planner, executor_tmpl)
    executor = ask_json(e_prompt, temp_first=0.3, temp_second=0.1, max_new=400)

    # Critic
    c_prompt = build_critic_prompt(task, code, planner, executor, rubric, critic_tmpl)
    critic = ask_json(c_prompt, temp_first=0.0, temp_second=0.0, max_new=400)

    # Weighted overall (same weights as before)
    scores = critic.get("scores", {})
    w = {"Relevance":1, "Depth":1, "ConceptAccuracy":1.5, "NonDisclosure":1.5, "Clarity":1}
    overall = round(sum(scores.get(k, 0)*w[k] for k in w) / sum(w.values()), 3)

    return {
        "id": sample["id"],
        "task": task,
        "code": code,
        "planner": planner,
        "executor": executor,
        "critic": critic,
        "overall_score": overall
    }

# ---------- Pretty console + artifact writers ----------

def print_pretty(result: dict):
    rid = result["id"]
    title = f"{rid} — Overall Score: {result['overall_score']}"
    print("\n" + _fmt_header(title, "="))

    # Task
    print(_fmt_header("Task", "-"))
    print(_wrap(result["task"], 100))

    # Planner summary (not raw JSON)
    p = result["planner"]
    print(_fmt_header("Planner Summary", "-"))
    print("Concepts:")
    print(_fmt_bullet_list(p.get("concepts", [])))
    print(f"\nBloom Level: {p.get('bloom','N/A')}\n")
    print("Likely Misconceptions:")
    print(_fmt_bullet_list(p.get("misconceptions", [])))

    # Executor questions (numbered list)
    ex = result["executor"]
    qs = ex.get("questions", [])
    print("\n" + _fmt_header("Socratic Questions", "-"))
    if not qs:
        print("  (No questions generated)")
    else:
        for i, q in enumerate(qs, 1):
            print(f"  {i}. {q}")
    if ex.get("rationale") is not None:
        print("\nRationale:")
        print(_wrap(_as_text(ex.get("rationale")), 100, indent="  "))

    # Critic rubric table + justification
    cr = result["critic"]
    sc = cr.get("scores", {})
    print("\n" + _fmt_header("Rubric Scores", "-"))
    print(_fmt_rubric_table(sc))
    if cr.get("justification"):
        print("\nJustification:")
        print(_wrap(cr["justification"], 100, indent="  "))

def write_artifacts(results: List[dict]):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    # JSONL (full payloads)
    with open(out_dir / f"pec_results_{ts}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV (flat: overall + rubric + questions joined)
    csv_path = out_dir / f"pec_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "overall_score", "Relevance", "Depth", "ConceptAccuracy", "NonDisclosure", "Clarity", "questions", "justification"])
        for r in results:
            s = r["critic"].get("scores", {})
            qs = r["executor"].get("questions", [])
            writer.writerow([
                r["id"],
                r["overall_score"],
                s.get("Relevance"),
                s.get("Depth"),
                s.get("ConceptAccuracy"),
                s.get("NonDisclosure"),
                s.get("Clarity"),
                " | ".join(qs),
                r["critic"].get("justification", ""),
            ])

    # Markdown (clean, non-JSON report)
    md_path = out_dir / f"pec_results_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PEC Results\n\n")
        for r in results:
            f.write(f"## {r['id']} — Overall: **{r['overall_score']}**\n\n")
            # Task
            f.write("**Task**\n\n")
            f.write(textwrap.fill(r["task"], width=100) + "\n\n")
            # Planner
            p = r["planner"]
            f.write("**Planner Summary**\n\n")
            f.write("- Concepts:\n")
            if p.get("concepts"):
                for c in p["concepts"]:
                    f.write(f"  - {c}\n")
            else:
                f.write("  - None\n")
            f.write(f"\n- Bloom Level: {p.get('bloom','N/A')}\n")
            f.write("- Likely Misconceptions:\n")
            if p.get("misconceptions"):
                for m in p["misconceptions"]:
                    f.write(f"  - {m}\n")
            else:
                f.write("  - None\n")
            f.write("\n")
            # Executor
            ex = r["executor"]
            f.write("**Socratic Questions**\n\n")
            if ex.get("questions"):
                for i, q in enumerate(ex["questions"], 1):
                    f.write(f"{i}. {q}\n")
            else:
                f.write("(No questions generated)\n")
            if ex.get("rationale") is not None:
                f.write(f"\n*Rationale:* {textwrap.fill(_as_text(ex['rationale']), width=100)}\n")
            f.write("\n")
            # Critic
            s = r["critic"].get("scores", {})
            f.write("**Rubric Scores**\n\n")
            f.write("| Criterion | Score |\n|---|---|\n")
            for k in ["Relevance","Depth","ConceptAccuracy","NonDisclosure","Clarity"]:
                f.write(f"| {k} | {s.get(k,'')} |\n")
            if r["critic"].get("justification"):
                f.write(f"\n**Justification**\n\n{textwrap.fill(r['critic']['justification'], width=100)}\n")
            f.write("\n---\n\n")

    print(f"\nArtifacts saved to: {out_dir}")

# ---------- Main ----------

if __name__ == "__main__":
    lines = [ln for ln in read_text("data", "student_samples.jsonl").splitlines() if ln.strip()]
    samples = [json.loads(ln) for ln in lines]

    results = [run_sample(s) for s in samples]

    for r in results:
        print_pretty(r)

    write_artifacts(results)
