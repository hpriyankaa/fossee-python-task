from .llm import chat
from .utils import extract_json
import json

def run_critic(task: str, code: str, planner: dict, executor: dict, rubric: dict, tmpl: str) -> dict:
    prompt = f"""{tmpl}

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
    out = chat(prompt, temperature=0.0, max_new_tokens=400)
    return extract_json(out)
