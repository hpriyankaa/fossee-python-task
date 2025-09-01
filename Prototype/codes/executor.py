from .llm import chat
from .utils import extract_json
import json

def run_executor(task: str, code: str, planner: dict, tmpl: str) -> dict:
    prompt = f"""{tmpl}

TASK:
{task}

STUDENT_CODE:
{code}

PLANNER_JSON:
{json.dumps(planner, ensure_ascii=False, indent=2)}

JSON:"""
    out = chat(prompt, temperature=0.3, max_new_tokens=400)
    return extract_json(out)
