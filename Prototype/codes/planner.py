from .llm import chat
from .utils import extract_json

def run_planner(task: str, code: str, tmpl: str) -> dict:
    prompt = f"""{tmpl}

TASK:
{task}

STUDENT_CODE:
{code}

JSON:"""
    out = chat(prompt, temperature=0.1, max_new_tokens=400)
    return extract_json(out)
