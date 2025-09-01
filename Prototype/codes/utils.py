import json, re

def extract_json(s: str):
 
    m = re.search(r'\{.*\}', s, re.S)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def aggregate_scores(scores: dict) -> float:
    w = {"Relevance":1, "Depth":1, "ConceptAccuracy":1.5, "NonDisclosure":1.5, "Clarity":1}
    num = sum(scores[k]*w[k] for k in w)
    den = sum(w.values())
    return round(num/den, 3)
