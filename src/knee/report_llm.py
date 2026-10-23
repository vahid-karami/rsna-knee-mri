"""Label reports with a small open-weight LLM (runs on a free Colab T4).

The model reads each multilingual report and answers with a JSON object of
12 findings, each 1 (present), 0 (absent / not mentioned) or 0.5 (uncertain).
Answers are appended to a JSONL file, so an interrupted Colab session resumes
where it stopped.

    python -m knee.report_llm --model Qwen/Qwen2.5-3B-Instruct --batch-size 16
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from . import config as C
from .data import load_train

DEFINITIONS = {
    "ACL": "anterior cruciate ligament tear (partial or complete)",
    "MCL": "medial collateral ligament tear or sprain",
    "Medial Meniscus": "tear of the medial meniscus",
    "Lateral Meniscus": "tear of the lateral meniscus",
    "Medial OA": "osteoarthritis / cartilage degeneration of the medial femorotibial compartment",
    "Lateral OA": "osteoarthritis / cartilage degeneration of the lateral femorotibial compartment",
    "PF OA": "osteoarthritis / cartilage degeneration of the patellofemoral compartment",
    "Effusion": "joint effusion (more than physiological fluid)",
    "Synovitis": "synovitis",
    "Baker's": "Baker's (popliteal) cyst",
    "Contusion": "bone contusion / bone bruise / traumatic bone marrow edema",
    "Fracture": "fracture of any bone in the knee",
}

SYSTEM = (
    "You are a musculoskeletal radiologist. You read knee MRI reports written in any "
    "language and extract findings. Reply with JSON only."
)


def build_prompt(report, labels):
    lines = "\n".join(f'- "{l}": {DEFINITIONS.get(l, l)}' for l in labels)
    return (
        "Knee MRI report:\n\"\"\"\n" + report[:4000] + "\n\"\"\"\n\n"
        "For each finding below answer 1 if the report says it is present, "
        "0 if it is absent or not mentioned, 0.5 if the report is uncertain "
        "(e.g. 'possible', 'cannot be excluded').\n" + lines + "\n\n"
        "Return one JSON object with exactly these keys."
    )


def parse_answer(text, labels):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    lower = {str(k).strip().lower(): v for k, v in obj.items()}
    out = {}
    for label in labels:
        v = lower.get(label.lower())
        try:
            out[label] = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            out[label] = None
    return out


def run(model_name, out_path, batch_size=16, max_new_tokens=200, limit=None, data_root=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    labels = C.load_label_names(data_root)
    train = load_train(data_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)[C.ID_COL] for line in f if line.strip()}
    todo = train[~train[C.ID_COL].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(done)} reports already labelled, {len(todo)} to go")
    if todo.empty:
        return

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()

    rows = todo[[C.ID_COL, C.REPORT_COL]].fillna("").values.tolist()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_prompt(r, labels)}],
                    tokenize=False, add_generation_prompt=True,
                )
                for _, r in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (sid, _), text in zip(chunk, texts):
                f.write(json.dumps({C.ID_COL: sid, "answer": text,
                                    "labels": parse_answer(text, labels)}) + "\n")
            f.flush()
            print(f"  {min(i + batch_size, len(rows))}/{len(rows)}", end="\r")
    print()


def load_llm_scores(path, labels):
    """JSONL -> DataFrame of scores (NaN where the model gave no usable answer)."""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            row = {C.ID_COL: str(r[C.ID_COL])}
            row.update({l: (r["labels"] or {}).get(l) for l in labels})
            recs.append(row)
    df = pd.DataFrame(recs).drop_duplicates(C.ID_COL, keep="last")
    df[labels] = df[labels].astype(float)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--out", default=str(C.WORK_DIR / "labels" / "llm_answers.jsonl"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="label only N reports (for a quick test)")
    a = p.parse_args()
    run(a.model, a.out, a.batch_size, limit=a.limit)
