"""Turn reports into training targets and check them against the gold labels.

Only a few dozen studies have expert labels. Every other study gets a soft
label from its report (rules, LLM, or their average); whichever source agrees
best with the experts is written to targets.csv. The gold studies keep their
expert labels and are held out as the validation set for the image model.

    python -m knee.build_labels
"""

import argparse
from pathlib import Path

import pandas as pd

from . import config as C
from .data import gold_mask, load_train
from .metrics import macro_auc, per_label_auc
from .report_llm import load_llm_scores
from .report_rules import score_reports


def main(llm_path, out_dir, data_root=None):
    labels = C.load_label_names(data_root)
    train = load_train(data_root)
    gold = gold_mask(train, labels)
    print(f"{len(train)} studies, {gold.sum()} with expert labels")

    sources = {"rules": score_reports(train, labels).set_index(C.ID_COL)}
    if llm_path and Path(llm_path).exists():
        llm = load_llm_scores(llm_path, labels).set_index(C.ID_COL).reindex(sources["rules"].index)
        print(f"LLM answers for {llm.notna().all(axis=1).sum()} studies")
        sources["llm"] = llm.fillna(sources["rules"])  # rules fill unparsable answers
        sources["mean"] = (sources["rules"] + sources["llm"]) / 2

    y_gold = train.loc[gold].set_index(C.ID_COL)[labels]
    report = {}
    for name, scores in sources.items():
        s = scores.loc[y_gold.index]
        report[name] = per_label_auc(y_gold.values, s.values, labels)
        report[name]["MACRO"] = macro_auc(y_gold.values, s.values, labels)
    report = pd.DataFrame(report)
    print("\nAgreement with expert labels (ROC-AUC on gold studies):")
    print(report.round(3).to_string())

    best = report.loc["MACRO"].idxmax()
    print(f"\nUsing '{best}' labels for training")
    targets = sources[best].copy()
    targets.loc[y_gold.index, labels] = y_gold.values
    targets["is_gold"] = targets.index.isin(y_gold.index)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets.reset_index().to_csv(out_dir / "targets.csv", index=False)
    report.to_csv(out_dir / "label_agreement.csv")
    print(f"Wrote {out_dir / 'targets.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default=str(C.WORK_DIR / "labels" / "llm_answers.jsonl"))
    p.add_argument("--out", default=str(C.WORK_DIR / "labels"))
    a = p.parse_args()
    main(a.llm, a.out)
