"""Write submission.csv for the test studies (run inside a Kaggle notebook).

Images are read straight from the DICOMs, so no cache is needed. If test.csv
carries report text, the image probabilities are blended with the rule-based
report scores.

    python -m knee.predict --checkpoints /kaggle/input/my-knee-models/*.pt
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import load_series, select_views
from .model import KneeNet
from .preprocess import study_volume
from .report_rules import score_reports


def main(checkpoints, out, text_weight, data_root=None):
    root = Path(data_root or C.DATA_ROOT)
    labels = C.load_label_names(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(root / "test.csv")
    test[C.ID_COL] = test[C.ID_COL].astype(str)
    views = select_views(load_series("test", root))

    models = []
    for path in checkpoints:
        ck = torch.load(path, map_location=device)
        m = KneeNet(ck["backbone"], n_views=len(C.VIEWS), n_labels=len(labels), pretrained=False)
        m.load_state_dict(ck["model"])
        models.append((m.to(device).eval(), ck["slices"], ck["crop"]))
        n_slices_cache, size = ck.get("cache_shape", (24, 256))  # all checkpoints share one cache format
    if not models:
        raise SystemExit("no checkpoints found")

    preds = np.full((len(test), len(labels)), 0.5, dtype=np.float32)
    for i, sid in enumerate(test[C.ID_COL]):
        try:
            vol, present = study_volume(root / "test_series" / sid, views.get(sid, {}), n_slices_cache, size)
        except Exception as e:
            print(f"{sid}: could not read images ({e}); predicting 0.5")
            continue
        if not present.any():
            continue
        outs = []
        for m, s, crop in models:
            idx = np.linspace(0, n_slices_cache - 1, s).round().astype(int)
            o = (size - crop) // 2
            x = torch.from_numpy(vol[:, idx, o:o + crop, o:o + crop].astype(np.float32) / 255.0)[None]
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                outs.append(torch.sigmoid(m(x.to(device), torch.from_numpy(present)[None].to(device))).float().cpu().numpy()[0])
        preds[i] = np.mean(outs, axis=0)

    if C.REPORT_COL in test.columns and test[C.REPORT_COL].notna().any() and text_weight > 0:
        text = score_reports(test, labels)[labels].values
        preds = (1 - text_weight) * preds + text_weight * text
        print(f"Blended with report scores (weight {text_weight})")

    sub = pd.DataFrame(np.clip(preds, 0, 1), columns=labels)
    sub.insert(0, C.ID_COL, test[C.ID_COL].values)
    sub.to_csv(out, index=False)
    print(f"Wrote {out} ({len(sub)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", default=[str(C.WORK_DIR / "models" / "best.pt")])
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--text-weight", type=float, default=0.5)
    a = p.parse_args()
    ckpts = [f for pat in a.checkpoints for f in glob.glob(pat)]
    main(ckpts, a.out, a.text_weight)
