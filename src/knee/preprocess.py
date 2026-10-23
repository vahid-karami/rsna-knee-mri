"""DICOM series -> small, fixed-size uint8 volumes (one .npz per study).

The raw training images are ~250 GB, far more than a Colab disk holds. Run
this once where the data lives (a Kaggle notebook has it mounted read-only),
then download the compact cache to Colab:

    python -m knee.preprocess --max-studies 1500 --slices 24 --size 256

Each .npz holds `vol` (views, slices, size, size) uint8 and `present` (views,).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .data import gold_mask, load_series, load_train, select_views


def read_series(folder):
    """Read a DICOM series into a (slices, H, W) float32 array, ordered in space."""
    import pydicom

    items = []
    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
            img = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if img.ndim == 3:  # multi-frame file
            frames = list(img)
        else:
            frames = [img]
        img_slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        img_icept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        invert = str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
        pos = _slice_position(ds)
        for k, f in enumerate(frames):
            f = f * img_slope + img_icept
            if invert:
                f = f.max() - f
            items.append((pos if pos is not None else float(getattr(ds, "InstanceNumber", 0) or 0), k, f))
    if not items:
        return None
    items.sort(key=lambda t: (t[0], t[1]))
    shape = items[len(items) // 2][2].shape  # drop the odd localiser of a different size
    return np.stack([f for _, _, f in items if f.shape == shape])


def _slice_position(ds):
    """Position of the slice along its normal, from the DICOM geometry tags."""
    try:
        o = np.array(ds.ImageOrientationPatient, dtype=float)
        p = np.array(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(np.cross(o[:3], o[3:]), p))
    except Exception:
        return None


def normalise(vol, n_slices, size):
    """Window to the 1st-99th percentile, resample slices, resize to size x size."""
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    idx = np.linspace(0, len(vol) - 1, n_slices).round().astype(int)
    out = np.stack([cv2.resize(vol[i], (size, size), interpolation=cv2.INTER_AREA) for i in idx])
    return (out * 255).astype(np.uint8)


def study_volume(study_dir, views, n_slices, size):
    """All views of one study -> (vol, present)."""
    vol = np.zeros((len(C.VIEWS), n_slices, size, size), np.uint8)
    present = np.zeros(len(C.VIEWS), bool)
    for v, view in enumerate(C.VIEWS):
        sid = views.get(view)
        if sid is None:
            continue
        arr = read_series(Path(study_dir) / sid)
        if arr is None or len(arr) < 3:
            continue
        vol[v] = normalise(arr, n_slices, size)
        present[v] = True
    return vol, present


def _job(args):
    study, study_dir, views, out, n_slices, size = args
    if out.exists():
        return study, True
    try:
        vol, present = study_volume(study_dir, views, n_slices, size)
        np.savez_compressed(out, vol=vol, present=present)
        return study, bool(present.any())
    except Exception as e:  # keep going; a bad study must not stop a 2-hour job
        print(f"failed {study}: {e}")
        return study, False


def main(max_studies, n_slices, size, out_dir, workers, split="train", data_root=None):
    root = Path(data_root or C.DATA_ROOT)
    series = load_series(split, root)
    views = select_views(series)
    studies = list(views)

    if split == "train" and max_studies:
        labels = C.load_label_names(root)
        train = load_train(root)
        gold = set(train.loc[gold_mask(train, labels), C.ID_COL])
        rng = np.random.default_rng(C.SEED)
        rest = [s for s in studies if s not in gold]
        rng.shuffle(rest)
        # Always keep the expert-labelled studies: they are the validation set.
        studies = [s for s in studies if s in gold] + rest[: max(0, max_studies - len(gold))]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, root / f"{split}_series" / s, views[s], out_dir / f"{s}.npz", n_slices, size)
            for s in studies]
    print(f"Preprocessing {len(jobs)} studies with {workers} workers -> {out_dir}")
    ok = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (_, good) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
            ok += good
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {ok} with images", flush=True)
    pd.DataFrame({C.ID_COL: studies}).to_csv(out_dir / "studies.csv", index=False)


if __name__ == "__main__":
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--max-studies", type=int, default=1500, help="0 = all studies")
    p.add_argument("--slices", type=int, default=24)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out", default=str(C.CACHE_DIR))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = p.parse_args()
    main(a.max_studies, a.slices, a.size, a.out, a.workers)
