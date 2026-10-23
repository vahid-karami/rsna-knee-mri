"""End-to-end run on a tiny fake dataset (CPU, ~1 minute).

Builds fake CSVs and DICOMs shaped like the competition data, then runs
label building, preprocessing, one training epoch and prediction. It checks
that the plumbing works, not that the model is any good.

    python tests/smoke_test.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee import config as C  # noqa: E402

REPORTS = [
    "Complete tear of the anterior cruciate ligament. Large joint effusion. Bone bruise lateral femoral condyle.",
    "Horizontal tear of the medial meniscus. Medial compartment osteoarthritis. Small Baker's cyst.",
    "No effusion. Cruciate and collateral ligaments intact. Menisci intact.",
    "Ruptur des vorderen Kreuzbandes. Gelenkerguss. Kein Hinweis auf Fraktur.",
    "Fractura no desplazada de la meseta tibial lateral. Derrame articular.",
]
PLANES = {"sagittal": ([0, 1, 0, 0, 0, -1], 0), "coronal": ([1, 0, 0, 0, 0, -1], 1), "axial": ([1, 0, 0, 0, 1, 0], 2)}


def write_dicom(path, img, orient, pos, inst):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.ImageOrientationPatient, ds.ImagePositionPatient = orient, pos
    ds.InstanceNumber = inst
    ds.PixelData = img.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def make_split(root, split, n, rng):
    series_rows, studies = [], []
    for i in range(n):
        sid = f"1.2.{split}.{i}"
        studies.append(sid)
        for plane, (orient, axis) in PLANES.items():
            ser = f"{sid}.{axis}"
            series_rows.append({C.ID_COL: sid, C.SERIES_COL: ser, C.FLUID_COL: 1,
                                C.FATSAT_COL: 1, C.PLANE_COL: plane.capitalize()})
            d = root / f"{split}_series" / sid / ser
            d.mkdir(parents=True)
            for k in range(10):
                pos = [0.0, 0.0, 0.0]
                pos[axis] = float(k * 3)
                write_dicom(d / f"{rng.integers(1e9)}.dcm", rng.integers(0, 2000, (64, 64)), orient, pos, k + 1)
    pd.DataFrame(series_rows).to_csv(root / f"{split}_series.csv", index=False)
    return studies


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        data, work = Path(tmp) / "data", Path(tmp) / "work"
        data.mkdir()
        train_ids = make_split(data, "train", 24, rng)
        test_ids = make_split(data, "test", 3, rng)

        train = pd.DataFrame({C.ID_COL: train_ids, C.REPORT_COL: [REPORTS[i % 5] for i in range(24)]})
        for label in C.LABELS:  # first 8 studies are "gold"
            train[label] = [float(rng.integers(0, 2)) if i < 8 else np.nan for i in range(24)]
            train.loc[0, label], train.loc[1, label] = 0.0, 1.0
        train.to_csv(data / "train.csv", index=False)
        pd.DataFrame({C.ID_COL: test_ids, C.REPORT_COL: REPORTS[:3]}).to_csv(data / "test.csv", index=False)
        pd.DataFrame({C.ID_COL: test_ids, **{l: 0.5 for l in C.LABELS}}).to_csv(data / "sample_submission.csv", index=False)

        env = {"KNEE_DATA_ROOT": str(data), "KNEE_WORK_DIR": str(work), "PYTHONPATH": str(ROOT / "src")}
        import os
        env = {**os.environ, **env}

        def run(*args):
            print("\n$ python -m", " ".join(args))
            subprocess.run([sys.executable, "-m", *args], check=True, env=env, cwd=tmp)

        run("knee.build_labels")
        run("knee.preprocess", "--max-studies", "0", "--slices", "8", "--size", "64", "--workers", "2")
        run("knee.train_image", "--epochs", "1", "--backbone", "resnet10t", "--no-pretrained",
            "--batch-size", "2", "--accum", "1", "--slices", "6", "--crop", "56", "--workers", "0")
        run("knee.predict", "--out", str(work / "submission.csv"))
        sub = pd.read_csv(work / "submission.csv")
        assert list(sub.columns) == [C.ID_COL] + C.LABELS and len(sub) == 3
        assert sub[C.LABELS].apply(np.isfinite).all().all()
        print("\nSmoke test passed")


if __name__ == "__main__":
    main()
