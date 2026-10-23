"""Paths, label names and shared constants.

Every path can be overridden with an environment variable so the same code
runs on Google Colab, in a Kaggle notebook, or on a local machine.
"""

import os
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
REPORT_COL = "Report"
PLANE_COL = "Anatomical_Plane"
FLUID_COL = "Fluid_Sensitive"
FATSAT_COL = "Fat_Suppression"

# The 12 findings scored by the competition, in submission order.
# load_label_names() prefers the order found in sample_submission.csv.
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Image views fed to the model: one series per slot.
VIEWS = {
    "sag_fluid": {"plane": "sagittal", "fluid": 1},
    "cor_fluid": {"plane": "coronal", "fluid": 1},
    "ax_fluid": {"plane": "axial", "fluid": 1},
}


def _first_existing(*candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(candidates[-1])


# Raw competition data (CSVs + train_series/ DICOM folders).
DATA_ROOT = _first_existing(
    os.environ.get("KNEE_DATA_ROOT"),
    "/kaggle/input/competitions/" + COMPETITION,
    "/kaggle/input/" + COMPETITION,
    "/content/data",
    "data",
)

# Everything this project writes: labels, caches, checkpoints, predictions.
WORK_DIR = Path(os.environ.get("KNEE_WORK_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "work"))

# Preprocessed image volumes (one .npz per study), see knee.preprocess.
CACHE_DIR = Path(os.environ.get("KNEE_CACHE_DIR", WORK_DIR / "cache"))

SEED = 42


def load_label_names(data_root=None):
    """Return label columns in the order the scorer expects."""
    import pandas as pd

    sample = Path(data_root or DATA_ROOT) / "sample_submission.csv"
    if sample.exists():
        cols = [c for c in pd.read_csv(sample, nrows=1).columns if c != ID_COL]
        if len(cols) == 12:
            return cols
    return list(LABELS)
