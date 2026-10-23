"""Loading the competition tables and choosing one MRI series per view."""

from pathlib import Path

import pandas as pd

from . import config as C


def load_train(data_root=None):
    """train.csv: one row per study with the report text and 12 label columns.

    Label columns are 1/0 for the small expert-labelled ("gold") subset and
    NaN for every other study.
    """
    root = Path(data_root or C.DATA_ROOT)
    df = pd.read_csv(root / "train.csv")
    df[C.ID_COL] = df[C.ID_COL].astype(str)
    return df


def load_series(split="train", data_root=None):
    root = Path(data_root or C.DATA_ROOT)
    df = pd.read_csv(root / f"{split}_series.csv")
    for col in (C.ID_COL, C.SERIES_COL):
        df[col] = df[col].astype(str)
    df["_plane"] = df[C.PLANE_COL].astype(str).str.strip().str.lower()
    for col in (C.FLUID_COL, C.FATSAT_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def gold_mask(train, labels):
    """Studies where every finding has an expert label."""
    return train[labels].notna().all(axis=1)


def select_views(series):
    """Map StudyInstanceUID -> {view: SeriesInstanceUID or None}.

    For each view we take a series in the right plane, preferring the right
    fluid-sensitivity and then fat suppression. Ties break on the UID so the
    choice is deterministic.
    """
    out = {}
    for study, grp in series.groupby(C.ID_COL, sort=False):
        picks = {}
        for view, spec in C.VIEWS.items():
            cand = grp[grp["_plane"] == spec["plane"]].copy()
            if cand.empty:
                picks[view] = None
                continue
            cand["_fluid_match"] = (cand[C.FLUID_COL] == spec["fluid"]).astype(int)
            cand = cand.sort_values(
                ["_fluid_match", C.FATSAT_COL, C.SERIES_COL],
                ascending=[False, False, True],
            )
            picks[view] = cand.iloc[0][C.SERIES_COL]
        out[study] = picks
    return out
