import numpy as np
from sklearn.metrics import roc_auc_score


def per_label_auc(y_true, y_pred, labels):
    """ROC-AUC per label; NaN where a label has only one class in y_true."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for j, label in enumerate(labels):
        t, p = y_true[:, j], y_pred[:, j]
        ok = ~np.isnan(t) & ~np.isnan(p)
        t, p = (t[ok] >= 0.5).astype(int), p[ok]
        out[label] = roc_auc_score(t, p) if 0 < t.sum() < len(t) else np.nan
    return out


def macro_auc(y_true, y_pred, labels):
    """The competition metric: mean ROC-AUC over the 12 findings."""
    return float(np.nanmean(list(per_label_auc(y_true, y_pred, labels).values())))
