"""Train the image model on report-derived soft labels.

Validation is done two ways:
  * a held-out 15% of the report-labelled studies (used to pick the best epoch)
  * the expert-labelled gold studies (never used for training or selection,
    so their score is an honest estimate)

    python -m knee.train_image --epochs 10 --batch-size 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .metrics import macro_auc, per_label_auc
from .model import KneeNet, soft_bce


class CacheDataset(Dataset):
    def __init__(self, ids, targets, cache_dir, n_slices, crop, train):
        self.ids, self.targets = list(ids), targets.astype(np.float32)
        self.cache_dir, self.n_slices, self.crop, self.train = Path(cache_dir), n_slices, crop, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = np.load(self.cache_dir / f"{self.ids[i]}.npz")
        vol, present = d["vol"], d["present"]  # (V, S, H, W) uint8, (V,)
        V, S, H, W = vol.shape

        if self.train:  # random slice subset (kept in order) and random crop
            idx = np.sort(np.random.choice(S, self.n_slices, replace=S < self.n_slices))
            y0, x0 = np.random.randint(0, H - self.crop + 1), np.random.randint(0, W - self.crop + 1)
        else:
            idx = np.linspace(0, S - 1, self.n_slices).round().astype(int)
            y0, x0 = (H - self.crop) // 2, (W - self.crop) // 2
        vol = vol[:, idx, y0:y0 + self.crop, x0:x0 + self.crop].astype(np.float32) / 255.0

        if self.train:  # per-view contrast / brightness jitter
            gain = np.random.uniform(0.8, 1.2, (V, 1, 1, 1))
            bias = np.random.uniform(-0.1, 0.1, (V, 1, 1, 1))
            vol = np.clip(vol * gain + bias, 0, 1).astype(np.float32)
        return torch.from_numpy(vol), torch.from_numpy(present), torch.from_numpy(self.targets[i])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for vol, present, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(vol.to(device), present.to(device))).float().cpu())
    return torch.cat(out).numpy()


def main(a):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = C.load_label_names(a.data_root)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(a.targets)
    t[C.ID_COL] = t[C.ID_COL].astype(str)
    cached = {p.stem for p in Path(a.cache).glob("*.npz")}
    t = t[t[C.ID_COL].isin(cached)].reset_index(drop=True)
    gold = t[t["is_gold"]]
    soft = t[~t["is_gold"]].sample(frac=1, random_state=C.SEED)
    n_val = int(len(soft) * 0.15)
    val, tr = soft.iloc[:n_val], soft.iloc[n_val:]
    print(f"train {len(tr)} | val {len(val)} | gold {len(gold)} studies (device: {device})")

    def loader(df, train):
        ds = CacheDataset(df[C.ID_COL], df[labels].values, a.cache, a.slices, a.crop, train)
        return DataLoader(ds, batch_size=a.batch_size if train else a.batch_size * 2, shuffle=train,
                          num_workers=a.workers, pin_memory=True, drop_last=train)

    dl_tr, dl_val, dl_gold = loader(tr, True), loader(val, False), loader(gold, False)
    # (slices, size) of the cache, so predict.py can preprocess test images the same way
    cache_shape = list(np.load(Path(a.cache) / f"{t[C.ID_COL].iloc[0]}.npz")["vol"].shape[1:3])

    model = KneeNet(a.backbone, n_views=len(C.VIEWS), n_labels=len(labels), pretrained=not a.no_pretrained)
    if a.grad_checkpointing:
        model.encoder.set_grad_checkpointing(True)
    model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * max(1, len(dl_tr) // a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for step, (vol, present, y) in enumerate(dl_tr, 1):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = soft_bce(model(vol.to(device), present.to(device)), y.to(device)) / a.accum
            scaler.scale(loss).backward()
            if step % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(loss.item() * a.accum)

        val_auc = macro_auc(val[labels].values, predict(model, dl_val, device), labels) if len(val) else float("nan")
        gold_auc = macro_auc(gold[labels].values, predict(model, dl_gold, device), labels) if len(gold) else float("nan")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auc": val_auc, "gold_auc": gold_auc})
        print(f"epoch {epoch:2d} | loss {np.mean(losses):.4f} | val AUC {val_auc:.4f} | "
              f"gold AUC {gold_auc:.4f} | {time.time() - t0:.0f}s", flush=True)
        if np.isnan(val_auc) or val_auc > best:
            best = val_auc
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "labels": labels,
                        "slices": a.slices, "crop": a.crop, "cache_shape": cache_shape}, out_dir / "best.pt")

    # Final, honest numbers from the checkpoint chosen on the soft-label val set.
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    gold_pred = predict(model, dl_gold, device)
    per_label = per_label_auc(gold[labels].values, gold_pred, labels)
    summary = {"best_val_auc": best, "gold_macro_auc": float(np.nanmean(list(per_label.values()))),
               "gold_per_label_auc": per_label, "history": history,
               "n_train": len(tr), "n_val": len(val), "n_gold": len(gold), "backbone": a.backbone}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.DataFrame(gold_pred, columns=labels, index=gold[C.ID_COL]).to_csv(out_dir / "gold_predictions.csv")
    print(json.dumps({k: summary[k] for k in ("best_val_auc", "gold_macro_auc")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default=str(C.WORK_DIR / "labels" / "targets.csv"))
    p.add_argument("--cache", default=str(C.CACHE_DIR))
    p.add_argument("--out", default=str(C.WORK_DIR / "models"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--slices", type=int, default=16, help="slices per view fed to the model")
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad-checkpointing", action="store_true", help="less GPU memory, slower")
    main(p.parse_args())
