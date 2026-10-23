"""2.5D multi-view CNN for knee MRI.

Each slice is stacked with its two neighbours into a 3-channel image and
encoded by an ImageNet-pretrained 2D backbone (shared across slices and
views). Attention pooling turns the slice features of a view into one vector;
the vectors of the sagittal, coronal and axial views are concatenated and a
linear head predicts the 12 findings. Missing views are masked out.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x):  # x: (B, S, D)
        w = self.score(x).softmax(dim=1)
        return (w * x).sum(dim=1), w.squeeze(-1)


class KneeNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", n_views=3, n_labels=12, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        dim = self.encoder.num_features
        self.view_embed = nn.Parameter(torch.zeros(n_views, dim))
        self.pool = AttentionPool(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim * n_views, n_labels))

    @staticmethod
    def to_triplets(vol):
        """(B, V, S, H, W) -> (B, V, S, 3, H, W): each slice with its neighbours."""
        prev = torch.cat([vol[:, :, :1], vol[:, :, :-1]], dim=2)
        nxt = torch.cat([vol[:, :, 1:], vol[:, :, -1:]], dim=2)
        return torch.stack([prev, vol, nxt], dim=3)

    def forward(self, vol, present):
        """vol: (B, V, S, H, W) float in [0, 1]; present: (B, V) bool."""
        B, V, S, H, W = vol.shape
        x = self.to_triplets(vol)
        x = (x - 0.45) / 0.25
        feats = self.encoder(x.reshape(B * V * S, 3, H, W)).reshape(B, V, S, -1)
        feats = feats + self.view_embed[None, :, None, :]
        pooled, _ = self.pool(feats.reshape(B * V, S, -1))
        pooled = pooled.reshape(B, V, -1) * present[..., None].float()
        return self.head(pooled.reshape(B, -1))


def soft_bce(logits, targets):
    """BCE that accepts soft (0-1) targets from the report labeller."""
    return F.binary_cross_entropy_with_logits(logits, targets)
