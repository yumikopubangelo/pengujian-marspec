"""
Baseline AMC architectures (IQ-only) — exact definitions matching the saved
weights in model/{CNN1D,ResNet1D,CLDNN,MCLDNN,MCformer}_weights.pt

Extracted from amc_benchmark_colab.ipynb (the script that trained these weights).
All take iq[B,2,1024] -> logits[B,24]. No context input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class CNN1D(nn.Module):
    """O'Shea et al. 2016 — Convolutional Radio Modulation Recognition."""

    def __init__(self, n_classes=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class ResNet1D(nn.Module):
    """O'Shea et al. 2017 — Over-the-Air Deep Learning Radio Signal Classification."""

    def __init__(self, n_classes=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(64), ResBlock(64), ResBlock(64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class CLDNN(nn.Module):
    """West & O'Shea 2017 — Deep Architectures for Modulation Recognition."""

    def __init__(self, n_classes=24, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes))

    def forward(self, x):
        c = self.conv(x).permute(0, 2, 1)
        _, (h, _) = self.lstm(c)
        return self.fc(h[-1])


class MCLDNN(nn.Module):
    """Multi-Channel CLDNN — parallel I, Q, IQ branches."""

    def __init__(self, n_classes=24, hidden=128):
        super().__init__()
        self.bi = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU())
        self.bq = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU())
        self.biq = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(96, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes))

    def forward(self, x):
        cat = torch.cat([self.bi(x[:, :1, :]), self.bq(x[:, 1:, :]), self.biq(x)], dim=1)
        _, (h, _) = self.lstm(cat.permute(0, 2, 1))
        return self.fc(h[-1])


class MCformer(nn.Module):
    """Hanna et al. 2022 — MCformer: Transformer-based AMC with SELU stem."""

    def __init__(self, n_classes=24, d_model=64, nhead=4, n_layers=2):
        super().__init__()
        seq_len = 1024 // 16
        self.stem = nn.Sequential(nn.Conv1d(2, d_model, 16, stride=16), nn.SELU())
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True, activation="gelu")
        self.tfm = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, n_classes))

    def forward(self, x):
        h = self.stem(x).permute(0, 2, 1)
        h = h + self.pos[:, :h.size(1), :]
        return self.fc(self.tfm(h).mean(dim=1))


BASELINE_REGISTRY = {
    "CNN1D": (CNN1D, "CNN1D_weights.pt"),
    "ResNet1D": (ResNet1D, "ResNet1D_weights.pt"),
    "CLDNN": (CLDNN, "CLDNN_weights.pt"),
    "MCLDNN": (MCLDNN, "MCLDNN_weights.pt"),
    "MCformer": (MCformer, "MCformer_weights.pt"),
}


# ---------------------------------------------------------------------------
# G0 (5 Okt 2026): baseline DILATIH dengan protokol MARSPEC yang sama persis
# lewat train_maritime_v3.py --arch mcformer|resnet1d. Bobot lama di atas
# (load_baseline) dilatih BERSIH saja -> perbandingannya mengukur protokol,
# bukan arsitektur. Lihat docs/g0_papan_skor_adil.md.
# ---------------------------------------------------------------------------
ARCH_BASELINE = {"mcformer": "MCformer", "resnet1d": "ResNet1D"}

# Kunci ukuran yang boleh diatur lewat --cfg, dengan nilai BAWAAN kelasnya.
# Disemai ke cfg supaya (a) `--cfg d_model=184` diterima pemeriksa kunci di
# train_maritime_v3.py, dan (b) config.json mencatat ukuran yang dipakai.
# Checkpoint lama (dilatih sebelum kunci ini ada, cfg-nya tanpa d_model dst.)
# tetap termuat: eval menyemai bawaan yang sama -> arsitektur identik.
# ResNet1D tak punya pengatur ukuran.
CFG_BASELINE_BAWAAN = {
    "mcformer": {"d_model": 64, "nhead": 4, "n_layers": 2},
    "resnet1d": {},
}


class AdaptorLatih(nn.Module):
    """Baseline -> antarmuka MaritimeAMCv3: forward(iq, ctx) -> (logits, iq_eq, gate).

    - konteks DIABAIKAN: baseline memang tidak sadar-kanal (ablasi 1 Okt juga
      menunjukkan konteks MARSPEC tak membawa informasi, jadi ini adil);
    - iq_eq = iq apa adanya: rugi equalizer jadi konstanta TANPA gradien,
      cuma pengukur kerusakan kanal (sama seperti dpdrsn tanpa DCCE);
    - gate = DUA kolom 0,5: std tak-bias atas satu elemen itu NaN, dua kolom
      sama -> moe_balance_loss tepat 0.
    Akibatnya lam_moe/lam_eq boleh SAMA dengan run MARSPEC (pagar setelan
    notebook tetap lolos) tanpa mengubah gradien baseline sedikit pun."""

    def __init__(self, dasar: nn.Module):
        super().__init__()
        self.dasar = dasar

    def forward(self, iq, context=None):
        return self.dasar(iq), iq, iq.new_full((iq.shape[0], 2), 0.5)


def bikin_baseline_latih(arch: str, n_classes: int = 24, cfg: dict | None = None) -> nn.Module:
    """'mcformer' | 'resnet1d' -> AdaptorLatih(model dengan bobot ACAK).

    `cfg` boleh memuat kunci ukuran (lihat CFG_BASELINE_BAWAAN); kunci lain
    (baseline, num_classes) diabaikan di sini."""
    cls, _ = BASELINE_REGISTRY[ARCH_BASELINE[arch]]
    kw = {k: (cfg or {}).get(k, v) for k, v in CFG_BASELINE_BAWAAN[arch].items()}
    return AdaptorLatih(cls(n_classes=n_classes, **kw))


def load_baseline(name: str, model_dir: Path, device: str = "cpu") -> nn.Module:
    """Load a baseline with its saved weights (strict — raises on mismatch)."""
    cls, weight_file = BASELINE_REGISTRY[name]
    model = cls(n_classes=24).to(device)
    sd = torch.load(Path(model_dir) / weight_file, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
