#!/usr/bin/env python3
"""
MaritimeAMC v3 — Training with Curriculum + Physics-Grounded Augmentation
=========================================================================

Trains MaritimeAMCv3 on RadioML 2018.01A with maritime channel augmentation
sampled from the OceanStateGenerator pools (easy/normal/hard), using three losses:

  1. Cross-entropy        — the AMC classification objective.
  2. MoE load-balance     — keeps all experts in use (soft-gating CV^2 penalty).
  3. Equalizer aux (MSE)  — supervises the equalizer toward the CLEAN IQ.

Curriculum: difficulty rises with epoch (easy -> normal -> hard pool).

IMPORTANT (lesson learned 2026-06-15): the ocean pools must use REALISTIC
receivable distances. A passive VHF receiver only hears vessels within the
radio/duct horizon (~150 km, OceanStateGenerator max_range_km=150). If pools
contain far vessels (300 km+), the channel SNR penalty turns most samples into
pure noise and accuracy collapses. The channel SNR loss is also capped at 8 dB
so 2-ray multipath (Paper 1's dominant mechanism) is the degradation, not a
path-loss cliff. Also: the train/val split shuffles positions because RadioML
is stored ordered by class (a contiguous slice would give disjoint classes).

Two data paths:
  --lazy  (default for full runs): reads each IQ sample from HDF5 on demand.
          Memory-safe for the full 2.56M-sample dataset (~21 GB would NOT fit
          in Colab RAM if loaded at once).
  in-RAM  (--smoke or small --max_samples): loads everything for speed.

Checkpointing: saves model+optim+scheduler+epoch every epoch and supports
--resume, so multi-hour training survives Colab disconnects.

Examples
--------
Fast local sanity check (tiny model + tiny data, CPU):
    python marspec_poc/train_maritime_v3.py --smoke

FULL run: 2.56M samples + 24M-param config (Colab GPU, lazy + checkpoints):
    python marspec_poc/train_maritime_v3.py --preset full24m \
        --hdf5 /content/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --pools /content/drive/MyDrive/MARSPEC/ocean_states \
        --out  /content/drive/MyDrive/MARSPEC/outputs/maritime_v3 --resume

  IMPORTANT for Colab: COPY the HDF5 to local /content/ first (NOT mounted Drive).
  Random per-sample reads from mounted Drive are unusably slow.
      !cp "/content/drive/MyDrive/MARSPEC/GOLD_XYZ_OSC.0001_1024.hdf5" /content/
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from maritime_amc_v3_arch import MaritimeAMCv3, CFG_V3, validate_config
from maritime_amc_lite import PRESET as PRESET_UKURAN

_ROOT = Path(__file__).resolve().parents[1]
_BENIGN_CTX = np.array([0.05, 0.0, 0.5, 0.25], np.float32)  # clean-sample context


# ---------------------------------------------------------------------------
# Eksperimen ORACLE grup D (4 Okt 2026, docs/uji_oracle_2sinar.md §8).
# Keduanya MATI secara bawaan; papan ukur lama tidak berubah sedikit pun.
# ---------------------------------------------------------------------------
def pita_sinyal(N: int = 1024) -> np.ndarray:
    """Bin tempat sinyal RadioML berada, urutan np.fft.fft (tak digeser).

    Diukur 4 Okt 2026: PSD rata-rata sinyal BERSIH lintas 24 kelas (1.728
    sinyal, SNR 10-30) >= 10x mediannya -> bin -79..+78, menyambung, tanpa
    lubang. Bin +79 cuma 0,1 dB di bawah ambang, jadi ditulis simetris
    |k| <= 79 (159 bin). Rata-rata lintas kelas -> tak membocorkan kelas."""
    return np.abs(np.fft.fftfreq(N) * N) <= round(79 * N / 1024)


def balik_2sinar(sig, rho, tau, phi, f_geser, pita=None):
    """Oracle zero-forcing: buang geser frekuensi, bagi H2, kenakan lagi.

        H2[k] = (1 + rho e^{j phi} e^{-j 2 pi k tau / N}) / (1 + rho)
        x^    = D * IFFT( FFT(y / D) / H2 ),   D[n] = e^{j 2 pi f_geser n}

    rho < 1 -> |H2| >= (1-rho)/(1+rho) > 0, jadi selalu bisa dibagi; harganya
    derau di bin null diperkuat sampai (1+rho)/(1-rho) (= 39x di rho 0,95).
    `pita`: kalau diberi, H2 hanya dibagi di bin itu; bin lain dibiarkan
    (tak ada sinyal di sana, hanya derau & interferer yang ikut membengkak).
    Rumus & dua cek (bit-identik + sisa tanpa derau): uji_oracle_2sinar.py."""
    N = len(sig)
    n = np.arange(N)
    D = np.exp(1j * 2 * np.pi * f_geser * n)
    Y = np.fft.fft(sig / D)
    H = (1 + rho * np.exp(1j * phi) * np.exp(-2j * np.pi * n * tau / N)) / (1 + rho)
    X = Y / H
    if pita is not None:
        X = np.where(pita, X, Y)
    return np.fft.ifft(X) * D


def ctx_dari_kanal(p) -> np.ndarray:
    """Konteks ORACLE = parameter 2-sinar SEBENARNYA: [tau/8, cos phi, sin phi, rho].

    Skalanya sengaja setara konteks pool (~0..2). cos/sin, bukan phi mentah:
    phi=0,01 dan phi=6,27 itu hampir sama, tapi angkanya berjauhan. p=None
    (sampel bersih, tanpa kanal) -> [0, 1, 0, 0] = H2 identitas."""
    if p is None or not p.get("tau"):
        return np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    return np.array([p["tau"] / 8.0, np.cos(p["phi"]), np.sin(p["phi"]), p["rho"]],
                    np.float32)


# ---------------------------------------------------------------------------
# Compact maritime channel — uses significant wave height (handles storms B7-9)
# ---------------------------------------------------------------------------
class MaritimeChannel:
    """Applies physics-based maritime distortion to a clean IQ window [2, N].

    Models (all physics-grounded from the OceanStateGenerator pool):
      - distance path loss (capped), sea-surface 2-ray multipath, wave Doppler,
      - EVAPORATION DUCT (SST-driven, `duct_h_m`): the headline tropical-maritime
        feature. A strong duct traps the signal -> recovers part of the distance
        loss (propagation beyond horizon) AND adds a trapped waveguide-mode
        replica (extra, longer-delay multipath). Was previously UNUSED (`sst_c`
        was a dead parameter) — now it reaches the signal.
      - co-channel interference (`n_interferers`, from AIS vessel density): busy
        shipping lanes overlap narrowband interferers onto the signal.
      - PhaseAug (dominant AMC degradation, Paper 1).

    `duct`/`interference` toggles let you ablate each mechanism.
    """

    def __init__(self, freq_mhz: float = 300.0, fs: float = 1e6, rng=None,
                 duct: bool = True, interference: bool = True,
                 lo_ppm: float = 0.0, dua_sinar: bool = True,
                 fasa_global: bool = True, balik: str = ""):
        self.freq_hz = freq_mhz * 1e6
        self.wavelength = 3e8 / self.freq_hz
        self.fs = fs
        self.rng = rng or np.random.default_rng()
        self.duct = duct
        self.interference = interference
        # Offset osilator lokal, dalam ppm. DEFAULT 0 = MATI, supaya seluruh
        # hasil lama tetap sebanding. Nyalakan eksplisit (--lo-ppm 20).
        self.lo_ppm = float(lo_ppm)
        # Ablasi dua komponen yang SEBELUMNYA tak bisa dimatikan. 2026-09-30:
        # penguraian dengan duct+interferensi saja hanya menjelaskan 9,3% dari
        # jurang SNR-tinggi -- karena pantulan permukaan (rho s/d 0,95, null
        # spektral dalam) dan fasa global acak tetap menyala di kedua ablasi.
        self.dua_sinar = dua_sinar
        self.fasa_global = fasa_global
        # Eksperimen ORACLE (grup D): "" = mati (bawaan) | "zf-pita" = balik
        # 2-sinar dengan parameter SEBENARNYA, hanya di pita sinyal, sebelum
        # normalisasi. Tak memakai rng -> undian kanal identik dengan mode mati.
        if balik not in ("", "zf-pita"):
            raise ValueError(f"balik: '' | 'zf-pita', dapat {balik!r}")
        self.balik = balik
        self._pita = None
        # Parameter undian TERAKHIR (tau=0 bila tak ada 2-sinar). Dibaca
        # dataset untuk konteks oracle; selalu diisi, mode apa pun.
        self.terakhir = None

    def apply(self, iq: np.ndarray, dist_km: float, hs_m: float,
              sst_c: float, snr_db: float, duct_h_m: float | None = None,
              n_interferers: int = 0) -> np.ndarray:
        N = iq.shape[1]
        sig = iq[0] + 1j * iq[1]

        # Evaporation-duct strength [0,1]. Prefer the pool's grounded duct_h_m
        # (OceanStateGenerator: Rogers-1988 SST + P.453 refractivity); fall back
        # to the SST-only approximation (propagation_model.evaporation_duct_height)
        # for older pools that predate the duct_h_m field.
        if duct_h_m is None:
            duct_h_m = float(np.clip(0.27 * (sst_c + 4), 0, 40))
        duct = float(np.clip(duct_h_m / 25.0, 0.0, 1.0)) if self.duct else 0.0

        d = max(dist_km, 0.1) * 1e3
        loss_db = 20 * np.log10(4 * np.pi * d / self.wavelength)
        # Distance reduces SNR, but CAPPED: the dominant AMC degradation is the
        # 2-ray multipath (Paper 1), not a path-loss cliff. Without the cap, far
        # vessels become pure noise and the model cannot learn (val collapses).
        snr_loss = min(max(loss_db - 100, 0) * 0.25, 8.0)   # 0..8 dB
        snr_loss *= (1.0 - 0.5 * duct)                      # ducting recovers up to 50%
        extra = snr_loss / 40.0
        sig = sig * (10 ** (-extra / 20))

        k = 2 * np.pi / self.wavelength
        sigma_h = hs_m / 4.0
        rho = float(np.clip(np.exp(-2 * (k * sigma_h * np.sin(np.radians(2))) ** 2), 0, 0.95))
        tau2, phi2 = 0, 0.0                                  # dicatat utk oracle
        if rho > 0.01 and self.dua_sinar:
            delay = int(self.rng.integers(1, 9))
            phase = self.rng.uniform(0, 2 * np.pi)
            sig = (sig + rho * np.roll(sig, delay) * np.exp(1j * phase)) / (1 + rho)
            tau2, phi2 = delay, phase

        # Trapped duct-mode multipath: a longer-delay replica than the sea-surface
        # bounce, present only when a duct exists (SST-driven).
        if duct > 0.05:
            d_delay = int(self.rng.integers(6, 24))
            d_phase = self.rng.uniform(0, 2 * np.pi)
            d_amp = 0.35 * duct
            sig = (sig + d_amp * np.roll(sig, d_delay) * np.exp(1j * d_phase)) / (1 + d_amp)

        fd = f_lo = 0.0                                      # dicatat utk oracle
        if hs_m >= 0.6:
            wave_speed = np.sqrt(9.81 * hs_m) * 0.5
            fd = self.rng.uniform(-1, 1) * (wave_speed / self.wavelength) / self.fs
            sig = sig * np.exp(1j * 2 * np.pi * fd * np.arange(N))

        # OFFSET OSILATOR LOKAL. Kristal penerima meleset; RTL-SDR murah
        # +-20 ppm. Di 300 MHz itu +-6 kHz = +-6e-3 ternormalisasi pada
        # fs=1 MHz -- sekitar 2.400x Doppler gelombang di atas (2,5 Hz), jadi
        # inilah gangguan frekuensi yang DOMINAN di perangkat nyata.
        # Dimatikan secara bawaan: menyalakannya mengubah papan ukur.
        if self.lo_ppm > 0:
            ppm = self.rng.uniform(-self.lo_ppm, self.lo_ppm)
            f_lo = (self.freq_hz * ppm * 1e-6) / self.fs        # ternormalisasi
            sig = sig * np.exp(1j * 2 * np.pi * f_lo * np.arange(N))

        # Co-channel interference from nearby vessels (AIS density). First-order
        # model: each interferer is a narrowband tone at a random frequency
        # offset — busy lanes (Malacca/Singapore) inject several.
        if self.interference and n_interferers > 0:
            for _ in range(int(min(n_interferers, 6))):
                f_off = self.rng.uniform(-0.25, 0.25)
                amp = 0.10 * self.rng.uniform(0.5, 1.5)
                ph = self.rng.uniform(0, 2 * np.pi)
                sig = sig + amp * np.exp(1j * (2 * np.pi * f_off * np.arange(N) + ph))

        # PhaseAug: random global phase rotation theta ~ U[0, 2pi).
        # Paper 1 identified this as the DOMINANT AMC degradation (a passive
        # receiver has no absolute phase reference). Training with it teaches
        # phase-invariant features — the single biggest robustness lever.
        if self.fasa_global:
            sig = sig * np.exp(1j * self.rng.uniform(0, 2 * np.pi))

        eff_snr = snr_db - extra * 40
        npow = 10 ** (-eff_snr / 10)
        sig = sig + np.sqrt(npow / 2) * (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N))

        # Doppler & LO = perkalian berubah-waktu SESUDAH 2-sinar, tak komutatif
        # dengan geser melingkar -> oracle wajib membuangnya dulu (sisa 5,2e-3
        # kalau diabaikan; uji_oracle_2sinar.md §5).
        self.terakhir = dict(rho=rho if tau2 else 0.0, tau=tau2, phi=phi2,
                             f_geser=fd + f_lo)
        if self.balik and tau2:
            if self._pita is None or len(self._pita) != N:
                self._pita = pita_sinyal(N)
            sig = balik_2sinar(sig, rho, tau2, phi2, fd + f_lo, self._pita)

        std = np.std(np.abs(sig))
        if std > 1e-8:
            sig = sig / std
        return np.stack([sig.real, sig.imag], 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class _AugMixin:
    """Shared maritime-augmentation logic for both in-RAM and lazy datasets."""

    def _init_aug(self, pool, aug_fraction, seed, channel_kw=None, gpu_channel=False,
                  ctx_oracle=False):
        self.pool = pool
        self.aug_fraction = aug_fraction
        self.rng = np.random.default_rng(seed)
        self.channel = MaritimeChannel(rng=self.rng, **(channel_kw or {}))
        # Grup D: konteks = parameter 2-sinar SEBENARNYA, bukan catatan pool.
        # Undian (j, kanal) tetap sama persis; hanya isi vektor konteks berganti.
        self.ctx_oracle = ctx_oracle
        # gpu_channel=True: workers return CLEAN iq + per-sample channel params
        # ("chp"); the batched GPUMaritimeChannel applies the physics on-device in
        # run_epoch. Fixes the CPU-augmentation bottleneck that starves fast GPUs.
        self.gpu_channel = gpu_channel

    def set_pool(self, pool: Dict):
        self.pool = pool

    def _make_item(self, iq_clean: np.ndarray, label: int, snr: float) -> Dict:
        chp = np.zeros(6, np.float32)               # [dist,hs,sst,duct_h,n_int,snr]
        if self.rng.random() < self.aug_fraction:
            j = self.rng.integers(0, len(self.pool["context"]))
            ctx = self.pool["context"][j].astype(np.float32)
            dist = float(self.pool["dist_km"][j])
            hs = float(self.pool["hs_m"][j])
            sst = float(self.pool["sst_c"][j])
            duct_h = float(self.pool["duct_h_m"][j]) if "duct_h_m" in self.pool \
                else float(np.clip(0.27 * (sst + 4), 0, 40))   # SST-only fallback
            n_int = int(self.pool["n_interferers"][j]) if "n_interferers" in self.pool else 0
            chp[:] = (dist, hs, sst, duct_h, n_int, snr)
            if self.gpu_channel:
                iq = iq_clean                       # physics applied on-GPU later
            else:
                iq = self.channel.apply(iq_clean, dist_km=dist, hs_m=hs, sst_c=sst,
                                        snr_db=snr, duct_h_m=duct_h, n_interferers=n_int)
                if self.ctx_oracle:
                    ctx = ctx_dari_kanal(self.channel.terakhir)
            is_aug = 1.0
        else:
            iq = iq_clean
            ctx = ctx_dari_kanal(None) if self.ctx_oracle else _BENIGN_CTX.copy()
            is_aug = 0.0
        return {
            "iq": torch.from_numpy(iq),
            "iq_clean": torch.from_numpy(iq_clean),
            "context": torch.from_numpy(ctx),
            "label": int(label),
            "is_aug": is_aug,
            "chp": torch.from_numpy(chp),
        }


class InMemoryMaritimeDataset(Dataset, _AugMixin):
    """All IQ held in RAM, with optional index views to avoid train/val copies."""

    def __init__(self, X, Y, SNR, pool, aug_fraction=0.5, seed=42, row_indices=None,
                 channel_kw=None, gpu_channel=False, ctx_oracle=False):
        self.X, self.Y, self.SNR = X, Y, SNR
        self.row_indices = None if row_indices is None else np.asarray(row_indices, dtype=np.int64)
        self._init_aug(pool, aug_fraction, seed, channel_kw, gpu_channel, ctx_oracle)

    def __len__(self):
        return len(self.row_indices) if self.row_indices is not None else len(self.X)

    def __getitem__(self, idx):
        row = int(self.row_indices[idx]) if self.row_indices is not None else idx
        return self._make_item(self.X[row].astype(np.float32, copy=False),
                               int(self.Y[row]), float(self.SNR[row]))


class LazyMaritimeDataset(Dataset, _AugMixin):
    """
    Reads each IQ sample from HDF5 on demand — memory-safe for the full 2.56M set.
    The h5py handle is opened lazily per worker process (fork-safe).
    """

    def __init__(self, hdf5_path, indices, labels, snrs, pool, aug_fraction=0.5, seed=42,
                 channel_kw=None, gpu_channel=False, ctx_oracle=False):
        self.hdf5_path = str(hdf5_path)
        self.indices = indices            # global row indices into HDF5 X
        self.labels = labels
        self.snrs = snrs
        self._init_aug(pool, aug_fraction, seed, channel_kw, gpu_channel, ctx_oracle)
        self._h5 = None

    def _x(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.hdf5_path, "r")["X"]
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        g = int(self.indices[i])
        iq_clean = self._x()[g].T.astype(np.float32)   # [1024,2] -> [2,1024]
        return self._make_item(iq_clean, int(self.labels[i]), float(self.snrs[i]))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model weights (eval/best use the smoothed copy)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def store(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}


def moe_balance_loss(gate_w: torch.Tensor) -> torch.Tensor:
    imp = gate_w.sum(0)
    imp = imp / (imp.sum() + 1e-9)
    return (imp.std() / (imp.mean() + 1e-9)) ** 2


def equalizer_aux_loss(iq_eq, iq_clean, is_aug) -> torch.Tensor:
    """
    Phase-INVARIANT MSE between equalised and clean IQ (only augmented samples).
    Because PhaseAug rotates the signal by an unknowable global phase, we first
    align the optimal global phase in closed form (phi = -angle(sum e*conj(c)))
    and then take MSE. This penalises the recoverable distortions (multipath,
    fading) without fighting phase-invariance.
    """
    mask = is_aug > 0.5
    if mask.sum() == 0:
        return torch.zeros((), device=iq_eq.device)
    e, c = iq_eq[mask], iq_clean[mask]                 # [M,2,N]
    er, ei, cr, ci = e[:, 0], e[:, 1], c[:, 0], c[:, 1]
    S_re = (er * cr + ei * ci).sum(-1, keepdim=True)   # Re(sum e*conj(c))
    S_im = (ei * cr - er * ci).sum(-1, keepdim=True)   # Im(sum e*conj(c))
    phi = -torch.atan2(S_im, S_re)                     # optimal alignment angle
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    er2 = er * cphi - ei * sphi                         # rotate e by phi
    ei2 = er * sphi + ei * cphi
    return ((er2 - cr) ** 2 + (ei2 - ci) ** 2).mean()


# ---------------------------------------------------------------------------
# Pabrik model
# ---------------------------------------------------------------------------
def bikin_model_sama(cfg, arch, device):
    """Satu-satunya tempat arch -> kelas. Jangkar WAJIB lewat sini juga.

    Kalau dispatch ini disalin dua kali, cukup satu cabang terlewat untuk
    membuat jangkar diam-diam memakai arsitektur LAIN daripada muridnya --
    dan load_state_dict(strict=False) akan menerimanya tanpa suara.
    """
    if arch.startswith('dpdrsn'):
        from dp_drsn import DPDRSN
        return DPDRSN(cfg).to(device)
    if arch in ('mcformer', 'resnet1d'):                 # G0: baseline berprotokol sama
        from baseline_models import bikin_baseline_latih
        return bikin_baseline_latih(arch, cfg['num_classes'], cfg).to(device)
    if arch == 'v4a1':
        from maritime_amc_v4_arch import MaritimeAMCv4
        return MaritimeAMCv4(cfg).to(device)
    if arch == 'v4a2':
        from maritime_amc_v4a2_arch import MaritimeAMCv4A2
        return MaritimeAMCv4A2(cfg).to(device)
    return MaritimeAMCv3(cfg).to(device)


# ---------------------------------------------------------------------------
# Jangkar (self-evolution): melanjutkan latihan tanpa membuang yang lama
# ---------------------------------------------------------------------------
# BACA INI SEBELUM MEMAKAI. Latihan MARSPEC bawaan TIDAK berurutan: tiap
# sampel dikenai kanal maritim dengan peluang --aug_fraction (bawaan 0,5),
# jadi bersih dan maritim bercampur di SETIAP batch sejak epoch 0. Dalam
# susunan itu tak ada lupa katastrofik untuk diukur, karena tak ada tahap
# "sebelum". Angka T_aug100 (bersih 0,7348 -> 0,3524) sering disalahbaca
# sebagai lupa; sebenarnya aug_fraction=1,0 berarti modelnya TAK PERNAH
# melihat data bersih, jadi 0,3524 itu transfer nol-tembakan, bukan lupa.
#
# Untuk benar-benar menguji self-evolution, susunan berurutannya harus
# DIBUAT dulu:
#   Fase 1  --aug_fraction 0.0                  -> model bersih, skor C1
#   Fase 2  --init-from <fase1> --aug_fraction 1.0
# Baru di situ lupa terdefinisi: C1 - C2.
#
# DUA RAGAM, dan biayanya tidak sama:
#
#   l2sp  denda di RUANG PARAMETER:  mean((theta - theta_ref)^2)
#         Nol tambahan lintasan maju. Praktis gratis. Ini L2-SP, pembanding
#         standar di continual learning.
#
#   kd    denda di RUANG FUNGSI: KL(murid || jangkar) atas masukan BERSIH,
#         disuhu T, dikali T^2 supaya besar gradiennya tak ikut berubah saat
#         T diubah. Butuh DUA lintasan maju tambahan (jangkar atas bersih +
#         murid atas bersih) -> sekitar 2-3x lebih lambat per epoch.
#
# KEJUJURAN YANG WAJIB DICATAT: di MARSPEC, replay sudah GRATIS dan sudah ada
# (--aug_fraction < 1 menyisakan sampel bersih BERLABEL di tiap batch), dan kd
# mengejar sasaran yang SAMA dengan replay -- hanya memakai prediksi jangkar
# sebagai label lunak alih-alih label sebenarnya yang sudah kita punya. GPI
# butuh jangkar karena pralatihannya tanpa label. MARSPEC punya label. Jadi
# jangkar di sini hanya berguna kalau ia MENGALAHKAN replay pada anggaran
# sampel-bersih yang setara. Itu kriteria lulusnya, bukan skornya naik.
def muat_bobot_saja(path, model, device):
    """Ambil HANYA bobot dari sebuah ckpt atau state_dict. Optimizer tidak."""
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    # Urutan: EMA dulu, baru mentah. EMA adalah bobot yang divalidasi dan
    # disimpan sebagai best.pt, jadi itulah yang dievaluasi di mana-mana.
    # Kalau fase 2 dimulai dari bobot MENTAH sementara skor fase 1 diukur
    # pada EMA, selisih 'lupa' ikut memuat selisih EMA-lawan-mentah.
    if isinstance(ck, dict) and ck.get('ema'):
        sd = ck['ema']
    elif isinstance(ck, dict) and 'model' in ck:
        sd = ck['model']
    else:
        sd = ck                      # best.pt = state_dict EMA telanjang
    # strict=False memaafkan kunci yang HILANG atau ASING, tapi TIDAK
    # memaafkan BENTUK yang beda -- itu tetap RuntimeError. Ketiganya berarti
    # hal yang sama bagi pemakai (ckpt ini bukan untuk arsitektur ini), jadi
    # ketiganya dijadikan satu pesan yang menyebut sebabnya.
    try:
        hilang, asing = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        baris = [b for b in str(e).split(chr(10)) if 'size mismatch' in b][:4]
        raise SystemExit(
            'bobot %s BENTUKNYA tak cocok dengan arsitektur ini.%s%s%s'
            'Hampir selalu --size atau --arch berbeda dari saat ckpt itu'
            ' dilatih. Lihat config.json di direktori ckpt-nya.'
            % (path, chr(10), chr(10).join('  ' + b.strip() for b in baris),
               chr(10))) from None
    if hilang or asing:
        # Digagalkan, TIDAK didiamkan: bobot yang tak cocok diam-diam adalah
        # cara paling rapi untuk menjalankan model separuh acak sambil
        # mengira kamu sedang mengukur efek jangkar.
        raise SystemExit(
            'bobot %s TIDAK cocok dengan arsitektur ini.%s  hilang: %s%s'
            '  asing: %s%sPeriksa --arch dan --size sama dengan saat ckpt'
            ' itu dilatih.'
            % (path, chr(10), list(hilang)[:6], chr(10),
               list(asing)[:6], chr(10)))
    return model


def putuskan_init_from(init_from, resume, ckpt_path, out_dir):
    """Kembalikan "muat" atau "lewati"; gagalkan bila ambigu.

    Penanda `init_from.txt` ditulis saat bobot --init-from benar-benar
    dimuat. Kalau kemudian sesi putus dan run diulang dengan --resume,
    penanda yang COCOK membuktikan ckpt yang ada adalah lanjutan run yang
    sama -> resume menang dan --init-from dilewati dengan sah. Penanda
    yang hilang atau berbeda berarti ckpt itu milik run LAIN -> gagal,
    karena resume akan menimpa bobot --init-from diam-diam.
    """
    from pathlib import Path as _P
    if not init_from:
        return 'lewati'
    if not (resume and _P(ckpt_path).exists()):
        return 'muat'
    penanda = _P(out_dir) / 'init_from.txt'
    if penanda.exists() and penanda.read_text().strip() == str(init_from):
        return 'lewati'
    raise SystemExit(
        '--init-from dan --resume bentrok: ckpt %s sudah ada tapi BUKAN '
        'lanjutan run --init-from %s (penanda init_from.txt %s). Resume akan '
        'menimpa bobot --init-from. Pakai --out direktori baru.'
        % (ckpt_path, init_from, 'berbeda' if penanda.exists() else 'tak ada'))


def bikin_jangkar(path, model_acuan, device, mode):
    """dict jangkar. l2sp -> tensor acuan; kd -> model beku."""
    muat_bobot_saja(path, model_acuan, device)
    if mode == 'l2sp':
        return dict(mode='l2sp',
                    acuan={n: p.detach().clone()
                           for n, p in model_acuan.named_parameters()})
    beku = model_acuan.to(device).eval()
    for p in beku.parameters():
        p.requires_grad_(False)
    return dict(mode='kd', model=beku)


def rugi_jangkar(jangkar, model, iq_clean, ctx, temp):
    """Denda penyimpangan. Skalar, bergradien ke parameter model."""
    if jangkar['mode'] == 'l2sp':
        # mean atas ELEMEN, bukan sum: bobotnya jadi bebas-skala, sehingga
        # --anchor-weight yang sama berarti hal yang sama di T (2,1jt) dan
        # XM (51,3jt). Dengan sum, bobot yang pas di T akan 25x terlalu
        # besar di XM dan latihannya membeku tanpa pesan error apa pun.
        s, n = None, 0
        for nm, p in model.named_parameters():
            r = jangkar['acuan'].get(nm)
            if r is None:
                continue
            d = ((p - r) ** 2).sum()
            s = d if s is None else s + d
            n += p.numel()
        if s is None:
            return torch.zeros((), device=iq_clean.device)
        return s / max(n, 1)

    with torch.no_grad():
        lg_a = jangkar['model'](iq_clean, ctx)[0].float()
    lg_m = model(iq_clean, ctx)[0].float()
    # T^2 mengembalikan besar gradien yang dikecilkan 1/T^2 oleh pembagian
    # suhu, jadi --anchor-weight tak perlu disetel ulang tiap kali T diubah.
    return F.kl_div(F.log_softmax(lg_m / temp, dim=-1),
                    F.softmax(lg_a / temp, dim=-1),
                    reduction='batchmean') * (temp ** 2)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
def curriculum_pool(epoch: int, total: int) -> str:
    frac = epoch / max(total - 1, 1)
    if frac < 0.30:
        return "easy"
    if frac < 0.65:
        return "normal"
    return "hard"


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def load_pools(pools_dir: Path) -> Dict[str, Dict]:
    out = {}
    for tag in ("easy", "normal", "hard"):
        p = pools_dir / f"ocean_pool_{tag}.npz"
        if p.exists():
            d = np.load(p)
            out[tag] = {k: d[k] for k in d.files}
    if not out:
        raise FileNotFoundError(f"No ocean_pool_*.npz in {pools_dir}")
    # If only some tags exist, alias missing ones to the closest available
    for tag in ("easy", "normal", "hard"):
        out.setdefault(tag, out[next(iter(out))])
    return out


def scan_indices(hdf5_path: Path, snr_min: int, max_samples: Optional[int], seed=42):
    """
    Load only Z (SNR) + Y (labels) to build the sample index — never loads X.
    Returns (indices, labels, snrs) as compact arrays.
    """
    import h5py
    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        Z = f["Z"][:].squeeze().astype(np.int16)        # ~5 MB
        keep = np.where(Z >= snr_min)[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(keep)
        if max_samples and len(keep) > max_samples:
            keep = keep[:max_samples]
        keep.sort()
        # Label dibaca PER BLOK BERURUTAN untuk SELURUH berkas (2,56 jt x 24, ~0,5 GB
        # dibaca bertahap), lalu dipilih di numpy. Dulu `f["Y"][keep]` = fancy
        # indexing h5py atas 600rb indeks: lambat (~0,7 ms/indeks), dan pada 11 Okt
        # 2026 tertahan berjam-jam di Colab. Hasilnya identik (diuji).
        Y = f["Y"]
        label_semua = np.empty(Y.shape[0], np.int16)
        for a in range(0, Y.shape[0], 262_144):
            label_semua[a:a + 262_144] = np.argmax(Y[a:a + 262_144], axis=1)
        labels = label_semua[keep]
        snrs = Z[keep].astype(np.float16)
    print(f"  indeks & label: {len(keep):,} sampel dipilih ({time.time() - t0:.0f} dtk)", flush=True)
    return keep.astype(np.int64), labels, snrs


def _ram_tersedia():
    """Byte RAM yang masih tersedia (Linux /proc/meminfo; None kalau tak terbaca)."""
    try:
        with open("/proc/meminfo") as fh:
            for baris in fh:
                if baris.startswith("MemAvailable:"):
                    return int(baris.split()[1]) * 1024
    except OSError:
        pass
    return None


def load_X_inmemory(hdf5_path: Path, indices, blok: int = 65536):
    """
    Load selected rows into RAM as [n,2,1024] float32, in the order of `indices`.

    Ditulis ulang 7 Okt 2026. Versi lama (`f["X"][indices]` lalu transpose) punya
    dua masalah pada 600rb sampel: (1) puncak RAM ~2x hasil akhir (array [n,1024,2]
    + salinan transpose = ~10 GB), cukup untuk mematikan runtime Colab biasa tanpa
    pesan -- sel terlihat "macet" di "Loading selected IQ into RAM"; (2) jalur
    > 50% berkas membaca SELURUH 21 GB ke RAM sekaligus.

    Sekarang: hasil dialokasikan SEKALI, berkas dibaca per blok berurutan
    (`blok` baris, ~0,5 GB), baris yang dipilih langsung di-transpose ke tempatnya.
    Puncak RAM = hasil + satu blok. Untuk pilihan kecil (<= 100rb) fancy indexing
    tetap dipakai: ia membaca hanya baris yang perlu, jadi lebih cepat dari
    memindai 21 GB.
    """
    import h5py
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    perlu = n * 2 * 1024 * 4
    sisa = _ram_tersedia()
    if sisa is not None and perlu > 0.8 * sisa:
        raise SystemExit(
            f"RAM tak cukup untuk memuat {n:,} sampel ke RAM: butuh {perlu / 1e9:.1f} GB, "
            f"tersedia {sisa / 1e9:.1f} GB. Pakai --lazy (notebook: DATA_MODE = \"lazy\") "
            f"atau runtime high-RAM. Tanpa cek ini runtime Colab mati diam-diam.")
    with h5py.File(hdf5_path, "r") as f:
        ds = f["X"]                                      # [N,1024,2]
        N = ds.shape[0]
        if n <= 100_000:
            urut_ok = n < 2 or bool(np.all(np.diff(indices) > 0))
            if urut_ok:
                return np.ascontiguousarray(np.transpose(ds[indices], (0, 2, 1)), dtype=np.float32)
        out = np.empty((n, ds.shape[2], ds.shape[1]), np.float32)
        urut = np.argsort(indices, kind="stable")
        idx_urut = indices[urut]
        t0, laporan = time.time(), 0.0
        for a in range(0, N, blok):
            b = min(a + blok, N)
            lo = np.searchsorted(idx_urut, a, "left")
            hi = np.searchsorted(idx_urut, b, "left")
            if hi > lo:
                slab = ds[a:b]                           # baca berurutan, cepat
                out[urut[lo:hi]] = np.transpose(slab[idx_urut[lo:hi] - a], (0, 2, 1))
            if b / N >= laporan + 0.1 or b == N:
                laporan = b / N
                print(f"  memuat IQ ke RAM: {b / N:5.1%} berkas dipindai, "
                      f"{hi:,}/{n:,} sampel ({time.time() - t0:.0f} dtk)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optim, scaler, device, train, lam_moe, lam_eq, use_amp,
              label_smoothing=0.0, ema=None, gpu_ch=None,
              jangkar=None, lam_anchor=0.0, anchor_temp=2.0):
    model.train(train)
    tot = correct = 0
    loss_sum = ce_sum = moe_sum = eq_sum = anc_sum = 0.0
    for batch in loader:
        iq = batch["iq"].to(device, non_blocking=True)
        iq_clean = batch["iq_clean"].to(device, non_blocking=True)
        ctx = batch["context"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        is_aug = batch["is_aug"].to(device, non_blocking=True)
        if gpu_ch is not None:
            # Batched on-device channel: apply the maritime physics HERE (the
            # workers delivered clean iq + params). Keeps the GPU fed.
            chp = batch["chp"].to(device, non_blocking=True)
            params = dict(dist_km=chp[:, 0], hs_m=chp[:, 1], sst_c=chp[:, 2],
                          duct_h_m=chp[:, 3], n_interf=chp[:, 4], snr=chp[:, 5])
            iq = gpu_ch(iq_clean, params, is_aug)

        with torch.set_grad_enabled(train), torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu",
                enabled=use_amp):
            logits, iq_eq, gate_w = model(iq, ctx)
            ce = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            lb = moe_balance_loss(gate_w)
            eq = equalizer_aux_loss(iq_eq.float(), iq_clean, is_aug)
            loss = ce + lam_moe * lb + lam_eq * eq
            if jangkar is not None and lam_anchor > 0.0:
                anc = rugi_jangkar(jangkar, model, iq_clean, ctx, anchor_temp)
                loss = loss + lam_anchor * anc
            else:
                anc = torch.zeros((), device=logits.device)

            # GAGAL KERAS pada rugi tak-hingga. Di bawah AMP, inf di GRADIEN itu
            # wajar (GradScaler menyelidik skalanya lalu melewati langkah itu);
            # inf di rugi MAJU tidak pernah wajar. Dan ia tak bisa dipulihkan:
            # lintasan maju yang sama sudah memperbarui running stats
            # BatchNorm, lalu EMA menyalinnya -> validasi rusak selamanya.
            # 1 Okt 2026 run A_ctx berjalan 7 epoch dengan ce=nan sebelum
            # ketahuan dari mata. Pemeriksaan ini tak menambah sinkronisasi:
            # loss.item() di bawah sudah menyinkronkan setiap batch.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"rugi tak-hingga/NaN saat {'LATIH' if train else 'VAL'} setelah "
                    f"{tot:,} sampel: ce={ce.item()} moe={lb.item()} eq={eq.item()} "
                    f"anc={float(anc)} | maks|iq| batch={iq.abs().max().item():.1f}. "
                    f"Running stats BatchNorm dan EMA sudah tercemar; melanjutkan "
                    f"hanya menghasilkan model rusak. Kalau ini --resume dari ckpt "
                    f"yang sudah rusak, HAPUS direktori --out lalu mulai ulang.")

        if train:
            optim.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            if ema is not None:
                ema.update(model)

        bs = y.size(0)
        tot += bs
        correct += (logits.argmax(1) == y).sum().item()
        loss_sum += loss.item() * bs
        ce_sum += ce.item() * bs; moe_sum += lb.item() * bs; eq_sum += eq.item() * bs
        anc_sum += float(anc) * bs
    return dict(acc=correct / tot, loss=loss_sum / tot,
                ce=ce_sum / tot, moe=moe_sum / tot, eq=eq_sum / tot,
                anc=anc_sum / tot)


def save_ckpt(path, model, optim, sched, scaler, epoch, best, history, ema=None):
    torch.save({
        "model": model.state_dict(), "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.shadow if ema else None,
        "epoch": epoch, "best": best, "history": history,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", type=str, default=str(_ROOT / "data/raw/radioml/archive/GOLD_XYZ_OSC.0001_1024.hdf5"))
    ap.add_argument("--pools", type=str, default=str(_ROOT / "data/ocean_states"))
    ap.add_argument("--out", type=str, default=str(_ROOT / "outputs/maritime_v3"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_samples", type=int, default=200_000, help="None-like: 0 = use ALL")
    ap.add_argument("--snr_min", type=int, default=-6)
    ap.add_argument("--aug_fraction", type=float, default=0.5)
    ap.add_argument("--lo-ppm", dest="lo_ppm", type=float, default=0.0,
                    help="offset osilator lokal +-ppm (0=mati). PAKAI RESIDU "
                         "SETELAH koreksi CFO, bukan galat kristal mentah: 20 ppm "
                         "di 300 MHz memutar konstelasi ~6 kali dalam satu jendela")
    ap.add_argument("--no_duct", action="store_true",
                    help="ablation: disable evaporation-duct channel effect in TRAINING (val stays realistic)")
    ap.add_argument("--no_interference", action="store_true",
                    help="ablation: disable co-channel interference in TRAINING")
    ap.add_argument("--gpu_channel", action="store_true",
                    help="apply the maritime channel as batched torch ops ON the GPU "
                         "(fixes CPU-augmentation bottleneck on fast GPUs)")
    ap.add_argument("--balik-2sinar", dest="balik_2sinar", choices=["tidak", "zf-pita"],
                    default="tidak",
                    help="ORACLE grup D: balik 2-sinar dgn tau/phi/rho SEBENARNYA, hanya di "
                         "pita sinyal, di latih DAN validasi. Bukan fitur penempatan: "
                         "penerima nyata tak tahu tau/phi. Mengukur batas atas.")
    ap.add_argument("--ctx-oracle", dest="ctx_oracle", action="store_true",
                    help="ORACLE grup D: konteks = [tau/8, cos phi, sin phi, rho] "
                         "2-sinar SEBENARNYA, bukan catatan pool. Latih DAN validasi.")
    ap.add_argument("--lam_moe", type=float, default=0.01)
    ap.add_argument("--lam_eq", type=float, default=0.10)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 disables EMA")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lazy", action="store_true", help="Stream IQ from HDF5 (memory-safe)")
    ap.add_argument("--amp", action="store_true", help="Mixed precision (recommended on GPU)")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint if present")
    ap.add_argument("--init-from", dest="init_from", default=None, metavar="CKPT",
                    help="Mulai dari BOBOT ckpt ini, optimizer/jadwal/epoch "
                         "TETAP dari nol. Bedanya dengan --resume: --resume "
                         "melanjutkan run yang SAMA, --init-from memulai run "
                         "BARU dari bobot lama. Ini yang dipakai untuk fase 2 "
                         "percobaan self-evolution.")
    ap.add_argument("--anchor-ckpt", dest="anchor_ckpt", default=None, metavar="CKPT",
                    help="Ckpt acuan untuk denda penyimpangan. Biasanya SAMA "
                         "dengan --init-from.")
    ap.add_argument("--anchor-weight", dest="anchor_weight", type=float, default=0.0,
                    help="Bobot denda jangkar. 0 = mati (bawaan). Skala l2sp "
                         "dan kd BERBEDA jauh: l2sp memakai mean kuadrat "
                         "selisih bobot (biasanya 1e2..1e5), kd memakai KL "
                         "(biasanya 0,1..10). Jangan pindahkan angka antar "
                         "ragam tanpa menyetel ulang.")
    ap.add_argument("--anchor-mode", dest="anchor_mode", choices=["l2sp", "kd"],
                    default="l2sp",
                    help="l2sp=denda ruang parameter, GRATIS. kd=distilasi atas "
                         "masukan bersih, 2-3x lebih lambat tapi menjaga "
                         "FUNGSI bukan bobot.")
    ap.add_argument("--anchor-temp", dest="anchor_temp", type=float, default=2.0,
                    help="Suhu untuk --anchor-mode kd. Diabaikan oleh l2sp.")
    ap.add_argument("--preset", choices=["full24m", "a100"], default=None)
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KUNCI=NILAI",
                    help="Timpa kunci cfg satuan SETELAH preset --size, mis. "
                         "--cfg mcf_nlayers=15 eq_layers=8 cnn_channels=[32,64,96]. "
                         "Kunci tak dikenal digagalkan, bukan diabaikan diam-diam.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Mengendalikan init bobot + augmentasi LATIH + urutan batch. "
                         "Belah train/val dan augmentasi VAL sengaja TIDAK ikut "
                         "berubah, supaya beda antar-benih murni dari pelatihan "
                         "dan bukan dari papan ukur yang bergeser.")
    ap.add_argument("--size", choices=["L", "S", "T", "XL", "XM"], default="L",
                    help="L=36,9jt (CFG_V3 apa adanya) | S=4,9jt | T=2,1jt | "
                         "XL=47,8jt (sinyal dilebarkan, MoE sama) | "
                         "XM=51,3jt (HANYA MoE dibesarkan). "
                         "Hanya mengubah angka cfg -- kode arsitekturnya sama "
                         "persis, jadi selisih skor hanya bisa dari kapasitas.")
    ap.add_argument("--arch", choices=["v3", "v4a1", "v4a2", "dpdrsn", "dpdrsn_dcce",
                                       "mcformer", "resnet1d"],
                    default="v3",
                    help="v3=MaritimeAMCv3 (real); v4a1=equalizer bernilai kompleks; "
                         "v4a2=v3 + cabang spektral (--cfg use_farel=1 use_moe=0 "
                         "moe_route=spectral); "
                         "dpdrsn=baseline ringan ~28rb param; dpdrsn_dcce=idem + DCCE. "
                         "mcformer (117.592 param) / resnet1d (86.424) = baseline "
                         "literatur, dilatih dgn protokol MARSPEC yang SAMA (G0). "
                         "dpdrsn/mcformer/resnet1d MENGABAIKAN --size (cfg sendiri).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # ── presets ───────────────────────────────────────────────────────────
    if args.preset == "full24m":
        # 24M config (CFG_V3); 2.56M samples, lazy+amp — for standard/T4 Colab.
        args.max_samples = 0          # use ALL
        args.epochs = 20 if args.epochs == 50 else args.epochs
        args.batch = 256 if args.batch == 128 else args.batch
        args.lazy = True
        args.amp = True
        args.workers = max(args.workers, 4)
    elif args.preset == "a100":
        # High-RAM A100/H100 (80GB RAM / 80GB VRAM): load all to RAM, big batch.
        args.max_samples = 0          # use ALL selected samples (~21GB max raw IQ)
        args.epochs = 20 if args.epochs == 50 else args.epochs
        args.batch = 1024 if args.batch == 128 else args.batch
        args.lazy = False             # in-RAM is ~2x faster than streaming
        args.amp = True               # TF32/FP16 on A100
        args.workers = max(args.workers, 8)  # parallelise channel augmentation

    # Ditetapkan SEBELUM model dibuat -- init bobot menarik dari RNG global torch.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    if device == "cuda":
        # TF32 + autotuned cudnn kernels: big speedup for fp32 conv/matmul
        # (negligible accuracy impact); matmul TF32 is OFF by default in PyTorch.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | AMP: {use_amp} | lazy: {args.lazy}")

    if args.arch.startswith("dpdrsn"):
        # DP-DRSN punya ruang cfg sendiri; CFG_V3 dan --size tak berlaku.
        from dp_drsn import CFG_DPDRSN
        cfg = dict(CFG_DPDRSN)
        cfg["use_dcce"] = args.arch == "dpdrsn_dcce"
        cfg["num_classes"] = CFG_V3["num_classes"]
        if args.size != "L":
            print(f"CATATAN: --size {args.size} diabaikan untuk --arch {args.arch}")
        if not cfg["use_dcce"] and args.lam_eq != 0.0:
            # Tanpa DCCE, iq_eq ADALAH masukannya: rugi eq tak punya jalur
            # gradien ke parameter mana pun. Angkanya tetap berguna (ia mengukur
            # kerusakan kanal mentah) tapi tak boleh dikira sedang dilatih.
            print("CATATAN: tanpa DCCE, rugi eq hanya PENGUKUR kerusakan kanal, "
                  "bukan yang dilatih (tak ada gradien ke parameter).")
        return_cfg_v3 = False
    elif args.arch in ("mcformer", "resnet1d"):
        # G0: baseline literatur dengan protokol MARSPEC yang SAMA. Arsitekturnya
        # tetap persis seperti di baseline_models.py (bentuk bobot lama), jadi
        # yang berubah dari angka baseline lama HANYA protokol latihnya.
        from baseline_models import CFG_BASELINE_BAWAAN
        # Kunci ukuran disemai dgn nilai bawaan kelasnya -> `--cfg d_model=184
        # n_layers=5` diterima, dan config.json mencatat ukuran sebenarnya.
        cfg = {"baseline": args.arch, "num_classes": CFG_V3["num_classes"],
               **CFG_BASELINE_BAWAAN[args.arch]}
        print(f"CATATAN: --arch {args.arch} = baseline; --size {args.size} hanya label, "
              f"konteks diabaikan, rugi eq/moe tanpa gradien (pengukur saja).")
        return_cfg_v3 = False
    else:
        cfg = dict(CFG_V3)
        cfg.update(PRESET_UKURAN[args.size])
        return_cfg_v3 = True
    if return_cfg_v3 and args.size in ("S", "T"):
        # grad checkpoint itu tukar-tambah VRAM-lawan-kecepatan yang hanya
        # masuk akal untuk model besar. Pada S/T modelnya sudah muat, jadi
        # menyalakannya cuma memperlambat tanpa imbalan.
        #
        # Syaratnya "S atau T", BUKAN "bukan L". Versi lama mematikan
        # checkpoint untuk setiap preset selain L -- yang berarti XL (47,8jt)
        # dan XM (51,3jt), dua preset yang paling MEMBUTUHKANNYA, justru
        # kehilangannya dan kemungkinan besar OOM di T4. Daftar putih tumbuh
        # dengan aman; daftar hitam tidak.
        cfg["grad_checkpoint"] = False
    if args.smoke:
        # Pengecilan CFG hanya berlaku untuk jalur v3/v4 -- dpdrsn punya ruang
        # cfg sendiri. Tapi pengecilan RUN (epoch/batch/sampel/worker) berlaku
        # untuk SEMUA arsitektur; menggerbangi keduanya bersama membuat --smoke
        # dpdrsn diam-diam menjalankan 50 epoch x 200rb sampel.
        if return_cfg_v3:
            cfg.update(eq_filters=16, eq_layers=4, cnn_channels=[16, 32, 64], cnn_pool=4,
                       mcf_d_model=32, mcf_nhead=4, mcf_nlayers=2, ctx_hidden=32,
                       fusion_d=64, n_experts=4, expert_hidden=64)
        args.epochs, args.batch, args.max_samples, args.workers = 2, 32, 512, 0
        args.lazy = False

    # Kunci khas v4a2 dibenihkan dengan NILAI BAWAANNYA sebelum --cfg dibaca.
    # Tanpa ini, --cfg use_farel=1 ditolak "kunci tak dikenal" walau teks
    # bantuan --arch justru menyuruh memakainya: kuncinya hidup di v4a2, bukan
    # di CFG_V3. Dibenihkan hanya untuk arch yang memakainya, supaya menjalankan
    # v3 dengan --cfg moe_route=... tetap gagal keras dan bukan diam-diam
    # diabaikan.
    if args.arch == "v4a2":
        cfg.setdefault("use_farel", True)
        cfg.setdefault("use_moe", True)
        cfg.setdefault("farel_hidden", 64)
        cfg.setdefault("farel_on", "raw")
        cfg.setdefault("farel_mode", "reim_logpow")
        cfg.setdefault("moe_route", "ctx")

    # --cfg diterapkan PALING AKHIR supaya selalu menang atas --size dan --smoke.
    for kv in args.cfg:
        if "=" not in kv:
            raise ValueError(f"--cfg butuh KUNCI=NILAI, dapat {kv!r}")
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise KeyError(f"kunci cfg tak dikenal: {k!r}. Yang ada: {sorted(cfg)}")
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v                       # biarkan string apa adanya (mis. rnn_kind)
        print(f"  cfg[{k}] = {cfg[k]!r}")

    if return_cfg_v3:
        validate_config(cfg)

    # ── RESUME HARUS SETELAN YANG SAMA ───────────────────────────────────
    # --resume melanjutkan ckpt di --out apa adanya. Kalau setelannya diubah
    # (mis. --epochs 20 -> 40, --max_samples 400rb -> 600rb), hasilnya run
    # CAMPURAN: separuh pertama dengan data lain, jadwal kosinus yang T_max-nya
    # dipulihkan dari ckpt lama -- dan config.json ditimpa sehingga MENGAKU
    # seluruh run memakai setelan baru. Diperiksa di sini, sebelum data dimuat,
    # memakai config.json LAMA (yang baru belum ditulis).
    _out = Path(args.out)
    if args.resume and (_out / "ckpt_last.pt").exists() and (_out / "config.json").exists():
        _lama = json.loads((_out / "config.json").read_text(encoding="utf-8")).get("args", {})
        _kini = vars(args)
        _kunci = ("arch", "size", "epochs", "batch", "max_samples", "seed", "lr",
                  "aug_fraction", "cfg", "snr_min", "label_smoothing", "ema_decay",
                  "lam_moe", "lam_eq", "init_from", "anchor_ckpt", "anchor_mode",
                  "anchor_weight", "balik_2sinar", "ctx_oracle")
        _beda = {k: (_lama.get(k), _kini.get(k)) for k in _kunci
                 if k in _lama and _lama.get(k) != _kini.get(k)}
        if _beda:
            raise SystemExit(
                "--resume DITOLAK: ckpt di %s dilatih dengan setelan lain:%s%s%s"
                "Melanjutkannya menghasilkan run campuran. Pakai --out direktori "
                "baru, atau kembalikan setelannya." % (
                    _out, chr(10), chr(10).join("  %s: lama %r -> kini %r" % (k, a, b)
                                                  for k, (a, b) in _beda.items()), chr(10)))

    # ── validasi argumen jangkar, SEBELUM apa pun yang mahal ─────────────
    # Diperiksa di sini dan bukan di tempat jangkarnya dibangun: tempat itu
    # berada setelah HDF5 dipindai dan model dibangun, jadi satu salah ketip
    # --anchor-ckpt baru ketahuan setelah 21 GB terbaca. Pemeriksaan murah
    # harus jatuh sebelum pekerjaan mahal.
    if args.anchor_weight > 0.0 and not args.anchor_ckpt:
        raise SystemExit(
            '--anchor-weight %g diberikan tanpa --anchor-ckpt. Tanpa ckpt acuan'
            ' tak ada yang bisa dijangkar, dan dendanya akan diam-diam nol.'
            % args.anchor_weight)
    for nm, pth in (('--init-from', args.init_from),
                    ('--anchor-ckpt', args.anchor_ckpt)):
        if pth and not Path(pth).exists():
            raise SystemExit('%s menunjuk berkas yang tidak ada: %s' % (nm, pth))
    if args.anchor_weight > 0.0 and args.anchor_mode == 'kd':
        print('CATATAN: --anchor-mode kd menambah DUA lintasan maju per batch '
              '(jangkar + murid atas iq_clean) -> perkirakan 2-3x lebih lambat.')
    # Oracle grup D butuh parameter kanal per-sampel dari MaritimeChannel (CPU).
    # Jalur GPU menerapkan fisikanya sendiri dan tak mencatat tau/phi -> mode
    # oracle di sana akan DIAM-DIAM tak aktif.
    if args.gpu_channel and (args.balik_2sinar != 'tidak' or args.ctx_oracle):
        raise SystemExit('--balik-2sinar / --ctx-oracle hanya jalan dengan kanal CPU; '
                         'buang --gpu_channel.')

    max_samples = None if args.max_samples in (0, None) else args.max_samples

    # ── data ──────────────────────────────────────────────────────────────
    # PENJAGA FASE PERSIAPAN (11 Okt 2026). Dua kali run tertahan berjam-jam di
    # sini tanpa jejak (sel Colab cuma diam). Kalau fase ini lewat 10 menit,
    # Python mencetak tumpukan panggilan SEMUA thread ke stderr tiap 10 menit:
    # baris persis tempat ia tertahan jadi terlihat. Dimatikan begitu loader siap.
    import faulthandler, sys
    faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
    pools = load_pools(Path(args.pools))
    print(f"Pools: {list(pools)}", flush=True)
    idx, labels, snrs = scan_indices(Path(args.hdf5), args.snr_min, max_samples)
    print(f"Selected samples: {len(idx):,} (SNR>={args.snr_min}, "
          f"max_samples={'ALL' if max_samples is None else max_samples})")

    # RadioML is stored ordered by class and `idx` is sorted (for fast HDF5 reads),
    # so a CONTIGUOUS slice would give train/val DISJOINT classes (val acc ~ 0!).
    # Shuffle split positions so train and val both cover all 24 classes.
    n_val = int(len(idx) * 0.15)
    _perm = np.random.default_rng(123).permutation(len(idx))
    val_sl = _perm[:n_val]; tr_sl = _perm[n_val:]

    # Ablation: the channel mechanism toggle applies to the TRAIN channel only;
    # val stays the realistic (duct+interference) channel so runs are comparable.
    # Oracle grup D: berlaku di latih DAN validasi. Validasi wajib ikut, karena
    # best.pt dipilih dari va_acc -- model yang dilatih pada masukan terbalik
    # tapi divalidasi pada masukan mentah akan memilih checkpoint yang salah.
    balik = "" if args.balik_2sinar == "tidak" else args.balik_2sinar
    ch_kw = dict(duct=not args.no_duct, interference=not args.no_interference,
                 lo_ppm=args.lo_ppm, balik=balik)
    va_kw = dict(balik=balik)
    gc, co = args.gpu_channel, args.ctx_oracle
    if balik or co:
        print(f"ORACLE grup D: balik_2sinar={args.balik_2sinar} ctx_oracle={co} "
              f"(latih + validasi)")
    if args.lazy:
        train_ds = LazyMaritimeDataset(args.hdf5, idx[tr_sl], labels[tr_sl], snrs[tr_sl],
                                       pools["easy"], args.aug_fraction, seed=args.seed,
                                       channel_kw=ch_kw, gpu_channel=gc, ctx_oracle=co)
        val_ds = LazyMaritimeDataset(args.hdf5, idx[val_sl], labels[val_sl], snrs[val_sl],
                                     pools["normal"], aug_fraction=1.0, seed=1234,
                                     channel_kw=va_kw, gpu_channel=gc, ctx_oracle=co)
    else:
        print("Loading selected IQ into RAM ...")
        X = load_X_inmemory(Path(args.hdf5), idx)
        train_ds = InMemoryMaritimeDataset(X, labels, snrs, pools["easy"],
                                           args.aug_fraction, seed=args.seed,
                                           row_indices=tr_sl, channel_kw=ch_kw,
                                           gpu_channel=gc, ctx_oracle=co)
        val_ds = InMemoryMaritimeDataset(X, labels, snrs, pools["normal"],
                                         aug_fraction=1.0, seed=1234,
                                         row_indices=val_sl, channel_kw=va_kw,
                                         gpu_channel=gc, ctx_oracle=co)
    # On-device channel instances (train follows the ablation toggles; val is
    # always the realistic full channel).
    gpu_ch_tr = gpu_ch_va = None
    if gc:
        from gpu_channel import GPUMaritimeChannel
        gpu_ch_tr = GPUMaritimeChannel(**ch_kw)
        gpu_ch_va = GPUMaritimeChannel()
        print("GPU channel: ON (batched on-device augmentation)")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    pin = device == "cuda"
    # IMPORTANT: train loader must NOT use persistent_workers — the curriculum
    # calls train_ds.set_pool() each epoch, and persistent workers hold a stale
    # copy of the dataset (the pool change would never reach them). Non-persistent
    # workers respawn per epoch and pick up the new pool. (val pool is constant.)
    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, pin_memory=pin, persistent_workers=False)
    val_ld = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=pin, persistent_workers=args.workers > 0)
    faulthandler.cancel_dump_traceback_later()       # persiapan selesai; latihan boleh lama

    # ── model ─────────────────────────────────────────────────────────────
    model = bikin_model_sama(cfg, args.arch, device)
    print(f"Arch: {args.arch} | Size: {args.size} | Model params: {sum(p.numel() for p in model.parameters()):,}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt_last.pt"

    # REKAM KONFIGURASI. Tanpa ini, run yang sudah jadi tak bisa direproduksi
    # maupun dibandingkan secara adil. 2026-09-26: direktori `ukuran_*_s42`
    # dan `ukuran_*_s72` terlihat seperti pasangan benih, ternyata beda
    # anggaran latihan (10 lawan 18-20 epoch) -- dan tak ada satu berkas pun
    # yang bisa membuktikannya selain menghitung baris history.json.
    # Lantai derau HANYA sah bila semua selain benih dipegang tetap, jadi
    # "semua yang lain" wajib tercatat.
    import time as _t
    (out_dir / "config.json").write_text(json.dumps(dict(
        args=vars(args),
        cfg={k: (list(v) if isinstance(v, (list, tuple)) else v)
             for k, v in cfg.items()},
        n_param=sum(p.numel() for p in model.parameters()),
        n_train=len(train_ds), n_val=len(val_ds),
        init_from=args.init_from, anchor_ckpt=args.anchor_ckpt,
        anchor_mode=args.anchor_mode, anchor_weight=args.anchor_weight,
        anchor_temp=args.anchor_temp,
        torch=torch.__version__, device=str(device),
        waktu=_t.strftime("%Y-%m-%d %H:%M:%S"),
    ), indent=2, default=str))
    history, best_acc, start_ep = [], 0.0, 0

    # ── init-from: bobot saja, run baru ───────────────────────────────────
    keputusan = putuskan_init_from(args.init_from, args.resume, ckpt_path, out_dir)
    if keputusan == 'muat':
        muat_bobot_saja(args.init_from, model, device)
        (out_dir / 'init_from.txt').write_text(str(args.init_from))
        # EMA dibuat di atas, SEBELUM bobot ini dimuat, jadi bayangannya
        # masih berisi bobot ACAK. Tanpa baris ini, EMA -- yang divalidasi
        # dan disimpan sebagai best.pt -- berangkat dari acak dan butuh
        # ~2 epoch (decay 0,999, ~1.560 langkah/epoch) untuk melupakannya.
        # Dua epoch pertama fase 2 justru tempat lupa diukur.
        if ema is not None:
            ema.shadow = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        print('Init dari bobot: %s (optimizer/jadwal/epoch dari NOL, EMA '
              'disalin dari bobot itu)' % args.init_from)
    elif args.init_from:
        print('Lanjutan run --init-from %s: resume dari ckpt, bobot init '
              'TIDAK dimuat ulang (penanda cocok).' % args.init_from)

    # ── jangkar ───────────────────────────────────────────────────────────
    jangkar = None
    if args.anchor_weight > 0.0:
        # Argumennya sudah divalidasi di atas; di sini tinggal membangun.
        acuan = bikin_model_sama(cfg, args.arch, device)
        jangkar = bikin_jangkar(args.anchor_ckpt, acuan, device, args.anchor_mode)
        print('Jangkar %s dari %s, bobot %g%s'
              % (args.anchor_mode, args.anchor_ckpt, args.anchor_weight,
                 '  (+2 lintasan maju/batch)' if args.anchor_mode == 'kd' else ''))
    elif args.anchor_ckpt:
        print('CATATAN: --anchor-ckpt diberikan tapi --anchor-weight 0 '
              '-> jangkar MATI.')

    # ── resume ────────────────────────────────────────────────────────────
    if args.resume and ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        if scaler and ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"): ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        start_ep = ck["epoch"] + 1; best_acc = ck["best"]; history = ck["history"]
        print(f"Resumed from epoch {start_ep} (best={best_acc:.4f})")

    # ── train ─────────────────────────────────────────────────────────────
    for ep in range(start_ep, args.epochs):
        phase = curriculum_pool(ep, args.epochs)
        train_ds.set_pool(pools[phase])
        t0 = time.time()
        tr = run_epoch(model, train_ld, optim, scaler, device, True, args.lam_moe, args.lam_eq,
                       use_amp, label_smoothing=args.label_smoothing, ema=ema, gpu_ch=gpu_ch_tr,
                       jangkar=jangkar, lam_anchor=args.anchor_weight,
                       anchor_temp=args.anchor_temp)
        # Validate with EMA weights (smoothed) — restore raw weights afterwards.
        if ema: ema.store(model); ema.copy_to(model)
        va = run_epoch(model, val_ld, optim, scaler, device, False, args.lam_moe, args.lam_eq,
                       use_amp, gpu_ch=gpu_ch_va)
        if ema: ema.restore(model)
        sched.step()
        history.append(dict(epoch=ep, phase=phase,
                            **{f"tr_{k}": v for k, v in tr.items()},
                            **{f"va_{k}": v for k, v in va.items()}))
        print(f"E{ep:02d} [{phase:6}] tr_acc={tr['acc']:.3f} va_acc={va['acc']:.3f} "
              f"ce={tr['ce']:.3f} moe={tr['moe']:.4f} eq={tr['eq']:.4f} "
              + (f"anc={tr['anc']:.4g} " if jangkar is not None else "")
              + f"({time.time()-t0:.0f}s)")
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            # save the EMA weights (what we validated with)
            torch.save(ema.shadow if ema else model.state_dict(), out_dir / "maritime_v3_best.pt")
        save_ckpt(ckpt_path, model, optim, sched, scaler, ep, best_acc, history, ema=ema)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nBest val acc: {best_acc:.4f}  |  saved to {out_dir}")


if __name__ == "__main__":
    main()
