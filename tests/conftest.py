# -*- coding: utf-8 -*-
"""Fixture bersama untuk SEMUA berkas uji.

Fixture = bahan uji yang disiapkan pytest otomatis. Cukup tulis namanya sebagai
argumen fungsi uji, misalnya `def test_x(cfg_kecil, iq_acak): ...`, dan pytest
yang membuatnya. Tidak perlu diimpor.

Semua bahan di sini SINTETIS dan KECIL: tak ada dataset 21 GB, tak ada GPU,
tak ada bobot model terlatih. Uji harus jalan di laptop siapa pun dalam detik.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def benih_tetap():
    """Dipakai OTOMATIS oleh setiap uji: hasil acak selalu sama di setiap mesin."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def cfg_kecil():
    """Konfigurasi MARSPEC kecil (~200 ribu parameter, bukan 2-37 juta).

    Arsitekturnya SAMA, hanya angkanya diperkecil, jadi jalur kodenya identik
    dengan model sungguhan tapi uji jalan cepat di CPU.
    """
    from maritime_amc_v3_arch import CFG_V3
    cfg = dict(CFG_V3)
    cfg.update(eq_filters=16, eq_layers=4, cnn_channels=[16, 32, 64], cnn_pool=4,
               mcf_d_model=32, mcf_nhead=4, mcf_nlayers=2, mcf_patch_len=32,
               ctx_hidden=32, ctx_mlp_hidden=64, ctx_layers=2, fusion_d=64,
               n_experts=4, expert_hidden=64, grad_checkpoint=False)
    return cfg


@pytest.fixture
def iq_acak():
    """Batch 4 sinyal IQ acak, bentuk [B=4, 2 kanal (I,Q), 1024 sampel]."""
    return torch.randn(4, 2, 1024)


@pytest.fixture
def ctx_acak():
    """Batch 4 vektor konteks laut acak, bentuk [4, 4]."""
    return torch.rand(4, 4)


@pytest.fixture
def pool_tiruan():
    """Pool kondisi laut tiruan, kunci-kuncinya sama dengan pool asli."""
    rng = np.random.default_rng(0)
    n = 50
    return {
        "context": rng.random((n, 4)).astype(np.float32),
        "dist_km": rng.uniform(1, 150, n),
        "hs_m": rng.uniform(0.1, 6.0, n),
        "sst_c": rng.uniform(26, 31, n),
        "duct_h_m": rng.uniform(0, 30, n),
        "n_interferers": rng.integers(0, 8, n),
    }


@pytest.fixture
def h5_tiruan(tmp_path):
    """Berkas HDF5 MINI berstruktur RadioML 2018 (X, Y, Z), di folder sementara.

    X [N, 1024, 2] float32 · Y one-hot [N, 24] · Z (SNR) [N, 1].
    N = 24 kelas x 6 SNR x 4 sampel = 576 baris (~4,7 MB, dihapus otomatis).
    Mengembalikan (path, Y_kelas, Z_snr) supaya uji bisa mencocokkan hasilnya.
    """
    import h5py
    snrs = [-10, -6, 0, 10, 20, 30]
    kelas, snr = [], []
    for k in range(24):
        for s in snrs:
            kelas += [k] * 4
            snr += [s] * 4
    kelas, snr = np.array(kelas), np.array(snr)
    n = len(kelas)
    X = np.random.default_rng(1).standard_normal((n, 1024, 2)).astype(np.float32)
    Y = np.eye(24, dtype=np.int64)[kelas]
    path = tmp_path / "radioml_mini.hdf5"
    with h5py.File(path, "w") as f:
        f["X"] = X
        f["Y"] = Y
        f["Z"] = snr.reshape(-1, 1)
    return path, kelas, snr
