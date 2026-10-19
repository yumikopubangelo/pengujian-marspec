# -*- coding: utf-8 -*-
"""BAGIAN 3 — Simulator kanal laut: MaritimeChannel.apply.

Pemilik : [Anggota 3]
Kode    : marspec/train_maritime_v3.py (kelas MaritimeChannel)
Catatan : apply() punya 12 titik keputusan -> bagian whitebox paling "gemuk".
          `ch.terakhir` mencatat parameter undian terakhir (rho, tau, phi,
          f_geser): berguna untuk membuktikan cabang mana yang diambil.
"""
import numpy as np
import pytest

from train_maritime_v3 import MaritimeChannel

TODO = pytest.mark.skip(reason="TODO: belum ditulis")


def _sinyal(benih=1):
    return np.random.default_rng(benih).standard_normal((2, 1024)).astype(np.float32)


# =============================================================================
# BLACKBOX
# =============================================================================
@pytest.mark.parametrize("hs", [0.3, 5.5])        # partisi: laut tenang & badai
def test_bb_keluaran_ternormalisasi(hs):
    """CONTOH. Apa pun kondisi lautnya, keluaran [2, 1024] float32 dengan
    simpangan baku magnitudo = 1 (normalisasi akhir simulator)."""
    ch = MaritimeChannel(rng=np.random.default_rng(0))
    y = ch.apply(_sinyal(), dist_km=30.0, hs_m=hs, sst_c=28.0, snr_db=20.0,
                 duct_h_m=10.0, n_interferers=2)
    assert y.shape == (2, 1024) and y.dtype == np.float32
    assert abs(np.std(np.abs(y[0] + 1j * y[1])) - 1.0) < 1e-4


@TODO
def test_bb_benih_sama_hasil_sama():
    """Dua kanal dengan benih rng sama + masukan sama -> keluaran identik."""


@TODO
def test_bb_snr_batas():
    """Nilai batas SNR yang dipakai latihan: -6 dan 30 dB. Apa yang bisa diukur
    dari luar yang membedakan keduanya?"""


@TODO
def test_bb_jarak_nol_atau_negatif():
    """Masukan janggal: dist_km = 0 atau negatif. Apakah tetap menghasilkan
    keluaran sah (tanpa NaN/inf)?"""


@TODO
def test_bb_semua_komponen_dimatikan():
    """duct=False, interference=False, dua_sinar=False, fasa_global=False:
    sifat apa yang harus tetap berlaku?"""


# =============================================================================
# WHITEBOX
# =============================================================================
@pytest.mark.parametrize("hs, ada_doppler", [(0.59, False), (0.6, True)])
def test_wb_cabang_doppler_batas_hs(hs, ada_doppler):
    """CONTOH. Cabang `if hs_m >= 0.6` (Doppler gelombang). Nilai batas di
    sekitar 0,6 membuktikan cabang diambil tepat mulai 0,6."""
    ch = MaritimeChannel(rng=np.random.default_rng(0))
    ch.apply(_sinyal(), dist_km=30.0, hs_m=hs, sst_c=28.0, snr_db=20.0,
             duct_h_m=10.0, n_interferers=0)
    assert (ch.terakhir["f_geser"] != 0.0) == ada_doppler


@TODO
def test_wb_cabang_dua_sinar_dimatikan():
    """Kondisi `rho > 0.01 and self.dua_sinar`: dengan dua_sinar=False,
    ch.terakhir['tau'] harus 0."""


@TODO
def test_wb_ambang_duct():
    """Cabang `duct > 0.05`: duct = duct_h_m / 25. Cari duct_h_m tepat di batas."""


@TODO
def test_wb_perulangan_interferer():
    """Perulangan `range(min(n_interferers, 6))`: uji perulangan dgn 0, 1, 6, dan 7
    (lebih dari batas). Bagaimana membuktikan jumlah iterasinya dari luar?"""


@TODO
def test_wb_cabang_balik_zf_pita():
    """Cabang `if self.balik and tau2`: MaritimeChannel(balik='zf-pita')."""


@TODO
def test_wb_duct_h_m_kosong():
    """Cabang `if duct_h_m is None` (pool lama tanpa kolom duct). Apa nilai
    penggantinya, dan apakah hasilnya sama dengan memberikan nilai itu langsung?"""


@TODO
def test_wb_masukan_nol_normalisasi():
    """Cabang `if std > 1e-8`: sinyal nol semua dengan SNR sangat tinggi.
    Apakah normalisasi menghindari pembagian dengan nol (tanpa NaN)?"""


@TODO
def test_wb_basis_path_apply():
    """Basis path MaritimeChannel.apply: graf alir, V(G), jalur independen.
    Fungsinya panjang -- boleh dipotong per blok asal disebut di laporan."""
