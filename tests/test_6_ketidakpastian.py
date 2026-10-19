# -*- coding: utf-8 -*-
"""BAGIAN 6 — Kepala ketidakpastian (abstain) + demo web.

Pemilik : [Anggota 6]
Kode    : marspec/marspec_ketidakpastian.py
          demo_web/index.html  -> diuji BLACKBOX secara manual, lihat docs/uji_demo_web.md
Konvensi: semua skor "makin besar makin yakin"; prediksi -1 = TIDAK_PASTI (abstain).
"""
import numpy as np
import pytest
import torch

from marspec_ketidakpastian import (TIDAK_PASTI, KepalaKetidakpastian, aurc,  # noqa: F401
                                    kurva_risiko_cakupan, skor_energi, skor_entropi, skor_msp)

TODO = pytest.mark.skip(reason="TODO: belum ditulis")


def _logit_val(n=200, benih=0):
    """Logit tiruan 24 kelas + label; sekitar separuhnya diprediksi benar."""
    g = torch.Generator().manual_seed(benih)
    y = torch.randint(0, 24, (n,), generator=g)
    logit = torch.randn(n, 24, generator=g)
    logit[torch.arange(n // 2), y[: n // 2]] += 4.0       # separuh pertama dibuat yakin & benar
    return logit, y


# =============================================================================
# BLACKBOX
# =============================================================================
def test_bb_skor_msp_rentang():
    """CONTOH. Nilai batas MSP pada 24 kelas: logit rata -> tepat 1/24 (paling
    ragu); satu logit sangat besar -> mendekati 1 (paling yakin)."""
    rata = skor_msp(torch.zeros(1, 24))
    yakin = skor_msp(torch.tensor([[50.0] + [0.0] * 23]))
    assert rata.item() == pytest.approx(1 / 24)
    assert yakin.item() == pytest.approx(1.0, abs=1e-6)


@TODO
def test_bb_tiga_skor_urutan_sama():
    """Untuk logit 'yakin' vs 'ragu', ketiga skor (msp, entropi, energi) harus
    sepakat mana yang lebih yakin."""


@TODO
def test_bb_aurc_sempurna_vs_terbalik():
    """Skor yang menaruh semua jawaban BENAR paling atas -> AURC lebih kecil
    daripada skor yang menaruh semua yang SALAH paling atas."""


@TODO
def test_bb_prediksi_abstain_bernilai_minus_satu():
    """Setelah pas(..., target_cakupan=0.5): kira-kira separuh prediksi = -1,
    sisanya kelas 0..23. Toleransi berapa yang wajar? Tulis alasanmu."""


@TODO
def test_bb_jenis_skor_tak_dikenal():
    """KepalaKetidakpastian('ngawur') -> KeyError yang menyebut pilihan yang sah."""


@TODO
def test_bb_target_risiko_mustahil():
    """target_risiko=0.0 pada data yang punya prediksi salah di urutan teratas:
    harus menolak SEMUA (ambang = inf, cakupan_val = 0), bukan diam-diam melanggar."""


# =============================================================================
# WHITEBOX
# =============================================================================
@pytest.mark.parametrize("argumen", [{}, {"target_risiko": 0.1, "target_cakupan": 0.5}])
def test_wb_pas_wajib_tepat_satu_target(argumen):
    """CONTOH. Kondisi `(target_risiko is None) == (target_cakupan is None)`
    benar di DUA situasi: keduanya kosong, atau keduanya diisi. Dua-duanya harus
    ValueError."""
    logit, y = _logit_val()
    with pytest.raises(ValueError, match="tepat satu"):
        KepalaKetidakpastian("msp").pas(logit, y, **argumen)


@TODO
def test_wb_pas_tanpa_kalibrasi():
    """Cabang `kalibrasi=False` -> self.T tetap 1.0."""


@TODO
def test_wb_pas_cabang_target_cakupan_batas():
    """Cabang target_cakupan: `min(searchsorted(...), n - 1)`. Nilai batas
    target_cakupan = 1.0 (dan > 1.0) tak boleh menghasilkan indeks di luar array."""


@TODO
def test_wb_aurc_cadangan_trapz(monkeypatch):
    """Cabang `hasattr(np, 'trapezoid')`: numpy lama vs baru. Bagaimana menguji
    cabang yang tak terambil di numpy yang terpasang? (petunjuk: monkeypatch.delattr)"""


@TODO
def test_wb_basis_path_pas():
    """Basis path KepalaKetidakpastian.pas: graf alir, V(G), jalur independen."""
