# -*- coding: utf-8 -*-
"""BAGIAN 6 — Kepala ketidakpastian (abstain) + demo web.

Pemilik : Hegira Musyafa Kartiwan
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



def test_bb_tiga_skor_urutan_sama():
    """Untuk logit 'yakin' vs 'ragu', ketiga skor (msp, entropi, energi) harus
    sepakat mana yang lebih yakin."""
    logit_yakin = torch.tensor([[50.0] + [0.0] * 23])
    logit_ragu = torch.zeros(1, 24)
    skor_yakin = [skor_msp(logit_yakin), skor_entropi(logit_yakin), skor_energi(logit_yakin)]
    skor_ragu = [ skor_msp(logit_ragu), skor_entropi(logit_ragu), skor_energi(logit_ragu)]
    assert all(sy > sr for sy, sr in zip(skor_yakin, skor_ragu))
    



def test_bb_aurc_sempurna_vs_terbalik():
    """Skor yang menaruh semua jawaban BENAR paling atas -> AURC lebih kecil
    daripada skor yang menaruh semua yang SALAH paling atas."""
    logit_sempurna = torch.tensor([[50.0] + [0.0] * 23, [0.0] * 24])
    logit_terbalik = torch.tensor([[0.0] * 24, [50.0] + [0.0] * 23])
    y = torch.tensor([0, 1])
    benar_sempurna = logit_sempurna.argmax(-1) == y
    benar_terbalik = logit_terbalik .argmax(-1) == y
    assert benar_sempurna.tolist() == [True, False]
    assert benar_terbalik.tolist() == [True, False]
    skor_sempurna = skor_msp(logit_sempurna).numpy()
    skor_terbalik = skor_msp(logit_terbalik).numpy()
    aurc_sempurna = aurc(skor_sempurna, benar_sempurna.numpy())
    aurc_terbalik = aurc(skor_terbalik, benar_terbalik.numpy())
    assert aurc_sempurna < aurc_terbalik
    



def test_bb_prediksi_abstain_bernilai_minus_satu():
    """Setelah pas(..., target_cakupan=0.5): kira-kira separuh prediksi = -1,
    sisanya kelas 0..23. Toleransi berapa yang wajar? Tulis alasanmu."""
    kepala = KepalaKetidakpastian("msp")
    logit, y = _logit_val()
    kepala.pas(logit, y, target_cakupan=0.5)
    prediksi = kepala.prediksi(logit)
    toleransi = 0.001 #karena yang akan diterima itu 0,499 - 0,501
    n = len(prediksi)
    cakupan = (prediksi != TIDAK_PASTI).sum().item() / n
    assert abs(cakupan - 0.5) < toleransi
    assert prediksi[prediksi != TIDAK_PASTI].min().item() >= 0
    assert prediksi[prediksi != TIDAK_PASTI].max().item() <= 23
    



def test_bb_jenis_skor_tak_dikenal():
    """KepalaKetidakpastian('ngawur') -> KeyError yang menyebut pilihan yang sah."""
    skor_tak_dikenal = "ngawur"
    pilihan_sah = list("msp entropi energi".split())
    jenis_skor = [f"'{j}'" for j in pilihan_sah]
    with pytest.raises(KeyError, match=skor_tak_dikenal) as info:
        KepalaKetidakpastian(skor_tak_dikenal)
    assert all(j in str(info.value) for j in jenis_skor)



def test_bb_target_risiko_mustahil():
    """target_risiko=0.0 pada data yang punya prediksi salah di urutan teratas:
    harus menolak SEMUA (ambang = inf, cakupan_val = 0), bukan diam-diam melanggar."""
    target_resiko = 0.0
    kepala = KepalaKetidakpastian("msp")
    logit_terbalik = torch.tensor([[0.0] * 24, [50.0] + [0.0] * 23])
    y = torch.tensor([0, 1])
    kepala.pas(logit_terbalik, y, target_risiko=target_resiko)
    cakupan_val = kepala.cakupan_val
    assert cakupan_val == 0.0
    #hasilnya tidak terdefinisikan namun tetap dipakai karena untuk mencatat perilaku nya
    assert kepala.risiko_val == 0.0
    assert kepala.ambang == float("inf")
    
    


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



def test_wb_pas_tanpa_kalibrasi():
    """Cabang `kalibrasi=False` -> self.T tetap 1.0."""
    logit, y = _logit_val()
    kepala = KepalaKetidakpastian("msp")
    kepala.pas(logit, y, target_cakupan=0.5, kalibrasi=False)
    assert kepala.T == 1.0
    kepala_kal = KepalaKetidakpastian("msp")
    kepala_kal.pas(logit, y, target_cakupan=0.5, kalibrasi=True)
    assert kepala_kal.T != 1.0


@pytest.mark.parametrize("target_cakupan", [1.0, 1.5])
def test_wb_pas_cabang_target_cakupan_batas(target_cakupan):
    """Cabang target_cakupan: `min(searchsorted(...), n - 1)`. Nilai batas
    target_cakupan = 1.0 (dan > 1.0) tak boleh menghasilkan indeks di luar array."""
    n = 200
    logit, y = _logit_val(n)
    kepala = KepalaKetidakpastian("msp")
    kepala.pas(logit, y, target_cakupan=target_cakupan)
    assert kepala.cakupan_val == pytest.approx(1.0)
    


def test_wb_aurc_cadangan_trapz(monkeypatch):
    """Cabang `hasattr(np, 'trapezoid')`: numpy lama vs baru. Kedua cabang dipaksa
    bergantian dengan monkeypatch, dan fungsi integralnya diganti "mata-mata" yang
    mencatat dirinya dipanggil: terbukti cabang MANA yang terambil, dan hasilnya sama."""
    # cakupan [1/4, 2/4, 3/4, 1], risiko [0, 1/2, 1/3, 1/4]
    # AURC = 1/4 * [(0 + 1/2) + (1/2 + 1/3) + (1/3 + 1/4)] / 2 = 23/96
    skor = np.array([0.9, 0.8, 0.3, 0.2])
    benar = np.array([1, 0, 1, 1])
    aurc_tangan = 23 / 96

    # Integral asli, apa pun versi numpy-nya. Pakai `or`, BUKAN getattr(np, "trapezoid",
    # np.trapz): argumen default dievaluasi lebih dulu, padahal np.trapz sudah tak ada
    # di numpy 2.x (jebakan yang sama dicatat di baris 88-89 modul).
    integral_asli = getattr(np, "trapezoid", None) or np.trapz
    terpanggil = []

    def mata_mata(nama):
        def integral(y, x):
            terpanggil.append(nama)
            return integral_asli(y, x)
        return integral

    # cabang `if`: numpy baru, np.trapezoid ada
    monkeypatch.setattr(np, "trapezoid", mata_mata("trapezoid"), raising=False)
    assert aurc(skor, benar) == pytest.approx(aurc_tangan)

    # cabang `else`: numpy lama, np.trapezoid belum ada tapi np.trapz ada
    monkeypatch.delattr(np, "trapezoid")
    monkeypatch.setattr(np, "trapz", mata_mata("trapz"), raising=False)
    assert aurc(skor, benar) == pytest.approx(aurc_tangan)

    assert terpanggil == ["trapezoid", "trapz"]       # tiap cabang terambil tepat sekali


# Basis path KepalaKetidakpastian.pas (marspec_ketidakpastian.py baris 119-160).
# Simpul keputusan: 1 (baris 126, dua target?), 3 (131, kalibrasi?), 6 (141,
# target_risiko?), 7 (146, ada titik risiko <= target?), 11 (150, k is None?).
# V(G) = E - N + 2 = 19 - 15 + 2 = 6  =  P + 1 = 5 + 1 = 6.
# Jalur J6 (simpul 10 -> 11 -> 12) MUSTAHIL: dari cabang target_cakupan, k selalu
# bilangan bulat, jadi `k is None` hanya bisa lahir dari simpul 9. Lima jalur J1-J5
# sudah melewati ke-19 sisi graf. Graf alir lengkap: lihat laporan bagian 6.
def _logit_terbalik():
    """2 sampel; yang paling yakin ([50, 0, ...]) justru SALAH (prediksi 0, label 1)."""
    return torch.tensor([[0.0] * 24, [50.0] + [0.0] * 23]), torch.tensor([0, 1])


@pytest.mark.parametrize("data, argumen, harapan", [
    pytest.param(_logit_val, dict(target_cakupan=0.5),
                 dict(T_satu=False, tolak_semua=False, cakupan_val=0.5),
                 id="J1-dasar_1-3-4-6-10-11-13-14"),
    pytest.param(_logit_val, dict(target_cakupan=0.5, kalibrasi=False),
                 dict(T_satu=True, tolak_semua=False, cakupan_val=0.5),
                 id="J3-tanpa_kalibrasi_1-3-5-6-10-11-13-14"),
    pytest.param(_logit_val, dict(target_risiko=0.1),
                 dict(T_satu=False, tolak_semua=False, cakupan_val=None),
                 id="J4-target_risiko_1-3-4-6-7-8-11-13-14"),
    pytest.param(_logit_terbalik, dict(target_risiko=0.0),
                 dict(T_satu=False, tolak_semua=True, cakupan_val=0.0),
                 id="J5-tolak_semua_1-3-4-6-7-9-11-12-14"),
])
def test_wb_basis_path_pas(data, argumen, harapan):
    """Basis path KepalaKetidakpastian.pas: satu kasus per jalur independen yang
    mungkin dilalui (J2 di fungsi terpisah karena keluar lewat galat)."""
    logit, y = data()
    kepala = KepalaKetidakpastian("msp").pas(logit, y, **argumen)

    assert (kepala.T == 1.0) == harapan["T_satu"]                     # simpul 3 -> 5 atau 4
    assert (kepala.ambang == float("inf")) == harapan["tolak_semua"]  # simpul 11 -> 12 atau 13
    if harapan["cakupan_val"] is not None:
        assert kepala.cakupan_val == pytest.approx(harapan["cakupan_val"])
    if "target_risiko" in argumen:                                      # janji cabang simpul 6 'ya'
        assert kepala.risiko_val <= argumen["target_risiko"]


def test_wb_basis_path_pas_J2():
    """Basis path J2 (1-2-K): dua target kosong -> ValueError di simpul 2, keluar
    SEBELUM simpul 3, jadi keadaan awal kepala (T = 1.0, ambang = -inf) masih utuh."""
    logit, y = _logit_val()
    kepala = KepalaKetidakpastian("msp")
    with pytest.raises(ValueError, match="tepat satu"):
        kepala.pas(logit, y)
    assert kepala.T == 1.0
    assert kepala.ambang == -np.inf
