# -*- coding: utf-8 -*-
"""BAGIAN 4 — Fungsi rugi & pemuat data: moe_balance_loss, equalizer_aux_loss,
scan_indices, load_X_inmemory.

Pemilik : [Anggota 4]
Kode    : marspec/train_maritime_v3.py
Catatan : fixture `h5_tiruan` membuat HDF5 mini (576 baris) di folder sementara,
          mengembalikan (path, kelas, snr). Tak perlu dataset 21 GB.
          Isi h5_tiruan: 24 kelas x 6 SNR (-10, -6, 0, 10, 20, 30) x 4 sampel,
          jadi setiap nilai SNR punya 24 x 4 = 96 baris.
"""
import numpy as np
import pytest
import torch

import train_maritime_v3 as tm
from train_maritime_v3 import equalizer_aux_loss, load_X_inmemory, moe_balance_loss, scan_indices

BARIS_PER_SNR = 96          # 24 kelas x 4 sampel per (kelas, SNR) di h5_tiruan
PENANDA_BLOK = "memuat IQ ke RAM"   # hanya dicetak jalur pindai-per-blok load_X_inmemory


def _x_asli(path):
    """Baca seluruh X [N, 1024, 2] dari HDF5 mini sebagai kebenaran dasar."""
    import h5py
    with h5py.File(path, "r") as f:
        return f["X"][:]


def _putar(iq, sudut):
    """Putar sinyal IQ [B, 2, N] dengan fasa global `sudut` (radian)."""
    c, s = float(np.cos(sudut)), float(np.sin(sudut))
    return torch.stack([iq[:, 0] * c - iq[:, 1] * s, iq[:, 0] * s + iq[:, 1] * c], dim=1)


# =============================================================================
# BLACKBOX
# =============================================================================
def test_bb_balance_gate_seragam_nol():
    """CONTOH. Partisi 'beban sudah seimbang': semua sampel membagi rata ke 4
    pakar -> rugi penyeimbang = 0."""
    gate = torch.full((8, 4), 0.25)
    assert moe_balance_loss(gate).item() == pytest.approx(0.0, abs=1e-9)


def test_bb_balance_satu_pakar_dominan():
    """Partisi 'beban timpang': semua bobot ke pakar 0. Rugi harus > 0, dan
    lebih besar daripada kasus timpang sebagian.

    Tiga partisi beban: timpang penuh > timpang sebagian > seimbang (= 0).
    Separuh sampel ke pakar 0, separuh seragam = timpang sebagian."""
    timpang_penuh = torch.zeros(8, 4)
    timpang_penuh[:, 0] = 1.0
    timpang_sebagian = torch.full((8, 4), 0.25)
    timpang_sebagian[:4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    seimbang = torch.full((8, 4), 0.25)

    r_penuh = moe_balance_loss(timpang_penuh).item()
    r_sebagian = moe_balance_loss(timpang_sebagian).item()
    r_seimbang = moe_balance_loss(seimbang).item()

    assert r_sebagian > 0
    assert r_penuh > r_sebagian > r_seimbang


def test_bb_eq_aux_identik_nol():
    """iq_eq == iq_clean, semua sampel ber-augmentasi -> rugi ~ 0.
    Nilai batas: equalizer sempurna adalah rugi terkecil yang mungkin."""
    iq = torch.randn(3, 2, 64)
    rugi = equalizer_aux_loss(iq.clone(), iq, torch.ones(3))
    assert rugi.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("sudut", [0.0, np.pi / 2, np.pi, 2.0287],
                         ids=["0", "pi/2", "pi", "acak_2.0287"])
def test_bb_eq_aux_kebal_rotasi_fasa(sudut):
    """Sifat yang dijanjikan docstring: memutar iq_eq dengan fasa global apa pun
    TIDAK mengubah rugi. Uji dengan beberapa sudut (partisi: 0, pi/2, pi, acak).

    Sudut 'acak' ditulis tetap supaya hasil uji bisa diulang. iq_eq diberi
    distorsi supaya rugi acuannya > 0 (rugi 0 tak membuktikan apa-apa)."""
    bersih = torch.randn(3, 2, 64)
    eq = bersih + 0.3 * torch.randn(3, 2, 64)
    aug = torch.ones(3)

    acuan = equalizer_aux_loss(eq, bersih, aug).item()
    diputar = equalizer_aux_loss(_putar(eq, sudut), bersih, aug).item()

    assert acuan > 0.01
    assert diputar == pytest.approx(acuan, rel=1e-4)


def test_bb_scan_indices_snr_min(h5_tiruan):
    """Semua snr yang dikembalikan >= snr_min. Nilai batas: snr_min tepat 0.

    Baris ber-SNR tepat 0 dB harus IKUT (>=, bukan >): 96 baris. Total yang lolos
    = SNR 0, 10, 20, 30 = 4 x 96 = 384. Label dan SNR harus cocok dengan baris aslinya."""
    path, kelas, snr = h5_tiruan
    idx, label, s = scan_indices(path, snr_min=0, max_samples=None)

    assert np.all(s >= 0)
    assert np.sum(s == 0) == BARIS_PER_SNR
    assert len(idx) == 4 * BARIS_PER_SNR
    np.testing.assert_array_equal(label, kelas[idx])
    np.testing.assert_array_equal(s, snr[idx])


def test_bb_scan_indices_max_samples(h5_tiruan):
    """max_samples=10 -> tepat 10 indeks, terurut naik, tanpa duplikat.
    (Naik tegas, diff > 0, sekaligus membuktikan tanpa duplikat.)"""
    path, kelas, _ = h5_tiruan
    idx, label, _ = scan_indices(path, snr_min=-10, max_samples=10)

    assert len(idx) == 10
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < len(kelas)
    np.testing.assert_array_equal(label, kelas[idx])


def test_bb_load_x_bentuk_dan_isi(h5_tiruan):
    """Hasil [n, 2, 1024] dan baris ke-i SAMA dengan X[indices[i]] di-transpose.
    Indeks dipilih di baris pertama, tengah, dan terakhir berkas."""
    path, _, _ = h5_tiruan
    idx = np.array([0, 7, 288, 575])
    X = load_X_inmemory(path, idx)
    asli = _x_asli(path)

    assert X.shape == (4, 2, 1024)
    assert X.dtype == np.float32
    for i, j in enumerate(idx):
        np.testing.assert_array_equal(X[i], asli[j].T)


def test_bb_load_x_indeks_duplikat(h5_tiruan):
    """Tebakan galat: indeks yang sama diminta dua kali. h5py menolak indeks
    berulang pada fancy indexing, jadi fungsi harus tetap memberi isi yang benar
    (lewat jalur blok), bukan galat atau baris kosong."""
    path, _, _ = h5_tiruan
    idx = np.array([3, 3, 250])
    X = load_X_inmemory(path, idx)
    asli = _x_asli(path)

    assert X.shape == (3, 2, 1024)
    for i, j in enumerate(idx):
        np.testing.assert_array_equal(X[i], asli[j].T)


@pytest.mark.xfail(strict=True, reason="BUG-03: jalur blok load_X_inmemory mengembalikan "
                                        "memori tak terinisialisasi untuk indeks di luar rentang")
@pytest.mark.parametrize("idx", [[600, 3], [5, -1]], ids=["600_dari_576", "minus_1"])
def test_bb_load_x_indeks_di_luar_rentang(h5_tiruan, idx):
    """Tebakan galat: indeks di luar [0, N). Jalur fancy (indeks terurut) sudah
    menolaknya dengan IndexError dari h5py, jadi jalur blok (indeks tak terurut)
    seharusnya juga menolak, bukan diam-diam mengisi baris itu dengan sisa memori
    (np.empty) lalu mengembalikannya seolah data."""
    path, _, _ = h5_tiruan
    with pytest.raises((IndexError, ValueError)):
        load_X_inmemory(path, idx)


# =============================================================================
# WHITEBOX
# =============================================================================
def test_wb_eq_aux_tanpa_sampel_aug():
    """CONTOH. Cabang `if mask.sum() == 0`: tak ada sampel augmentasi ->
    fungsi pulang lebih awal dengan tensor nol (bukan NaN dari rata-rata kosong)."""
    iq = torch.randn(3, 2, 64)
    rugi = equalizer_aux_loss(iq, torch.randn(3, 2, 64), torch.zeros(3))
    assert rugi.item() == 0.0


@pytest.mark.parametrize("idx", [[575, 3, 250, 99], [3, 3, 250]], ids=["tak_terurut", "duplikat"])
def test_wb_load_x_jalur_blok_indeks_acak(h5_tiruan, monkeypatch, capsys, idx):
    """load_X_inmemory punya DUA jalur: fancy indexing (n <= 100rb DAN terurut)
    vs pindai per blok. Indeks TAK terurut memaksa jalur blok. Kedua jalur harus
    memberi isi yang sama. Petunjuk: perkecil `blok` supaya perulangannya > 1 kali.

    blok=100 pada N=576 -> 6 putaran. 99|100 mengapit batas blok. Jalur blok
    dikenali dari cetakan kemajuannya; jalur fancy tidak mencetak apa pun."""
    monkeypatch.setattr(tm, "_ram_tersedia", lambda: None)
    path, _, _ = h5_tiruan
    idx = np.array(idx)
    asli = _x_asli(path)

    X_blok = load_X_inmemory(path, idx, blok=100)
    assert PENANDA_BLOK in capsys.readouterr().out
    for i, j in enumerate(idx):
        np.testing.assert_array_equal(X_blok[i], asli[j].T)

    # pembanding jalur fancy: indeks unik yang terurut
    unik = np.unique(idx)
    X_fancy = load_X_inmemory(path, unik)
    assert PENANDA_BLOK not in capsys.readouterr().out
    for i, j in enumerate(idx):
        np.testing.assert_array_equal(X_blok[i], X_fancy[np.searchsorted(unik, j)])


@pytest.mark.parametrize("sisa, ditolak", [(10_239, True), (10_240, False)])
def test_wb_load_x_ram_tak_cukup(h5_tiruan, monkeypatch, sisa, ditolak):
    """Cabang `perlu > 0.8 * sisa` -> SystemExit. Petunjuk:
    monkeypatch.setattr(tm, "_ram_tersedia", lambda: 1000).

    Nilai batas: 1 sampel butuh 2 x 1024 x 4 = 8192 byte. Ditolak bila
    8192 > 0,8 x sisa, jadi batasnya sisa = 8192 / 0,8 = 10240 byte:
    10239 ditolak, 10240 masih boleh (pembandingnya '>' tegas)."""
    monkeypatch.setattr(tm, "_ram_tersedia", lambda: sisa)
    path, _, _ = h5_tiruan
    if ditolak:
        with pytest.raises(SystemExit, match="RAM tak cukup"):
            load_X_inmemory(path, [5])
    else:
        assert load_X_inmemory(path, [5]).shape == (1, 2, 1024)


def test_wb_load_x_ram_tak_terbaca(h5_tiruan, monkeypatch):
    """Cabang `sisa is None` (Windows / tanpa /proc/meminfo): pemeriksaan RAM
    dilewati, pemuatan tetap jalan.

    Pembanding: permintaan yang SAMA ditolak bila RAM terbaca tapi kecil. Jadi
    lolosnya kasus None memang karena pemeriksaannya dilewati, bukan karena lulus."""
    path, _, _ = h5_tiruan
    idx = np.arange(0, 576, 5)

    monkeypatch.setattr(tm, "_ram_tersedia", lambda: None)
    assert load_X_inmemory(path, idx).shape == (len(idx), 2, 1024)

    monkeypatch.setattr(tm, "_ram_tersedia", lambda: 1000)
    with pytest.raises(SystemExit):
        load_X_inmemory(path, idx)


@pytest.mark.parametrize("max_samples, n_harap",
                         [(None, 192), (0, 192), (10, 10), (192, 192), (1000, 192)],
                         ids=["None", "nol", "lebih_kecil", "sama_persis", "lebih_besar"])
def test_wb_scan_indices_cabang_max_samples(h5_tiruan, max_samples, n_harap):
    """Kondisi `max_samples and len(keep) > max_samples`: rancang kasus untuk
    max_samples None, 0, lebih kecil, dan lebih besar dari jumlah yang lolos.

    snr_min=20 -> SNR 20 dan 30 lolos = 2 x 96 = 192 baris.
    - None dan 0: operan pertama salah -> tak dipotong. 0 berarti 'pakai SEMUA'
      sesuai teks bantuan --max_samples di main() ("0 = use ALL").
    - 192 (sama persis): operan kedua salah ('>' tegas) -> tak dipotong."""
    path, _, snr = h5_tiruan
    idx, _, _ = scan_indices(path, snr_min=20, max_samples=max_samples)
    assert len(idx) == n_harap
    assert np.all(snr[idx] >= 20)


# --- basis path: load_X_inmemory ---------------------------------------------
# Kondisi sederhana (kondisi majemuk dipecah, sesuai docs/templat_basis_path.md):
#   K1 sisa is not None     K2 perlu > 0.8*sisa     K3 n <= 100_000
#   K4 n < 2                K5 indeks naik tegas    (K4 or K5 = urut_ok)
#   K6 for a in range(...)  K7 hi > lo
#   K8 b/N >= laporan+0.1   K9 b == N               (K8 or K9 = cetak kemajuan)
# V(G) = 9 + 1 = 10. Jalur J1 (dasar) lalu tiap jalur membalik satu keputusan.
# Tidak diuji: J8 n > 100_000 (layak, tapi keluaran 100.001 x 8 KB = 819 MB RAM)
#              J9 perulangan nol kali (butuh berkas X kosong; indeks apa pun ke
#                 berkas kosong tidak sah, lihat BUG-03).
_JALUR_LOAD_X = [
    # id                 sisa RAM   indeks                         blok    jalur        cetakan
    ("J1_dasar_fancy",    None,     [0, 7, 288, 575],              65536, "fancy",      None),
    ("J2_ram_cukup",      10**12,   [0, 7, 288, 575],              65536, "fancy",      None),
    ("J3_ram_kurang",     1,        [0, 7, 288, 575],              65536, "SystemExit", None),
    ("J4_satu_indeks",    None,     [42],                          65536, "fancy",      None),
    ("J5_tak_terurut",    None,     [575, 3, 250, 120, 333, 450],  100,   "blok",       [1, 2, 3, 4, 5, 6]),
    ("J6_blok_tanpa_idx", None,     [3, 550, 20],                  100,   "blok",       [2, 2, 2, 2, 2, 3]),
    ("J7_cetak_jarang",   None,     [575, 3, 250, 120, 333, 450],  10,    "blok",       10),
]


@pytest.mark.parametrize("sisa, idx, blok, jalur, cetakan",
                         [pytest.param(*r[1:], id=r[0]) for r in _JALUR_LOAD_X])
def test_wb_basis_path_load_x(h5_tiruan, monkeypatch, capsys, sisa, idx, blok, jalur, cetakan):
    """Satu kasus per jalur independen load_X_inmemory (lihat komentar di atas).

    Jalur yang diambil dibuktikan dari cetakan kemajuan:
    - fancy: tak mencetak apa pun;
    - J5: tiap blok berisi indeks (K7 benar) -> jumlah sampel naik di tiap baris;
    - J6: blok 100-499 kosong (K7 salah) -> jumlah sampel tertahan di 2;
    - J7: blok=10 -> cetakan dilewati bila kemajuan < 10% (K8 salah), dan baris
      terakhir dicetak karena b == N (K9 benar): total 10 baris, bukan 58."""
    monkeypatch.setattr(tm, "_ram_tersedia", lambda: sisa)
    path, _, _ = h5_tiruan
    idx = np.array(idx)

    if jalur == "SystemExit":
        with pytest.raises(SystemExit, match="RAM tak cukup"):
            load_X_inmemory(path, idx, blok=blok)
        return

    X = load_X_inmemory(path, idx, blok=blok)
    baris_cetak = [b for b in capsys.readouterr().out.splitlines() if PENANDA_BLOK in b]
    asli = _x_asli(path)
    for i, j in enumerate(idx):
        np.testing.assert_array_equal(X[i], asli[j].T)

    if jalur == "fancy":
        assert baris_cetak == []
    elif isinstance(cetakan, list):
        jumlah = [int(b.split(",")[1].split("/")[0]) for b in baris_cetak]
        assert jumlah == cetakan
    else:
        assert len(baris_cetak) == cetakan
        assert "100.0%" in baris_cetak[-1]
