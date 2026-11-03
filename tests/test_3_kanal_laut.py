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


def test_bb_benih_sama_hasil_sama():
    """Partisi ekivalen: benih sama menghasilkan keluaran yang identik."""
    iq = _sinyal()

    ch1 = MaritimeChannel(rng=np.random.default_rng(123))
    ch2 = MaritimeChannel(rng=np.random.default_rng(123))

    y1 = ch1.apply(
        iq, dist_km=30.0, hs_m=0.3, sst_c=28.0,
        snr_db=20.0, duct_h_m=10.0, n_interferers=2
    )
    y2 = ch2.apply(
        iq, dist_km=30.0, hs_m=0.3, sst_c=28.0,
        snr_db=20.0, duct_h_m=10.0, n_interferers=2
    )

    assert np.array_equal(y1, y2)


@pytest.mark.parametrize("snr_db", [-6.0, 30.0])
def test_bb_snr_batas(snr_db):
    """Nilai batas: SNR rendah dan tinggi menghasilkan keluaran yang valid."""
    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        duct=False,
        interference=False,
        dua_sinar=False,
        fasa_global=False
    )

    y = ch.apply(
        _sinyal(), dist_km=30.0, hs_m=0.3,
        sst_c=28.0, snr_db=snr_db,
        duct_h_m=0.0, n_interferers=0
    )

    assert y.shape == (2, 1024)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("dist_km", [0.0, -5.0])
def test_bb_jarak_nol_atau_negatif(dist_km):
    """Tebakan galat: jarak nol dan negatif tetap menghasilkan keluaran valid."""
    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        duct=False,
        interference=False,
        dua_sinar=False,
        fasa_global=False
    )

    y = ch.apply(
        _sinyal(), dist_km=dist_km, hs_m=0.3,
        sst_c=28.0, snr_db=20.0,
        duct_h_m=0.0, n_interferers=0
    )

    assert y.shape == (2, 1024)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


def test_bb_semua_komponen_dimatikan():
    """Partisi ekivalen: kanal tanpa komponen tambahan tetap menghasilkan keluaran valid."""
    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        duct=False,
        interference=False,
        dua_sinar=False,
        fasa_global=False
    )

    y = ch.apply(
        _sinyal(), dist_km=30.0, hs_m=0.3,
        sst_c=28.0, snr_db=20.0,
        duct_h_m=10.0, n_interferers=2
    )

    assert y.shape == (2, 1024)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert abs(np.std(np.abs(y[0] + 1j * y[1])) - 1.0) < 1e-4


@pytest.mark.parametrize("jumlah_sampel", [512, 2048])
def test_bb_panjang_sinyal(jumlah_sampel):
    """Partisi ekivalen: panjang sinyal berbeda tetap dipertahankan pada keluaran."""
    iq = np.random.default_rng(7).standard_normal(
        (2, jumlah_sampel)
    ).astype(np.float32)

    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        duct=False,
        interference=False,
        dua_sinar=False,
        fasa_global=False
    )

    y = ch.apply(
        iq, dist_km=30.0, hs_m=0.3,
        sst_c=28.0, snr_db=20.0,
        duct_h_m=0.0, n_interferers=0
    )

    assert y.shape == (2, jumlah_sampel)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


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



def test_wb_cabang_dua_sinar_dimatikan():
    """Pengujian cabang: dua_sinar=False membuat tau bernilai 0."""
    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        dua_sinar=False
    )

    ch.apply(
        _sinyal(), dist_km=30.0, hs_m=5.5,
        sst_c=28.0, snr_db=20.0,
        duct_h_m=10.0, n_interferers=0
    )

    assert ch.terakhir["tau"] == 0


@pytest.mark.parametrize(
    "n_interferers, jumlah_diharapkan",
    [(0, 0), (1, 1), (6, 6), (7, 6)]
)
def test_wb_perulangan_interferer(n_interferers, jumlah_diharapkan):
    """Pengujian batas perulangan: maksimal enam interferer diproses."""

    class RNGPenghitung:
        def __init__(self, benih):
            self.rng = np.random.default_rng(benih)
            self.jumlah_uniform = 0

        def uniform(self, *args, **kwargs):
            self.jumlah_uniform += 1
            return self.rng.uniform(*args, **kwargs)

        def integers(self, *args, **kwargs):
            return self.rng.integers(*args, **kwargs)

        def standard_normal(self, *args, **kwargs):
            return self.rng.standard_normal(*args, **kwargs)

    rng = RNGPenghitung(42)

    ch = MaritimeChannel(
        rng=rng,
        duct=False,
        interference=True,
        dua_sinar=False,
        fasa_global=False
    )

    ch.apply(
        _sinyal(), dist_km=30.0, hs_m=0.3,
        sst_c=28.0, snr_db=20.0,
        duct_h_m=0.0, n_interferers=n_interferers
    )

    assert rng.jumlah_uniform == jumlah_diharapkan * 3


def test_wb_cabang_balik_zf_pita():
    """Pengujian cabang: metode zf-pita dijalankan ketika dua sinar aktif."""

    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        balik="zf-pita",
        dua_sinar=True
    )

    y = ch.apply(
        _sinyal(),
        dist_km=30.0,
        hs_m=5.5,
        sst_c=28.0,
        snr_db=20.0,
        duct_h_m=0.0,
        n_interferers=0
    )

    assert ch._pita is not None
    assert len(ch._pita) == 1024
    assert y.shape == (2, 1024)
    assert np.all(np.isfinite(y))


def test_wb_duct_h_m_kosong():
    """Pengujian kondisi: duct_h_m None menggunakan nilai otomatis dari sst_c."""

    ch1 = MaritimeChannel(rng=np.random.default_rng(42))
    ch2 = MaritimeChannel(rng=np.random.default_rng(42))

    y_otomatis = ch1.apply(
        _sinyal(),
        dist_km=30.0,
        hs_m=0.3,
        sst_c=28.0,
        snr_db=20.0,
        duct_h_m=None,
        n_interferers=0
    )

    y_manual = ch2.apply(
        _sinyal(),
        dist_km=30.0,
        hs_m=0.3,
        sst_c=28.0,
        snr_db=20.0,
        duct_h_m=8.64,
        n_interferers=0
    )

    assert np.allclose(y_otomatis, y_manual)


def test_wb_masukan_nol_normalisasi():
    """Pengujian kondisi batas: normalisasi dilewati jika std <= 1e-8."""

    ch = MaritimeChannel(
        rng=np.random.default_rng(42),
        duct=False,
        interference=False,
        dua_sinar=False,
        fasa_global=False
    )

    sinyal_nol = np.zeros((2, 1024), dtype=np.float32)

    y = ch.apply(
        sinyal_nol,
        dist_km=30.0,
        hs_m=0.3,
        sst_c=28.0,
        snr_db=300.0,
        duct_h_m=0.0,
        n_interferers=0
    )

    assert y.shape == (2, 1024)
    assert np.all(np.isfinite(y))
    assert np.std(np.abs(y)) <= 1e-8


# =============================================================================
# WHITEBOX TEST — MaritimeChannel.apply()
# Pengujian jalur, percabangan, loop, dan cache
# =============================================================================

class _RNGJejak:
    """Mencatat pemanggilan RNG tanpa mengubah hasil undian."""

    def __init__(self, benih=42):
        self._rng = np.random.default_rng(benih)
        self.jejak = []

    def integers(self, *a, **k):
        self.jejak.append(("integers", a))
        return self._rng.integers(*a, **k)

    def uniform(self, *a, **k):
        self.jejak.append(("uniform", a))
        return self._rng.uniform(*a, **k)

    def standard_normal(self, *a, **k):
        self.jejak.append(("standard_normal", a))
        return self._rng.standard_normal(*a, **k)


_2PI = 2 * np.pi

_J_2SINAR = [
    ("integers", (1, 9)),
    ("uniform", (0, _2PI)),
]

_J_DUCT = [
    ("integers", (6, 24)),
    ("uniform", (0, _2PI)),
]

_J_DOPPLER = [("uniform", (-1, 1))]
_J_LO = [("uniform", (-20.0, 20.0))]

_J_INTF = [
    ("uniform", (-0.25, 0.25)),
    ("uniform", (0.5, 1.5)),
    ("uniform", (0, _2PI)),
]

_J_FASA = [("uniform", (0, _2PI))]
_J_NOISE = [("standard_normal", (1024,))] * 2


def _jejak(dua=True, duct=True, doppler=True,
           lo=False, n_intf=1, fasa=True):
    return (
        _J_2SINAR * dua
        + _J_DUCT * duct
        + _J_DOPPLER * doppler
        + _J_LO * lo
        + _J_INTF * n_intf
        + _J_FASA * fasa
        + _J_NOISE
    )


_KTOR_DASAR = dict(
    duct=True,
    interference=True,
    dua_sinar=True,
    fasa_global=True,
    balik="",
    lo_ppm=0.0,
)

_APPLY_DASAR = dict(
    dist_km=30.0,
    hs_m=5.5,
    sst_c=28.0,
    snr_db=20.0,
    duct_h_m=10.0,
    n_interferers=1,
)


_JALUR = [
    ("J1_dasar", {}, {}, _jejak(), True, True),
    ("J2_duct_otomatis", {}, dict(duct_h_m=None), _jejak(), True, True),
    ("J3_duct_dimatikan", dict(duct=False), {},
     _jejak(duct=False), True, True),
    ("J4_duct_nol", {}, dict(duct_h_m=0.0),
     _jejak(duct=False), True, True),
    ("J5_rho_kecil", dict(freq_mhz=3000.0), {},
     _jejak(dua=False), False, True),
    ("J6_dua_sinar_mati", dict(dua_sinar=False), {},
     _jejak(dua=False), False, True),
    ("J7_laut_tenang", {}, dict(hs_m=0.3),
     _jejak(doppler=False), True, False),
    ("J8_offset_lo", dict(lo_ppm=20.0), {},
     _jejak(lo=True), True, True),
    ("J9_interferensi_mati", dict(interference=False), {},
     _jejak(n_intf=0), True, True),
    ("J10_nol_interferer", {}, dict(n_interferers=0),
     _jejak(n_intf=0), True, True),
    ("J11_dua_interferer", {}, dict(n_interferers=2),
     _jejak(n_intf=2), True, True),
    ("J12_fasa_global_mati", dict(fasa_global=False), {},
     _jejak(fasa=False), True, True),
    ("J13_balik_tanpa_dua_sinar",
     dict(balik="zf-pita", dua_sinar=False), {},
     _jejak(dua=False), False, True),
]


@pytest.mark.parametrize(
    "ubah_k, ubah_a, jejak, tau_aktif, ada_geser",
    [pytest.param(*r[1:], id=r[0]) for r in _JALUR],
)
def test_wb_jalur_apply(
    ubah_k, ubah_a, jejak, tau_aktif, ada_geser
):
    """Menguji variasi jalur apply melalui jejak RNG dan hasil keluaran."""

    rng = _RNGJejak(42)

    ch = MaritimeChannel(
        rng=rng,
        **{**_KTOR_DASAR, **ubah_k}
    )

    y = ch.apply(
        _sinyal(),
        **{**_APPLY_DASAR, **ubah_a}
    )

    assert rng.jejak == jejak
    assert (ch.terakhir["tau"] != 0) == tau_aktif
    assert (ch.terakhir["f_geser"] != 0.0) == ada_geser

    assert y.shape == (2, 1024)
    assert y.dtype == np.float32


def test_wb_jalur_balik_pita_cache():
    """Menguji pembuatan, penggunaan ulang, dan pembuatan ulang cache pita."""

    ch = MaritimeChannel(
        rng=np.random.default_rng(0),
        balik="zf-pita",
    )

    ch.apply(_sinyal(), **_APPLY_DASAR)

    pita_awal = ch._pita
    assert pita_awal is not None
    assert len(pita_awal) == 1024
    assert ch.terakhir["tau"] != 0

    ch.apply(_sinyal(2), **_APPLY_DASAR)

    assert ch._pita is pita_awal

    iq512 = np.random.default_rng(7).standard_normal(
        (2, 512)
    ).astype(np.float32)

    y = ch.apply(iq512, **_APPLY_DASAR)

    assert ch._pita is not pita_awal
    assert len(ch._pita) == 512
    assert y.shape == (2, 512)
    assert np.all(np.isfinite(y))


def test_wb_jalur_std_nol():
    """Menguji kondisi standar deviasi kecil sehingga normalisasi dilewati."""

    rng = _RNGJejak(42)
    ch = MaritimeChannel(rng=rng)

    y = ch.apply(
        np.zeros((2, 1024), dtype=np.float32),
        **{**_APPLY_DASAR, "snr_db": 300.0},
    )

    assert rng.jejak == _jejak()

    mag = np.abs(y[0] + 1j * y[1])

    assert np.std(mag) < 1e-6
    assert 0.01 < np.mean(mag) < 0.2


def test_wb_jalur_keluaran_setara():
    """Membandingkan konfigurasi berbeda yang menghasilkan duct atau sinyal setara."""

    def jalankan(ktor, **ubah):
        ch = MaritimeChannel(
            rng=np.random.default_rng(5),
            **ktor,
        )

        return ch.apply(
            _sinyal(),
            **{**_APPLY_DASAR, **ubah},
        )

    assert np.array_equal(
        jalankan(dict(duct=False)),
        jalankan({}, duct_h_m=0.0),
    )

    assert np.array_equal(
        jalankan(dict(freq_mhz=3000.0)),
        jalankan(
            dict(freq_mhz=3000.0, dua_sinar=False)
        ),
    )
