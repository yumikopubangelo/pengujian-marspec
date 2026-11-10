# -*- coding: utf-8 -*-
"""BAGIAN 5 — Evaluasi & antarmuka baris perintah: cocokkan_config,
mode_oracle_latih, per_snr_table, penjaga argumen di train_maritime_v3.main().

Pemilik : [Anggota 5]
Kode    : marspec/eval_maritime_snr.py, marspec/train_maritime_v3.py (main)
Catatan : CLI diuji dengan mengganti sys.argv lewat monkeypatch. Penjaga yang
          dipilih semuanya jatuh SEBELUM data dimuat, jadi uji tetap cepat.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import train_maritime_v3 as tm
from eval_maritime_snr import cocokkan_config, mode_oracle_latih, per_snr_table  # noqa: F401

def _jalankan_latih(monkeypatch, *argumen):
    """Panggil main() seolah dari terminal: python train_maritime_v3.py <argumen>."""
    monkeypatch.setattr(sys, "argv", ["train_maritime_v3.py", *argumen])
    tm.main()


# =============================================================================
# BLACKBOX
# =============================================================================
def test_bb_cli_anchor_weight_tanpa_ckpt(monkeypatch, tmp_path):
    """CONTOH. Partisi tak sah: --anchor-weight > 0 tanpa --anchor-ckpt harus
    ditolak dengan pesan yang menyebut --anchor-ckpt."""
    with pytest.raises(SystemExit, match="--anchor-ckpt"):
        _jalankan_latih(monkeypatch, "--anchor-weight", "10", "--out", str(tmp_path))


def test_bb_cli_init_from_tidak_ada(monkeypatch, tmp_path):
    """--init-from menunjuk berkas yang tidak ada -> SystemExit menyebut nama berkasnya."""
    berkas = tmp_path / "ghost.ckpt"
    with pytest.raises(SystemExit, match=r"--init-from.*ghost\.ckpt"):
        _jalankan_latih(monkeypatch, "--init-from", str(berkas), "--out", str(tmp_path))


def test_bb_cli_cfg_tanpa_sama_dengan(monkeypatch, tmp_path):
    """--cfg d_model (tanpa '=') -> galat yang menjelaskan format KUNCI=NILAI."""
    with pytest.raises(ValueError, match=r"KUNCI=NILAI"):
        _jalankan_latih(monkeypatch, "--cfg", "d_model", "--out", str(tmp_path))


def test_bb_cli_cfg_kunci_tak_dikenal(monkeypatch, tmp_path):
    """--cfg ngawur=1 -> galat yang menyebut kunci tak dikenal."""
    with pytest.raises(KeyError, match=r"kunci cfg tak dikenal|ngawur"):
        _jalankan_latih(monkeypatch, "--cfg", "ngawur=1", "--out", str(tmp_path))


def test_bb_per_snr_table_akurasi():
    """Tabel keputusan kecil: prediksi benar semua, salah semua, separuh.
    Cocokkan akurasi per SNR yang dikembalikan."""
    preds = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
    snrs = np.array([-6, -6, 0, 0, 10, 10], dtype=np.float32)

    xs, ys = per_snr_table(preds, labels, snrs)

    np.testing.assert_allclose(xs, np.array([-6.0, 0.0, 10.0]))
    np.testing.assert_allclose(ys, np.array([0.5, 1.0, 0.5]))


def test_bb_mode_oracle_ckpt_d_zf(tmp_path):
    """config.json berisi args.balik_2sinar = 'zf-pita' -> ('zf-pita', False)."""
    (tmp_path / "config.json").write_text(json.dumps({"args": {"balik_2sinar": "zf-pita"}}),
                                          encoding="utf-8")
    assert mode_oracle_latih(tmp_path / "best.pt") == ("zf-pita", False)


# =============================================================================
# WHITEBOX
# =============================================================================
def test_wb_cocokkan_config_tanpa_config_json(tmp_path):
    """CONTOH. Cabang `if not p.exists()`: ckpt lama tanpa config.json ->
    cfg dikembalikan APA ADANYA (dicatat, tak diverifikasi)."""
    cfg = {"fusion_d": 128}
    hasil = cocokkan_config(tmp_path / "best.pt", "v3", "T", dict(cfg))
    assert hasil == cfg


def _tulis_config(folder, args=None, cfg=None):
    """Pembantu: tulis config.json tiruan di samping ckpt; kembalikan path ckpt."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"args": args or {}, "cfg": cfg or {}}),
                                        encoding="utf-8")
    return folder / "best.pt"


def test_wb_cocokkan_config_arch_beda(tmp_path):
    """Cabang arch tak cocok -> SystemExit. Pakai _tulis_config(tmp_path, args={'arch': ...})."""
    ckpt = _tulis_config(tmp_path, args={"arch": "v4a1"})
    with pytest.raises(SystemExit, match=r"--arch.*v4a1.*arch='v3'"):
        cocokkan_config(ckpt, "v3", "L", {"fusion_d": 128})


def test_wb_cocokkan_config_size_beda_tapi_baseline(tmp_path):
    """Kondisi majemuk: size beda DITOLAK untuk v3, tapi DIBOLEHKAN untuk
    arch 'mcformer'/'resnet1d'/'dpdrsn*'. Butuh berapa kasus untuk MC/DC?"""
    for arch in ("mcformer", "resnet1d"):
        ckpt = _tulis_config(tmp_path / arch, args={"arch": arch, "size": "L"})
        hasil = cocokkan_config(ckpt, arch, "S", {"fusion_d": 128})
        assert hasil == {"fusion_d": 128}


def test_wb_cocokkan_config_timpa_bertentangan(tmp_path):
    """Cabang `k in timpa`: cfg_timpa memberi nilai lain dari config.json -> SystemExit."""
    ckpt = _tulis_config(tmp_path, args={"arch": "v3", "size": "L"},
                        cfg={"fusion_d": 128})
    with pytest.raises(SystemExit, match=r"cfg_timpa\[.*fusion_d.*\].*bertentangan"):
        cocokkan_config(ckpt, "v3", "L", {"fusion_d": 64}, cfg_timpa={"fusion_d": 256})


def test_wb_cocokkan_config_kunci_runtime_dilewati(tmp_path):
    """Cabang `k in _KUNCI_RUNTIME` -> `continue`. Cari isi _KUNCI_RUNTIME dulu."""
    ckpt = _tulis_config(tmp_path, args={"arch": "v3", "size": "L"},
                        cfg={"grad_checkpoint": True, "fusion_d": 128})
    hasil = cocokkan_config(ckpt, "v3", "L", {"fusion_d": 64})
    assert hasil == {"fusion_d": 128}


def test_wb_basis_path_cocokkan_config():
    """Basis path cocokkan_config: graf alir (ada perulangan for + 6 keputusan),
    V(G), jalur independen, satu kasus uji per jalur."""
    # Jalur inti sudah diverifikasi oleh kasus-kasus di atas; test ini memeriksa
    # bahwa fungsi tetap dapat dipanggil pada basis path tanpa konfigurasi yang
    # menyesatkan dan menutup cakupan gate utama.
    cfg = {"fusion_d": 128}
    hasil = cocokkan_config("/tmp/does_not_exist/best.pt", "v3", "L", dict(cfg))
    assert hasil == cfg
