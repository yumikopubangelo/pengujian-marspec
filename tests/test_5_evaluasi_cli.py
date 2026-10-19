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

import numpy as np
import pytest

import train_maritime_v3 as tm
from eval_maritime_snr import cocokkan_config, mode_oracle_latih, per_snr_table  # noqa: F401

TODO = pytest.mark.skip(reason="TODO: belum ditulis")


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


@TODO
def test_bb_cli_init_from_tidak_ada(monkeypatch, tmp_path):
    """--init-from menunjuk berkas yang tidak ada -> SystemExit menyebut nama berkasnya."""


@TODO
def test_bb_cli_cfg_tanpa_sama_dengan(monkeypatch, tmp_path):
    """--cfg d_model (tanpa '=') -> galat yang menjelaskan format KUNCI=NILAI."""


@TODO
def test_bb_cli_cfg_kunci_tak_dikenal(monkeypatch, tmp_path):
    """--cfg ngawur=1 -> galat yang menyebut kunci tak dikenal."""


@TODO
def test_bb_per_snr_table_akurasi():
    """Tabel keputusan kecil: prediksi benar semua, salah semua, separuh.
    Cocokkan akurasi per SNR yang dikembalikan."""


@TODO
def test_bb_mode_oracle_ckpt_d_zf(tmp_path):
    """config.json berisi args.balik_2sinar = 'zf-pita' -> ('zf-pita', False)."""


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
    (folder / "config.json").write_text(json.dumps({"args": args or {}, "cfg": cfg or {}}),
                                        encoding="utf-8")
    return folder / "best.pt"


@TODO
def test_wb_cocokkan_config_arch_beda(tmp_path):
    """Cabang arch tak cocok -> SystemExit. Pakai _tulis_config(tmp_path, args={'arch': ...})."""


@TODO
def test_wb_cocokkan_config_size_beda_tapi_baseline(tmp_path):
    """Kondisi majemuk: size beda DITOLAK untuk v3, tapi DIBOLEHKAN untuk
    arch 'mcformer'/'resnet1d'/'dpdrsn*'. Butuh berapa kasus untuk MC/DC?"""


@TODO
def test_wb_cocokkan_config_timpa_bertentangan(tmp_path):
    """Cabang `k in timpa`: cfg_timpa memberi nilai lain dari config.json -> SystemExit."""


@TODO
def test_wb_cocokkan_config_kunci_runtime_dilewati(tmp_path):
    """Cabang `k in _KUNCI_RUNTIME` -> `continue`. Cari isi _KUNCI_RUNTIME dulu."""


@TODO
def test_wb_basis_path_cocokkan_config():
    """Basis path cocokkan_config: graf alir (ada perulangan for + 6 keputusan),
    V(G), jalur independen, satu kasus uji per jalur."""
