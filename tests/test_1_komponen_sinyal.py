# -*- coding: utf-8 -*-
"""BAGIAN 1 — Komponen sinyal: SE1d, ChannelEqualizer, CNN1DExtractor, MCformerEncoder.

Pemilik : [Anggota 1]
Kode    : marspec/maritime_amc_v3_arch.py
Dua contoh di bawah SUDAH JALAN sebagai pola. Ganti setiap uji ber-@TODO dengan
uji sungguhan (hapus @TODO-nya), lalu tambah uji lain sampai target tercapai
(lihat PEMBAGIAN_TUGAS.md).
"""
import pytest
import torch

from maritime_amc_v3_arch import CNN1DExtractor, ChannelEqualizer, MCformerEncoder, SE1d


# =============================================================================
# BLACKBOX — hanya masukan & keluaran; jangan bergantung pada isi kode
# =============================================================================
@pytest.mark.parametrize("B", [1, 8])            # nilai batas: batch terkecil & batch biasa
def test_bb_equalizer_mempertahankan_bentuk(B):
    """CONTOH. Teknik: nilai batas pada ukuran batch.
    Harapan: keluaran equalizer berbentuk SAMA dengan masukannya."""
    m = ChannelEqualizer(n_filters=16, n_layers=4, ctx_dim=8).eval()
    x = torch.randn(B, 2, 1024)
    with torch.no_grad():
        y = m(x, torch.rand(B, 8))
    assert y.shape == x.shape


def test_bb_se1d_bentuk_dan_rentang_bobot():
    """Partisi ekivalen. SE1d(c, r): keluaran berbentuk sama dengan masukan,
    dan skala per kanal dibatasi pada rentang (0, 1) untuk sinyal non-negatif."""
    m = SE1d(c=8, r=4).eval()
    x = torch.rand(2, 8, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.all((y >= 0) & (y <= 1))


@pytest.mark.parametrize("B", [1, 8])
def test_bb_cnn_keluaran_out_dim(B):
    """CNN1DExtractor((16, 32, 64), out_dim=32, pool=4) pada [B, 2, 1024]
    harus menghasilkan [B, 32]. Coba juga B = 1."""
    m = CNN1DExtractor((16, 32, 64), out_dim=32, pool=4).eval()
    x = torch.randn(B, 2, 1024)
    y = m(x)
    assert y.shape == (B, 32)


def test_bb_mcformer_deterministik_saat_eval():
    """Dua kali forward pada masukan yang sama dalam mode eval -> hasil identik."""
    m = MCformerEncoder(d_model=32, nhead=4, nlayers=2, patch_len=32, out_dim=16).eval()
    x = torch.randn(2, 2, 1024)
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x)
    assert torch.allclose(y1, y2, rtol=1e-5, atol=1e-6)


def test_bb_mcformer_masukan_nan():
    """Masukan NaN menyebar ke keluaran, bukan dibersihkan secara diam-diam."""
    m = MCformerEncoder(d_model=32, nhead=4, nlayers=2, patch_len=32, out_dim=16).eval()
    x = torch.randn(2, 2, 1024)
    x[0, 0, 0] = float("nan")
    with torch.no_grad():
        y = m(x)
    assert torch.isnan(y).any()


def test_bb_mcformer_panjang_masukan_salah():
    """Masukan [B, 2, 1000] (bukan 1024) harus ditolak dengan galat yang jelas."""
    m = MCformerEncoder(d_model=32, nhead=4, nlayers=2, patch_len=32, out_dim=16).eval()
    x = torch.randn(2, 2, 1000)
    with pytest.raises((RuntimeError, ValueError)):
        m(x)


# =============================================================================
# WHITEBOX — kasus uji dirancang dari STRUKTUR kode (cabang, jalur, perulangan)
# =============================================================================
def test_wb_equalizer_cabang_film(iq_acak):
    """CONTOH. Kondisi di ChannelEqualizer.forward:
        film is not None AND ctx_feat is not None AND i == film_at
    ctx diberikan -> cabang FiLM DIAMBIL; ctx None -> cabang DILEWATI.
    Dua kasus ini menutup kedua arah cabang (branch coverage)."""
    m = ChannelEqualizer(16, 4, ctx_dim=8).eval()
    with torch.no_grad():
        dengan_film = m(iq_acak, torch.rand(4, 8))
        tanpa_film = m(iq_acak, None)
    assert not torch.allclose(dengan_film, tanpa_film)


def test_wb_equalizer_tanpa_film_ctx_dim_nol():
    """Cabang `film is None` (ctx_dim=0): konteks apa pun tak boleh mengubah keluaran."""
    m = ChannelEqualizer(16, 4, ctx_dim=0).eval()
    x = torch.randn(4, 2, 1024)
    ctx = torch.rand(4, 8)
    with torch.no_grad():
        y_ctx = m(x, ctx)
        y_none = m(x, None)
    assert torch.allclose(y_ctx, y_none, rtol=1e-5, atol=1e-6)


def test_wb_mcformer_layerdrop_hanya_saat_latih():
    """Kondisi `self.training and self.drop_path > 0 and rand < drop_path`.
    drop_path=1.0 membuat lapis SELALU dilewati saat latih; saat eval tidak pernah."""
    x = torch.randn(2, 2, 1024)
    m = MCformerEncoder(d_model=32, nhead=4, nlayers=2, patch_len=32, out_dim=16, drop_path=1.0)
    m.train()
    with torch.no_grad():
        y_train = m(x)
    m.eval()
    with torch.no_grad():
        y_eval = m(x)
    assert y_train.shape == y_eval.shape
    assert not torch.allclose(y_train, y_eval, rtol=1e-5, atol=1e-6)


def test_wb_mcformer_cabang_grad_checkpoint():
    """Cabang `grad_checkpoint and training`: hasil harus sama dengan tanpa checkpoint
    ketika acak dropout di-reset sebelum setiap forward."""
    x = torch.randn(3, 2, 1024)
    m1 = MCformerEncoder(d_model=32, nhead=4, nlayers=2, patch_len=32, out_dim=16,
                        grad_checkpoint=True, drop_path=0.0).train()
    m2 = MCformerEncoder(d_model=32, nhead=4, nlayers=2, patch_len=32, out_dim=16,
                        grad_checkpoint=False, drop_path=0.0).train()
    m2.load_state_dict(m1.state_dict())
    torch.manual_seed(0)
    with torch.no_grad():
        y1 = m1(x)
    torch.manual_seed(0)
    with torch.no_grad():
        y2 = m2(x)
    assert torch.allclose(y1, y2, rtol=1e-5, atol=1e-6)


def test_wb_basis_path_equalizer_forward():
    """Basis path ChannelEqualizer.forward: branch FiLM aktif hanya saat ctx valid.
    Kasus ini menutup jalur dengan film aktif dan tanpa film."""
    m = ChannelEqualizer(16, 4, ctx_dim=8).eval()
    x = torch.randn(2, 2, 1024)
    ctx = torch.rand(2, 8)
    with torch.no_grad():
        y_film = m(x, ctx)
        y_none = m(x, None)
    assert y_film.shape == x.shape
    assert torch.all(torch.isfinite(y_film))
    assert not torch.allclose(y_film, y_none, rtol=1e-5, atol=1e-6)
