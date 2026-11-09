# -*- coding: utf-8 -*-
"""BAGIAN 2 — Konteks & model utuh: pengujian Blackbox dan Whitebox."""
import pytest
import torch

from maritime_amc_v3_arch import (
    ContextEncoder,
    ContextGatedFusion,
    MaritimeAMCv3,
    MoEChannelRouter,
    validate_config,
)


# =============================================================================
# BLACKBOX
# =============================================================================
def test_bb_model_keluaran_tiga_tensor(cfg_kecil, iq_acak, ctx_acak):
    """Model utuh menghasilkan logits, IQ hasil equalizer, dan bobot gate."""
    model = MaritimeAMCv3(cfg_kecil).eval()
    with torch.no_grad():
        logits, iq_eq, gate = model(iq_acak, ctx_acak)
    assert logits.shape == (4, 24)
    assert iq_eq.shape == iq_acak.shape
    assert gate.shape == (4, cfg_kecil["n_experts"])
    assert torch.allclose(gate.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_bb_moe_gate_tak_bergantung_sinyal():
    """Gate hanya bergantung pada konteks, bukan sinyal x."""
    model = MoEChannelRouter(dim=64, ctx_dim=32, n_experts=4, hidden=64, dropout=0.0).eval()
    x1 = torch.randn(3, 64)
    x2 = torch.randn(3, 64) * 5
    context = torch.randn(3, 32)
    with torch.no_grad():
        _, gate1 = model(x1, context)
        _, gate2 = model(x2, context)
    assert torch.allclose(gate1, gate2, atol=1e-7)


def test_bb_context_encoder_bentuk():
    """ContextEncoder menerima [B,4] dan mengeluarkan [B,32]."""
    model = ContextEncoder(4, 64, 32, 3).eval()
    x = torch.randn(4, 4)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (4, 32)


def test_bb_fusion_bentuk_keluaran():
    """Fusion menerima signal [B,64] dan konteks [B,32], keluaran [B,64]."""
    model = ContextGatedFusion(64, 32, 64).eval()
    signal = torch.randn(4, 64)
    context = torch.randn(4, 32)
    with torch.no_grad():
        output = model(signal, context)
    assert output.shape == (4, 64)


def test_bb_model_batch_satu(cfg_kecil):
    """Nilai batas batch=1 tetap menghasilkan bentuk keluaran yang benar."""
    model = MaritimeAMCv3(cfg_kecil).eval()
    iq = torch.randn(1, 2, 1024)
    context = torch.rand(1, 4)
    with torch.no_grad():
        logits, iq_eq, gate = model(iq, context)
    assert logits.shape == (1, 24)
    assert iq_eq.shape == (1, 2, 1024)
    assert gate.shape == (1, cfg_kecil["n_experts"])
    assert torch.allclose(gate.sum(-1), torch.ones(1), atol=1e-5)


def test_bb_model_konteks_salah_ukuran(cfg_kecil):
    """Konteks dengan 3 fitur (seharusnya 4) tidak diterima model."""
    model = MaritimeAMCv3(cfg_kecil).eval()
    iq = torch.randn(2, 2, 1024)
    context_salah = torch.rand(2, 3)
    with pytest.raises((RuntimeError, ValueError)):
        model(iq, context_salah)


# =============================================================================
# WHITEBOX
# =============================================================================
def test_wb_validate_config_fusion_ganjil(cfg_kecil):
    """Cabang fusion_d ganjil menghasilkan ValueError yang menyebut fusion_d."""
    with pytest.raises(ValueError, match="fusion_d"):
        validate_config(dict(cfg_kecil, fusion_d=63))


def test_wb_basis_path_validate_config():
    """Basis paths validate_config.

    Ada 3 keputusan validasi parameter (d_model, fusion, patch) dan keputusan
    akhir `if errs`; V(G)=4. Basis: jalur normal tanpa error, lalu tiap kondisi
    validasi parameter menjadi true secara terpisah. Jalur yang menyalahi lebih
    dari satu parameter sekaligus bukan jalur basis tambahan yang diperlukan.
    """
    from copy import deepcopy
    from maritime_amc_v3_arch import CFG_V3

    valid = deepcopy(CFG_V3)
    assert validate_config(valid) is None

    # Jalur dengan pelanggaran d_model % nhead != 0.
    with pytest.raises(ValueError, match="mcf_d_model"):
        validate_config(dict(valid, mcf_d_model=33))

    # Jalur dengan pelanggaran fusion_d % 2 != 0.
    with pytest.raises(ValueError, match="fusion_d"):
        validate_config(dict(valid, fusion_d=383))

    # Jalur dengan pelanggaran 1024 % patch != 0.
    with pytest.raises(ValueError, match="mcf_patch_len"):
        validate_config(dict(valid, mcf_patch_len=30))


def test_wb_validate_config_dua_pelanggaran():
    """Dua pelanggaran sekaligus harus dicantumkan pada pesan ValueError."""
    from copy import deepcopy
    from maritime_amc_v3_arch import CFG_V3

    cfg = deepcopy(CFG_V3)
    cfg["fusion_d"] = 383
    cfg["mcf_patch_len"] = 30
    with pytest.raises(ValueError) as exc:
        validate_config(cfg)
    message = str(exc.value)
    assert "fusion_d" in message
    assert "mcf_patch_len" in message


def test_wb_fusion_rumus_film():
    """Saat gamma=-1 dan beta=0, masukan FiLM menjadi nol sebelum norm1."""
    model = ContextGatedFusion(64, 32, 64).eval()
    with torch.no_grad():
        model.film.weight.zero_()
        model.film.bias[:64].fill_(-1.0)  # gamma = -1
        model.film.bias[64:].zero_()      # beta = 0

    signal = torch.randn(2, 64)
    context = torch.randn(2, 32)
    captured = {}

    def capture_norm1_input(_module, inputs):
        captured["input"] = inputs[0].detach().clone()

    hook = model.norm1.register_forward_pre_hook(capture_norm1_input)
    try:
        with torch.no_grad():
            _ = model(signal, context)
    finally:
        hook.remove()

    # s * (1 + (-1)) + 0 = 0
    assert torch.allclose(captured["input"], torch.zeros_like(captured["input"]), atol=1e-6)
