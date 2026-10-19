# -*- coding: utf-8 -*-
"""BAGIAN 2 — Konteks & model utuh: ContextEncoder, ContextGatedFusion,
MoEChannelRouter, MaritimeAMCv3, validate_config.

Pemilik : [Anggota 2]
Kode    : marspec/maritime_amc_v3_arch.py
"""
import pytest
import torch

from maritime_amc_v3_arch import (ContextEncoder, ContextGatedFusion,  # noqa: F401
                                  MaritimeAMCv3, MoEChannelRouter, validate_config)

TODO = pytest.mark.skip(reason="TODO: belum ditulis")


# =============================================================================
# BLACKBOX
# =============================================================================
def test_bb_model_keluaran_tiga_tensor(cfg_kecil, iq_acak, ctx_acak):
    """CONTOH. Model utuh: keluaran = (logits 24 kelas, iq_eq, gate MoE),
    dan bobot gate tiap sampel berjumlah 1."""
    m = MaritimeAMCv3(cfg_kecil).eval()
    with torch.no_grad():
        logits, iq_eq, gate = m(iq_acak, ctx_acak)
    assert logits.shape == (4, 24)
    assert iq_eq.shape == iq_acak.shape
    assert torch.allclose(gate.sum(-1), torch.ones(4), atol=1e-5)


@TODO
def test_bb_moe_gate_tak_bergantung_sinyal():
    """MoEChannelRouter: ganti x tapi konteks tetap -> gate harus SAMA."""


@TODO
def test_bb_context_encoder_bentuk():
    """ContextEncoder(4, 64, 32, 3) pada [B, 4] -> [B, 32]."""


@TODO
def test_bb_fusion_bentuk_keluaran():
    """ContextGatedFusion(64, 32, 64): (sinyal [B, 64], konteks [B, 32]) -> [B, 64]."""


@TODO
def test_bb_model_batch_satu():
    """Nilai batas: model utuh dengan B = 1."""


@TODO
def test_bb_model_konteks_salah_ukuran():
    """Konteks [B, 3] (seharusnya 4): apakah ditolak dengan galat yang jelas?"""


# =============================================================================
# WHITEBOX
# =============================================================================
def test_wb_validate_config_fusion_ganjil(cfg_kecil):
    """CONTOH. Cabang `fusion % 2 != 0` di validate_config: harus ValueError
    yang menyebut fusion_d."""
    with pytest.raises(ValueError, match="fusion_d"):
        validate_config(dict(cfg_kecil, fusion_d=63))


@TODO
def test_wb_basis_path_validate_config():
    """Basis path validate_config: ada 4 keputusan. Hitung V(G), daftar jalur
    independen, lalu tunjukkan mana yang MUSTAHIL dilalui (infeasible) dan kenapa."""


@TODO
def test_wb_validate_config_dua_pelanggaran():
    """Dua pelanggaran sekaligus: pesan galat harus memuat KEDUANYA."""


@TODO
def test_wb_fusion_rumus_film():
    """ContextGatedFusion memakai s * (1 + gamma) + beta. Uji dengan memaksa
    bobot Linear FiLM nol dan bias gamma = -1: apa yang seharusnya terjadi?"""
