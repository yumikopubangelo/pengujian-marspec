# -*- coding: utf-8 -*-
"""Preset ringan untuk MaritimeAMCv3 — tanpa arsitektur baru.

Arsitektur v3 sudah sepenuhnya diparameterkan lewat dict cfg, jadi
"model ringan" TIDAK memerlukan kelas baru. Yang berubah cuma angkanya.
Itu penting secara metodologis: S dan T menjalankan **kode yang sama persis**
dengan L, sehingga selisih skornya hanya bisa berasal dari kapasitas --
bukan dari perbedaan implementasi yang tak sengaja ikut terbawa.

    L  36,94 jt   v3.2 apa adanya
    S   4,92 jt   -86,7%
    T   2,07 jt   -94,4%

Sebaran pada L: mcf 33,0% · moe 26,3% · eq 19,1% · cnn 15,2%. Empat blok itu
memegang 93,6% bobot, jadi hanya keempatnya yang layak dipangkas; menyentuh
classifier (0,4%) tak akan mengubah apa pun.

Kedua preset memilih transformer yang LEBIH DALAM ketimbang lebih lebar pada
anggaran yang sama (S: 8 lapis x 160 = 4,92 jt, sedangkan 6 lapis x 192 =
5,17 jt). Alasannya bukan selera: Paper 1 menemukan cabang transformer paling
tahan terhadap rotasi fasa, yaitu degradasi dominan di kanal maritim.

Pemakaian:
    python maritime_amc_lite.py            # tabel parameter ketiga preset
    python maritime_amc_lite.py --forward  # + uji lintasan maju

    from maritime_amc_lite import bikin_model
    model = bikin_model("S")
"""
from __future__ import annotations

import copy
from typing import Dict

from maritime_amc_v3_arch import CFG_V3, MaritimeAMCv3, validate_config

# --------------------------------------------------------------------------- #
# preset
# --------------------------------------------------------------------------- #
# Hanya kunci yang BERBEDA dari CFG_V3 yang ditulis di sini. Sisanya diwarisi,
# supaya perubahan pada CFG_V3 tidak diam-diam terputus dari preset ringannya.
UBAH_S: Dict = dict(
    eq_filters=128, eq_layers=8,               # 384/16 -> 7,05 jt jadi 0,38 jt
    cnn_channels=[48, 96, 160], cnn_pool=16,
    mcf_d_model=160, mcf_nhead=8, mcf_nlayers=8, mcf_patch_len=32,
    ctx_hidden=128, ctx_mlp_hidden=128, ctx_layers=2,
    fusion_d=192, n_experts=4, expert_hidden=256,
    grad_checkpoint=False,                     # muat di VRAM tanpa ini
)

UBAH_T: Dict = dict(
    eq_filters=80, eq_layers=6,
    cnn_channels=[32, 64, 96], cnn_pool=16,
    mcf_d_model=128, mcf_nhead=8, mcf_nlayers=6, mcf_patch_len=32,
    ctx_hidden=64, ctx_mlp_hidden=64, ctx_layers=2,
    fusion_d=128, n_experts=2, expert_hidden=128,
    grad_checkpoint=False,
)

# ---- preset BESAR, untuk menguji dua pertanyaan yang TERPISAH ----------- #
# Dirancang agar dua sumbu tidak tercampur: XL menaikkan kapasitas SINYAL
# (equalizer, cnn, transformer, fusion) dan membiarkan MoE apa adanya; XM
# menaikkan HANYA MoE dan membiarkan sisanya persis seperti L.
#
# PERINGATAN DARI DATA SENDIRI, baca sebelum memakai: akurasi maritim nyaris
# DATAR di rentang 1.300x parameter (0,3554 pada 28 ribu -> 0,4433 pada 37
# juta), dan papan bersih MEMUNCAK di ~5 juta lalu TURUN (S 0,7946 lawan L
# 0,7659). Ablasi 1 Okt 2026 juga menunjukkan 98% kerusakan maritim adalah
# informasi yang HANCUR di null spektral -- jumlah parameter tak bisa
# mengembalikan informasi yang sudah tidak ada. Prediksi: XL dan XM TIDAK
# membayar. Preset ini ada supaya prediksi itu diuji, bukan diasumsikan.
#
# XM punya peringatan kedua: MoE dirutekan KONTEKS, dan konteks terbukti tak
# membawa informasi (matched - shuffled = 0,0005). Memperbesar router yang
# sinyal rutenya derau kemungkinan besar hanya memperbesar modul kosong.
# Kalau mau MoE yang berguna, rutekan dari cabang spektral: lihat
# maritime_amc_v4a2_arch.py, cfg moe_route="spectral".
UBAH_XL: Dict = dict(                # ~47jt: LEBAR, bukan DALAM
    # Kenapa melebar dan bukan memperdalam: parameter transformer ~ n*12d^2,
    # jadi d adalah tuas KUADRAT dan n hanya LINEAR. Pada 15 lapis, d 256->320
    # (+56% parameter mcf) sedangkan menambah 6 lapis di d=256 hanya +40%.
    # Kedalaman juga menambah panjang lintasan gradien tanpa menambah
    # kekayaan per-token; jangkauan equalizer (127 sampel) sudah 4x lebih
    # panjang daripada tunda duktus terukur (31), jadi bukan kedalaman yang
    # kurang. MoE dibiarkan SAMA PERSIS seperti L supaya sumbu ini bersih.
    eq_filters=384, eq_layers=16,
    cnn_channels=[112, 224, 448], cnn_pool=16,
    mcf_d_model=320, mcf_nhead=16, mcf_nlayers=15, mcf_patch_len=32,
    fusion_d=448,
    grad_checkpoint=True,            # 50jt param: tanpa ini VRAM habis
)

UBAH_XM: Dict = dict(            # HANYA MoE yang berubah dari L
    n_experts=20, expert_hidden=1536,
    grad_checkpoint=True,
)

PRESET = {"L": {}, "S": UBAH_S, "T": UBAH_T, "XL": UBAH_XL, "XM": UBAH_XM}


def cfg_preset(nama: str, **timpa) -> Dict:
    """CFG_V3 + selisih preset + timpaan pemanggil. Divalidasi sebelum pulang."""
    if nama not in PRESET:
        raise KeyError(f"preset {nama!r} tak dikenal; pilih {list(PRESET)}")
    cfg = copy.deepcopy(CFG_V3)
    cfg.update(PRESET[nama])
    cfg.update(timpa)
    validate_config(cfg)
    return cfg


def bikin_model(nama: str = "S", **timpa) -> MaritimeAMCv3:
    return MaritimeAMCv3(cfg_preset(nama, **timpa))


def cacah(m) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# laporan
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse, io, contextlib, warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Tabel parameter preset MARSPEC ringan.")
    ap.add_argument("--forward", action="store_true", help="uji lintasan maju")
    args = ap.parse_args()

    model, total = {}, {}
    for nama in PRESET:
        with contextlib.redirect_stderr(io.StringIO()):
            model[nama] = bikin_model(nama)
        total[nama] = cacah(model[nama])

    blok = ["mcf_encoder", "moe", "equalizer", "cnn_extractor",
            "fusion", "ctx_encoder", "classifier"]
    print(f"{'blok':<16}" + "".join(f"{n:>14}" for n in PRESET))
    print("-" * (16 + 14 * len(PRESET)))
    for b in blok:
        baris = f"{b:<16}"
        for n in PRESET:
            baris += f"{cacah(getattr(model[n], b)):>14,}"
        print(baris)
    print("-" * (16 + 14 * len(PRESET)))
    print(f"{'TOTAL':<16}" + "".join(f"{total[n]:>14,}" for n in PRESET))
    print(f"{'juta':<16}" + "".join(f"{total[n]/1e6:>13.2f} " for n in PRESET))
    print(f"{'fp32 MB':<16}" + "".join(f"{total[n]*4/1e6:>13.1f} " for n in PRESET))
    print(f"{'int8 MB':<16}" + "".join(f"{total[n]/1e6:>13.1f} " for n in PRESET))
    print(f"{'vs L':<16}" + "".join(
        f"{'-':>12}  " if n == "L" else f"{-100*(total['L']-total[n])/total['L']:>12.1f}% "
        for n in PRESET))

    if args.forward:
        import torch
        print()
        iq, ctx = torch.randn(2, 2, 1024), torch.randn(2, 4)
        for n in PRESET:
            model[n].eval()
            with torch.no_grad():
                logits, iq_eq, gate = model[n](iq, ctx)
            print(f"  {n}: logits{tuple(logits.shape)} iq_eq{tuple(iq_eq.shape)} "
                  f"gate{tuple(gate.shape)}  OK")


if __name__ == "__main__":
    main()
