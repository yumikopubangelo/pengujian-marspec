#!/usr/bin/env python3
"""Per-SNR evaluation for MaritimeAMC v3 — CLEAN vs MARITIME, well-defined.

Resolves the "v3 looks worse than v2" confusion by making the eval condition
EXPLICIT (the notebook cell was ambiguous about this):

  --channel none      clean RadioML + benign context      -> CLEAN accuracy
  --channel maritime  full maritime channel + MATCHED ctx -> MARITIME accuracy

Why this matters: v3 is context-conditioned (FiLM equalizer + MoE routed by
context[B,4]). Two silent bugs tank v3 but not v2:
  (1) evaluating on the maritime channel while comparing to v2's CLEAN number
      (apples-to-oranges: maritime accuracy is *supposed* to be lower), and
  (2) applying the channel but feeding BENIGN context (equalizer thinks the sea
      is calm -> under-corrects -> collapse).
This script feeds context that MATCHES the signal, and lets you pick RAW vs EMA
weights, so you get a like-for-like number. Compare v3 --channel none against
v2's clean curve BEFORE concluding v3 is worse.

Run from marspec_poc/ (same working dir as training):
    # clean accuracy (compare to v2's ~85% high-SNR)
    python eval_maritime_snr.py --channel none
    # maritime accuracy (retention story)
    python eval_maritime_snr.py --channel maritime --pool normal
    # raw weights instead of EMA, + confusion matrix at +18 dB
    python eval_maritime_snr.py --channel none --key model \
        --ckpt outputs/maritime_v3/ckpt_last.pt --cm-snr 18
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from maritime_amc_v3_arch import MaritimeAMCv3, CFG_V3, validate_config
from train_maritime_v3 import (
    MaritimeChannel, _BENIGN_CTX, load_pools, scan_indices, load_X_inmemory, _ROOT,
    ctx_dari_kanal,
)

# RadioML 2018.01A modulation order (for a readable confusion matrix). Falls
# back to indices if the head width != 24.
MODS_24 = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


def load_weights(model, ckpt_path: str, key: str, device):
    """key: 'auto' (bare state_dict = best.pt/EMA), 'model' (raw), 'ema'."""
    # Coba tiga cara, kumpulkan galatnya. `weights_only=False` yang benar untuk
    # checkpoint milik sendiri; torch >= 2.6 mengubah bawaannya ke True, dan
    # torch lama tak kenal parameternya sama sekali.
    #
    # JANGAN menimpa fungsi ini dengan versi yang memaksa weights_only=True.
    # 2026-09-27: sebuah sel notebook melakukannya dan gagal dengan
    # "module 'torch' has no attribute '_utils'" -- galat yang datang DARI
    # patch itu sendiri, bukan dari checkpoint maupun dari fungsi ini.
    sd, galat = None, []
    for kw in ({"weights_only": False}, {"weights_only": True}, {}):
        try:
            sd = torch.load(ckpt_path, map_location=device, **kw)
            break
        except TypeError:
            continue                      # torch lama: parameter tak dikenal
        except Exception as e:
            galat.append(f"{kw or {}}: {type(e).__name__}: {e}")
    if sd is None:
        raise RuntimeError(
            f"Gagal memuat checkpoint {ckpt_path!r}. Percobaan:"
            + "\n  " + "\n  ".join(galat)
            + "\n  Periksa: berkas utuh? versi torch? ada sel yang menimpa"
            + "\n  load_weights()? Fungsi ini sudah menangani ketiga jalur.")
    if isinstance(sd, dict) and "model" in sd:          # full checkpoint
        state = sd["ema"] if key == "ema" else sd["model"]
        note = f"{'EMA' if key == 'ema' else 'RAW'} weights from full ckpt"
    else:                                                # bare state_dict
        state = sd
        note = "bare state_dict (best.pt = EMA)"
    model.load_state_dict(state, strict=True)
    print(f"Loaded {note}: {ckpt_path}")
    return model


@torch.no_grad()
def evaluate(model, X, labels, snrs, channel, pool, device, batch, use_amp, seed=0,
             return_logits=False, ctx_mode="matched"):
    """Returns preds[n], atau (preds, logits[n,C]) bila return_logits=True.

    channel=None -> clean+benign; else maritime+matched ctx.

    Logit diperlukan untuk apa pun yang bukan keputusan keras: kurva
    risiko-cakupan, kalibrasi suhu, voting multi-jendela. argmax membuang
    persis informasi yang dibutuhkan ketiganya.
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    # RNG TERPISAH untuk konteks. Kalau undian konteks diambil dari `rng` yang
    # sama, urutannya bergeser dan undian KANAL ikut berubah -- perbandingan
    # matched lawan shuffled lalu mencampur dua perubahan sekaligus.
    rng_ctx = np.random.default_rng(seed + 12345)
    if ctx_mode not in ("matched", "shuffled", "benign", "zero", "oracle"):
        raise ValueError("ctx_mode: matched | shuffled | benign | zero | oracle")
    preds = np.empty(n, np.int64)
    semua_logit = [] if return_logits else None
    model.eval()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        iqs, ctxs = [], []
        for k in range(start, end):
            iq_clean = X[k]                              # [2,1024] float32
            if channel is None:
                iq = iq_clean
                # oracle: [0,1,0,0] = "tanpa 2-sinar", sama seperti saat latih
                ctx = ctx_dari_kanal(None) if ctx_mode == "oracle" else _BENIGN_CTX
            else:
                j = int(rng.integers(0, len(pool["context"])))
                iq = channel.apply(
                    iq_clean, dist_km=float(pool["dist_km"][j]),
                    hs_m=float(pool["hs_m"][j]), sst_c=float(pool["sst_c"][j]),
                    snr_db=float(snrs[k]),
                    duct_h_m=float(pool["duct_h_m"][j]) if "duct_h_m" in pool else None,
                    n_interferers=int(pool["n_interferers"][j]) if "n_interferers" in pool else 0)
                # ABLASI KONTEKS. `shuffled` memakai undian LAIN yang bebas:
                # distribusi marjinal konteks IDENTIK, hanya kaitan
                # konteks<->sinyal yang hancur. Itu mengisolasi INFORMASI,
                # bukan distribusi -- `zero`/`benign` menggeser distribusinya
                # juga, sehingga penurunan bisa datang dari masukan di luar
                # distribusi, bukan dari hilangnya informasi.
                if ctx_mode == "matched":
                    ctx = pool["context"][j].astype(np.float32)
                elif ctx_mode == "oracle":
                    # parameter 2-sinar SEBENARNYA dari undian barusan
                    ctx = ctx_dari_kanal(channel.terakhir)
                elif ctx_mode == "shuffled":
                    j2 = int(rng_ctx.integers(0, len(pool["context"])))
                    ctx = pool["context"][j2].astype(np.float32)
                elif ctx_mode == "benign":
                    ctx = _BENIGN_CTX
                else:
                    ctx = np.zeros_like(_BENIGN_CTX)
            iqs.append(iq); ctxs.append(ctx)
        iqb = torch.from_numpy(np.stack(iqs)).to(device)
        cxb = torch.from_numpy(np.stack(ctxs)).to(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            enabled=use_amp):
            logits, _, _ = model(iqb, cxb)
        preds[start:end] = logits.argmax(1).cpu().numpy()
        if return_logits:
            semua_logit.append(logits.float().cpu().numpy())
    if return_logits:
        return preds, np.concatenate(semua_logit, 0)
    return preds


# Kunci yang hanya mengatur cara MENJALANKAN, bukan bentuk/perilaku model.
_KUNCI_RUNTIME = {"grad_checkpoint"}


def cocokkan_config(ckpt, arch, size, cfg, cfg_timpa=None):
    """Selaraskan cfg eval dengan config.json yang ditulis saat LATIH.

    Kenapa wajib: load_state_dict(strict=True) hanya menangkap BENTUK yang
    beda. Kunci yang mengubah PERILAKU tanpa mengubah bentuk lolos diam-diam.
    Contoh nyata: pada preset T, ctx_hidden (64) = fusion_d/2 (64), jadi ckpt
    moe_route='spectral' termuat sempurna ke model moe_route='ctx' -- lalu
    dievaluasi dengan rute yang salah tanpa satu pesan pun.

    Aturan: arch/size beda -> gagal. Kunci cfg yang beda -> ambil dari
    config.json, KECUALI pemanggil menimpanya eksplisit lewat cfg_timpa
    dengan nilai lain -> gagal (dua sumber kebenaran yang bertentangan).
    Tanpa config.json (ckpt lama) -> perilaku lama, dengan catatan.
    """
    import json as _json
    p = Path(ckpt).parent / "config.json"
    if not p.exists():
        print(f"  [config] tak ada config.json di {p.parent} -- cfg dari preset "
              f"saja, TIDAK diverifikasi.")
        return cfg
    rek = _json.loads(p.read_text(encoding="utf-8"))
    a = rek.get("args", {})
    if a.get("arch", arch) != arch:
        raise SystemExit(f"ckpt dilatih dengan --arch {a.get('arch')}, dievaluasi "
                         f"dengan arch={arch!r}. Samakan.")
    # dpdrsn & baseline G0 punya cfg sendiri; --size bagi mereka cuma label.
    if not (arch.startswith("dpdrsn") or arch in ("mcformer", "resnet1d")) \
            and a.get("size", size) != size:
        raise SystemExit(f"ckpt dilatih dengan --size {a.get('size')}, dievaluasi "
                         f"dengan size={size!r}. Samakan.")
    timpa = cfg_timpa or {}
    for k, v in rek.get("cfg", {}).items():
        if k in _KUNCI_RUNTIME:
            continue
        v_kini = cfg.get(k)
        if isinstance(v_kini, tuple):
            v_kini = list(v_kini)
        if v_kini == v:
            continue
        if k in timpa:
            raise SystemExit(f"cfg_timpa[{k!r}]={timpa[k]!r} bertentangan dengan "
                             f"config.json latih ({v!r}).")
        print(f"  [config] {k} = {v!r} (dari config.json; preset memberi {v_kini!r})")
        cfg[k] = v
    return cfg


def mode_oracle_latih(ckpt):
    """(balik, ctx_oracle) dari config.json latih; ('', False) kalau tak ada.

    Alasannya sama dengan cocokkan_config(): model D_zf yang dievaluasi dengan
    masukan MENTAH tetap jalan tanpa satu galat pun -- hanya angkanya salah."""
    import json as _json
    p = Path(ckpt).parent / "config.json"
    if not p.exists():
        return "", False
    a = _json.loads(p.read_text(encoding="utf-8")).get("args", {})
    b = a.get("balik_2sinar", "tidak")
    return ("" if b in (None, "tidak") else b), bool(a.get("ctx_oracle", False))


def run_eval(ckpt, hdf5, pools_dir, key="auto", channel="none", pool="normal",
             max_samples=120_000, snr_min=-6, batch=256, amp=False, device=None,
             arch="v3", size="L", cfg_timpa=None, return_logits=False,
             ctx_mode=None, duct=True, interference=True,
             dua_sinar=True, fasa_global=True, balik=None):
    """Importable entry point. Returns (preds, labels, snrs) as numpy arrays so a
    notebook can plot/compare directly. channel: 'none' (clean) | 'maritime'.
    arch: 'v3' (real) | 'v4a1' (complex equalizer) — must match the checkpoint.

    ctx_mode / balik = None -> diambil dari config.json latih (oracle grup D);
    tanpa config.json -> 'matched' / '' seperti dulu. Nilai eksplisit dihormati
    (mis. mengevaluasi model D_zf pada masukan mentah dengan sengaja)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device == "cuda"
    balik_latih, ctx_oracle_latih = mode_oracle_latih(ckpt)
    if balik is None:
        balik = balik_latih
        if balik:
            print(f"  [config] balik_2sinar = {balik!r} (dari config.json latih)")
    if ctx_mode is None:
        ctx_mode = "oracle" if ctx_oracle_latih else "matched"
        if ctx_oracle_latih:
            print("  [config] ctx = 'oracle' (dari config.json latih)")

    # cfg HARUS dibangun sama persis seperti saat latih, kalau tidak
    # load_state_dict(strict=True) menolak dengan size mismatch.
    if arch.startswith("dpdrsn"):
        from dp_drsn import CFG_DPDRSN, DPDRSN
        cfg = dict(CFG_DPDRSN)
        cfg["use_dcce"] = arch == "dpdrsn_dcce"
        cfg["num_classes"] = CFG_V3["num_classes"]
        cfg.update(cfg_timpa or {})
        cfg = cocokkan_config(ckpt, arch, size, cfg, cfg_timpa)
        model = DPDRSN(cfg).to(device)
    elif arch in ("mcformer", "resnet1d"):
        # G0: baseline yang DILATIH lewat train_maritime_v3.py (protokol sama),
        # dibungkus AdaptorLatih yang sama -> kunci state_dict "dasar.*" cocok.
        from baseline_models import bikin_baseline_latih, CFG_BASELINE_BAWAAN
        # Bawaan disemai DULU, lalu cocokkan_config menimpanya dari config.json.
        # Checkpoint lama tanpa kunci ukuran -> bawaan = ukuran yang dulu dilatih.
        cfg = {"baseline": arch, "num_classes": CFG_V3["num_classes"],
               **CFG_BASELINE_BAWAAN[arch]}
        cfg.update(cfg_timpa or {})
        cfg = cocokkan_config(ckpt, arch, size, cfg, cfg_timpa)
        model = bikin_baseline_latih(arch, cfg["num_classes"], cfg).to(device)
    else:
        from maritime_amc_lite import PRESET
        cfg = dict(CFG_V3)
        cfg.update(PRESET[size])
        if size in ("S", "T"):      # bukan '!= "L"': XL/XM paling butuh
            cfg["grad_checkpoint"] = False
        cfg.update(cfg_timpa or {})
        cfg = cocokkan_config(ckpt, arch, size, cfg, cfg_timpa)
        validate_config(cfg)
        if arch == "v4a1":
            from maritime_amc_v4_arch import MaritimeAMCv4
            model = MaritimeAMCv4(cfg).to(device)
        elif arch == "v4a2":
            from maritime_amc_v4a2_arch import MaritimeAMCv4A2
            model = MaritimeAMCv4A2(cfg).to(device)
        else:
            model = MaritimeAMCv3(cfg).to(device)
    print(f"Arch: {arch} | Size: {size} | params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    load_weights(model, ckpt, key, device)

    idx, labels, snrs = scan_indices(Path(hdf5), snr_min,
                                     None if max_samples in (0, None) else max_samples)
    snrs = np.asarray(snrs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    print(f"Eval samples: {len(idx):,}  | channel={channel}  | device={device}")
    X = load_X_inmemory(Path(hdf5), idx)                 # [n,2,1024]

    ch = pl = None
    if channel == "maritime":
        pl = load_pools(Path(pools_dir))[pool]
        # Komponen kanal bisa diablasi SATU-SATU. Kurva per-SNR menunjukkan
        # maritim MASIH NAIK di +30 dB sementara bersih mendatar sejak +10 dB,
        # jadi plafon maritim BUKAN dibatasi derau. Ablasi ini menguraikan
        # jurangnya: multipath duct lawan interferensi tonal.
        ch = MaritimeChannel(rng=np.random.default_rng(0),
                             duct=duct, interference=interference,
                             dua_sinar=dua_sinar, fasa_global=fasa_global,
                             balik=balik)
    if return_logits:
        preds, logit = evaluate(model, X, labels, snrs, ch, pl, device, batch,
                                use_amp, return_logits=True, ctx_mode=ctx_mode)
        return preds, labels, snrs, logit
    preds = evaluate(model, X, labels, snrs, ch, pl, device, batch, use_amp,
                     ctx_mode=ctx_mode)
    return preds, labels, snrs


def per_snr_table(preds, labels, snrs):
    """-> (snr_values, acc_values) plus prints the table + summary."""
    print("\nsnr\tacc\tn")
    xs, ys = [], []
    for s in np.unique(snrs):
        m = snrs == s
        a = float((preds[m] == labels[m]).mean())
        xs.append(float(s)); ys.append(a)
        print(f"{s:+.0f}\t{a:.4f}\t{int(m.sum())}")
    overall = float((preds == labels).mean())
    hi = [a for s, a in zip(xs, ys) if s >= 10]
    msg = f"\nALL-SNR acc = {overall:.4f}"
    if hi:
        msg += f"   |   HIGH-SNR(>=10dB) mean = {np.mean(hi):.4f}"
    print(msg)
    return np.array(xs), np.array(ys)


def top_confusions(preds, labels, snrs, cm_snr, k=12):
    """Print the k most-confused (true -> pred) modulation pairs at one SNR."""
    m = snrs == np.float32(cm_snr)
    if m.sum() == 0:
        print(f"\n[cm] no samples at SNR={cm_snr}"); return
    C = int(max(labels.max(), preds.max())) + 1
    cm = np.zeros((C, C), np.int64)
    for t, p in zip(labels[m], preds[m]):
        cm[t, p] += 1
    names = MODS_24 if C == len(MODS_24) else [str(i) for i in range(C)]
    pairs = []
    for t in range(C):
        denom = cm[t].sum()
        for p in range(C):
            if t != p and cm[t, p] > 0:
                pairs.append((cm[t, p] / denom if denom else 0, cm[t, p],
                              names[t], names[p]))
    pairs.sort(reverse=True)
    print(f"\nTop confusions at SNR={cm_snr:+.0f} dB (true -> pred):")
    for frac, cnt, tn, pn in pairs[:k]:
        print(f"  {tn:>10} -> {pn:<10}  {cnt:4d}  ({frac:.0%} of {tn})")


def _urai_cfg(daftar):
    """['mcf_nlayers=15'] -> {'mcf_nlayers': 15}"""
    import ast as _ast
    keluar = {}
    for kv in daftar:
        if "=" not in kv:
            raise ValueError(f"--cfg butuh KUNCI=NILAI, dapat {kv!r}")
        k, v = kv.split("=", 1)
        try:
            keluar[k] = _ast.literal_eval(v)
        except (ValueError, SyntaxError):
            keluar[k] = v
    return keluar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", default=str(_ROOT / "data/raw/radioml/archive/GOLD_XYZ_OSC.0001_1024.hdf5"))
    ap.add_argument("--pools", default=str(_ROOT / "data/ocean_states"))
    ap.add_argument("--ckpt", default=str(_ROOT / "outputs/maritime_v3/maritime_v3_best.pt"))
    ap.add_argument("--key", choices=["auto", "model", "ema"], default="auto",
                    help="auto=best.pt(EMA); model=RAW from ckpt_last.pt; ema=EMA from ckpt_last.pt")
    ap.add_argument("--channel", choices=["none", "maritime"], default="none",
                    help="none=clean RadioML (compare to v2); maritime=full channel")
    ap.add_argument("--pool", choices=["easy", "normal", "hard"], default="normal")
    ap.add_argument("--max_samples", type=int, default=120_000)
    ap.add_argument("--snr_min", type=int, default=-6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cm-snr", type=float, default=None,
                    help="if set, print a confusion matrix + top confused pairs at this SNR")
    ap.add_argument("--size", choices=["L", "S", "T", "XL", "XM"], default="L",
                    help="HARUS sama dengan --size saat latih (strict=True akan menolak).")
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KUNCI=NILAI",
                    help="Timpa cfg, harus sama dengan saat latih (mis. mcf_nlayers=15).")
    ap.add_argument("--arch", choices=["v3", "v4a1", "v4a2", "dpdrsn", "dpdrsn_dcce",
                                       "mcformer", "resnet1d"], default="v3",
                    help="must match the checkpoint's architecture")
    ap.add_argument("--ctx", dest="ctx_mode", default=None,
                    choices=["matched", "shuffled", "benign", "zero", "oracle"],
                    help="ablasi konteks. shuffled = undian bebas, distribusi "
                         "marjinal identik, kaitan konteks-sinyal hancur. oracle = "
                         "parameter 2-sinar sebenarnya. Kosong = dari config.json "
                         "latih (oracle kalau dilatih --ctx-oracle, selain itu matched)")
    ap.add_argument("--balik-2sinar", dest="balik", default=None,
                    choices=["tidak", "zf-pita"],
                    help="Kosong = dari config.json latih.")
    ap.add_argument("--no-duct", dest="duct", action="store_false",
                    help="matikan multipath moda-duct (ablasi komponen kanal)")
    ap.add_argument("--no-interference", dest="interference", action="store_false",
                    help="matikan interferensi tonal ko-kanal")
    ap.add_argument("--no-2ray", dest="dua_sinar", action="store_false",
                    help="matikan multipath 2-sinar PERMUKAAN LAUT (rho s/d 0,95)")
    ap.add_argument("--no-phase", dest="fasa_global", action="store_false",
                    help="matikan rotasi fasa global acak (PhaseAug)")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    preds, labels, snrs = run_eval(
        args.ckpt, args.hdf5, args.pools, key=args.key, channel=args.channel,
        size=args.size, cfg_timpa=_urai_cfg(args.cfg),
        pool=args.pool, max_samples=args.max_samples, snr_min=args.snr_min,
        batch=args.batch, amp=args.amp, arch=args.arch, ctx_mode=args.ctx_mode,
        duct=args.duct, interference=args.interference,
        dua_sinar=args.dua_sinar, fasa_global=args.fasa_global,
        balik=None if args.balik is None else ("" if args.balik == "tidak" else args.balik))

    per_snr_table(preds, labels, snrs)
    if args.cm_snr is not None:
        top_confusions(preds, labels, snrs, args.cm_snr)


if __name__ == "__main__":
    main()
