# -*- coding: utf-8 -*-
"""Penolakan set-terbuka untuk MaritimeAMC: model boleh menjawab "tidak tahu".

KENAPA BUKAN KELAS KE-25
------------------------
Menambah satu neuron keluaran berlabel "tidak pasti" terdengar paling langsung,
tapi tak bisa dilatih: tidak ada satu pun sampel di RadioML yang berlabel
"tidak pasti", jadi kelas itu tak pernah menerima gradien positif dan bobotnya
mati. Ketidakpastian bukan kelas ke-25 -- ia sifat dari 24 logit yang sudah ada.

Jadi keputusannya diambil SESUDAH softmax, dan modul ini PASCA-LATIH: tidak
mengubah arsitektur, tidak melatih ulang apa pun, cukup logit validasi.

    kepala = KepalaKetidakpastian().pas(logit_val, y_val, target_risiko=0.10)
    kelas  = kepala.prediksi(logit_uji)      # -1 = TIDAK PASTI

TIGA SKOR
---------
msp      max softmax. Paling sederhana, paling terkenal, paling lemah -- jaringan
         modern terlalu percaya diri, jadi msp tinggi pun bisa salah.
entropi  sebar seluruh distribusi, bukan cuma puncaknya. Menangkap "dua kelas
         bersaing" yang luput dari msp.
energi   logsumexp(logit). Tidak dinormalkan, jadi ia melihat BESARAN logit dan
         bukan cuma bentuknya. Untuk masukan di luar distribusi -- pemancar yang
         tak ada di 24 kelas -- inilah yang biasanya paling tajam, dan justru
         kasus itulah misi kapal gelapmu.

KALIBRASI SUHU
--------------
Satu skalar T, dioptimalkan pada NLL validasi. Tidak mengubah ranking, jadi
akurasi TETAP SAMA persis -- yang berubah cuma angka keyakinannya jadi jujur.
Tanpa ini ambangnya tak bermakna: 0,9 pada jaringan tak terkalibrasi bisa
berarti benar 60 persen saja.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

TIDAK_PASTI = -1


# --------------------------------------------------------------------------- #
# skor -- konvensi: MAKIN BESAR MAKIN YAKIN (biar ambangnya selalu satu arah)
# --------------------------------------------------------------------------- #
def skor_msp(logit: torch.Tensor) -> torch.Tensor:
    return F.softmax(logit, -1).max(-1).values


def skor_entropi(logit: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logit, -1)
    h = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    return -h


def skor_energi(logit: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logit, -1)


SKOR = {"msp": skor_msp, "entropi": skor_entropi, "energi": skor_energi}


# --------------------------------------------------------------------------- #
# kurva risiko-cakupan
# --------------------------------------------------------------------------- #
def kurva_risiko_cakupan(skor: np.ndarray, benar: np.ndarray):
    """Urut dari paling yakin; kembalikan (cakupan, risiko) kumulatif.

    risiko = bagian yang SALAH di antara yang DITERIMA. Kurva ini satu-satunya
    cara jujur membaca abstain: akurasi naik itu sepele kalau cakupannya bebas
    turun, jadi keduanya harus selalu dibaca berpasangan.
    """
    urut = np.argsort(-skor)
    b = benar[urut].astype(np.float64)
    n = len(b)
    cakupan = np.arange(1, n + 1) / n
    risiko = np.cumsum(1.0 - b) / np.arange(1, n + 1)
    return cakupan, risiko


def aurc(skor: np.ndarray, benar: np.ndarray) -> float:
    """Luas di bawah kurva risiko-cakupan. Makin KECIL makin baik.

    Satu angka untuk membandingkan ketiga skor tanpa harus memilih ambang dulu.
    """
    cak, ris = kurva_risiko_cakupan(skor, benar)
    # numpy 2.x membuang np.trapz; getattr(..., np.trapz) TIDAK bisa dipakai
    # sebagai cadangan karena argumen default dievaluasi lebih dulu -> AttributeError.
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(ris, cak))


# --------------------------------------------------------------------------- #
# kepala
# --------------------------------------------------------------------------- #
class KepalaKetidakpastian:
    def __init__(self, jenis: str = "energi"):
        if jenis not in SKOR:
            raise KeyError(f"skor {jenis!r} tak dikenal; pilih {list(SKOR)}")
        self.jenis = jenis
        self.T = 1.0
        self.ambang = -np.inf
        self.cakupan_val = self.risiko_val = float("nan")

    def _pas_suhu(self, logit: torch.Tensor, y: torch.Tensor) -> float:
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

        def tutup():
            opt.zero_grad()
            rugi = F.cross_entropy(logit / logT.exp(), y)
            rugi.backward()
            return rugi

        opt.step(tutup)
        return float(logT.exp().item())

    def pas(self, logit_val, y_val, target_risiko=None, target_cakupan=None,
            kalibrasi=True):
        """Pasang suhu + pilih ambang, pada data VALIDASI.

        Beri TEPAT SATU: target_risiko (galat maksimum yang ditoleransi di
        antara yang diterima) atau target_cakupan (bagian yang ingin dijawab).
        """
        if (target_risiko is None) == (target_cakupan is None):
            raise ValueError("beri tepat satu: target_risiko ATAU target_cakupan")
        logit = torch.as_tensor(logit_val).float()
        y = torch.as_tensor(np.asarray(y_val)).long()

        self.T = self._pas_suhu(logit, y) if kalibrasi else 1.0
        s = SKOR[self.jenis](logit / self.T).numpy()
        benar = (logit.argmax(-1) == y).numpy()

        urut = np.argsort(-s)
        s_urut, b_urut = s[urut], benar[urut].astype(np.float64)
        n = len(s_urut)
        risiko = np.cumsum(1.0 - b_urut) / np.arange(1, n + 1)
        cakupan = np.arange(1, n + 1) / n

        if target_risiko is not None:
            ok = np.flatnonzero(risiko <= target_risiko)
            # cakupan TERBESAR yang masih memenuhi risiko. Kalau tak satu pun
            # memenuhi, tolak semuanya dan katakan begitu -- jangan diam-diam
            # mengembalikan ambang yang melanggar target yang diminta.
            k = int(ok[-1]) if len(ok) else None
        else:
            k = min(int(np.searchsorted(cakupan, target_cakupan)), n - 1)

        if k is None:
            self.ambang = float("inf")
            self.cakupan_val = self.risiko_val = 0.0
        else:
            self.ambang = float(s_urut[k])
            self.cakupan_val = float(cakupan[k])
            self.risiko_val = float(risiko[k])

        self.aurc_val = aurc(s, benar)
        self.akurasi_mentah_val = float(benar.mean())
        return self

    def skor(self, logit) -> np.ndarray:
        logit = torch.as_tensor(logit).float()
        return SKOR[self.jenis](logit / self.T).numpy()

    def prediksi(self, logit) -> np.ndarray:
        """kelas argmax, atau TIDAK_PASTI (-1) bila skor di bawah ambang."""
        logit = torch.as_tensor(logit).float()
        kelas = logit.argmax(-1).numpy()
        return np.where(self.skor(logit) >= self.ambang, kelas, TIDAK_PASTI)

    def __repr__(self):
        return (f"KepalaKetidakpastian(jenis={self.jenis!r}, T={self.T:.3f}, "
                f"ambang={self.ambang:.4f}, cakupan_val={self.cakupan_val:.3f}, "
                f"risiko_val={self.risiko_val:.3f})")


def laporan(logit_uji, y_uji, kepala) -> dict:
    """Ringkasan siap-skripsi untuk satu himpunan uji."""
    logit = torch.as_tensor(logit_uji).float()
    y = np.asarray(y_uji)
    pred = kepala.prediksi(logit)
    terima = pred != TIDAK_PASTI
    benar_mentah = (logit.argmax(-1).numpy() == y)
    nan = float("nan")
    return dict(
        n=len(y),
        akurasi_mentah=float(benar_mentah.mean()),
        cakupan=float(terima.mean()),
        akurasi_diterima=float((pred[terima] == y[terima]).mean()) if terima.any() else nan,
        risiko_diterima=float((pred[terima] != y[terima]).mean()) if terima.any() else nan,
        aurc=aurc(kepala.skor(logit), benar_mentah),
    )
