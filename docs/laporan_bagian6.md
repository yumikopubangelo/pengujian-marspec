# Test Design Specification — Bagian 6: Kepala Ketidakpastian & Demo Web MARSPEC

| | |
|---|---|
| Proyek | Pengujian MARSPEC (Maritime Spectrum Intelligence System) |
| Mata kuliah | Pengujian dan Pengembangan Perangkat Lunak |
| Butir yang diuji | `marspec/marspec_ketidakpastian.py` dan `demo_web/index.html` |
| Penguji | Hegira Musyafa Kartiwan (bagian 6) |
| Tanggal | 9 November 2026 |
| Versi kode | salinan beku 19 Oktober 2026 (`marspec/VERSI.md`): `marspec_ketidakpastian.py` 193 baris, SHA-256 `32d3927ef352`; `demo_web/index.html` 609 baris, SHA-256 `80d4b082cf14` (keduanya dicek ulang 9 Nov, tidak berubah) |
| Acuan format | IEEE Std 829-1998, *Test Design Specification* (templat strongqa.com) |

---

## 1. Test Design Specification Identifier

**TDS-MARSPEC-06**, versi 1.0.

Rencana uji induk: `PEMBAGIAN_TUGAS.md` (target per anggota, tahapan) dan `docs/panduan_pengujian.md`.
Dokumen terkait: `docs/uji_demo_web.md` (rincian uji manual demo web), `LOG_BUG.md` (bug),
`tests/test_6_ketidakpastian.py` (kode uji otomatis).

## 2. Features to be Tested

| ID fitur | Fitur | Unit / halaman |
|---|---|---|
| F6-1 | Skor keyakinan: makin besar makin yakin, tiga jenis skor | `skor_msp`, `skor_entropi`, `skor_energi` |
| F6-2 | Kurva risiko–cakupan dan luasnya (AURC) | `kurva_risiko_cakupan`, `aurc` |
| F6-3 | Pemilihan ambang tolak dari target risiko **atau** target cakupan, dengan/tanpa kalibrasi suhu | `KepalaKetidakpastian.pas` |
| F6-4 | Prediksi dengan penolakan: kelas 0–23, atau −1 (`TIDAK_PASTI`) | `KepalaKetidakpastian.prediksi` |
| F6-5 | Penolakan jenis skor yang tidak dikenal | `KepalaKetidakpastian.__init__` |
| F6-6 | Demo web — mode Jelajahi: kekuatan sinyal dan kondisi laut | `demo_web/index.html` |
| F6-7 | Demo web — mode Tantang AI: alur ronde, papan skor, penyimpanan skor | `demo_web/index.html` |
| F6-8 | Demo web — mode pameran, parameter alamat (`?tantang`) | `demo_web/index.html` |
| F6-9 | Demo web — ketahanan: tanpa internet, jendela privat, data hilang, lebar ponsel | `demo_web/index.html` |

**Tidak diuji** (di luar cakupan bagian ini):
- `laporan()` dan `__repr__` di `marspec_ketidakpastian.py`. Keduanya terbaca di laporan cakupan (§8).
- Isi bagian dalam `_pas_suhu` (optimasi LBFGS); yang diuji hanya akibatnya pada `T`.
- Kebenaran angka akurasi model di demo web (dihasilkan `bangun_demo.py` di repositori pengembang).
- Kinerja dan beban.

## 3. Approach Refinements

**Blackbox.** Harapan setiap kasus diambil dari docstring dan komentar modul (unit Python) atau dari
teks di layar (demo web), bukan dari isi fungsi. Teknik yang dipakai:

| Teknik | Dipakai untuk |
|---|---|
| Nilai batas (BVA) | MSP pada logit rata (tepat 1/24) dan logit ekstrem (→ 1); target risiko 0,0; ujung penggeser SNR |
| Partisi ekivalen (EP) | logit yakin vs ragu; kelas sah vs −1; jenis skor sah vs tak sah; parameter alamat dikenal vs salah ketik |
| Transisi keadaan (ST) | mode Jelajahi ↔ Tantang AI ↔ pameran |
| Tebakan galat (EG) | jendela privat, berkas data hilang, klik ganda |
| Tabel keputusan | dua argumen target di `pas` (§6.3) |

**Whitebox.**
- **Basis path** pada `KepalaKetidakpastian.pas` (§7): graf alir, V(G), jalur independen, satu kasus per jalur.
- **Cakupan cabang** dengan `pytest-cov --cov-branch`, sebelum dan sesudah (§8).
- **Tambahan: uji mutasi** (§9), untuk membuktikan setiap uji memang bisa gagal.

**Data uji.** Logit tiruan dari `torch.Generator` berbenih tetap, sehingga hasil bisa diulang persis:
`_logit_val()` (200 sampel, 24 kelas, separuh pertama dibuat yakin dan benar) dan `_logit_terbalik()`
(2 sampel; yang paling yakin justru salah).

**Lingkungan.** Windows 11, Python 3.14.5, torch 2.13.0+cpu, numpy 2.5.2, pytest 9.2.0, coverage 7.17.2.
Demo web: Chrome 155 dan Firefox 151.0.3 (uji manual), Chrome 157 dan Firefox 151.0.3 (otomasi peramban,
puppeteer-core 24.43), lebar ponsel 375 px.

**Uji manual dan otomasi.** 13 kasus demo web diuji manual oleh penguji. Enam kasus (W-06, W-09, W-10,
W-15, W-16, W-17) dijalankan dengan otomasi peramban, dengan skrip yang disertakan di `docs/otomasi/`
supaya bisa diulang. W-16 dijalankan pada salinan folder sehingga berkas asli tidak diubah.

**Independensi.** Penguji bagian ini juga pembuat MARSPEC, jadi ada risiko menguji apa yang *dimaksud*,
bukan apa yang *dilakukan* kode. Mitigasinya: harapan ditulis dari docstring/layar sebelum menjalankan uji,
kolom "Dasar harapan" wajib diisi, harapan yang direvisi setelah uji ditandai (W-14, W-19), dan uji
mutasi membuktikan uji bisa gagal.

## 4. Test Identification

### 4.1 Uji otomatis (17 kasus, `tests/test_6_ketidakpastian.py`)

| ID | Fitur | Teknik | Fungsi uji | Ringkasan |
|---|---|---|---|---|
| B6-01 | F6-1 | Nilai batas | `test_bb_skor_msp_rentang` | MSP: logit rata = 1/24, logit sangat besar ≈ 1 |
| B6-02 | F6-1 | Partisi ekivalen | `test_bb_tiga_skor_urutan_sama` | ketiga skor sepakat mana yang lebih yakin |
| B6-03 | F6-2 | Nilai batas | `test_bb_aurc_sempurna_vs_terbalik` | urutan terbaik memberi AURC lebih kecil dari urutan terburuk |
| B6-04 | F6-3, F6-4 | Partisi ekivalen | `test_bb_prediksi_abstain_bernilai_minus_satu` | target cakupan 0,5 → separuh −1, sisanya kelas 0–23 |
| B6-05 | F6-5 | Partisi ekivalen (tak sah) | `test_bb_jenis_skor_tak_dikenal` | jenis `'ngawur'` → KeyError yang menyebut pilihan sah |
| B6-06 | F6-3 | Nilai batas | `test_bb_target_risiko_mustahil` | target risiko 0,0 tak tercapai → tolak semua |
| W6-01, W6-02 | F6-3 | Tabel keputusan / cabang | `test_wb_pas_wajib_tepat_satu_target[…]` | kedua target kosong, atau keduanya diisi → ValueError |
| W6-03 | F6-3 | Cabang | `test_wb_pas_tanpa_kalibrasi` | `kalibrasi=False` → T = 1; `True` → T ≠ 1 |
| W6-04, W6-05 | F6-3 | Cabang + nilai batas | `test_wb_pas_cabang_target_cakupan_batas[1.0/1.5]` | target cakupan 1,0 dan > 1 tak membuat indeks keluar larik |
| W6-06 | F6-2 | Cabang | `test_wb_aurc_cadangan_trapz` | cabang `np.trapezoid` dan cadangan `np.trapz` sama-sama terambil, hasil 23/96 |
| W6-07…W6-10 | F6-3 | Basis path | `test_wb_basis_path_pas[J1/J3/J4/J5]` | satu kasus per jalur independen |
| W6-11 | F6-3 | Basis path | `test_wb_basis_path_pas_J2` | jalur J2 keluar lewat ValueError sebelum keadaan berubah |

### 4.2 Uji manual demo web (19 kasus, `docs/uji_demo_web.md`)

W-01–W-19. Rincian langkah, harapan, dan hasil aktual ada di dokumen itu (IEEE 829 mengizinkan isi
diletakkan di dokumen lain asal dirujuk dan dilampirkan). Ringkasan hasil ada di §10.

## 5. Feature Pass/Fail Criteria

Satu kasus **lulus** bila hasil aktual sama dengan hasil yang diharapkan, sesuai dasar harapan yang
tertulis sebelum uji dijalankan. Satu **fitur lulus** bila semua kasusnya lulus. Kasus yang gagal dicatat di
`LOG_BUG.md` dengan langkah reproduksi. Fitur dengan bug tingkat **minor** dinyatakan *lulus bersyarat*:
fungsinya benar, tetapi ada perilaku yang perlu diputuskan atau diperbaiki pemilik kode.

Untuk whitebox: setiap jalur basis path yang **mungkin dilalui** punya satu kasus yang lulus, dan setiap
bagian kode yang tidak tertutup cakupan dijelaskan alasannya.

**Hasil:**

| Fitur | Kasus | Hasil | Status fitur |
|---|---|---|---|
| F6-1 | B6-01, B6-02 | 2/2 lulus | Lulus |
| F6-2 | B6-03, W6-06 | 2/2 lulus | Lulus |
| F6-3 | B6-04, B6-06, W6-01…W6-05, W6-07…W6-11 | 12/12 lulus | Lulus |
| F6-4 | B6-04 | 1/1 lulus | Lulus |
| F6-5 | B6-05 | 1/1 lulus | Lulus |
| F6-6 | W-02, W-03, W-04 | 3/3 lulus | Lulus |
| F6-7 | W-05–W-11, W-18 | 7/8 lulus; W-09 gagal (BUG-02, minor) | Lulus bersyarat |
| F6-8 | W-12, W-13, W-14, W-19 | 4/4 lulus | Lulus |
| F6-9 | W-01, W-15, W-16, W-17 | 3/4 lulus; W-16 gagal (BUG-01, minor) | Lulus bersyarat |

Total: **17/17 uji otomatis lulus**; **17/19 uji demo web lulus**, 2 gagal tercatat sebagai bug minor.

---

## 6. Rincian kasus uji blackbox otomatis

Prasyarat umum: repositori di-clone, dependensi terpasang (`requirements.txt`), modul diimpor dari
`marspec/` (lewat `pytest.ini`). Cara menjalankan: `pytest tests/test_6_ketidakpastian.py`.

### 6.1 Tabel kasus uji

| ID | Teknik | Fungsi diuji | Kasus | Masukan | Hasil yang diharapkan | Dasar harapan | Hasil aktual | Status | Fungsi uji |
|---|---|---|---|---|---|---|---|---|---|
| B6-01 | Nilai batas | `skor_msp` | logit rata dan logit ekstrem, 24 kelas | `zeros(1, 24)`; `[50, 0, …, 0]` | tepat 1/24; ≈ 1 | softmax dari logit rata = seragam; satu logit dominan → peluang ≈ 1 | 0,041667; 1,000000 | Lulus | `test_bb_skor_msp_rentang` |
| B6-02 | Partisi ekivalen | `skor_msp`, `skor_entropi`, `skor_energi` | logit yakin vs ragu | `[50, 0, …, 0]` vs `zeros(1, 24)` | ketiga skor: yakin > ragu | konvensi modul: semua skor "makin besar makin yakin" | msp 1,0 > 0,0417; entropi −0,0 > −3,1781; energi 50,0 > 3,1781 | Lulus | `test_bb_tiga_skor_urutan_sama` |
| B6-03 | Nilai batas | `aurc` | dua urutan ekstrem pada benar/salah yang sama | 2 sampel, benar = [ya, tidak]; skor tinggi di sampel benar vs di sampel salah | AURC urutan terbaik < urutan terburuk | AURC = luas di bawah kurva risiko; jawaban salah di urutan atas menaikkan risiko di cakupan kecil | 0,125 < 0,375 | Lulus | `test_bb_aurc_sempurna_vs_terbalik` |
| B6-04 | Partisi ekivalen | `pas`, `prediksi` | target cakupan 0,5 | `_logit_val()` 200 sampel, `target_cakupan=0.5` | cakupan 0,5 (toleransi 0,001); yang diterima berlabel 0–23; sisanya −1 | docstring: −1 = `TIDAK_PASTI`; n = 200 → langkah cakupan 1/200 = 0,005, jadi toleransi 0,001 hanya meloloskan tepat 100/200 | cakupan 0,5000; 100 sampel −1; kelas diterima 0–23 | Lulus | `test_bb_prediksi_abstain_bernilai_minus_satu` |
| B6-05 | Partisi ekivalen (tak sah) | `KepalaKetidakpastian.__init__` | jenis skor tak dikenal | `'ngawur'` | `KeyError` yang menyebut `'ngawur'` dan ketiga pilihan sah | pesan galat yang baik menyebut pilihan yang sah | `KeyError("skor 'ngawur' tak dikenal; pilih ['msp', 'entropi', 'energi']")` | Lulus | `test_bb_jenis_skor_tak_dikenal` |
| B6-06 | Nilai batas | `pas` | target risiko 0,0 mustahil dicapai | `_logit_terbalik()`, `target_risiko=0.0` | tolak semua: ambang = ∞, `cakupan_val` = 0 | docstring: target mustahil harus menolak semua, bukan diam-diam melanggar target | ambang ∞; `cakupan_val` 0,0; `risiko_val` 0,0 | Lulus | `test_bb_target_risiko_mustahil` |

### 6.2 Partisi ekivalen dan nilai batas

| Masukan | Partisi sah | Partisi tak sah | Nilai batas yang diuji |
|---|---|---|---|
| logit 24 kelas | yakin (satu logit dominan) · ragu (logit rata) | — | rata persis (MSP = 1/24) · logit 50 (MSP ≈ 1) |
| jenis skor | `msp`, `entropi`, `energi` | string lain (`'ngawur'`) | — |
| `target_risiko` | 0 ≤ r ≤ 1, tercapai | tak tercapai (ditangani: tolak semua) | r = 0,0 |
| `target_cakupan` | 0 < c ≤ 1 | c > 1 (ditangani: dipotong ke indeks terakhir) | c = 0,5 · 1,0 · 1,5 |
| keluaran `prediksi` | kelas 0–23 | — | −1 (`TIDAK_PASTI`) |

### 6.3 Tabel keputusan: argumen target di `pas`

| Aturan | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Kondisi: `target_risiko` diisi | Y | Y | T | T |
| Kondisi: `target_cakupan` diisi | Y | T | Y | T |
| Aksi: ValueError "tepat satu" | ✓ (W6-02) | | | ✓ (W6-01, W6-11) |
| Aksi: ambang dari target risiko | | ✓ (B6-06, W6-09, W6-10) | | |
| Aksi: ambang dari target cakupan | | | ✓ (B6-04, W6-07, W6-08) | |

### 6.4 Contoh spesifikasi kasus uji (format strongqa)

| [ID] Nama | B6-06 Target risiko mustahil |
|---|---|
| Ringkasan | Bila tidak ada ambang yang memenuhi target risiko, kepala harus menolak semua prediksi |
| Butir diuji | `KepalaKetidakpastian.pas`, cabang `target_risiko` |
| Prasyarat | modul `marspec_ketidakpastian` dapat diimpor |
| Langkah | 1. Buat `KepalaKetidakpastian("msp")`. 2. Panggil `pas(logit, y, target_risiko=0.0)` dengan data yang sampel paling yakinnya salah. 3. Baca `ambang`, `cakupan_val`, `risiko_val` |
| Jalur alternatif | data yang sampel paling yakinnya benar → ambang hingga (diuji di W6-09, J4) |
| Masukan | `_logit_terbalik()`: logit `[[0]*24, [50]+[0]*23]`, label `[0, 1]` |
| Hasil yang diharapkan | `ambang = inf`, `cakupan_val = 0.0` |
| Hasil aktual | `ambang = inf`, `cakupan_val = 0.0`, `risiko_val = 0.0` · Lulus |

## 7. Basis path: `KepalaKetidakpastian.pas`

Berkas `marspec/marspec_ketidakpastian.py` baris 119–160.

**Simpul**

| Simpul | Isi (baris kode) |
|---|---|
| 1 | `if (target_risiko is None) == (target_cakupan is None)` (126) |
| 2 | `raise ValueError("beri tepat satu ...")` (127) |
| 3 | `logit = ...`, `y = ...` (128–129); keputusan `kalibrasi` dari ekspresi bersyarat baris 131 |
| 4 | `self._pas_suhu(logit, y)` (131, cabang ya) |
| 5 | `1.0` (131, cabang tidak) |
| 6 | `self.T = ...` s/d `cakupan = ...` (131–139); `if target_risiko is not None` (141) |
| 7 | `ok = np.flatnonzero(risiko <= target_risiko)` (142); keputusan `len(ok)` dari ekspresi bersyarat baris 146 |
| 8 | `k = int(ok[-1])` (146, ya) |
| 9 | `k = None` (146, tidak) |
| 10 | `k = min(int(np.searchsorted(cakupan, target_cakupan)), n - 1)` (148) |
| 11 | `if k is None` (150) |
| 12 | `ambang = inf`, `cakupan_val = risiko_val = 0.0` (151–152) |
| 13 | `ambang = s_urut[k]`, `cakupan_val`, `risiko_val` (154–156) |
| 14 | `aurc_val`, `akurasi_mentah_val`, `return self` (158–160) |
| K | keluar |

Ekspresi bersyarat `a if c else b` (baris 131 dan 146) dihitung sebagai keputusan. Tanpa itu V(G) keliru
menjadi 4. `min(...)` (baris 148) tidak dihitung karena merupakan fungsi bawaan; perilaku batasnya diuji
terpisah di W6-04 dan W6-05.

**Graf alir**

```
          (1)
       ya/   \tidak
       (2)    (3)
        |   ya/  \tidak
        K   (4)  (5)
              \  /
               (6)
          ya /    \ tidak
           (7)     (10)
        ya/  \tidak   |
        (8)  (9)      |
          \   |      /
            (11)
         ya/    \tidak
         (12)   (13)
            \   /
             (14)
              |
              K
```

**V(G)** = E − N + 2 = 19 − 15 + 2 = **6** · P + 1 = 5 + 1 = **6** (simpul keputusan 1, 3, 6, 7, 11).

**Jalur independen dan kasus uji**

| Jalur | Urutan simpul | Mungkin dilalui? | Masukan | Hasil yang diharapkan | Fungsi uji | Status |
|---|---|---|---|---|---|---|
| J1 | 1-3-4-6-10-11-13-14-K | ya | `_logit_val()`, `target_cakupan=0.5` | T ≠ 1; ambang hingga; `cakupan_val` = 0,5 | `test_wb_basis_path_pas[J1-…]` | Lulus |
| J2 | 1-2-K | ya | `_logit_val()`, tanpa target | ValueError; T = 1 dan ambang = −∞ tak tersentuh | `test_wb_basis_path_pas_J2` | Lulus |
| J3 | 1-3-5-6-10-11-13-14-K | ya | `target_cakupan=0.5`, `kalibrasi=False` | T = 1; `cakupan_val` = 0,5 | `test_wb_basis_path_pas[J3-…]` | Lulus |
| J4 | 1-3-4-6-7-8-11-13-14-K | ya | `_logit_val()`, `target_risiko=0.1` | ambang hingga; `risiko_val` ≤ 0,1 | `test_wb_basis_path_pas[J4-…]` | Lulus |
| J5 | 1-3-4-6-7-9-11-12-14-K | ya | `_logit_terbalik()`, `target_risiko=0.0` | ambang = ∞; `cakupan_val` = 0 | `test_wb_basis_path_pas[J5-…]` | Lulus |
| J6 | 1-3-4-6-10-11-12-14-K | **tidak** | — | — | — | — |

**Kenapa J6 mustahil.** `k` bernilai `None` jika dan hanya jika simpul 6 "ya" **dan** simpul 7 "tidak".
Dari simpul 10, `searchsorted` selalu memberi bilangan bulat, jadi sisi 11→12 tidak bisa dicapai lewat
cabang target cakupan. J1–J5 sudah melewati ke-19 sisi graf, sehingga V(G) = 6 adalah batas atas dan
fungsi ini hanya punya 5 jalur yang layak.

## 8. Cakupan cabang

Diukur dengan:
```
pytest tests/test_6_ketidakpastian.py --cov=marspec_ketidakpastian --cov-branch --cov-report=term-missing
```

| | Uji | Pernyataan | Terlewat | Cabang | Cabang sebagian | Cakupan |
|---|---|---|---|---|---|---|
| Sebelum (kerangka awal, commit `d4714b7`) | 3 lulus, 9 dilewati | 86 | 59 | 8 | 2 | **31%** |
| Sesudah (9 Nov 2026) | 17 lulus | 86 | 8 | 8 | 0 | **91%** |

Angka "Cakupan" dari coverage.py menggabungkan pernyataan dan cabang. Sesudah pengujian, kedelapan cabang
yang dikenali coverage.py tertutup penuh.

**Yang tidak tertutup** (baris 173 dan 180–193):
- `__repr__` (baris 173) dan seluruh fungsi `laporan()` (baris 180–193).
- Keduanya **bisa dijangkau**: tidak butuh GPU, berkas 21 GB, atau kondisi mustahil. Tidak ada uji yang
  memanggilnya karena keduanya di luar fitur yang dirancang di §2 (`laporan()` hanya merangkum keluaran
  fungsi lain yang sudah diuji).

**Batas alat ukur.** Di dalam `laporan()` ada dua keputusan satu baris
(`... if terima.any() else nan`, baris 190–191). coverage.py **tidak menghitung ekspresi bersyarat sebagai
cabang**, jadi angka "8/8 cabang tertutup" lebih tinggi dari keadaan sebenarnya. Keputusan yang sama di
`pas` (baris 131 dan 146) juga tidak dihitung alat ini, tetapi sudah tertutup lewat basis path (§7).

## 9. Uji mutasi (tambahan)

"Lulus" saja tidak membuktikan uji itu benar, karena uji yang tidak bisa gagal juga selalu lulus. Satu bug
ditanam di **salinan** modul (bukan `marspec/` yang beku), lalu semua uji dijalankan. Bug dianggap
tertangkap bila ada uji yang gagal.

| Mutan | Bug yang ditanam (baris) | Uji yang gagal |
|---|---|---|
| M1 | `skor_entropi` mengembalikan `h`, bukan `-h` (54) | B6-02 |
| M2 | kurva diurutkan naik, bukan turun (74) | B6-03, W6-06 |
| M3 | ambang off-by-one: `searchsorted(...) + 1` (148) | B6-04, J1, J3 |
| M4 | `prediksi` memakai `>`, bukan `>=` (170) | B6-04 |
| M5 | pesan KeyError tanpa daftar pilihan (100) | B6-05 |
| M6 | target risiko mustahil diam-diam memakai `k = 0` (146) | B6-06, J5 |
| M7 | `kalibrasi=False` diabaikan (131) | W6-03, J3 |
| M8 | pengaman `min(…, n − 1)` dihapus (148) | W6-05 (target 1,5) saja |
| M9 | kode abstain −2, bukan −1 (170) | B6-04 |
| M10 | cabang cadangan memanggil nama yang salah (90) | W6-06 |

**10/10 mutan tertangkap.** M8 hanya tertangkap oleh kasus target cakupan 1,5, dan M4 hanya tertangkap
karena toleransi B6-04 dipilih 0,001 (dengan toleransi 0,1, mutan M4 lolos). Batasnya: bug yang mengubah
**nilai** skor energi tanpa mengubah urutannya (misalnya `max(logit)`) tidak tertangkap, karena tidak ada
uji yang mematok nilai energi.

## 10. Hasil uji manual demo web

Rincian lengkap: `docs/uji_demo_web.md`. Tangkapan layar: `docs/tangkapan/`.

| ID | Teknik | Langkah (ringkas) | Status |
|---|---|---|---|
| W-01 | EP | buka tanpa internet | Lulus |
| W-02 | BVA | penggeser SNR paling kiri (−10 dB) | Lulus |
| W-03 | BVA | penggeser SNR paling kanan (30 dB) | Lulus |
| W-04 | EP | empat kondisi laut | Lulus |
| W-05 | ST | Jelajahi → Tantang AI | Lulus |
| W-06 | ST | jawab salah dan jawab benar | Lulus |
| W-07 | ST | ganti jawaban setelah menjawab | Lulus |
| W-08 | ST | Ronde berikutnya | Lulus |
| W-09 | ST | soal terbuka → Jelajahi → kembali | **Gagal → BUG-02** |
| W-10 | EP | skor setelah F5 | Lulus |
| W-11 | EP | atur ulang papan skor | Lulus |
| W-12 | ST | diam 30 detik → pameran | Lulus |
| W-13 | ST | gerak tetikus menghentikan pameran | Lulus |
| W-14 | EP | `index.html?tantang` | Lulus (harapan direvisi, ditandai) |
| W-15 | EG | jendela privat | Lulus |
| W-16 | EG | `data/demo.js` hilang | **Gagal → BUG-01** |
| W-17 | EP | lebar ponsel 375 px | Lulus |
| W-18 | EG | klik jawaban dua kali cepat | Lulus |
| W-19 | EP (tak sah) | `index.html?tantangan` (salah ketik) | Lulus |

Diagram keadaan halaman yang diperbarui dari hasil uji ada di `docs/uji_demo_web.md`. Transisi baru:
masuk lewat `?tantang` (W-14), soal terbuka dibuang saat pindah ke Jelajahi (W-09), dan F5 kembali ke
Jelajahi dengan skor tetap (W-10).

## 11. Bug yang ditemukan

| ID | Ringkasan | Tingkat | Status |
|---|---|---|---|
| BUG-01 | Bila `data/demo.js` tidak termuat, halaman hanya menampilkan kerangka kosong tanpa pesan. Console: `TypeError: Cannot read properties of undefined (reading 'keluarga')` | minor | baru |
| BUG-02 | Soal Tantang AI yang belum dijawab dibuang saat pindah ke Jelajahi; kembali ke Tantang AI memberi soal baru. Mungkin disengaja | minor | baru |

Langkah reproduksi lengkap: `LOG_BUG.md`. Perbaikan dilakukan di repositori pengembang, bukan di salinan
beku `marspec/`.

## 12. Batasan pengujian

1. **Independensi penguji.** Penguji juga pembuat kode; mitigasinya dijelaskan di §3.
2. **Enam kasus demo web memakai otomasi peramban**, bukan tangan. Hasilnya bisa diulang dengan skrip di
   `docs/otomasi/`. Chrome sempat diperbarui dari 155 (uji manual) ke 157 (otomasi).
3. **Cakupan 91% dari alat yang tidak menghitung ekspresi bersyarat** (§8).
4. **Uji mutasi hanya membuktikan tertangkapnya bug yang ditanam** (§9).
5. Uji otomatis memakai logit tiruan, bukan keluaran model terlatih. Ini sesuai untuk menguji logika
   pemilihan ambang, tetapi tidak menguji kualitas kalibrasi pada data nyata.

## Kontrol dokumen

| Versi | Tanggal | Perubahan | Penulis |
|---|---|---|---|
| 1.0 | 9 Nov 2026 | Versi pertama | Hegira Musyafa Kartiwan |

Disimpan di `docs/laporan_bagian6.md`, repositori `pengujian-marspec`. Pemilik dokumen: penguji bagian 6.
