# PENGUJIAN MARSPEC

## TEST SUMMARY REPORT

**Versi 0.95 (draf)**
**10 November 2026**

Pengenal dokumen: **TSR-MARSPEC-01**. Rencana uji yang dirangkum: **TP-MARSPEC-01** (`docs/test_plan.md`).

---

## VERSION HISTORY

| Versi | Disusun oleh | Tanggal revisi | Disetujui oleh | Tanggal persetujuan | Alasan |
|---|---|---|---|---|---|
| 0.9 | [isi] | 10 Nov 2026 | — | — | Draf pertama, bersusunan IEEE Std 829-1998 (berkas lama `laporan_gabungan.md`) |
| 0.95 | [isi] | 10 Nov 2026 | — | — | Disusun ulang mengikuti templat *Test Summary Report* dari dosen. Isi dan angka tidak berubah. Ditambah Lampiran B (referensi) dan C (istilah) |

---

## DAFTAR ISI

1. Introduction
   1.1 Purpose
2. Test Summary
   2.1–2.6 Unit test, bagian 1–6
   2.7 System test (manual), demo web
3. Test Assessment
4. Test Results
5. Suggested Actions

Lampiran A: Test Summary Report Approval
Lampiran B: References
Lampiran C: Key Terms
Lampiran D: Kasus uji blackbox
Lampiran E: Whitebox, basis path dan uji cabang
Lampiran F: Cara mengulang angka laporan ini

---

## 1 INTRODUCTION

### 1.1 PURPOSE

Laporan uji Pengujian MARSPEC ini merangkum hasil pengujian yang dijalankan sesuai rencana uji
TP-MARSPEC-01 (`docs/test_plan.md`, yang memformalkan `PEMBAGIAN_TUGAS.md` dan `docs/panduan_pengujian.md`).
Dokumen sumber lainnya tercantum di Lampiran B.

## 2 TEST SUMMARY

**Project Name:** Pengujian MARSPEC, tugas kelompok mata kuliah Pengujian dan Pengembangan Perangkat Lunak

**System Name:** MARSPEC (Maritime Spectrum Intelligence System). Sistem ini adalah model PyTorch untuk
mengenali modulasi sinyal radio di kanal laut, beserta simulator kanal, pemuat data, alat evaluasi, kepala
ketidakpastian, dan demo web.

**Version Number:** salinan beku 19 Oktober 2026 (`marspec/`, `demo_web/`). Sidik SHA-256 tiap berkas ada di
`marspec/VERSI.md`.

**Additional Comments:** pengujian dibagi enam bagian. Setiap anggota menguji satu bagian dengan **blackbox**
(partisi ekivalen, nilai batas, tabel keputusan, transisi keadaan, tebakan galat) dan **whitebox** (basis path,
cakupan cabang). Semua uji otomatis memakai data sintetis kecil, jadi tidak membutuhkan GPU, dataset 21 GB,
atau bobot model terlatih. Laporan ini masih **draf**: uji bagian 1, 4, dan 5 diambil dari salinan kerja yang
belum di-commit (§4.3).

**Hasil utama (seluruh rangkaian dijalankan 10 November 2026):**

| Ukuran | Hasil |
|---|---|
| Uji otomatis (`pytest`) | **117 lulus, 2 xfail, 0 gagal, 0 dilewati** (di bawah 15 detik di laptop). Kedua xfail adalah uji yang sengaja menandai BUG-03 |
| Uji demo web (19 kasus) | **17 lulus, 2 gagal** |
| Bug | **3**: dua *minor* di demo web (BUG-01, BUG-02) dan satu di pemuat data Python (BUG-03; *kritis* menurut definisi tabel bug, dampak praktisnya rendah) |
| Cakupan per bagian | 87–100% atas fungsi milik tiap bagian (§3.2) |
| Cakupan seluruh paket `marspec/` | 29% → 44%. Angka ini **bukan** ukuran yang tepat karena paket berisi banyak kode di luar lingkup (§3.4) |

**Lingkungan uji:**

| Lingkungan | Versi |
|---|---|
| Lokal (angka laporan ini) | Windows 11 Pro 10.0.26200, Python 3.14.5, torch 2.13.0+cpu, numpy 2.5.2, pytest 9.2.0, pytest-cov 7.1.0, coverage 7.17.2 |
| GitHub Actions (`.github/workflows/uji.yml`) | ubuntu-latest, Python 3.11, torch CPU |
| Demo web | Chrome 157.0.8103.38, Firefox 151.0.3; otomasi dengan puppeteer-core 24.43.1 |

**Ringkasan kegiatan:**

| Tanggal | Kegiatan |
|---|---|
| 19 Okt 2026 | Kerangka repositori (commit `d4714b7`): 16 lulus / 57 dilewati, cakupan paket 29% |
| 1–2 Nov 2026 | Pembagian tugas diperbarui (`PEMBAGIAN_TUGAS.md`) |
| 3 Nov 2026 | Bagian 3 menulis uji kanal laut (commit `6d8d7ec`). Bagian 6 menulis uji blackbox dan menjalankan 13 kasus demo web secara manual |
| 9 Nov 2026 | PR #1 dan #2 digabung. Bagian 2 dan 6 di-commit (`c11db92`). 6 kasus demo web dijalankan lewat otomasi peramban, lalu BUG-01 dan BUG-02 dicatat |
| 10 Nov 2026 | Uji bagian 1 dan 5 masuk salinan kerja. Uji bagian 4 ditulis dan BUG-03 dicatat. Seluruh rangkaian dijalankan dan cakupan diukur ulang untuk laporan ini |

### 2.1 UNIT TEST: BAGIAN 1, KOMPONEN SINYAL (BLACKBOX + WHITEBOX)

Butir yang diuji: `SE1d`, `ChannelEqualizer`, `CNN1DExtractor`, `MCformerEncoder`.

**Test Owner:** [isi nama]

**Test Date:** 19 Oktober – 10 November 2026. Berkas uji terakhir diubah 9 November dan belum di-commit.

**Test Results:** 13 lulus (8 blackbox, 5 whitebox). Cakupan fungsi bagian ini 94% → **99%**. Kontrak tiap
komponen terbukti dipenuhi:
- bentuk keluaran tetap untuk batch 1 dan 8;
- mode eval deterministik;
- NaN tidak disembunyikan;
- panjang masukan yang salah ditolak;
- FiLM hanya berpengaruh bila konteks diberikan;
- *layerdrop* hanya aktif saat latih;
- *gradient checkpoint* tidak mengubah hasil.

**Additional Comments:**
- Kasus blackbox baru 8, di bawah target 10.
- V(G) dan graf alir `ChannelEqualizer.forward` belum ditulis pemilik (catatan T-4). Penyusun menghitung V(G) = 5
  (Lampiran E.1).
- Satu-satunya cabang yang tidak tertutup (132→129) mustahil dilalui tanpa mengubah kode (§3.3).

### 2.2 UNIT TEST: BAGIAN 2, KONTEKS & MODEL UTUH (BLACKBOX + WHITEBOX)

Butir yang diuji: `ContextEncoder`, `ContextGatedFusion`, `MoEChannelRouter`, `MaritimeAMCv3`, `validate_config`.

**Test Owner:** Elsa Novia

**Test Date:** 19 Oktober – 9 November 2026 (commit `c11db92`, PR #1).

**Test Results:** 10 lulus (6 blackbox, 4 whitebox). Cakupan 94% → **100%**. Klaim desain inti MARSPEC
terbukti: **gate MoE hanya bergantung pada konteks, bukan sinyal**. Model utuh mengeluarkan tiga tensor
berbentuk benar, bobot gate-nya berjumlah 1, dan konteks yang salah ukuran ditolak. `validate_config`
melaporkan **semua** pelanggaran sekaligus, bukan hanya yang pertama.

**Additional Comments:**
- Kasus blackbox baru 6, di bawah target 10.
- Docstring basis path menulis V(G) = 4, padahal seharusnya 5. Himpunan lima jalurnya sebenarnya sudah lengkap
  (catatan T-1, Lampiran E.2).

### 2.3 UNIT TEST: BAGIAN 3, SIMULATOR KANAL LAUT (BLACKBOX + WHITEBOX)

Butir yang diuji: `MaritimeChannel.apply`.

**Test Owner:** Maitsa [lengkapi nama]

**Test Date:** 19 Oktober – 3 November 2026 (commit `6d8d7ec`, digabung lewat PR #2 pada 9 November).

**Test Results:** 36 lulus (10 blackbox, 26 whitebox). Cakupan 80% → **98%**. Ini whitebox paling lengkap di
kelompok. Setiap saklar simulator dibuktikan lewat **jejak pemanggilan RNG**: komponen yang mati tidak
mengundi angka acak sama sekali. Normalisasi keluaran, keterulangan dengan benih yang sama, batas Doppler di
0,6 m, batas 6 interferer, dan cache pita juga terbukti. Ke-17 kondisi sederhana di `apply` sudah diambil ke
dua arah.

**Additional Comments:** dua dari 10 kasus blackbox adalah contoh kerangka. Baris 161 (`balik` tak sah →
`ValueError`) belum diuji (catatan T-5).

### 2.4 UNIT TEST: BAGIAN 4, RUGI & PEMUAT DATA (BLACKBOX + WHITEBOX)

Butir yang diuji: `moe_balance_loss`, `equalizer_aux_loss`, `scan_indices`, `load_X_inmemory`.

**Test Owner:** M Tegar Fathan (pemilik bagian menurut `PEMBAGIAN_TUGAS.md`). Uji ditulis oleh **[isi]** (§4.3
butir 1).

**Test Date:** 10 November 2026 (belum di-commit).

**Test Results:** 29 lulus + 2 xfail (13 blackbox, 18 whitebox). Cakupan 9% → **99%**. Hal-hal berikut terbukti:
- kedua jalur `load_X_inmemory` (fancy dan per blok) memberi isi yang sama dengan berkas aslinya, termasuk untuk
  indeks berulang;
- pemeriksaan RAM jatuh tepat di batasnya: 10.239 byte ditolak, 10.240 boleh;
- `scan_indices` tetap menyertakan baris ber-SNR tepat `snr_min`, dan memperlakukan `max_samples=0` sebagai
  "pakai semua" sesuai teks bantuan `--max_samples`;
- rugi equalizer kebal terhadap rotasi fasa global.

Temuannya **BUG-03**: indeks di luar rentang menghasilkan sisa memori tanpa galat (§4.4).

**Additional Comments:** basis path `load_X_inmemory` V(G) = 10, dengan 7 jalur diuji. Dua jalur tidak diuji,
dan alasannya ada di Lampiran E.4. Pada uji mutasi, 7 dari 7 mutan non-ekuivalen terbunuh.

### 2.5 UNIT TEST: BAGIAN 5, EVALUASI & CLI (BLACKBOX + WHITEBOX)

Butir yang diuji: `cocokkan_config`, `mode_oracle_latih`, `per_snr_table`, penjaga argumen `main()`.

**Test Owner:** [isi nama]

**Test Date:** 19 Oktober – 10 November 2026. Berkas uji terakhir diubah 9 November dan belum di-commit.

**Test Results:** 12 lulus (6 blackbox, 6 whitebox). Cakupan 9% → **87%**. Penjaga CLI terbukti jatuh
**sebelum** data dimuat, dan `cocokkan_config` menolak arsitektur yang beda serta penimpaan yang bertentangan.

**Additional Comments:**
- Kasus blackbox baru 6, di bawah target 10.
- Basis path belum lengkap: ujinya menjalankan ulang jalur tanpa `config.json` (catatan T-2), dan 4 arah
  kondisi belum diuji (Lampiran E.5).
- Penjaga yang paling sering terpicu dalam praktik, yaitu `--size` beda pada v3 (baris 192), justru belum
  diuji.
- Aturan "salah semua" di `per_snr_table` belum diuji (catatan T-3).

### 2.6 UNIT TEST: BAGIAN 6, KEPALA KETIDAKPASTIAN (BLACKBOX + WHITEBOX)

Butir yang diuji: `marspec_ketidakpastian.py`.

**Test Owner:** Hegira Musyafa Kartiwan

**Test Date:** 2 – 9 November 2026 (commit `c11db92`).

**Test Results:** 17 lulus (6 blackbox, 11 whitebox). Cakupan 31% → **91%**, cabang 8/8, dan 10 dari 10 mutan
terbunuh. Basis path `KepalaKetidakpastian.pas` V(G) = 6, dengan 5 jalur layak diuji.

**Additional Comments:** `__repr__` dan `laporan()` berada di luar fitur yang dirancang. Rinciannya di
`docs/laporan_bagian6.md`.

### 2.7 SYSTEM TEST (MANUAL): DEMO WEB

Butir yang diuji: `demo_web/index.html`, dari sudut pandang pengunjung (blackbox, termasuk transisi keadaan).

**Test Owner:** Hegira Musyafa Kartiwan

**Test Date:** 3 November 2026 (13 kasus manual) dan 9 November 2026 (6 kasus lewat otomasi peramban).

**Test Results:** 19 kasus, **17 lulus dan 2 gagal**. Kedua kasus yang gagal menjadi BUG-01 (W-16) dan
BUG-02 (W-09).

**Additional Comments:** kasus W-06, 09, 10, 15, 16, dan 17 dijalankan dengan otomasi peramban (Chrome dan
Firefox sungguhan), bukan manual (§4.3 butir 5). Daftar kasusnya ada di `docs/uji_demo_web.md`.

## 3 TEST ASSESSMENT

### 3.1 Kelengkapan terhadap kriteria rencana uji

Kriteria minimum di rencana uji (TP-MARSPEC-01 §6, berasal dari `PEMBAGIAN_TUGAS.md` §2):
- minimal 10 kasus blackbox dengan minimal 2 teknik;
- basis path untuk satu fungsi;
- cakupan cabang diukur, dan cabang yang tidak tertutup dijelaskan;
- semua TODO terisi;
- `pytest` hijau di GitHub Actions.

Rencana uji **tidak** menetapkan ambang persentase cakupan.

| Bagian | Kasus blackbox (contoh kerangka) | Teknik blackbox | Basis path | Cakupan diukur | TODO tersisa |
|---|---|---|---|---|---|
| 1 | 8 (2) ✗ | nilai batas, partisi ekivalen, tebakan galat ✓ | ada; V(G) belum ditulis pemilik | ✓ | 0 |
| 2 | 6 (1) ✗ | partisi ekivalen, nilai batas, tebakan galat ✓ | ada; V(G) tertulis 4, seharusnya 5 (Lampiran E) | ✓ | 0 |
| 3 | 10 (2) ✓ | partisi ekivalen, nilai batas, tebakan galat ✓ | ada, 13 jalur + 2 uji jalur tambahan | ✓ | 0 |
| 4 | 13 (1) ✓ | partisi ekivalen, nilai batas, tebakan galat ✓ | V(G) = 10, 7 jalur diuji ✓ | ✓ | 0 |
| 5 | 6 (1) ✗ | partisi ekivalen, tebakan galat, tabel keputusan ✓ | belum lengkap ✗ | ✓ | 0 |
| 6 | 6 (1) + 19 demo web ✓ | 5 teknik ✓ | V(G) = 6, 5 jalur layak ✓ | ✓ | 0 |

Kriteria "`pytest` hijau di GitHub Actions" **belum terpenuhi** untuk bagian 1, 4, dan 5, karena berkasnya belum
di-commit (§4.3).

### 3.2 Cakupan per bagian

"Cakupan" = (baris tertutup + cabang tertutup) ÷ (baris + cabang). Rumusnya sama dengan angka total
coverage.py, tetapi dihitung hanya atas fungsi milik bagian itu (cara ukur di Lampiran F). Angka "sebelum"
diukur pada commit kerangka `d4714b7`.

| Bagian | Fungsi yang diukur | Baris sebelum → sesudah | Cabang sebelum → sesudah | Cakupan |
|---|---|---|---|---|
| 1 | `SE1d`, `ChannelEqualizer`, `CNN1DExtractor`, `MCformerEncoder` | 64/66 → 66/66 | 17/20 → 19/20 | 94% → **99%** |
| 2 | `ContextEncoder`, `ContextGatedFusion`, `MoEChannelRouter`, `MaritimeAMCv3`, `validate_config` | 54/56 → 56/56 | 8/10 → 10/10 | 94% → **100%** |
| 3 | `MaritimeChannel` (`__init__`, `apply`) | 60/68 → 67/68 | 14/24 → 23/24 | 80% → **98%** |
| 4 | `moe_balance_loss`, `equalizer_aux_loss`, `scan_indices`, `load_X_inmemory` | 6/62 → 62/62 | 1/18 → 17/18 | 9% → **99%** |
| 5 | `cocokkan_config`, `mode_oracle_latih`, `per_snr_table` | 5/46 → 42/46 | 1/22 → 17/22 | 9% → **87%** |
| 5 | `main()` (hanya penjaga argumen yang termasuk lingkup) | 60/212 → 68/212 | 13/68 → 20/68 | — |
| 6 | seluruh `marspec_ketidakpastian.py` | 27/86 → 78/86 | 2/8 → 8/8 | 31% → **91%** |

**Catatan: cakupan awal bagian 1 dan 2 sudah 94%.** Contoh bawaan kerangka menjalankan `MaritimeAMCv3`
utuh, dan model utuh memanggil hampir semua komponen. Akibatnya hampir semua baris sudah *dieksekusi*
sebelum satu pun uji bagian 1–2 ditulis, walaupun belum ada yang *memeriksa* perilaku komponen itu. Nilai
tambah uji bagian 1–2 terletak pada asersinya, misalnya "gate tidak bergantung sinyal" atau "NaN tidak
disembunyikan", bukan pada kenaikan persentase. Ini contoh nyata bahwa cakupan baris yang tinggi tidak sama
dengan uji yang kuat.

### 3.3 Yang belum teruji dengan cukup

**Baris dan cabang yang tidak tertutup:**

| Bagian | Letak | Isi | Alasan | Bisa dijangkau? |
|---|---|---|---|---|
| 1 | cabang 132→129, `ChannelEqualizer.__init__` | `if m.bias is not None` bernilai salah | semua `Conv1d` di kelas ini dibuat dengan bias bawaan (`True`) | **Tidak**, mustahil tanpa mengubah kode |
| 3 | baris 161, `MaritimeChannel.__init__` | `raise ValueError` untuk `balik` yang tak sah | belum ada uji | Ya, cukup satu `pytest.raises` |
| 4 | cabang 697→701, `load_X_inmemory` | `if n <= 100_000` bernilai salah (lebih dari 100.000 indeks) | butuh 100.001 indeks; keluarannya saja 819 MB RAM | Ya, tetapi terlalu mahal untuk uji unit |
| 5 | baris 192 | `--size` beda pada arsitektur v3 → `SystemExit` | belum ada uji | Ya |
| 5 | baris 200 | nilai cfg bertipe tuple diubah ke list sebelum dibandingkan | belum ada uji | Ya |
| 5 | baris 202 | nilai cfg sama → `continue` | belum ada uji | Ya |
| 5 | baris 219 | `mode_oracle_latih` tanpa `config.json` → `('', False)` | belum ada uji | Ya |
| 5 | cabang 330→332 | `per_snr_table` tanpa SNR ≥ 10 dB | belum ada uji | Ya |
| 5 | `main()` baris 1034, 1054 | penjaga `--resume` dengan setelan berbeda; `--anchor-ckpt` menunjuk berkas yang tidak ada | belum ada uji | Ya, keduanya jatuh sebelum data dimuat |
| 6 | baris 173, 180–193 | `__repr__`, `laporan()` | di luar fitur yang dirancang (`laporan_bagian6.md` §8) | Ya |

Hanya **satu** celah yang mustahil dilalui (cabang bagian 1) dan satu yang terlalu mahal (cabang bagian 4).
Celah lain tidak membutuhkan GPU atau dataset. Celah itu tersisa karena ujinya belum ditulis.

**Catatan tinjauan silang.** Daftar ini untuk tahap 4 (Tinjau). Semuanya bisa diperbaiki tanpa mengubah
`marspec/`.

| # | Bagian | Temuan | Saran |
|---|---|---|---|
| T-1 | 2 | Docstring `test_wb_basis_path_validate_config` menulis V(G) = 4. `validate_config` punya **4** keputusan (tiga `if … % … != 0` dan `if errs`), jadi V(G) = 4 + 1 = **5** (13 sisi, 10 simpul) | Ganti angkanya jadi 5. Jalur ke-5 sudah ada: `test_wb_validate_config_dua_pelanggaran` bebas linier dari empat jalur lainnya, jadi himpunan basisnya sebenarnya sudah lengkap |
| T-2 | 5 | `test_wb_basis_path_cocokkan_config` menjalankan jalur yang sama dengan `test_wb_cocokkan_config_tanpa_config_json` (tanpa `config.json`). Basis path-nya belum benar-benar dikerjakan | Lengkapi graf alir; V(G) = 11 (Lampiran E.5). Tambah kasus untuk baris 192, 200, 202 dan `arch='dpdrsn…'` |
| T-3 | 5 | Docstring `test_bb_per_snr_table_akurasi` menjanjikan tiga aturan: benar semua, **salah semua**, separuh. Datanya hanya memberi akurasi 0,5 / 1,0 / 0,5, jadi aturan "salah semua" (0,0) tidak pernah diuji | Ubah satu kelompok SNR supaya semua prediksinya salah |
| T-4 | 1 | V(G) dan graf alir `ChannelEqualizer.forward` belum ditulis | Lihat Lampiran E.1: V(G) = 5. Jalur nol iterasi hanya layak bila `n_layers=0` **dan** `n_filters=2` (diperiksa: `n_filters=16` → `RuntimeError`) |
| T-5 | 3 | Baris 161 (`balik` tak sah → `ValueError`) belum diuji | Satu uji tebakan galat, misalnya `MaritimeChannel(balik="zf")` |
| T-6 | 1–5 | Kolom teknik dan dasar harapan belum didokumentasikan pemilik | Isi `templat_kasus_uji.md` per bagian, atau koreksi Lampiran D |

### 3.4 Di luar lingkup

Kode berikut sengaja tidak diuji, sesuai TP-MARSPEC-01 §5. Kode inilah yang menahan cakupan paket di 44%:
- `baseline_models.py` (0%) dan `maritime_amc_lite.py` (21%): model pembanding, bukan MARSPEC.
- Perulangan latih di `train_maritime_v3.main()` dan `run_eval()` di `eval_maritime_snr.py`: butuh dataset
  RadioML 21 GB dan waktu latih berjam-jam.
- Bobot model terlatih: akurasi model tidak diuji. Yang diuji adalah **kebenaran perilaku kode**.

**Batas alat ukur.** coverage.py tidak menghitung ekspresi bersyarat satu baris (`a if c else b`) sebagai
cabang. `MaritimeChannel.apply` punya dua ekspresi seperti ini, dan keduanya diperiksa lewat basis path
(Lampiran E.3).

## 4 TEST RESULTS

### 4.1 Uji otomatis

| Bagian | Uji blackbox | Uji whitebox | Lulus | Gagal | Dilewati (TODO) | xfail (bug) |
|---|---|---|---|---|---|---|
| 1 | 8 | 5 | 13 | 0 | 0 | 0 |
| 2 | 6 | 4 | 10 | 0 | 0 | 0 |
| 3 | 10 | 26 | 36 | 0 | 0 | 0 |
| 4 | 13 | 18 | 29 | 0 | 0 | 2 |
| 5 | 6 | 6 | 12 | 0 | 0 | 0 |
| 6 | 6 | 11 | 17 | 0 | 0 | 0 |
| **Jumlah** | **49** | **70** | **117** | **0** | **0** | **2** |

Uji berparameter (`@pytest.mark.parametrize`) dihitung per kasus, sama seperti cara pytest menghitungnya.
Semua kasus blackbox di Lampiran D berstatus **Lulus** (hasil aktual sesuai harapan), kecuali B4-12 dan B4-13
yang berstatus xfail karena BUG-03.

### 4.2 Uji demo web

19 kasus (W-01 sampai W-19): **17 lulus, 2 gagal** (W-09 → BUG-02, W-16 → BUG-01). Hasil per kasus beserta
tangkapan layarnya ada di `docs/uji_demo_web.md` dan `docs/tangkapan/`.

### 4.3 Penyimpangan dari rencana uji

1. **Bagian 4 dikerjakan belakangan**, pada 10 November 2026, setelah draf pertama laporan ini. Penulis
   ujinya: **[isi]**. Sejak itu tidak ada lagi uji TODO yang dilewati.
2. **Target 10 kasus blackbox belum tercapai di bagian 1, 2, dan 5** (8, 6, dan 6 kasus). Bagian 3
   tepat 10, tetapi 2 di antaranya contoh kerangka. Rinciannya di §3.1.
3. **Dokumen rancangan per anggota baru ada untuk bagian 6.** Tabel kasus uji (`templat_kasus_uji.md`)
   dan graf alir basis path (`templat_basis_path.md`) bagian 1–5 belum masuk repositori. Lampiran D dan E
   disusun penyusun laporan dari kode uji. Kolom "Teknik" bagian 1–5 **disimpulkan dari docstring**, dan
   V(G) bagian 1, 3, dan 5 **dihitung penyusun**. Pemilik bagian perlu memeriksa keduanya.
4. **Berkas uji bagian 1, 4, dan 5 belum di-commit.** Ketiganya sudah ada di salinan kerja penyusun dan lulus
   di sana, tetapi belum dijalankan GitHub Actions. Angka di laporan ini berasal dari menjalankan lokal (§2).
5. **Demo web: 6 dari 19 kasus dijalankan dengan otomasi peramban, bukan manual.** Kasusnya W-06, 09, 10, 15,
   16, dan 17, dijalankan dengan Chrome 157 dan Firefox 151 sungguhan. Skripnya ada di `docs/otomasi/` dan
   tangkapan layarnya di `docs/tangkapan/`. Pembagiannya dicatat di `docs/uji_demo_web.md`.

### 4.4 Bug yang ditemukan

| ID | Bagian | Uji | Ringkasan | Tingkat | Status |
|---|---|---|---|---|---|
| BUG-01 | 6 | W-16 | `data/demo.js` gagal dimuat → kerangka kosong tanpa pesan, padahal kode sudah menyiapkan pesan galat yang ramah. `TypeError` di `Object.keys(D.keluarga)` terjadi sebelum pesan itu sempat tampil | minor | baru |
| BUG-02 | 6 | W-09 | soal Tantang AI yang belum dijawab dibuang saat pindah ke Jelajahi, lalu muncul soal baru saat kembali. Skor tetap | minor (mungkin disengaja) | baru |
| BUG-03 | 4 | B4-12, B4-13 | `load_X_inmemory` dengan indeks tak terurut yang memuat indeks di luar [0, N) (misalnya 600 pada berkas 576 baris, atau −1) mengembalikan baris berisi sisa memori tanpa galat. Untuk masukan serupa yang terurut, jalur satunya melempar `IndexError` | kritis menurut definisi (hasil salah tanpa peringatan); dampak praktis rendah karena `scan_indices` selalu memberi indeks sah | baru |

Langkah reproduksi lengkap ada di `LOG_BUG.md`.

**Penyebab BUG-03.** Jalur pindai-per-blok menyiapkan keluaran dengan `np.empty`, lalu hanya mengisi baris
yang indeksnya jatuh di salah satu blok `[a, b)` dengan 0 ≤ a < N. Indeks ≥ N atau negatif tidak pernah jatuh di
blok mana pun, jadi barisnya tetap berisi sisa memori. Nilainya berubah setiap kali dijalankan (contoh: `3,3e-27`,
`1,6e+29`). Selain dua uji BUG-03, bagian 1–5 tidak menandai uji lain sebagai xfail.

### 4.5 Kesimpulan hasil

Pada semua kasus yang diuji di bagian 1, 2, 3, 5, dan 6, **kode Python MARSPEC berperilaku sesuai
spesifikasinya**. Bagian 4 menemukan satu bug di pemuat data (BUG-03), dan dua bug minor ditemukan di demo web.
Laporan ini **belum final**. Syarat yang belum dipenuhi tercantum di §5.1.

## 5 SUGGESTED ACTIONS

### 5.1 Sebelum laporan ini final (kelompok)

| # | Tindakan | Penanggung jawab |
|---|---|---|
| 1 | Commit berkas uji bagian 1, 4, dan 5, lalu pastikan lulus di GitHub Actions | pemilik bagian 1, 4, 5 |
| 2 | Periksa Lampiran D dan E, lalu tutup catatan T-1 sampai T-6 atau tolak dengan alasan | pemilik bagian 1–5 |
| 3 | Isi tabel kasus uji (`templat_kasus_uji.md`) dan graf alir (`templat_basis_path.md`) bagian 1–5 | pemilik bagian 1–5 |
| 4 | Isi semua `[isi]`: nama penguji bagian 1 dan 5, nama lengkap Maitsa, penulis uji bagian 4, penemu BUG-03 di `LOG_BUG.md`, penyusun versi di Version History | kelompok |
| 5 | Ubah status BUG-01 sampai BUG-03 di `LOG_BUG.md` dari `baru` menjadi `dikonfirmasi` atau `bukan bug` setelah dibicarakan dengan pengembang MARSPEC | pencatat bug |

### 5.2 Untuk pengembang MARSPEC (di repositori asli, bukan `marspec/` yang beku)

| Bug | Tindakan yang disarankan |
|---|---|
| BUG-03 | Periksa rentang indeks di awal `load_X_inmemory`, misalnya `if n and (indices.min() < 0 or indices.max() >= N): raise IndexError(...)`. Setelah diperbaiki dan salinan beku diperbarui, uji B4-12/B4-13 akan XPASS; `strict=True` membuatnya gagal sebagai tanda penanda `xfail`-nya harus dihapus |
| BUG-01 | Periksa apakah data demo sudah termuat **sebelum** baris pertama yang memakainya (`Object.keys(D.keluarga)`), supaya pesan galat yang sudah disiapkan bisa tampil |
| BUG-02 | Putuskan apakah membuang soal yang belum dijawab memang disengaja. Kalau ya, tulis perilaku itu di halaman. Kalau tidak, simpan soal yang sedang terbuka saat pindah ke Jelajahi |

### 5.3 Uji yang layak ditambah

Semua celah di §3.3 yang bertanda "Ya" bisa dijangkau tanpa GPU atau dataset. Urutan yang disarankan, dari
yang paling berguna dalam praktik:
1. bagian 5, baris 192 (`--size` beda pada v3);
2. bagian 5, baris 200 dan 202, serta `arch='dpdrsn…'`;
3. bagian 5, aturan "salah semua" di `per_snr_table`;
4. bagian 3, baris 161;
5. bagian 5, baris 219, cabang 330→332, dan `main()` baris 1034 dan 1054.

---

## LAMPIRAN A: TEST SUMMARY REPORT APPROVAL

Yang bertanda tangan di bawah ini menyatakan telah meninjau Test Summary Report Pengujian MARSPEC ini dan
menyetujui pendekatan yang disajikannya. Perubahan pada laporan ini dikoordinasikan dengan dan disetujui oleh
yang bertanda tangan atau wakil yang mereka tunjuk.

| Nama | Jabatan | Peran | Tanda tangan | Tanggal |
|---|---|---|---|---|
| [isi] | Mahasiswa | Penguji bagian 1 | | |
| Elsa Novia | Mahasiswa | Penguji bagian 2 | | |
| Maitsa [lengkapi] | Mahasiswa | Penguji bagian 3 | | |
| M Tegar Fathan | Mahasiswa | Penguji bagian 4 | | |
| [isi] | Mahasiswa | Penguji bagian 5 | | |
| Hegira Musyafa Kartiwan | Mahasiswa | Penguji bagian 6 | | |

## LAMPIRAN B: REFERENCES

Tabel berikut merangkum dokumen yang dirujuk dalam laporan ini.

| Nama dokumen & versi | Deskripsi | Lokasi |
|---|---|---|
| TP-MARSPEC-01 | Rencana uji (IEEE 829-1983) | `docs/test_plan.md` |
| Pembagian Tugas | Bagian, pemilik, target per anggota, tahapan | `PEMBAGIAN_TUGAS.md` |
| Panduan Pengujian | Teknik blackbox/whitebox, konvensi penamaan, cara ukur cakupan, penanganan bug | `docs/panduan_pengujian.md` |
| Templat kasus uji & basis path | Format rancangan uji per anggota | `docs/templat_kasus_uji.md`, `docs/templat_basis_path.md` |
| Laporan bagian 6 | *Test Design Specification* bagian 6: rincian kasus, basis path, cakupan, mutasi | `docs/laporan_bagian6.md` |
| Uji demo web | Daftar periksa dan hasil 19 kasus demo web | `docs/uji_demo_web.md`, `docs/otomasi/`, `docs/tangkapan/` |
| Log Bug | Laporan insiden BUG-01 sampai BUG-03 beserta langkah reproduksi | `LOG_BUG.md` |
| Versi kode yang diuji | Sidik SHA-256 salinan beku 19 Okt 2026 | `marspec/VERSI.md` |
| Kode uji | `test_1_…py` sampai `test_6_…py`, fixture di `conftest.py` | `tests/` |
| Alur CI | Menjalankan `pytest` + cakupan di setiap push | `.github/workflows/uji.yml` |
| IEEE Std 829-1983 (ringkasan) | Kerangka *Test Plan* | `docs/IEEEStandardTestPlans.doc` |
| Templat Test Summary Report (UP Template Version 12/31/08) | Templat laporan ini | `docs/test-summary-report-template.doc` |

## LAMPIRAN C: KEY TERMS

Tabel berikut memuat definisi istilah yang dipakai dalam laporan ini.

| Istilah | Definisi |
|---|---|
| Blackbox | Kasus uji dirancang dari apa yang dijanjikan (docstring, README, teks bantuan, perilaku yang masuk akal), tanpa melihat isi kode |
| Whitebox | Kasus uji dirancang dari struktur kode: percabangan, perulangan, dan jalur |
| Partisi ekivalen | Masukan dibagi menjadi kelompok yang perilakunya sama, lalu satu wakil per kelompok diuji |
| Nilai batas | Uji tepat di batas, tepat di bawah, dan tepat di atasnya |
| Tabel keputusan | Uji atas gabungan beberapa kondisi beserta hasil yang diharapkan untuk tiap gabungan |
| Transisi keadaan | Uji perpindahan antar-mode sistem, misalnya Jelajahi ↔ Tantang AI di demo web |
| Tebakan galat | Masukan yang sering membuat program salah: NaN, panjang salah, berkas tidak ada, indeks di luar rentang |
| Basis path | Satu kasus uji untuk tiap jalur independen dalam graf alir sebuah fungsi |
| V(G) | Kompleksitas siklomatik = jumlah jalur independen = jumlah kondisi sederhana + 1 = sisi − simpul + 2 |
| Kondisi sederhana / majemuk | Kondisi majemuk (`a and b`) dipecah menjadi kondisi sederhana; setiap operan dihitung satu simpul keputusan |
| Cakupan baris | Persentase baris yang pernah dijalankan oleh uji |
| Cakupan cabang | Persentase arah percabangan (benar/salah) yang pernah diambil oleh uji |
| Fixture | Bahan uji bersama yang disiapkan pytest, misalnya `cfg_kecil` atau `h5_tiruan` |
| xfail (`strict=True`) | Uji yang diharapkan gagal karena bug yang sudah dicatat. Kalau bugnya diperbaiki, uji itu lulus (XPASS), dan `strict` membuat pytest melaporkannya sebagai gagal |
| Uji mutasi | Kode sengaja dirusak sedikit (mutan) untuk memeriksa apakah uji mendeteksinya. Mutan "terbunuh" bila ada uji yang gagal |
| Mutan ekuivalen | Mutan yang perilakunya identik dengan kode asli, sehingga tidak mungkin terbunuh oleh uji mana pun |
| Salinan beku | Salinan `marspec/` 19 Okt 2026 yang tidak boleh diubah selama pengujian; perbaikan bug dilakukan di repositori asli |
| Jalur fancy / jalur blok | Dua cara `load_X_inmemory` membaca berkas HDF5: langsung dengan indeks terurut, atau memindai per blok untuk indeks acak |
| IQ | Sinyal radio sebagai pasangan komponen *in-phase* dan *quadrature*, di sini tensor `[2, 1024]` |
| SNR | *Signal-to-noise ratio*, perbandingan daya sinyal terhadap derau, dalam dB |
| MoE / gate | *Mixture of experts*: beberapa sub-jaringan (pakar) yang dibobot oleh gate. Di MARSPEC, gate hanya membaca konteks laut |
| GitHub Actions | Layanan CI yang menjalankan seluruh uji otomatis setiap kali ada push atau pull request |

---

## LAMPIRAN D: KASUS UJI BLACKBOX

Semua kasus di lampiran ini **Lulus** (hasil aktual sesuai harapan), kecuali B4-12 dan B4-13 yang berstatus
xfail karena BUG-03. "(contoh)" menandai kasus bawaan kerangka. Kolom "Teknik" bagian 1–5 disimpulkan dari
docstring (lihat §4.3 butir 3).

### D.1 Bagian 1: Komponen sinyal

| ID | Teknik | Kasus & masukan | Hasil yang diharapkan | Fungsi uji |
|---|---|---|---|---|
| B1-01, B1-02 | Nilai batas | `ChannelEqualizer(16, 4, ctx_dim=8)`, batch 1 dan 8, `x [B, 2, 1024]` | keluaran berbentuk sama dengan masukan | `test_bb_equalizer_mempertahankan_bentuk[1/8]` (contoh) |
| B1-03 | Partisi ekivalen | `SE1d(c=8, r=4)`, masukan non-negatif `rand [2, 8, 64]` | bentuk sama, semua nilai di [0, 1] | `test_bb_se1d_bentuk_dan_rentang_bobot` |
| B1-04, B1-05 | Nilai batas | `CNN1DExtractor((16, 32, 64), out_dim=32, pool=4)`, batch 1 dan 8 | keluaran `[B, 32]` | `test_bb_cnn_keluaran_out_dim[1/8]` |
| B1-06 | Partisi ekivalen | `MCformerEncoder` mode eval, dua kali forward pada masukan yang sama | kedua hasil identik | `test_bb_mcformer_deterministik_saat_eval` |
| B1-07 | Tebakan galat | `MCformerEncoder`, satu nilai masukan NaN | NaN muncul di keluaran (tidak dibersihkan diam-diam) | `test_bb_mcformer_masukan_nan` |
| B1-08 | Partisi ekivalen (tak sah) | `MCformerEncoder`, panjang 1000 (bukan 1024) | `RuntimeError`/`ValueError` | `test_bb_mcformer_panjang_masukan_salah` |

### D.2 Bagian 2: Konteks & model utuh

| ID | Teknik | Kasus & masukan | Hasil yang diharapkan | Fungsi uji |
|---|---|---|---|---|
| B2-01 | Partisi ekivalen | `MaritimeAMCv3(cfg_kecil)`, IQ `[4, 2, 1024]`, konteks `[4, 4]` | logits `[4, 24]`, IQ hasil equalizer berbentuk sama, gate `[4, n_experts]` berjumlah 1 | `test_bb_model_keluaran_tiga_tensor` (contoh) |
| B2-02 | Partisi ekivalen | `MoEChannelRouter`: dua sinyal berbeda (skala ×5), konteks sama | bobot gate identik | `test_bb_moe_gate_tak_bergantung_sinyal` |
| B2-03 | Partisi ekivalen | `ContextEncoder(4, 64, 32, 3)`, masukan `[4, 4]` | `[4, 32]` | `test_bb_context_encoder_bentuk` |
| B2-04 | Partisi ekivalen | `ContextGatedFusion(64, 32, 64)`, sinyal `[4, 64]`, konteks `[4, 32]` | `[4, 64]` | `test_bb_fusion_bentuk_keluaran` |
| B2-05 | Nilai batas | model utuh, batch 1 | bentuk benar, gate berjumlah 1 | `test_bb_model_batch_satu` |
| B2-06 | Tebakan galat | konteks 3 fitur (seharusnya 4) | `RuntimeError`/`ValueError` | `test_bb_model_konteks_salah_ukuran` |

### D.3 Bagian 3: Simulator kanal laut

Masukan dasar: sinyal acak `[2, 1024]`, jarak 30 km, suhu laut 28 °C, SNR 20 dB.

| ID | Teknik | Kasus & masukan | Hasil yang diharapkan | Fungsi uji |
|---|---|---|---|---|
| B3-01, B3-02 | Partisi ekivalen | laut tenang (hs 0,3 m) dan badai (hs 5,5 m) | `[2, 1024]` float32, simpangan baku magnitudo = 1 | `test_bb_keluaran_ternormalisasi[0.3/5.5]` (contoh) |
| B3-03 | Partisi ekivalen | dua kanal dengan benih 123 | keluaran identik | `test_bb_benih_sama_hasil_sama` |
| B3-04, B3-05 | Nilai batas | SNR −6 dan 30 dB, komponen tambahan mati | bentuk benar, float32, semua nilai hingga | `test_bb_snr_batas[-6.0/30.0]` |
| B3-06, B3-07 | Tebakan galat | jarak 0 dan −5 km | keluaran valid dan hingga (kode memotong jarak ke ≥ 0,1 km) | `test_bb_jarak_nol_atau_negatif[0.0/-5.0]` |
| B3-08 | Partisi ekivalen | duct, interferensi, dua sinar, dan fasa global dimatikan | valid, simpangan baku magnitudo = 1 | `test_bb_semua_komponen_dimatikan` |
| B3-09, B3-10 | Partisi ekivalen | panjang sinyal 512 dan 2048 | panjang dipertahankan | `test_bb_panjang_sinyal[512/2048]` |

### D.4 Bagian 4: Rugi & pemuat data

| ID | Teknik | Kasus & masukan | Hasil yang diharapkan | Fungsi uji |
|---|---|---|---|---|
| B4-01 | Partisi ekivalen | gate seragam: 8 sampel × 4 pakar, masing-masing 0,25 | rugi penyeimbang = 0 | `test_bb_balance_gate_seragam_nol` (contoh) |
| B4-02 | Partisi ekivalen | tiga partisi beban: timpang penuh (semua ke pakar 0), timpang sebagian (separuh sampel ke pakar 0), seimbang | penuh > sebagian > seimbang, dan sebagian > 0 (aktual 4,0 / 1,0 / 0,0) | `test_bb_balance_satu_pakar_dominan` |
| B4-03 | Nilai batas | `iq_eq` identik dengan `iq_clean`, semua sampel augmentasi | rugi ≈ 0 (selisih ≤ 1e-6) | `test_bb_eq_aux_identik_nol` |
| B4-04–B4-07 | Partisi ekivalen | `iq_eq` diputar fasa global 0, π/2, π, dan 2,0287 rad | rugi sama dengan tanpa putaran (relatif 1e-4); rugi acuan > 0,01 | `test_bb_eq_aux_kebal_rotasi_fasa[…]` |
| B4-08 | Nilai batas | `scan_indices` dengan `snr_min=0` | semua SNR ≥ 0; 96 baris ber-SNR tepat 0 ikut; total 384; label dan SNR cocok dengan baris aslinya | `test_bb_scan_indices_snr_min` |
| B4-09 | Partisi ekivalen | `max_samples=10` | tepat 10 indeks, naik tegas (tanpa duplikat), label cocok | `test_bb_scan_indices_max_samples` |
| B4-10 | Partisi ekivalen | `load_X_inmemory`, indeks 0, 7, 288, 575 | `[4, 2, 1024]` float32; baris ke-i = `X[indices[i]]` di-transpose | `test_bb_load_x_bentuk_dan_isi` |
| B4-11 | Tebakan galat | indeks berulang `[3, 3, 250]` | isi ketiga baris benar | `test_bb_load_x_indeks_duplikat` |
| B4-12, B4-13 | Tebakan galat | indeks di luar rentang, `[600, 3]` dan `[5, -1]`, pada berkas 576 baris | ditolak dengan galat, seperti jalur fancy | `test_bb_load_x_indeks_di_luar_rentang[…]`: **xfail, BUG-03** |

### D.5 Bagian 5: Evaluasi & CLI

| ID | Teknik | Kasus & masukan | Hasil yang diharapkan | Fungsi uji |
|---|---|---|---|---|
| B5-01 | Partisi ekivalen (tak sah) | `--anchor-weight 10` tanpa `--anchor-ckpt` | `SystemExit` yang menyebut `--anchor-ckpt` | `test_bb_cli_anchor_weight_tanpa_ckpt` (contoh) |
| B5-02 | Partisi ekivalen (tak sah) | `--init-from` menunjuk berkas yang tidak ada | `SystemExit` yang menyebut `--init-from` dan nama berkasnya | `test_bb_cli_init_from_tidak_ada` |
| B5-03 | Tebakan galat | `--cfg d_model` (tanpa `=`) | `ValueError` yang menjelaskan format `KUNCI=NILAI` | `test_bb_cli_cfg_tanpa_sama_dengan` |
| B5-04 | Tebakan galat | `--cfg ngawur=1` | `KeyError` "kunci cfg tak dikenal" | `test_bb_cli_cfg_kunci_tak_dikenal` |
| B5-05 | Tabel keputusan | `per_snr_table`: 3 SNR × 2 sampel | SNR [−6, 0, 10], akurasi [0,5; 1,0; 0,5] | `test_bb_per_snr_table_akurasi` (lihat T-3) |
| B5-06 | Partisi ekivalen | `config.json` berisi `balik_2sinar='zf-pita'` | `mode_oracle_latih` → `('zf-pita', False)` | `test_bb_mode_oracle_ckpt_d_zf` |

### D.6 Bagian 6: Ketidakpastian + demo web

Enam kasus otomatis B6-01–B6-06 dirinci di `docs/laporan_bagian6.md` §4.1 dan §6, termasuk nilai aktual,
partisi, dan tabel keputusan. Sembilan belas kasus demo web W-01–W-19 dirinci di `docs/uji_demo_web.md`.

---

## LAMPIRAN E: WHITEBOX, BASIS PATH DAN UJI CABANG

Aturan yang dipakai mengikuti `docs/templat_basis_path.md`: **kondisi majemuk dipecah**, sehingga setiap
operan `and`/`or` menjadi satu simpul keputusan. V(G) = jumlah simpul keputusan + 1.

### E.1 Bagian 1: `ChannelEqualizer.forward` (V(G) dihitung penyusun)

Keputusan: perulangan `for` + tiga operan dari `film is not None and ctx_feat is not None and i == film_at`.
Ada 4 simpul keputusan, jadi **V(G) = 5** (11 sisi, 8 simpul).

| Kondisi | Benar diuji oleh | Salah diuji oleh |
|---|---|---|
| `for`: masih ada lapis | semua uji | semua uji (keluar perulangan) |
| `film is not None` | `test_wb_equalizer_cabang_film` | `test_wb_equalizer_tanpa_film_ctx_dim_nol` (`ctx_dim=0`) |
| `ctx_feat is not None` | `test_wb_basis_path_equalizer_forward` | idem, panggilan dengan `ctx=None` |
| `i == film_at` | lapis berindeks 2 (`film_at = 4 // 2`) | lapis berindeks 0, 1, 3 |

Setiap kondisi sudah diuji ke dua arah. Satu jalur basis, yaitu perulangan nol kali, belum diuji. Jalur itu
hanya layak bila `n_layers=0` dan `n_filters=2`.

Uji whitebox lain di bagian ini:
- `test_wb_mcformer_layerdrop_hanya_saat_latih`: kondisi `training and drop_path > 0 and rand < drop_path`.
- `test_wb_mcformer_cabang_grad_checkpoint`: cabang `grad_checkpoint and training` memberi hasil yang sama
  dengan jalur biasa.

### E.2 Bagian 2: `validate_config`

Keputusan: `d_model % nhead != 0`, `fusion % 2 != 0`, `1024 % patch != 0`, dan `if errs`. Ada 4 simpul
keputusan, jadi **V(G) = 5**.

| Jalur | Masukan (dari `CFG_V3` yang sah) | Hasil | Fungsi uji |
|---|---|---|---|
| J1 | semua sah | `None` | `test_wb_basis_path_validate_config` |
| J2 | `mcf_d_model=33` | `ValueError` menyebut `mcf_d_model` | idem |
| J3 | `fusion_d=383` | `ValueError` menyebut `fusion_d` | idem |
| J4 | `mcf_patch_len=30` | `ValueError` menyebut `mcf_patch_len` | idem |
| J5 | `fusion_d=383` **dan** `mcf_patch_len=30` | pesan menyebut keduanya | `test_wb_validate_config_dua_pelanggaran` |

J5 bebas linier dari J1–J4, karena J1–J4 tidak pernah melewati dua cabang "tambah galat" sekaligus. Jadi
lima jalur basis sudah lengkap (lihat T-1). Uji whitebox lain di bagian ini:
- `test_wb_validate_config_fusion_ganjil` (contoh).
- `test_wb_fusion_rumus_film`: dengan γ = −1 dan β = 0, masukan `norm1` tepat nol, sesuai rumus
  FiLM s·(1+γ)+β.

### E.3 Bagian 3: `MaritimeChannel.apply`

`apply` punya 11 pernyataan `if`/`for` dan 2 ekspresi bersyarat satu baris. Empat di antaranya majemuk
(`rho > 0.01 and dua_sinar`, `interference and n > 0`, `balik and tau2`, `_pita is None or len != N`).
Setelah dipecah, ada **17 kondisi sederhana, jadi V(G) = 18**. Catatan kerangka menyebut 12 keputusan karena
tidak memecah kondisi majemuk.

Penguji memakai metode **jalur dasar + balik satu keputusan**. J1 adalah jalur dasar dengan semua komponen
menyala kecuali offset LO, hs 5,5 m, dan 1 interferer. Tiap jalur lain membalik satu keputusan. Cabang yang
diambil dibuktikan lewat urutan pemanggilan RNG yang dicatat `_RNGJejak`.

| Jalur | Yang dibalik dari J1 | Keputusan |
|---|---|---|
| J1 | — | jalur dasar |
| J2 | `duct_h_m=None` | tinggi duct dihitung dari suhu laut |
| J3 | `duct=False` | ekspresi bersyarat `… if self.duct else 0.0` |
| J4 | `duct_h_m=0` | `duct > 0.05` salah |
| J5 | `freq_mhz=3000` | `rho > 0.01` salah |
| J6 | `dua_sinar=False` | `rho > 0.01` benar, `dua_sinar` salah |
| J7 | `hs_m=0.3` | `hs_m >= 0.6` salah (tanpa Doppler) |
| J8 | `lo_ppm=20` | `lo_ppm > 0` benar |
| J9 | `interference=False` | operan pertama salah |
| J10 | `n_interferers=0` | operan kedua salah |
| J11 | `n_interferers=2` | perulangan dua kali |
| J12 | `fasa_global=False` | fasa global salah |
| J13 | `balik="zf-pita"`, `dua_sinar=False` | `balik` benar, `tau2` salah |

Uji jalur tambahan:
- `test_wb_jalur_balik_pita_cache` mencakup tiga panggilan: `balik` dan `tau2` benar, lalu cache dibuat,
  dipakai ulang, dan dibuat ulang ketika panjangnya berubah.
- `test_wb_jalur_std_nol`: `std > 1e-8` salah.
- `test_wb_perulangan_interferer[0/1/6/7]`: batas perulangan maksimal 6.

Hasilnya, **ke-17 kondisi sederhana sudah diambil ke dua arah**. Sebagian jalur basis menurut rumus tidak
layak. Ekspresi `rho if tau2 else 0.0` dan operan `tau2` di `if self.balik and tau2` sepenuhnya ditentukan
oleh keputusan dua-sinar: `tau2 ≠ 0` tepat ketika cabang dua-sinar diambil, karena `delay ≥ 1`. Ini pola yang
sama dengan J6 di bagian 6.

### E.4 Bagian 4: `load_X_inmemory`

Kondisi sederhana:
1. `sisa is not None`
2. `perlu > 0.8 * sisa`
3. `n <= 100_000`
4. `n < 2`
5. indeks naik tegas (`np.all(np.diff(indices) > 0)`)
6. `for a in range(0, N, blok)`
7. `hi > lo`
8. `b / N >= laporan + 0.1`
9. `b == N`

`urut_ok = n < 2 or …` langsung diuji di `if urut_ok`, jadi diperlakukan sebagai satu kondisi majemuk
(kondisi 4 dan 5). Begitu juga cetakan kemajuan (kondisi 8 dan 9). Jadi **V(G) = 10**.

Semua jalur yang diuji ada di `test_wb_basis_path_load_x[J1…J7]`. Jalur yang diambil dibuktikan dari cetakan
kemajuan, karena jalur fancy tidak mencetak apa pun.

| Jalur | Yang dibalik dari J1 | Masukan | Bukti jalur |
|---|---|---|---|
| J1 | — | RAM tak terbaca, indeks terurut `[0, 7, 288, 575]` | jalur fancy, tanpa cetakan |
| J2 | RAM terbaca dan cukup | sisa 10¹² byte | jalur fancy |
| J3 | RAM terbaca tapi kurang | sisa 1 byte | `SystemExit` "RAM tak cukup" |
| J4 | `n < 2` benar | satu indeks `[42]` | jalur fancy |
| J5 | indeks tak terurut | `[575, 3, 250, 120, 333, 450]`, blok 100 | jalur blok; jumlah sampel naik di tiap cetakan: 1, 2, 3, 4, 5, 6 |
| J6 | `hi > lo` salah | `[3, 550, 20]`, blok 100 | blok 100–499 kosong; jumlah sampel tertahan: 2, 2, 2, 2, 2, 3 |
| J7 | `b/N >= laporan+0.1` salah, `b == N` benar | indeks J5, blok 10 | 10 cetakan dari 58 putaran; cetakan terakhir 100,0% |
| J8 | `n <= 100_000` salah | ≥ 100.001 indeks | tidak diuji: keluarannya saja 819 MB RAM |
| J9 | perulangan nol kali | berkas X kosong | tidak diuji terpisah: semua indeks ke berkas kosong tidak sah, jadi jalur ini kasus khusus BUG-03 |

Uji whitebox lain di bagian ini:
- `test_wb_load_x_jalur_blok_indeks_acak[tak_terurut/duplikat]`: isi jalur blok sama dengan berkas asli dan
  dengan jalur fancy.
- `test_wb_load_x_ram_tak_cukup[10239/10240]`: nilai batas. Satu sampel butuh 8192 byte, dan ditolak bila
  8192 > 0,8 × sisa, jadi batasnya 10.240 byte.
- `test_wb_load_x_ram_tak_terbaca`: RAM `None` membuat pemeriksaan dilewati, sedangkan permintaan yang sama
  dengan RAM 1000 byte ditolak.
- `test_wb_scan_indices_cabang_max_samples[None/0/10/192/1000]`: dari 192 baris yang lolos, hanya `10`
  yang dipotong. `0` berarti "pakai semua", dan `192` tidak dipotong karena pembandingnya `>` tegas.
- `test_wb_eq_aux_tanpa_sampel_aug` (contoh).

**Uji mutasi** (8 mutan, dijalankan pada salinan kode):
- mutan yang terbunuh: jalur fancy dimatikan, `>` jadi `>=` di batas RAM, `>=` jadi `>` di `snr_min`, tanda
  sudut fasa dibalik, `max_samples=0` ikut memotong, pulang-awal `mask` nol dimatikan, `searchsorted` kiri jadi
  kanan;
- mutan ekuivalen: menghapus normalisasi `imp` tidak mengubah (std/mean)², jadi tidak ada uji yang bisa
  membunuhnya.

Jadi 7 dari 7 mutan non-ekuivalen terbunuh.

### E.5 Bagian 5: `cocokkan_config` (V(G) dihitung penyusun)

Kondisi sederhana:
1. `not p.exists()`
2. `arch != …`
3. `arch.startswith("dpdrsn")`
4. `arch in ("mcformer", "resnet1d")`
5. `size != …`
6. `for`
7. `k in _KUNCI_RUNTIME`
8. `isinstance(v_kini, tuple)`
9. `v_kini == v`
10. `k in timpa`

Jadi **V(G) = 11**. `cfg_timpa or {}` tidak dihitung karena berupa ekspresi nilai.

| Kondisi | Benar | Salah |
|---|---|---|
| 1 tak ada `config.json` | `…_tanpa_config_json` | uji lain |
| 2 arch beda | `…_arch_beda` | uji lain |
| 3 arch `dpdrsn…` | **belum** | semua |
| 4 arch baseline | `…_size_beda_tapi_baseline` | arch v3 |
| 5 size beda (v3) | **belum** (baris 192) | `…_timpa_bertentangan` |
| 6 `for` | `…_kunci_runtime_dilewati` | `…_size_beda_tapi_baseline` (cfg kosong) |
| 7 kunci runtime | `…_kunci_runtime_dilewati` | idem (`fusion_d`) |
| 8 tuple | **belum** (baris 200) | semua |
| 9 nilai sama | **belum** (baris 202) | `…_timpa_bertentangan` |
| 10 ditimpa | `…_timpa_bertentangan` | `…_kunci_runtime_dilewati` |

Ada **4 arah kondisi** yang belum diuji. `test_wb_basis_path_cocokkan_config` belum menambah jalur baru
(T-2). Untuk `mode_oracle_latih`, cabang tanpa `config.json` juga belum diuji (baris 219).

### E.6 Bagian 6: `KepalaKetidakpastian.pas`

**V(G) = 6** (19 sisi, 15 simpul). Ada 5 jalur layak, masing-masing diuji satu kasus. Jalur J6 mustahil
karena `k is None` terjadi tepat ketika simpul 6 bernilai ya dan simpul 7 bernilai tidak. Graf alir, tabel
jalur, cakupan, dan uji mutasi ada di `docs/laporan_bagian6.md` §7–§9.

---

## LAMPIRAN F: CARA MENGULANG ANGKA LAPORAN INI

```bash
pytest                                                        # 117 passed, 2 xfailed
pytest --cov=marspec --cov-branch --cov-report=term-missing   # cakupan per berkas (paket 44%)
pytest --cov=marspec --cov-branch --cov-report=json:cov.json  # bahan cakupan per fungsi
```

- **Cakupan per bagian** dihitung dari `cov.json`. Pada coverage.py 7.17.2 (format JSON 3), berkas ini
  memuat ringkasan per fungsi (`files → functions`). Untuk tiap bagian, `num_statements`, `covered_lines`,
  `num_branches`, dan `covered_branches` dijumlahkan atas fungsi dan metode kelas yang tercantum di §3.2.
- **Angka "sebelum"** diperoleh dengan perintah yang sama pada commit kerangka `d4714b7` (16 lulus / 57
  dilewati), misalnya `git worktree add ../kerangka d4714b7`. Isi `marspec/` di commit itu identik dengan
  sekarang; perbedaannya hanya akhir baris CRLF/LF, jadi kedua angka mengukur kode yang sama.
