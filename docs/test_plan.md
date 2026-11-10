# Rencana Uji (Test Plan) — Pengujian MARSPEC

| | |
|---|---|
| Acuan format | ANSI/IEEE Std 829-1983, *Test Plan* (ringkasan dari dosen: `docs/IEEEStandardTestPlans.doc`) |
| Mata kuliah | Pengujian dan Pengembangan Perangkat Lunak |
| Status | **Draf**, menunggu persetujuan kelompok (§16) |
| Laporan hasil | TSR-MARSPEC-01 (`docs/test_summary_report.md`) |

**Asal isi dokumen ini.** Rencana uji kelompok sudah disepakati sejak 19 Oktober 2026, tetapi tersebar di
beberapa berkas: `PEMBAGIAN_TUGAS.md` (diperbarui 1–2 November), `docs/panduan_pengujian.md`, `README.md`,
`docs/templat_kasus_uji.md`, `docs/templat_basis_path.md`, dan `marspec/VERSI.md`. Dokumen ini, disusun
10 November 2026, mengumpulkannya ke dalam kerangka IEEE 829. Isi yang **tidak** ada di berkas-berkas tersebut
dan baru ditambahkan saat penyusunan ini diberi tanda **[baru]**.

---

## 1. Test Plan Identifier

**TP-MARSPEC-01**, versi 0.9 (draf).

## 2. Introduction

**Ringkasan butir dan fitur yang diuji.** MARSPEC (Maritime Spectrum Intelligence System) adalah model
PyTorch untuk mengenali modulasi sinyal radio di kanal laut. Yang diuji adalah komponen model, model utuh,
simulator kanal laut, fungsi rugi, pemuat data, alat evaluasi dan CLI, kepala ketidakpastian, serta demo web.
Tujuan pengujian adalah memeriksa **kebenaran perilaku kode** dengan teknik blackbox dan whitebox, **bukan**
akurasi model terlatih.

**Kebutuhan dan riwayat butir uji.** MARSPEC dikembangkan sebagai bahan skripsi salah satu anggota (Hegira
Musyafa Kartiwan). Untuk tugas ini, kodenya disalin dari repositori pengembang pada 19 Oktober 2026 dan
dibekukan (§3).

**Dokumen terkait:**

| Dokumen | Isi |
|---|---|
| `PEMBAGIAN_TUGAS.md` | bagian sistem, pemilik, target per anggota, peran pendukung, tahapan |
| `docs/panduan_pengujian.md` | teknik blackbox/whitebox, konvensi penamaan, cara ukur cakupan, fixture, penanganan bug, jebakan |
| `README.md` | pemasangan lingkungan, isi repositori, perintah sehari-hari, alur kerja Git |
| `LOG_BUG.md` | format dan definisi tingkat laporan insiden |
| `marspec/VERSI.md` | versi dan sidik SHA-256 butir uji |

Proyek ini tidak punya rencana proyek, rencana QA, atau rencana manajemen konfigurasi formal. Fungsi
manajemen konfigurasi dijalankan oleh Git, salinan beku, dan aturan alur kerja di `README.md` §4.

**Rencana uji tingkat bawah.** Setiap anggota menyusun rancangan uji untuk bagiannya sendiri dengan dua
templat: `docs/templat_kasus_uji.md` (tabel kasus blackbox) dan `docs/templat_basis_path.md` (graf alir, V(G),
jalur independen). Contoh lengkapnya `docs/laporan_bagian6.md` (bagian 6) dan `docs/uji_demo_web.md` (demo web).

## 3. Test Items

| Butir uji | Baris | SHA-256 (12 karakter awal) |
|---|---|---|
| `marspec/maritime_amc_v3_arch.py` | 416 | `62e15f0cf278` |
| `marspec/train_maritime_v3.py` | 1250 | `c7ca71b7cc68` |
| `marspec/eval_maritime_snr.py` | 432 | `6d2fd85ace5b` |
| `marspec/marspec_ketidakpastian.py` | 193 | `32d3927ef352` |
| `demo_web/index.html` (+ `demo_web/data/`) | 609 | `80d4b082cf14` |

**Versi.** Salinan beku 19 Oktober 2026. Folder `marspec/` dan `demo_web/` **tidak boleh diubah** selama
pengujian. Bug dicatat, tidak diperbaiki di sini (§7).

**Media penyerahan.** Repositori Git kelompok di GitHub (privat, karena kode skripsi belum terbit). Keaslian
salinan diperiksa dengan membandingkan sidik SHA-256 di atas.

**Sumber spesifikasi.** MARSPEC tidak punya dokumen spesifikasi kebutuhan formal. Harapan uji blackbox
diturunkan dari: docstring dan komentar kode, teks bantuan argumen CLI, `README` pengembang, teks di layar demo
web, serta perilaku yang masuk akal. Setiap harapan harus bisa ditunjukkan dasarnya (`panduan_pengujian.md`
§6). Kalau dasarnya tidak jelas, tanyakan ke pengembang lalu catat jawabannya.

**Laporan bug terkait.** `LOG_BUG.md`. Saat rencana ini disepakati belum ada bug tercatat.

**Butir yang tidak diuji:** `marspec/baseline_models.py` dan `marspec/maritime_amc_lite.py` (lihat §5).

## 4. Features to be Tested

| Bagian | Fitur | Kode | Rancangan uji |
|---|---|---|---|
| 1 | Komponen sinyal: penyeimbang kanal, ekstraktor CNN, encoder MCformer, SE | `SE1d`, `ChannelEqualizer`, `CNN1DExtractor`, `MCformerEncoder` | `templat_kasus_uji.md` + `templat_basis_path.md`, bagian 1 |
| 2 | Konteks & model utuh: encoder konteks laut, fusi bergerbang konteks, router MoE, model lengkap, validasi konfigurasi | `ContextEncoder`, `ContextGatedFusion`, `MoEChannelRouter`, `MaritimeAMCv3`, `validate_config` | idem, bagian 2 |
| 3 | Simulator kanal laut: duct, dua sinar, Doppler gelombang, offset LO, interferensi, fasa global, pembalik pita, normalisasi | `MaritimeChannel.apply` | idem, bagian 3 |
| 4 | Rugi & pemuat data: rugi penyeimbang MoE, rugi bantu equalizer, pemilihan indeks dataset, pemuatan IQ ke RAM | `moe_balance_loss`, `equalizer_aux_loss`, `scan_indices`, `load_X_inmemory` | idem, bagian 4 |
| 5 | Evaluasi & CLI: pencocokan konfigurasi checkpoint, mode oracle, tabel akurasi per SNR, penjaga argumen | `cocokkan_config`, `mode_oracle_latih`, `per_snr_table`, penjaga argumen `main()` | idem, bagian 5 |
| 6 | Kepala ketidakpastian: penolakan set-terbuka (model boleh menjawab "tidak pasti"), skor msp/entropi/energi, ambang dari target risiko atau cakupan, AURC | `marspec_ketidakpastian.py` | `docs/laporan_bagian6.md` |
| 6 | Demo web: mode Jelajahi, Tantang AI, Pameran; tampilan ponsel; penanganan data gagal dimuat | `demo_web/` (manual) | `docs/uji_demo_web.md` |

## 5. Features Not to Be Tested

| Fitur / kode | Alasan |
|---|---|
| Akurasi model terlatih | Butuh bobot hasil latih dan dataset RadioML 21 GB. Pengujian ini tentang kebenaran kode, bukan kualitas model |
| Perulangan latih `train_maritime_v3.main()` dan `run_eval()` di `eval_maritime_snr.py` | Butuh dataset 21 GB dan waktu latih berjam-jam; tidak bisa jalan di GitHub Actions. Hanya penjaga argumen yang jatuh sebelum data dimuat yang diuji (bagian 5) |
| `baseline_models.py`, `maritime_amc_lite.py` | Model pembanding, bukan MARSPEC |
| Jalur khusus GPU | Tidak tersedia di laptop anggota maupun GitHub Actions. Cabang yang hanya bisa dicapai dengan GPU dicatat sebagai tidak tertutup beserta alasannya |
| Kinerja (kecepatan, memori) | Tidak termasuk tujuan tugas |
| Fungsi PyTorch itu sendiri | Yang diuji adalah keputusan kode MARSPEC, bukan pustaka (`panduan_pengujian.md` §7 butir 6) |
| Demo web di peramban lain selain dua peramban + satu lebar ponsel | Dibatasi sesuai `uji_demo_web.md` |

## 6. Approach

**Pendekatan umum.** Sistem dibagi enam bagian. **Setiap anggota mengerjakan blackbox dan whitebox** pada satu
bagian, sehingga bebannya setara dan semua orang mempraktikkan kedua teknik. Semua uji Python otomatis dengan
pytest pada data sintetis kecil (fixture di `tests/conftest.py`). Demo web diuji manual dari sudut pandang
pengunjung.

**Independensi penguji.** Pembuat MARSPEC tidak menguji bagian inti (1–5), karena penguji yang menulis kodenya
sendiri cenderung menguji apa yang ia maksudkan, bukan apa yang kodenya lakukan. Ia mengambil bagian 6 dan
menjadi narasumber tentang perilaku yang seharusnya, tanpa menulis uji anggota lain.

**Teknik:**

| Jenis | Teknik | Alat |
|---|---|---|
| Blackbox (pilih minimal 2) | partisi ekivalen, nilai batas, tabel keputusan, transisi keadaan, tebakan galat | pytest; demo web: peramban + DevTools |
| Whitebox | basis path untuk satu fungsi (graf alir, V(G), satu kasus per jalur independen; kondisi majemuk dipecah); cakupan cabang fungsi-fungsi bagiannya | pytest, pytest-cov (`--cov --cov-branch`) |
| Pendukung (opsional) | cakupan kondisi, MC/DC, uji perulangan, uji mutasi | dirancang manual |

**Tingkat kelengkapan minimum** (`PEMBAGIAN_TUGAS.md` §2), per anggota:
- minimal **10 kasus blackbox** dengan minimal **2 teknik**, terdokumentasi di `templat_kasus_uji.md`;
- **basis path** untuk satu fungsi;
- **cakupan cabang** fungsi bagiannya diukur **sebelum dan sesudah**, dan setiap cabang yang tidak tertutup
  dijelaskan alasannya (mustahil dilalui, butuh GPU, atau butuh berkas 21 GB).

Rencana ini **tidak** menetapkan ambang persentase cakupan.

**Cara menilai kelengkapan.** Cakupan cabang dihitung hanya atas fungsi milik bagian itu, bukan seluruh
berkas (`panduan_pengujian.md` §4). Untuk basis path, kelengkapan dinilai dari apakah setiap jalur independen
punya satu kasus uji.

**Kriteria selesai tambahan.**
- Semua TODO di berkas uji terisi, atau dihapus dengan alasan.
- `pytest` hijau di GitHub Actions.

**Ketertelusuran.** Setiap uji diberi nama `test_bb_…` atau `test_wb_…`. Baris pertama docstring-nya menyebut
teknik dan harapan. Kasus di tabel rancangan diberi ID (B*n*-xx untuk blackbox bagian *n*, W-xx untuk demo
web) dan dirujuk ke nama fungsi ujinya. Bug diberi ID BUG-xx yang sama di `LOG_BUG.md` dan di alasan
`xfail`-nya.

**Batasan:**
- tanpa GPU, tanpa dataset 21 GB, dan tanpa bobot terlatih di uji mana pun;
- `marspec/` beku;
- OS anggota berbeda (Windows) dari GitHub Actions (Linux), dan sebagian cabang bergantung OS, misalnya
  `_ram_tersedia()`. Uji yang menyentuh cabang seperti itu harus menimpanya secara eksplisit, dan OS-nya
  disebut di laporan;
- tenggat dari dosen.

## 7. Item Pass/Fail Criteria

- **Kasus uji otomatis lulus** bila pytest melaporkannya *passed*, artinya hasil aktual sama dengan harapan.
  Untuk bilangan pecahan, toleransinya ditulis bersama alasannya dan tidak diperbesar hanya supaya lulus
  (`panduan_pengujian.md` §7 butir 1).
- **Kasus yang menemukan bug** tidak diubah harapannya. Uji itu ditandai
  `@pytest.mark.xfail(strict=True, reason="BUG-xx: …")` dan bugnya dicatat di `LOG_BUG.md`. Butir uji yang
  memiliki uji xfail dinyatakan **gagal, dengan bug tercatat**.
- **Butir uji (fungsi) lulus** bila semua kasusnya lulus dan tidak ada xfail.
- **Kasus demo web lulus** bila perilaku yang teramati sesuai harapan di `uji_demo_web.md` pada kedua peramban
  dan lebar ponsel.
- **Sebelum menyebut sesuatu bug**, penguji harus bisa mengutip dasar harapannya (§3, sumber spesifikasi).
- **Uji yang tidak bisa gagal tidak dihitung.** Setelah lulus, harapannya dirusak sementara untuk memastikan uji
  itu benar-benar gagal (`panduan_pengujian.md` §7 butir 3).

## 8. Suspension Criteria and Resumption Requirements **[baru]**

| Kriteria penghentian | Yang dihentikan | Syarat dilanjutkan |
|---|---|---|
| Sidik SHA-256 berkas di `marspec/` atau `demo_web/` tidak lagi cocok dengan `VERSI.md` (butir uji berubah) | seluruh pengujian | Kembalikan salinan beku, **atau** perbarui `VERSI.md` dengan persetujuan kelompok. Lalu jalankan ulang seluruh rangkaian dan ukur ulang cakupan semua bagian |
| `pytest` di cabang `main` merah di GitHub Actions karena fixture atau `conftest.py` rusak | penggabungan PR | Penjaga templat memperbaiki `conftest.py`. Seluruh rangkaian dijalankan ulang dan harus hijau sebelum PR berikutnya digabung |
| Lingkungan seorang anggota tidak bisa menjalankan `pytest` (misalnya torch gagal dipasang) | pengujian lokal anggota itu | Penyiap lingkungan membantu memasang ulang. Sementara itu anggota memakai GitHub Actions sebagai pelari uji |

## 9. Test Deliverables

| Dokumen menurut IEEE 829 | Wujud di proyek ini |
|---|---|
| Test plan | dokumen ini |
| Test design specification | `templat_kasus_uji.md` dan `templat_basis_path.md` yang diisi per bagian; `docs/laporan_bagian6.md` |
| Test case specification | tabel kasus di rancangan + docstring setiap fungsi uji |
| Test procedure specification | `README.md` §3 (perintah pytest); langkah per kasus di `docs/uji_demo_web.md` |
| Test item transmittal report | `marspec/VERSI.md` |
| Test log | keluaran `pytest` lokal dan log GitHub Actions; tangkapan layar demo web di `docs/tangkapan/` |
| Test incident report | `LOG_BUG.md` |
| Test summary report | `docs/test_summary_report.md` |

**Data masukan uji.** Seluruhnya sintetis dan dibuat oleh fixture di `tests/conftest.py`:
- `cfg_kecil`: arsitektur sama dengan MARSPEC, sekitar 200 ribu parameter;
- `iq_acak`: tensor sinyal `[4, 2, 1024]`;
- `ctx_acak`: tensor konteks laut `[4, 4]`;
- `pool_tiruan`: 50 kondisi laut tiruan;
- `h5_tiruan`: berkas HDF5 mini 576 baris;
- `benih_tetap`: benih acak 0 di setiap uji.

**Data keluaran uji.** Hasil pytest, laporan cakupan (`term-missing`, HTML, atau JSON), dan tangkapan layar demo
web.

**Alat uji.** pytest, pytest-cov, dan GitHub Actions (`.github/workflows/uji.yml`).

## 10. Testing Tasks

| Tahap | Tugas | Keluaran | Bergantung pada |
|---|---|---|---|
| 1. Persiapan | clone, pasang lingkungan, buat cabang masing-masing | `pytest` = 16 passed / 57 skipped di setiap laptop | repositori kerangka + `conftest.py` siap (penjaga templat) |
| 2. Rancang | susun tabel kasus blackbox dan graf alir basis path, **sebelum** menulis kode | rancangan per bagian | tahap 1; dokumen spesifikasi (§3) |
| 3. Tulis uji | isi TODO, tambah uji sampai target, buka PR per anggota | berkas `test_n_….py` + PR | tahap 2 |
| 4. Tinjau | tinjau PR secara silang, catat bug, ukur ulang cakupan | PR digabung, `LOG_BUG.md`, angka cakupan | tahap 3; GitHub Actions hijau |
| 5. Laporan | susun laporan ringkasan dan slide | `test_summary_report.md` + slide | tahap 4 untuk **semua** bagian |

**Keterampilan khusus yang dibutuhkan:**
- Python dasar dan pytest (fixture, `parametrize`, `monkeypatch`, `pytest.raises`);
- Git dan pull request di GitHub;
- membaca kode PyTorch tingkat dasar (bentuk tensor, `eval()` vs latih);
- menggambar graf alir dan menghitung V(G);
- untuk demo web: DevTools peramban.

## 11. Environmental Needs

| Kebutuhan | Rincian |
|---|---|
| Perangkat keras | Laptop biasa; CPU saja, tanpa GPU. Ruang disk sekitar 1 GB (torch CPU sekitar 200 MB) |
| Perangkat lunak | Python ≥ 3.10; torch ≥ 2.1 versi CPU (dipasang dulu dari indeks CPU PyTorch); numpy ≥ 1.24; h5py ≥ 3.8; pytest ≥ 7.0; pytest-cov ≥ 4.0 (`requirements.txt`); Git |
| CI | GitHub Actions: ubuntu-latest, Python 3.11, menjalankan `pytest --cov=marspec --cov-branch` di setiap push dan PR |
| Demo web | Dua peramban (misalnya Chrome dan Firefox) dan satu lebar ponsel (DevTools, 375 px). Dibuka langsung dari berkas (`file:///`), tanpa server dan tanpa internet |
| Mode pemakaian | Mandiri (*stand-alone*); semua uji jalan luring di laptop masing-masing |
| Keamanan | Repositori **privat**, karena kode MARSPEC adalah bahan skripsi yang belum terbit. Dosen bisa diundang sebagai kolaborator. Tidak ada data pribadi dalam data uji |
| Alat khusus | Tidak ada, selain pytest-cov |
| Sumber kebutuhan yang belum tersedia | Semua bisa dipasang dari pip dan indeks CPU PyTorch; peramban dipasang masing-masing anggota |

## 12. Responsibilities

| Tanggung jawab | Pihak |
|---|---|
| Mengelola pengujian dan repositori, menyediakan butir uji (salinan beku) | pemilik repositori / pengembang MARSPEC: Hegira Musyafa Kartiwan |
| Merancang, menyiapkan, dan menjalankan uji bagian *n* | pemilik bagian *n* (tabel di bawah) |
| Menyaksikan dan memeriksa | peninjau silang di setiap PR (satu anggota lain) |
| Menyelesaikan bug | pengembang MARSPEC, di repositori asli, **bukan** di `marspec/` beku |
| Menyediakan lingkungan | penyiap lingkungan |
| Menjaga `conftest.py` dan konsistensi penamaan | penjaga templat (satu-satunya yang boleh menyunting `conftest.py`) |
| Merawat `LOG_BUG.md` | pencatat bug |
| Menyusun slide | penyusun slide |

| Bagian | Pemilik |
|---|---|
| 1 Komponen sinyal | [isi] |
| 2 Konteks & model utuh | Elsa Novia |
| 3 Simulator kanal laut | Maitsa [lengkapi nama] |
| 4 Rugi & pemuat data | M Tegar Fathan |
| 5 Evaluasi & CLI | [isi] |
| 6 Ketidakpastian + demo web | Hegira Musyafa Kartiwan |

Peran pendukung (penyiap lingkungan, penjaga templat, pencatat bug, slide): **[isi]**. Di `PEMBAGIAN_TUGAS.md`
§3 kolom anggotanya masih kosong.

## 13. Staffing and Training Needs

**Staf.** Enam orang, satu bagian per orang. Bobot bagian tidak persis sama. Bagian 3 paling gemuk whitebox-nya,
dan bagian 6 menambah uji manual demo web. Anggota yang selesai lebih dulu mengambil TODO dari bagian 3.

**Pelatihan:**
- `docs/panduan_pengujian.md`: wajib dibaca sebelum menulis uji pertama (sekitar 15 menit).
- `docs/templat_basis_path.md`: langkah basis path beserta satu contoh lengkap.
- Uji bertanda **CONTOH**, dua di setiap berkas uji (16 kasus setelah parametrisasi), sebagai pola untuk ditiru.
- Pembuat MARSPEC menjadi narasumber tentang perilaku yang seharusnya, tanpa menuliskan uji anggota lain.

## 14. Schedule

`PEMBAGIAN_TUGAS.md` §4 menyerahkan tanggal tiap tahap kepada kelompok sesuai tenggat dosen, dan tanggal
rencananya belum pernah diisi. Kolom realisasi diambil dari riwayat Git.

| Tonggak | Rencana | Realisasi |
|---|---|---|
| Penyerahan butir uji (salinan beku, satu-satunya) | — | 19 Okt 2026, commit `d4714b7` |
| Tahap 1 Persiapan selesai | [isi] | 19 Okt 2026 (kerangka: 16 lulus / 57 dilewati) |
| Tahap 2 Rancang selesai | [isi] | hanya bagian 6 yang punya dokumen rancangan tersendiri |
| Tahap 3 Tulis uji selesai | [isi] | bagian 3: 3 Nov; bagian 2 dan 6: 9 Nov; bagian 1, 4, 5: 10 Nov (belum di-commit) |
| Tahap 4 Tinjau | [isi] | PR #1 dan #2 digabung 9 Nov; tinjauan silang bagian 1–5 belum selesai |
| Tahap 5 Laporan | [isi] (tenggat dosen) | draf TSR 10 Nov 2026 |

**Periode pemakaian sumber daya:**
- GitHub Actions: di setiap push dan PR selama tahap 3–4.
- Peramban untuk demo web: tahap 3, bagian 6.

## 15. Risks and Contingencies **[baru]**

Daftar risiko ini baru disusun. Sebagian besar langkah pencegahannya sudah tertulis di `README.md`,
`PEMBAGIAN_TUGAS.md`, dan `panduan_pengujian.md` sejak 19 Oktober.

| # | Asumsi berisiko | Rencana cadangan / pencegahan |
|---|---|---|
| R1 | Semua anggota bisa memasang torch dan menjalankan uji di laptopnya | Pasang torch versi CPU (sekitar 200 MB, bukan versi GPU sekitar 2 GB). Kalau tetap gagal, GitHub Actions menjadi pelari uji (§8) |
| R2 | Hasil uji sama di Windows dan Linux | Cabang yang bergantung OS ditimpa secara eksplisit di uji (`monkeypatch`), dan OS disebut di laporan |
| R3 | Kode MARSPEC tidak berubah selama pengujian | Salinan beku + sidik SHA-256 di `VERSI.md` (§3, §8) |
| R4 | Anggota tidak saling menimpa pekerjaan | Satu berkas uji per anggota, satu cabang per anggota, `conftest.py` hanya disunting penjaga templat, dan PR merah tidak digabung |
| R5 | Setiap bagian selesai sebelum tenggat | Anggota yang selesai lebih dulu mengambil TODO bagian lain. Penyimpangan dicatat di laporan ringkasan |
| R6 | Penguji tidak bias terhadap kodenya sendiri | Pembuat MARSPEC hanya menguji bagian 6 (§6) |
| R7 | Uji benar-benar memeriksa sesuatu | Periksa dengan merusak harapan (§7). Uji mutasi opsional |
| R8 | Harapan uji punya dasar, sehingga tidak ada bug palsu | Harus mengutip docstring atau komentar. Catat di `LOG_BUG.md` dulu, lalu konfirmasi dengan pengembang |
| R9 | Bug yang ditemukan tidak "dibetulkan" diam-diam supaya uji lulus | `marspec/` beku, dan bug ditandai `xfail(strict=True)` (§7) |

## 16. Approvals

| Nama | Jabatan | Peran | Tanda tangan | Tanggal |
|---|---|---|---|---|
| [isi] | Mahasiswa | Penguji bagian 1 | | |
| Elsa Novia | Mahasiswa | Penguji bagian 2 | | |
| Maitsa [lengkapi] | Mahasiswa | Penguji bagian 3 | | |
| M Tegar Fathan | Mahasiswa | Penguji bagian 4 | | |
| [isi] | Mahasiswa | Penguji bagian 5 | | |
| Hegira Musyafa Kartiwan | Mahasiswa | Penguji bagian 6, pemilik repositori | | |
