# Pembagian Tugas

Prinsip: **setiap anggota mengerjakan blackbox DAN whitebox** pada satu bagian
sistem. Dengan begitu bebannya setara, dan semua orang mempraktikkan kedua teknik
(bukan tiga orang hanya blackbox, tiga orang hanya whitebox).

## 1. Bagian sistem

| # | Bagian | Kode yang diuji | Berkas uji | Pemilik |
|---|---|---|---|---|
| 1 | Komponen sinyal | `SE1d`, `ChannelEqualizer`, `CNN1DExtractor`, `MCformerEncoder` | `test_1_komponen_sinyal.py` | [Anggota 1] |
| 2 | Konteks & model utuh | `ContextEncoder`, `ContextGatedFusion`, `MoEChannelRouter`, `MaritimeAMCv3`, `validate_config` | `test_2_konteks_model.py` | [Anggota 2] |
| 3 | Simulator kanal laut | `MaritimeChannel.apply` | `test_3_kanal_laut.py` | [Anggota 3] |
| 4 | Rugi & pemuat data | `moe_balance_loss`, `equalizer_aux_loss`, `scan_indices`, `load_X_inmemory` | `test_4_rugi_data.py` | [Anggota 4] |
| 5 | Evaluasi & CLI | `cocokkan_config`, `mode_oracle_latih`, `per_snr_table`, penjaga argumen `main()` | `test_5_evaluasi_cli.py` | [Anggota 5] |
| 6 | Ketidakpastian + demo web | `marspec_ketidakpastian.py` + `demo_web/` (manual) | `test_6_ketidakpastian.py` + `docs/uji_demo_web.md` | [Anggota 6] |

Catatan: **pembuat MARSPEC sebaiknya tidak menguji bagian inti (1–5).** Penguji
yang menulis kodenya sendiri cenderung menguji apa yang ia *maksudkan*, bukan apa
yang kodenya *lakukan* (prinsip independensi penguji). Bagian 6 + peran penyiap
lingkungan cocok untuknya; ia juga jadi narasumber kalau anggota lain bingung
*apa yang seharusnya terjadi*, tanpa menulis ujinya.

Bobotnya tidak persis sama: bagian 3 paling gemuk whitebox-nya (12 keputusan di
satu fungsi), bagian 6 menambah uji manual demo web. Kalau ada yang selesai lebih
dulu, ambil TODO dari bagian 3.

## 2. Target per anggota (sama untuk semua)

**Blackbox**
- minimal **10 kasus uji**, memakai minimal **2 teknik** (partisi ekivalen,
  nilai batas, tabel keputusan, transisi keadaan, tebakan galat)
- didokumentasikan dengan `docs/templat_kasus_uji.md`
- setiap kasus punya fungsi uji `test_bb_...` di berkasnya (kecuali kasus demo web
  yang manual)

**Whitebox**
- **basis path** untuk satu fungsi: graf alir, V(G), jalur independen, satu kasus
  per jalur (`docs/templat_basis_path.md`)
- **cakupan cabang** fungsi-fungsi bagiannya, diukur dengan pytest-cov; tempel
  angka sebelum & sesudah ke laporan, dan jelaskan cabang yang TIDAK tertutup
  (mustahil dilalui? butuh GPU? butuh berkas 21 GB?)
- setiap kasus punya fungsi uji `test_wb_...`

**Kode**
- semua TODO di berkasnya terisi atau dihapus dengan alasan
- `pytest` hijau di GitHub Actions

## 3. Peran pendukung (satu per orang, di luar bagian ujinya)

| Peran | Tugas | Anggota |
|---|---|---|
| Penyiap lingkungan | memastikan semua bisa clone + `pytest` hijau di laptopnya di minggu pertama | [ ] |
| Penjaga templat | satu-satunya yang menyunting `conftest.py`; menjaga konsistensi penamaan | [ ] |
| Pencatat bug | merawat `LOG_BUG.md`, memastikan setiap bug punya langkah reproduksi | [ ] |
| Slide presentasi | menyusun slide dari laporan | [ ] |

## 4. Tahapan

| Tahap | Keluaran |
|---|---|
| 1. Persiapan | semua anggota clone, `pytest` = 16 passed / 57 skipped, cabang masing-masing dibuat |
| 2. Rancang | tabel kasus blackbox + graf alir basis path di kertas/dokumen (BELUM kode) |
| 3. Tulis uji | TODO diisi, uji tambahan sampai target, PR per anggota |
| 4. Tinjau | PR ditinjau silang, bug dicatat, cakupan diukur ulang |
| 5. Laporan | laporan gabungan + slide |

Tanggal tiap tahap diisi kelompok sesuai tenggat dari dosen.
