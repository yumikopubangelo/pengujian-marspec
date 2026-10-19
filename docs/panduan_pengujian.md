# Panduan Pengujian

Baca ini sebelum menulis uji pertama. Lama baca ±15 menit.

## 1. Blackbox vs whitebox dalam satu kalimat

- **Blackbox**: kasus uji dirancang dari **apa yang dijanjikan** (docstring, nama
  fungsi, README, perilaku yang masuk akal), *tanpa* melihat isi kodenya.
  Pertanyaannya: "Apakah fungsinya melakukan yang dijanjikan?"
- **Whitebox**: kasus uji dirancang dari **struktur kode** (if, perulangan, jalur).
  Pertanyaannya: "Apakah setiap cabang dan jalur pernah dijalankan dan benar?"

Satu fungsi boleh diuji dengan keduanya. Yang membedakan adalah *dari mana kasus
ujinya berasal*, jadi tulis asalnya di docstring setiap uji.

## 2. Konvensi penamaan

```
test_bb_<apa_yang_diuji>    # blackbox
test_wb_<apa_yang_diuji>    # whitebox
```

Docstring baris pertama menyebut **teknik** dan **harapan**. Contoh dari berkas:

```python
@pytest.mark.parametrize("hs, ada_doppler", [(0.59, False), (0.6, True)])
def test_wb_cabang_doppler_batas_hs(hs, ada_doppler):
    """CONTOH. Cabang `if hs_m >= 0.6` (Doppler gelombang). Nilai batas di
    sekitar 0,6 membuktikan cabang diambil tepat mulai 0,6."""
```

## 3. Teknik blackbox (pilih minimal 2)

| Teknik | Inti | Contoh di MARSPEC |
|---|---|---|
| Partisi ekivalen | bagi masukan ke kelompok yang perilakunya sama; satu wakil per kelompok | laut tenang vs badai; gate seimbang vs timpang |
| Nilai batas | uji tepat di batas, satu di bawah, satu di atas | batch 1; SNR −10 dan 30; `hs` 0,59 vs 0,6 |
| Tabel keputusan | gabungan beberapa kondisi → hasil | `target_risiko` × `target_cakupan` (isi/kosong) |
| Transisi keadaan | sistem punya mode; uji perpindahan antarmodenya | demo web: Jelajahi ↔ Tantang AI |
| Tebakan galat | masukan yang sering membuat program salah | NaN, panjang sinyal salah, konteks 3 kolom, berkas tak ada |

## 4. Teknik whitebox

| Kriteria | Artinya | Diukur dengan |
|---|---|---|
| Cakupan pernyataan | setiap baris pernah jalan | `--cov` |
| Cakupan cabang | setiap `if` pernah BENAR dan pernah SALAH | `--cov --cov-branch` |
| Cakupan kondisi | setiap sub-kondisi dalam `a and b and c` pernah benar & salah | dirancang manual |
| MC/DC | setiap sub-kondisi terbukti *sendirian* bisa mengubah hasil keputusan | dirancang manual |
| Basis path | satu kasus per jalur independen; jumlahnya V(G) | `docs/templat_basis_path.md` |
| Uji perulangan | perulangan dijalankan 0 kali, 1 kali, maksimum, dan lebih dari maksimum | perulangan interferer di bagian 3 |

### Mengukur cakupan bagianmu saja

```bash
pytest tests/test_4_rugi_data.py --cov=marspec --cov-branch --cov-report=term-missing
```

Kolom yang dibaca:
- `Miss` = baris yang belum pernah jalan; nomor barisnya di kolom `Missing`
- `BrPart` = cabang yang baru diambil satu arahnya
- Notasi `712->715` di kolom `Missing` artinya: lompatan dari baris 712 ke 715
  (biasanya arah "SALAH" dari sebuah `if`) belum pernah terjadi

Angka cakupan berkas itu **seluruh berkas**, termasuk bagian anggota lain. Untuk
laporan, sebut cabang **fungsi bagianmu** saja: buka `htmlcov/index.html`
(`--cov-report=html`) dan hitung di fungsi yang kamu uji.

Awas: cakupan bisa **beda antar-OS**. Contoh: `_ram_tersedia()` membaca
`/proc/meminfo`, yang hanya ada di Linux. Di Windows fungsinya mengembalikan `None`
sehingga cabang lain yang terambil. GitHub Actions memakai Linux; laptopmu mungkin
Windows. Sebut OS-nya di laporan.

## 5. Bahan uji (fixture) yang sudah tersedia

Tulis namanya sebagai argumen fungsi uji; pytest yang membuatnya.

| Fixture | Isi |
|---|---|
| `cfg_kecil` | konfigurasi MARSPEC berarsitektur sama tapi ±200 ribu parameter (bukan 2–37 juta) |
| `iq_acak` | tensor sinyal `[4, 2, 1024]` |
| `ctx_acak` | tensor konteks laut `[4, 4]` |
| `pool_tiruan` | 50 kondisi laut tiruan (kunci sama dengan pool asli) |
| `h5_tiruan` | berkas HDF5 mini 576 baris di folder sementara → `(path, kelas, snr)` |
| `tmp_path` | (bawaan pytest) folder sementara baru untuk setiap uji |
| `monkeypatch` | (bawaan pytest) mengganti fungsi/variabel sementara selama satu uji |
| `benih_tetap` | otomatis aktif: benih acak torch & numpy = 0 di setiap uji |

## 6. Kalau uji menemukan bug

Jangan ubah kode di `marspec/` dan jangan ubah harapan ujinya supaya lulus.

```python
@pytest.mark.xfail(strict=True, reason="BUG-03: konteks 3 kolom diterima tanpa galat")
def test_bb_model_konteks_salah_ukuran(...):
    ...
```

- `xfail` = "diharapkan gagal karena bug yang sudah diketahui". Rangkaian uji tetap hijau.
- `strict=True` = kalau suatu hari bugnya diperbaiki dan uji tiba-tiba lulus, pytest
  memberi tahu (XPASS dianggap gagal), jadi tanda `xfail`-nya tidak tertinggal.
- Catat di `LOG_BUG.md` dengan nomor yang sama.

Sebelum menyebutnya bug, pastikan harapanmu punya dasar: kutip docstring atau
komentar yang menjanjikannya. "Saya kira begitu" bukan dasar; tanyakan dulu ke
pemilik kode, lalu catat jawabannya.

## 7. Jebakan yang sering terjadi

1. **Membandingkan float dengan `==`.** Pakai `pytest.approx(nilai, abs=1e-6)` atau
   `torch.allclose(a, b, atol=1e-5)`. Toleransinya ditulis dengan alasan, bukan
   diperbesar sampai lulus.
2. **Lupa `.eval()`.** Dropout dan LayerDrop aktif dalam mode latih, jadi dua forward
   pada masukan yang sama bisa beda. Untuk uji yang mengharapkan hasil tetap, panggil
   `model.eval()` dan bungkus dengan `torch.no_grad()`.
3. **Uji yang tak bisa gagal.** Setelah uji lulus, rusak sementara harapannya (misal
   ganti `== 24` jadi `== 25`) dan pastikan ia GAGAL. Kalau tetap lulus, ujinya
   tidak memeriksa apa-apa. Kembalikan setelah dicoba.
4. **Uji yang bergantung pada urutan.** Setiap uji harus bisa jalan sendirian:
   `pytest tests/x.py::test_y`. Jangan berbagi variabel global antar-uji.
5. **Dataset asli atau GPU.** Jangan. Semua uji harus jalan di GitHub Actions
   (CPU, tanpa berkas besar). Kalau sebuah cabang hanya bisa dicapai dengan GPU,
   tulis itu di laporan sebagai cabang yang tidak tertutup beserta alasannya.
6. **Menguji ulang PyTorch.** "Linear menghasilkan bentuk yang benar" itu menguji
   PyTorch, bukan MARSPEC. Ujilah keputusan yang dibuat kode MARSPEC.

## 8. Asal-usul angka di kerangka uji

- `cfg_kecil` menyalin pengecilan yang sudah dipakai mode `--smoke` di
  `train_maritime_v3.py`, jadi kombinasi ukurannya diketahui sah.
- 16 contoh lulus / 57 TODO = hitungan saat kerangka dibuat (19 Okt 2026);
  cakupan cabang awal seluruh `marspec/` = 29% (Windows, Python 3.14).
- Batas Doppler 0,6 m, batas interferer 6, ambang duct 0,05, dan batas fancy
  indexing 100 000 diambil langsung dari kode di `marspec/` versi VERSI.md.
