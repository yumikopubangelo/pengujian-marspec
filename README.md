# Pengujian MARSPEC — Blackbox & Whitebox

Repositori tugas kelompok **Pengujian dan Pengembangan Perangkat Lunak**.
Sistem yang diuji: **MARSPEC**, model pengenal modulasi sinyal radio di kanal laut
(PyTorch), beserta simulator kanalnya, pemuat datanya, alat evaluasinya, dan demo webnya.

Semua uji di sini memakai **data sintetis kecil**: tidak perlu dataset 21 GB,
tidak perlu GPU, tidak perlu bobot model terlatih. Seluruh rangkaian jalan dalam
hitungan detik di laptop biasa.

---

## 1. Mulai (sekali saja)

Butuh: Python 3.10 atau lebih baru, dan Git. (Diperiksa jalan di Python 3.11 lewat
GitHub Actions dan di 3.14 + torch 2.13 CPU di Windows.)

```bash
git clone <URL-repo-ini>
cd pengujian_marspec

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS / Linux:         source .venv/bin/activate

# torch versi CPU DULU (±200 MB; versi GPU bawaan pip ±2 GB dan tak dibutuhkan)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Periksa pemasangan:

```bash
pytest
```

Hasil yang benar saat repositori baru di-clone: **16 passed, 57 skipped**.
Yang *passed* adalah contoh; yang *skipped* adalah TODO yang menunggu ditulis.

## 2. Isi repositori

```
marspec/            kode yang DIUJI (salinan beku, lihat marspec/VERSI.md) -- JANGAN diubah
demo_web/           demo web MARSPEC, diuji blackbox secara manual
tests/
  conftest.py       bahan uji bersama (fixture): cfg_kecil, iq_acak, h5_tiruan, ...
  test_1_komponen_sinyal.py   Anggota 1
  test_2_konteks_model.py     Anggota 2
  test_3_kanal_laut.py        Anggota 3
  test_4_rugi_data.py         Anggota 4
  test_5_evaluasi_cli.py      Anggota 5
  test_6_ketidakpastian.py    Anggota 6
docs/
  panduan_pengujian.md   cara menulis uji, teknik, cara ukur cakupan  <- BACA DULU
  templat_kasus_uji.md   tabel kasus uji blackbox untuk laporan
  templat_basis_path.md  langkah basis path + contoh lengkap
  uji_demo_web.md        daftar periksa blackbox demo web
PEMBAGIAN_TUGAS.md  siapa menguji apa, target per orang
LOG_BUG.md          catatan bug yang ditemukan
```

## 3. Perintah sehari-hari

| tujuan | perintah |
|---|---|
| jalankan semua uji | `pytest` |
| hanya berkas sendiri | `pytest tests/test_3_kanal_laut.py` |
| hanya satu uji | `pytest tests/test_3_kanal_laut.py::test_wb_cabang_doppler_batas_hs` |
| hanya blackbox / whitebox | `pytest -k bb` · `pytest -k wb` |
| cakupan cabang (untuk laporan whitebox) | `pytest --cov=marspec --cov-branch --cov-report=term-missing` |
| laporan cakupan HTML (baris merah = belum teruji) | `pytest --cov=marspec --cov-branch --cov-report=html` lalu buka `htmlcov/index.html` |

## 4. Alur kerja Git (satu cabang per anggota)

```bash
git checkout -b uji/anggota-3          # sekali, di awal
# ... tulis uji ...
pytest                                 # WAJIB hijau sebelum push
git add tests/test_3_kanal_laut.py
git commit -m "uji kanal laut: batas SNR & perulangan interferer"
git push -u origin uji/anggota-3
```

Lalu buka **Pull Request** ke `main` di GitHub. Satu anggota lain meninjau
(lihat peran *peninjau silang* di PEMBAGIAN_TUGAS.md) sebelum digabung.
GitHub Actions menjalankan semua uji otomatis di setiap push; PR dengan tanda
merah jangan digabung.

Aturan supaya tidak saling bentrok:
- Setiap orang hanya menyunting **berkas ujinya sendiri**.
- `conftest.py` hanya disunting oleh *penjaga templat*. Butuh fixture baru? Minta di sana,
  atau buat fixture lokal di berkas ujimu sendiri.
- **Folder `marspec/` tidak boleh diubah.** Kalau uji menemukan bug, jangan "betulkan"
  kodenya supaya uji lulus. Catat di LOG_BUG.md dan tandai ujinya `xfail`
  (caranya di docs/panduan_pengujian.md §6). Bug yang ditemukan itu *hasil* pengujian.

## 5. Untuk pemilik repositori (sekali)

```bash
git init -b main
git add .
git commit -m "kerangka pengujian MARSPEC"
# buat repo KOSONG di github.com (tanpa README), lalu:
git remote add origin https://github.com/<akun>/pengujian-marspec.git
git push -u origin main
```

Undang anggota: *Settings → Collaborators → Add people*.

Sarankan repositori **Private**: kode MARSPEC adalah bahan skripsi yang belum
terbit, dan repositori publik membuatnya bisa disalin sebelum waktunya.
Dosen bisa diundang sebagai kolaborator untuk melihatnya.
