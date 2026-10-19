# Templat Kasus Uji Blackbox

Satu tabel per anggota di laporan. Dua baris pertama adalah contoh yang sudah ada
di kode (lihat berkas uji masing-masing); hapus dan ganti dengan kasusmu.

## Bagian [n] — [nama bagian] · Penguji: [nama]

Versi kode: sesuai `marspec/VERSI.md` · OS: [Windows/Linux] · Python: [versi]

| ID | Teknik | Fungsi diuji | Kasus | Masukan | Hasil yang diharapkan | Dasar harapan | Hasil aktual | Status | Fungsi uji |
|---|---|---|---|---|---|---|---|---|---|
| B1-01 | Nilai batas | `ChannelEqualizer` | batch terkecil | `x` acak `[1, 2, 1024]`, ctx `[1, 8]` | keluaran `[1, 2, 1024]` | equalizer hanya membersihkan sinyal, tak mengubah ukurannya | `[1, 2, 1024]` | Lulus | `test_bb_equalizer_mempertahankan_bentuk[1]` |
| B6-01 | Nilai batas | `skor_msp` | logit rata, 24 kelas | `zeros(1, 24)` | tepat 1/24 | softmax dari logit rata = seragam | 0,041667 | Lulus | `test_bb_skor_msp_rentang` |
| | | | | | | | | | |

Penjelasan kolom:
- **ID**: `B<bagian>-<nomor>`. Whitebox pakai awalan `W`.
- **Teknik**: partisi ekivalen, nilai batas, tabel keputusan, transisi keadaan,
  tebakan galat. Minimal dua teknik berbeda di tabelmu.
- **Dasar harapan**: dari mana kamu tahu hasil yang benar? Kutip docstring, komentar,
  README, atau tulis alasan fisik/matematisnya. Kolom ini yang membedakan uji yang
  kuat dari uji yang menebak.
- **Status**: Lulus / Gagal (→ nomor BUG di LOG_BUG.md) / Tak dapat diuji (+ alasan).

## Untuk partisi ekivalen dan nilai batas, tunjukkan pembagiannya

| Masukan | Partisi sah | Partisi tak sah | Nilai batas yang diuji |
|---|---|---|---|
| contoh: `hs_m` (tinggi gelombang, m) | 0 ≤ hs < 0,6 (tanpa Doppler) · hs ≥ 0,6 (dengan Doppler) | hs < 0 | 0,59 · 0,6 |
| | | | |

## Untuk tabel keputusan

| Aturan | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Kondisi: `target_risiko` diisi | Y | Y | T | T |
| Kondisi: `target_cakupan` diisi | Y | T | Y | T |
| Aksi: ValueError | ✓ | | | ✓ |
| Aksi: ambang dipilih dari risiko | | ✓ | | |
| Aksi: ambang dipilih dari cakupan | | | ✓ | |
