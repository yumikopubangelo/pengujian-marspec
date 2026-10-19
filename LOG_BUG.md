# Log Bug

Bug = perilaku yang **berbeda dari yang diharapkan** menurut spesifikasi (docstring,
komentar kode, README, atau akal sehat yang bisa dipertanggungjawabkan). Tulis di
sini SEBELUM bertanya ke pemilik kode, supaya temuannya tercatat walau ternyata
bukan bug.

Cara mencatat:
1. Ambil nomor berikutnya (BUG-01, BUG-02, ...).
2. Tandai uji yang memicunya dengan
   `@pytest.mark.xfail(strict=True, reason="BUG-01: ringkasan")`.
3. Isi satu baris tabel di bawah. Langkah reproduksi harus bisa diikuti orang lain
   tanpa bertanya.
4. Status berubah hanya setelah dibicarakan: `baru` → `dikonfirmasi` / `bukan bug`.
   Perbaikan dilakukan di repositori pengembang MARSPEC, **bukan** di folder
   `marspec/` sini (salinan beku).

Tingkat: **kritis** (hasil salah tanpa peringatan, atau program mati) ·
**mayor** (fungsi tak sesuai janji tapi ada jalan lain) · **minor** (pesan galat
kurang jelas, kosmetik).

| ID | Tanggal | Penemu | Bagian | Uji pemicu | Langkah reproduksi | Harapan | Aktual | Tingkat | Status |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
