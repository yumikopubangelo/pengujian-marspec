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
| BUG-01 | 3 Nov 2026 (diulang 9 Nov) | Hegira Musyafa Kartiwan | 6 | W-16 (manual, `docs/uji_demo_web.md`) | 1. Ganti nama `demo_web/data/demo.js` jadi `demo.js.hilang` (atau salin folder `demo_web/` dan ganti nama di salinannya). 2. Buka `index.html` di Chrome. 3. Buka Console (F12). | halaman menampilkan pesan yang menjelaskan bahwa data tidak termuat dan cara memperbaikinya | kerangka dashboard kosong tanpa pesan apa pun; semua kontrol tak berfungsi. Console: `net::ERR_FILE_NOT_FOUND` (data/demo.js) lalu `Uncaught TypeError: Cannot read properties of undefined (reading 'keluarga')` | minor (pesan galat tak muncul; tak ada hasil salah) | baru |
| BUG-02 | 9 Nov 2026 | Hegira Musyafa Kartiwan | 6 | W-09 (otomasi peramban, `docs/otomasi/uji_6_baris.js`) | 1. Buka `index.html`, klik "Tantang AI". 2. Jangan menjawab; catat soalnya (tangkapan layar). 3. Klik "Jelajahi". 4. Klik "Tantang AI" lagi. | soal yang tadi belum dijawab masih terbuka; skor tetap | skor tetap, tetapi soal berganti (contoh: 32PSK 10 dB → 32QAM 26 dB); ronde yang ditinggal dibuang tanpa dihitung | minor; mungkin disengaja (ronde yang ditinggal tidak merugikan skor), perlu diputuskan pemilik kode | baru |
