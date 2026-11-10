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
| BUG-03 | 10 Nov 2026 | [isi: penulis uji bagian 4] | 4 | `test_bb_load_x_indeks_di_luar_rentang` (xfail, 2 kasus) | 1. Siapkan HDF5 berstruktur RadioML 576 baris (fixture `h5_tiruan`). 2. Panggil `load_X_inmemory(path, [600, 3])`: indeks tak terurut, jadi jalur pindai per blok yang dipakai. 3. Bandingkan dengan `load_X_inmemory(path, [3, 600])`: indeks terurut, jadi jalur fancy. 4. Ulangi dengan `[5, -1]`. | indeks di luar [0, N) ditolak dengan galat, sama seperti jalur fancy (`IndexError: Fancy indexing out of range for (0-575)`) | langkah 3 melempar `IndexError`, tetapi langkah 2 dan 4 jalan tanpa galat. Baris untuk indeks 600 atau −1 berisi sisa memori `np.empty` (contoh: `3.3e-27`, `1.6e+29`) yang berubah setiap kali dijalankan. Penyebabnya, indeks ≥ N atau negatif tidak pernah jatuh di blok `[a, b)` mana pun, jadi barisnya tak pernah diisi | kritis menurut definisi (hasil salah tanpa peringatan); dampak praktis rendah karena satu-satunya pemanggil, lewat `scan_indices`, selalu memberi indeks sah | baru |
