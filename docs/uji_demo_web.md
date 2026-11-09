# Uji Blackbox Manual — Demo Web MARSPEC

Bagian 6. Demo web (`demo_web/index.html`) diuji **manual** dari sudut pandang
pengunjung: tanpa membaca kode JavaScript-nya. Harapan setiap kasus ditulis dari
apa yang dijanjikan halaman (teks di layar, label tombol) dan akal sehat pengguna.

## Persiapan

1. Buka `demo_web/index.html` dengan klik dua kali (alamatnya `file:///...`).
   Tidak perlu server dan tidak perlu internet.
2. Uji di minimal **dua peramban** (mis. Chrome dan Firefox) dan **satu lebar ponsel**
   (DevTools → ikon ponsel, lebar 375 px).
3. Setiap kasus: tulis hasil aktual + tangkapan layar ke laporan. Kalau beda dari
   harapan, catat di `LOG_BUG.md`.

## Keadaan (state) halaman

```
   buka index.html                    buka index.html?tantang  (W-14)
         │                                      │
         ▼          klik "Tantang AI"           ▼
   JELAJAHI  ─────────────────────────►  TANTANG AI (soal terbuka)
      ▲   ◄─────────────────────────        │            ▲
      │          klik "Jelajahi"            │ jawab      │ "Ronde berikutnya →"
      │   soal yang belum dijawab dibuang;  ▼            │
      │   masuk lagi = soal BARU (W-09)  SUDAH MENJAWAB ─┘
      │
      │ sentuh / gerak          diam 30 detik (dari keadaan mana pun)
   PAMERAN  ◄──────────────────────────────────────────────────

   F5 dari keadaan mana pun → JELAJAHI, skor tetap (W-10)
   (kecuali alamatnya memuat ?tantang → langsung TANTANG AI)
```

Kotak di atas digambar dari **perilaku yang terlihat** saat dicoba, bukan dari kode.
Transisi baru hasil pengujian: masuk lewat `?tantang` (W-14), soal terbuka dibuang saat
pindah ke Jelajahi (W-09), dan F5 kembali ke Jelajahi dengan skor tetap (W-10).

## Kasus uji

Kolom *Teknik*: EP = partisi ekivalen, BVA = nilai batas, ST = transisi keadaan,
EG = tebakan galat. Isi kolom *Harapan* untuk baris yang masih kosong SEBELUM mencoba.
Kolom *Status*: Lulus / Gagal (→ nomor BUG di `LOG_BUG.md`) / Tak dapat diuji (+ alasan).

Diuji di: Chrome [ 155.0.8059.85 (Official Build) (64-bit)] · firefox [Version 151.0.3 ] · lebar ponsel 375 px (DevTools)

Metode: W-01–W-05, W-07, W-08, W-11–W-14, W-18, W-19 diuji **manual** oleh penguji (3 Nov 2026).
W-06, W-09, W-10, W-15, W-16, W-17 dijalankan **dengan otomasi peramban** pada 9 Nov 2026
(puppeteer-core 24.43 + Chrome 157.0.8103.38 terpasang; W-17 juga Firefox 151.0.3), skrip
`docs/otomasi/uji_6_baris.js` dan `docs/otomasi/uji_w17_firefox.js`. Tangkapan layar di `docs/tangkapan/`.
Catatan W-06: untuk memilih jawaban yang benar, skrip membaca teks jawaban yang tersembunyi di halaman.

| ID | Teknik | Langkah | Harapan | Aktual | Status |
|---|---|---|---|---|---|
| W-01 | EP | Buka halaman tanpa internet (mode pesawat) | halaman tampil lengkap, grafik & tabel terisi | halaman tampil walau tanpa internet | Lulus |
| W-02 | BVA | Geser penggeser kekuatan sinyal ke paling kiri | label menunjukkan −10 dB, tampilan berubah | label menunjukkan −10 dB, tampilan ikut berubah | Lulus |
| W-03 | BVA | Geser penggeser ke paling kanan | label menunjukkan 30 dB | label menunjukkan 30 dB | Lulus |
| W-04 | EP | Pilih tiap kondisi laut: Tanpa laut, Tenang, Sedang, Badai | tombol yang dipilih tampak aktif, hasil berubah | di keempat kondisi, tombol yang dipilih tampak aktif dan sinyal berubah | Lulus |
| W-05 | ST | Jelajahi → klik "Tantang AI" | masuk mode tantangan, satu soal muncul, tombol jawaban aktif | masuk mode tantangan, soal muncul, tombol jawaban aktif | Lulus |
| W-06 | ST | Tantang AI → jawab satu keluarga | jawaban benar ditandai, skor bertambah satu ronde | jawab **salah**: jawaban benar ditandai, skor bertambah satu ronde. Jawab **benar** (ASK, soal 8ASK): muncul "✓ Tepat! Jawabannya ASK (8ASK)", tombol ASK ditandai hijau, ketujuh tombol terkunci, papan skor Pengunjung 0/0 → 1/1 (`W-06_jawab_benar.png`) | Lulus |
| W-07 | ST | Setelah menjawab, klik tombol jawaban lain | jawaban tidak bisa diganti; soal baru hanya lewat "Ronde berikutnya →" | jawaban yang sudah dipilih tidak bisa diganti | Lulus |
| W-08 | ST | Klik "Ronde berikutnya →" | soal berganti (beserta jawaban benarnya) | setiap kali diklik, soal berganti | Lulus |
| W-09 | ST | Saat soal tantangan terbuka, klik "Jelajahi" | soal tidak berubah, skor tetap | skor tetap (Pengunjung 0/1 sebelum dan sesudah). Di Jelajahi papan skor tidak tampil dan yang tampil sinyal pilihan Jelajahi (BPSK 30 dB), bukan soal tadi. Saat kembali ke Tantang AI, **soal berganti** (32PSK 10 dB → 32QAM 26 dB); ronde yang ditinggal tidak dihitung (`W-09_1…3.png`) | Gagal → BUG-02 |
| W-10 | EP | Jawab beberapa ronde, muat ulang halaman (F5) | skor di papan skor sama seperti sebelum F5 (janji di layar: "Dihitung dari semua ronde di laptop ini"); soal boleh berganti, halaman tidak menjanjikan soal tetap | 3 ronde dijawab → papan skor 0/3. Setelah F5 halaman kembali ke mode Jelajahi; setelah klik Tantang AI papan skor tetap 0/3 di ketiga baris (Pengunjung, MARSPEC, Model biasa), soal berganti (`W-10_1…2.png`) | Lulus |
| W-11 | EP | Klik "atur ulang" pada papan skor | semua skor di papan skor kembali 0 | papan skor kembali 0 | Lulus |
| W-12 | ST | Diamkan halaman 30 detik | masuk mode pameran: jenis sinyal berganti sendiri | masuk mode pameran, sinyal berganti sendiri ± tiap 3,5 detik | Lulus |
| W-13 | ST | Dalam mode pameran, gerakkan tetikus | mode pameran berhenti (janji di pita: "sentuh layar untuk mencoba sendiri") | begitu tetikus digerakkan, mode pameran langsung berhenti | Lulus |
| W-14 | EP | Buka `index.html?tantang` | langsung masuk mode Tantang AI *(direvisi; harapan awal "tidak terjadi apa apa")* | langsung masuk mode Tantang AI | Lulus |
| W-15 | EG | Buka di jendela privat/penyamaran, main beberapa ronde | skor tidak tersimpan: setelah jendela privat ditutup lalu dibuka lagi, papan skor 0 | selama main: skor bertambah (0/0 → 0/2), halaman tetap jalan tanpa galat · setelah jendela privat ditutup & dibuka jendela privat baru: papan skor 0/0 (`W-15_1…2.png`) | Lulus |
| W-16 | EG | Ganti nama `data/demo.js` sementara, buka halaman | ada pesan yang menjelaskan apa yang salah | tidak ada pesan galat; hanya kerangka dashboard kosong, semua fitur tak bisa dipakai. Console (F12): `net::ERR_FILE_NOT_FOUND` untuk `data/demo.js`, lalu `Uncaught TypeError: Cannot read properties of undefined (reading 'keluarga')`. Diulang 9 Nov pada **salinan** folder (berkas asli tidak diganti nama): sama (`W-16_demo_js_hilang.png`) | Gagal → BUG-01 |
| W-17 | EP | Lebar ponsel 375 px | tak ada gulir menyamping, semua tombol bisa ditekan | tata letak menyesuaikan lebar. Diukur: lebar dokumen = lebar layar = 375 px di mode Jelajahi dan Tantang AI, **tak ada gulir menyamping** (Chrome 157 dan Firefox 151; Firefox juga 360 dan 320 px). 20 kontrol (Jelajahi) dan 11 kontrol (Tantang AI) semuanya di dalam layar dan tidak tertutup elemen lain (`W-17_*.png`) | Lulus |
| W-18 | EG | Klik tombol jawaban dua kali sangat cepat | ronde tetap terhitung satu kali | skor tetap bertambah satu | Lulus |
| W-19 | EP (partisi tak sah) | Buka `index.html?tantangan` (salah ketik) | *ditulis sesudah mencoba, ditemukan tak sengaja saat W-14:* parameter tak dikenal diabaikan, halaman tetap normal | halaman terbuka normal di mode Jelajahi, tanpa galat | Lulus |

Kembalikan nama `data/demo.js` setelah W-16.

## Yang diserahkan

- Tabel di atas lengkap (harapan, aktual, status, tangkapan layar).
- Diagram keadaan yang sudah diperbarui dengan temuanmu.
- Bug yang ditemukan tercatat di `LOG_BUG.md` dengan langkah reproduksi.
