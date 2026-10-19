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
                  klik "Tantang AI"
   JELAJAHI  ─────────────────────────►  TANTANG AI (soal terbuka)
      ▲   ◄─────────────────────────        │            ▲
      │          klik "Jelajahi"            │ jawab      │ "Ronde berikutnya →"
      │                                     ▼            │
      │                               SUDAH MENJAWAB ────┘
      │
      │ sentuh / gerak          diam 30 detik (dari keadaan mana pun)
   PAMERAN  ◄──────────────────────────────────────────────────
```

Kotak di atas digambar dari **perilaku yang terlihat** saat dicoba, bukan dari kode.
Kalau kamu menemukan transisi yang tidak ada di gambar, itu temuan: tambahkan.

## Kasus uji

Kolom *Teknik*: EP = partisi ekivalen, BVA = nilai batas, ST = transisi keadaan,
EG = tebakan galat. Isi kolom *Harapan* untuk baris yang masih kosong SEBELUM mencoba.

| ID | Teknik | Langkah | Harapan | Aktual | Status |
|---|---|---|---|---|---|
| W-01 | EP | Buka halaman tanpa internet (mode pesawat) | halaman tampil lengkap, grafik & tabel terisi | | |
| W-02 | BVA | Geser penggeser kekuatan sinyal ke paling kiri | label menunjukkan −10 dB, tampilan berubah | | |
| W-03 | BVA | Geser penggeser ke paling kanan | label menunjukkan 30 dB | | |
| W-04 | EP | Pilih tiap kondisi laut: Tanpa laut, Tenang, Sedang, Badai | tombol yang dipilih tampak aktif, hasil berubah | | |
| W-05 | ST | Jelajahi → klik "Tantang AI" | masuk mode tantangan, satu soal muncul, tombol jawaban aktif | | |
| W-06 | ST | Tantang AI → jawab satu keluarga | jawaban benar ditandai, skor bertambah satu ronde | | |
| W-07 | ST | Setelah menjawab, klik tombol jawaban lain | | | |
| W-08 | ST | Klik "Ronde berikutnya →" | | | |
| W-09 | ST | Saat soal tantangan terbuka, klik "Jelajahi" | | | |
| W-10 | EP | Jawab beberapa ronde, muat ulang halaman (F5) | | | |
| W-11 | EP | Klik "atur ulang" pada papan skor | | | |
| W-12 | ST | Diamkan halaman 30 detik | | | |
| W-13 | ST | Dalam mode pameran, gerakkan tetikus | | | |
| W-14 | EP | Buka `index.html?tantang` | | | |
| W-15 | EG | Buka di jendela privat/penyamaran, main beberapa ronde | | | |
| W-16 | EG | Ganti nama `data/demo.js` sementara, buka halaman | ada pesan yang menjelaskan apa yang salah | | |
| W-17 | EP | Lebar ponsel 375 px | tak ada gulir menyamping, semua tombol bisa ditekan | | |
| W-18 | EG | Klik tombol jawaban dua kali sangat cepat | | | |

Kembalikan nama `data/demo.js` setelah W-16.

## Yang diserahkan

- Tabel di atas lengkap (harapan, aktual, status, tangkapan layar).
- Diagram keadaan yang sudah diperbarui dengan temuanmu.
- Bug yang ditemukan tercatat di `LOG_BUG.md` dengan langkah reproduksi.
