# Otomasi peramban untuk uji demo web (bagian 6)

Skrip ini menjalankan ulang kasus W-06, W-09, W-10, W-15, W-16, W-17 dari `docs/uji_demo_web.md`
dengan peramban sungguhan (bukan simulasi). Tidak mengubah berkas apa pun di `demo_web/`:
W-16 mengganti nama `demo.js` di **salinan** folder di direktori sementara.

```bash
cd docs/otomasi
npm install puppeteer-core@24      # tanpa unduhan peramban; memakai Chrome/Firefox terpasang
node uji_6_baris.js                # Chrome; hasil: ../tangkapan/hasil_6_baris.json + *.png
node uji_w17_firefox.js            # W-17 di Firefox (375, 360, 320 px)
```

Lokasi peramban bisa diganti lewat variabel lingkungan `CHROME` dan `FIREFOX`.
`node_modules/` jangan di-commit.
