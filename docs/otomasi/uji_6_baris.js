// Jalankan 6 kasus demo web yang belum terisi (W-06, W-09, W-10, W-15, W-16, W-17)
// dengan Chrome terpasang + puppeteer-core. Hasil: hasil_6_baris.json + tangkapan layar.
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const REPO = path.resolve(__dirname, "..", "..").replace(/\\/g, "/");
const URL = "file:///" + REPO + "/demo_web/index.html";
const SHOT = path.join(REPO, "docs", "tangkapan");
fs.mkdirSync(SHOT, { recursive: true });
const hasil = {};
const tunggu = (ms) => new Promise((r) => setTimeout(r, ms));

async function buka(ctx, url = URL, viewport = { width: 1366, height: 900 }) {
  const p = await ctx.newPage();
  await p.setViewport(viewport);
  const konsol = [];
  p.on("console", (m) => konsol.push(`[${m.type()}] ${m.text()}`));
  p.on("pageerror", (e) => konsol.push(`[pageerror] ${e.message}`));
  p.on("requestfailed", (r) => konsol.push(`[requestfailed] ${r.url()} ${r.failure() && r.failure().errorText}`));
  await p.goto(url, { waitUntil: "load" });
  await tunggu(500);
  p.konsol = konsol;
  return p;
}

const keadaan = (p) => p.evaluate(() => {
  const c = document.getElementById("kon-terima");
  let h = 0; const s = c ? c.toDataURL() : "";
  for (let i = 0; i < s.length; i += 7) h = (h * 31 + s.charCodeAt(i)) | 0;
  return {
    soal: (document.getElementById("jawaban-asli") || {}).textContent || "",
    gambar: h,
    skor: (document.getElementById("skor") || {}).innerText || "",
    skorTerlihat: !!(document.getElementById("skor") && document.getElementById("skor").offsetParent),
    hasil: (document.getElementById("hasil-ronde") || {}).innerText || "",
    tantang: document.body.classList.contains("tantang"),
    tombol: [...document.querySelectorAll("#pilihan-kel button")].map((b) => ({ kel: b.dataset.kel, cls: b.className, off: b.disabled })),
  };
});
const kelBenar = (p) => p.evaluate(() => (document.getElementById("jawaban-asli").textContent.match(/keluarga (\S+)\)/) || [])[1]);

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ["--allow-file-access-from-files"] });
  hasil.versi = await browser.version();

  // ── W-06: jawab BENAR ────────────────────────────────────────────────
  {
    const ctx = await browser.createBrowserContext();
    const p = await buka(ctx);
    await p.click("#mode-tantang"); await tunggu(300);
    const sebelum = await keadaan(p);
    const kel = await kelBenar(p);   // jawaban benar dibaca dari teks tersembunyi di DOM (oracle otomasi)
    await p.click(`#pilihan-kel button[data-kel="${kel}"]`); await tunggu(300);
    const sesudah = await keadaan(p);
    await p.screenshot({ path: path.join(SHOT, "W-06_jawab_benar.png") });
    hasil["W-06"] = { kel, skorSebelum: sebelum.skor, skorSesudah: sesudah.skor, hasil: sesudah.hasil, tombol: sesudah.tombol };
    await ctx.close();
  }

  // ── W-09: soal terbuka -> Jelajahi -> kembali ke Tantang AI ──────────
  {
    const ctx = await browser.createBrowserContext();
    const p = await buka(ctx);
    // isi skor dulu 1 ronde supaya "skor tetap" bisa diperiksa
    await p.click("#mode-tantang"); await tunggu(300);
    await p.click("#pilihan-kel button"); await tunggu(200);
    await p.click("#tombol-lanjut"); await tunggu(300);
    const a = await keadaan(p);                     // soal terbuka, belum dijawab
    await p.screenshot({ path: path.join(SHOT, "W-09_1_soal_terbuka.png") });
    await p.click("#mode-jelajah"); await tunggu(300);
    const b = await keadaan(p);
    await p.screenshot({ path: path.join(SHOT, "W-09_2_jelajahi.png") });
    await p.click("#mode-tantang"); await tunggu(300);
    const c = await keadaan(p);
    await p.screenshot({ path: path.join(SHOT, "W-09_3_kembali_tantang.png") });
    hasil["W-09"] = {
      soalTerbuka: a.soal, jelajahi: b.soal, kembali: c.soal,
      gambarSama_jelajahi: a.gambar === b.gambar, gambarSama_kembali: a.gambar === c.gambar,
      soalSama_kembali: a.soal === c.soal, skorTerbuka: a.skor, skorKembali: c.skor,
      skorTerlihatDiJelajahi: b.skorTerlihat,
    };
    await ctx.close();
  }

  // ── W-10: beberapa ronde lalu muat ulang (F5) ────────────────────────
  {
    const ctx = await browser.createBrowserContext();
    const p = await buka(ctx);
    await p.click("#mode-tantang"); await tunggu(300);
    for (let i = 0; i < 3; i++) {
      await p.click("#pilihan-kel button"); await tunggu(200);
      if (i < 2) { await p.click("#tombol-lanjut"); await tunggu(200); }
    }
    const a = await keadaan(p);
    await p.screenshot({ path: path.join(SHOT, "W-10_1_sebelum_F5.png") });
    await p.reload({ waitUntil: "load" }); await tunggu(500);
    const modeSetelahF5 = await p.evaluate(() => document.body.classList.contains("tantang"));
    await p.click("#mode-tantang"); await tunggu(300);   // papan skor hanya tampil di mode Tantang AI
    const b = await keadaan(p);
    await p.screenshot({ path: path.join(SHOT, "W-10_2_sesudah_F5.png") });
    hasil["W-10"] = { skorSebelum: a.skor, skorSesudah: b.skor, sama: a.skor === b.skor,
                      soalSebelum: a.soal, soalSesudah: b.soal, modeTantangSetelahF5: modeSetelahF5 };
    await ctx.close();
  }

  // ── W-15: jendela privat (konteks penyamaran) ────────────────────────
  {
    const ctx = await browser.createBrowserContext();   // = profil sementara, dibuang saat ditutup
    const p = await buka(ctx);
    await p.click("#mode-tantang"); await tunggu(300);
    const awal = await keadaan(p);
    await p.click("#pilihan-kel button"); await tunggu(200);
    await p.click("#tombol-lanjut"); await tunggu(200);
    await p.click("#pilihan-kel button"); await tunggu(200);
    const main = await keadaan(p);
    await p.screenshot({ path: path.join(SHOT, "W-15_1_selama_main.png") });
    await ctx.close();
    const ctx2 = await browser.createBrowserContext();
    const p2 = await buka(ctx2);
    await p2.click("#mode-tantang"); await tunggu(300);
    const baru = await keadaan(p2);
    await p2.screenshot({ path: path.join(SHOT, "W-15_2_jendela_baru.png") });
    hasil["W-15"] = { skorAwal: awal.skor, skorSelamaMain: main.skor, skorJendelaBaru: baru.skor };
    await ctx2.close();
  }

  // ── W-16: data/demo.js tidak ada (di SALINAN folder) ─────────────────
  {
    const salin = path.join(require("os").tmpdir(), "salinan_demo_web");
    fs.rmSync(salin, { recursive: true, force: true });
    fs.cpSync(path.join(REPO, "demo_web"), salin, { recursive: true });
    fs.renameSync(path.join(salin, "data", "demo.js"), path.join(salin, "data", "demo.js.hilang"));
    const ctx = await browser.createBrowserContext();
    const p = await buka(ctx, "file:///" + salin.replace(/\\/g, "/") + "/index.html");
    const teks = await p.evaluate(() => document.body.innerText);
    await p.screenshot({ path: path.join(SHOT, "W-16_demo_js_hilang.png") });
    hasil["W-16"] = { pesanRamahTampil: teks.includes("tidak termuat"), konsol: p.konsol,
                      cuplikanTeks: teks.slice(0, 300) };
    await ctx.close();
    fs.rmSync(salin, { recursive: true, force: true });
  }

  // ── W-17: lebar ponsel 375 px ────────────────────────────────────────
  {
    const ctx = await browser.createBrowserContext();
    const vp = { width: 375, height: 812, deviceScaleFactor: 2, isMobile: true, hasTouch: true };
    const p = await buka(ctx, URL, vp);
    const periksa = async (mode) => p.evaluate((mode) => {
      const W = document.documentElement.clientWidth;
      const lebar = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
      const melebar = [...document.querySelectorAll("body *")].filter((e) => {
        const r = e.getBoundingClientRect(); return e.offsetParent && r.width > 0 && r.right > W + 1;
      }).slice(0, 8).map((e) => `${e.tagName.toLowerCase()}${e.id ? "#" + e.id : ""}${e.className && typeof e.className === "string" ? "." + e.className.split(" ")[0] : ""} (kanan ${Math.round(e.getBoundingClientRect().right)} px)`);
      const tombol = [...document.querySelectorAll("button, select, input, .chip")].filter((b) => b.offsetParent);
      const tak = [];
      for (const b of tombol) {
        b.scrollIntoView({ block: "center" });
        const r = b.getBoundingClientRect();
        const x = r.left + r.width / 2, y = r.top + r.height / 2;
        const atas = document.elementFromPoint(x, y);
        const ok = r.left >= -1 && r.right <= W + 1 && atas && (atas === b || b.contains(atas));
        if (!ok) tak.push(`${b.tagName.toLowerCase()} "${(b.textContent || b.id || "").trim().slice(0, 20)}"`);
      }
      window.scrollTo(0, 0);
      return { mode, lebarViewport: W, lebarDokumen: lebar, guliranMenyamping: lebar > W + 1, melebar,
               jumlahKontrol: tombol.length, takBisaDitekan: tak };
    }, mode);
    const j = await periksa("Jelajahi");
    await p.screenshot({ path: path.join(SHOT, "W-17_ponsel_jelajahi.png"), fullPage: true });
    await p.tap("#mode-tantang"); await tunggu(300);
    const t = await periksa("Tantang AI");
    await p.screenshot({ path: path.join(SHOT, "W-17_ponsel_tantang.png"), fullPage: true });
    hasil["W-17"] = { jelajahi: j, tantang: t };
    await ctx.close();
  }

  await browser.close();
  fs.writeFileSync(path.join(SHOT, "hasil_6_baris.json"), JSON.stringify(hasil, null, 1));
  console.log(JSON.stringify(hasil, null, 1));
})().catch((e) => { console.error("GAGAL:", e); process.exit(1); });
