// W-17 di Firefox 151 (WebDriver BiDi): gulir menyamping di lebar 375 px?
const puppeteer = require("puppeteer-core");
const path = require("path");
const URL = "file:///" + path.resolve(__dirname, "..", "..", "demo_web", "index.html").replace(/\\/g, "/");
const SHOT = path.resolve(__dirname, "..", "tangkapan");

(async () => {
  const browser = await puppeteer.launch({ browser: "firefox", executablePath: process.env.FIREFOX || "C:/Program Files/Mozilla Firefox/firefox.exe", headless: true });
  const hasil = { versi: await browser.version() };
  for (const lebar of [375, 360, 320]) {
    const p = await browser.newPage();
    await p.setViewport({ width: lebar, height: 812 });
    await p.goto(URL, { waitUntil: "load" });
    await new Promise((r) => setTimeout(r, 800));
    for (const mode of ["Jelajahi", "Tantang AI"]) {
      if (mode === "Tantang AI") { await p.click("#mode-tantang"); await new Promise((r) => setTimeout(r, 300)); }
      hasil[`${lebar}px ${mode}`] = await p.evaluate(() => {
        const W = document.documentElement.clientWidth;
        const lebarDok = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
        const melebar = [...document.querySelectorAll("body *")].filter((e) => {
          const r = e.getBoundingClientRect(); return e.offsetParent && r.width > 0 && r.right > W + 1;
        }).slice(0, 6).map((e) => `${e.tagName.toLowerCase()}${e.id ? "#" + e.id : ""} kanan=${Math.round(e.getBoundingClientRect().right)}`);
        return { W, lebarDok, guliran: lebarDok > W + 1, melebar };
      });
      if (lebar === 375) await p.screenshot({ path: path.join(SHOT, `W-17_firefox_${mode === "Jelajahi" ? "jelajahi" : "tantang"}.png`), fullPage: true });
    }
    await p.close();
  }
  await browser.close();
  console.log(JSON.stringify(hasil, null, 1));
})().catch((e) => { console.error("GAGAL:", e.message); process.exit(1); });
