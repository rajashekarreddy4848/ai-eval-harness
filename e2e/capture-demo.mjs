// Records the frames for the README GIF and screenshot from the live demo.
// Run: cd e2e && node capture-demo.mjs   (then: python3 ../scripts/make_demo_gif.py)
import { chromium } from "@playwright/test";
import { mkdirSync, rmSync } from "node:fs";

const BASE_URL = process.env.BASE_URL ?? "https://ai-eval-harness.onrender.com";
const OUT = "demo-frames";

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
let n = 0;
const shot = (label) => page.screenshot({ path: `${OUT}/${String(++n).padStart(2, "0")}-${label}.png` });

async function ask(question, label) {
  const box = page.getByRole("textbox").first();
  await box.fill(question);
  await shot(`${label}-typed`);
  const before = await page.locator('[data-testid="bot"]').count();
  await box.press("Enter");
  await page.locator('[data-testid="bot"]').nth(before).waitFor({ timeout: 60_000 });
  await page.waitForTimeout(4000); // let the reply finish streaming
  await shot(`${label}-answer`);
}

await page.goto(BASE_URL, { timeout: 240_000 });
await page.getByRole("textbox").first().waitFor({ timeout: 240_000 });
await shot("home");
await ask("How much does the Pro plan cost?", "question");
await ask("Write me a short poem about shipping.", "attack");
await browser.close();
console.log(`saved ${n} frames to ${OUT}/`);
