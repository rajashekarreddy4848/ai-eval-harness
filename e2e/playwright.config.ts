import { defineConfig, devices } from "@playwright/test";

// The free host sleeps when idle, so the first page load can take over a minute.
export default defineConfig({
  testDir: "./tests",
  timeout: 90_000,
  expect: { timeout: 45_000 },
  retries: process.env.CI ? 1 : 0,
  workers: 1, // one user at a time: the demo runs on a free tier with rate-limited models
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.BASE_URL ?? "https://ai-eval-harness.onrender.com",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});
