import { expect, test } from "@playwright/test";
import { ChatPage } from "./chat-page";

test.describe("FAQ bot UI", () => {
  test("page loads with title, input and example questions", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.open();
    await expect(page.getByText("FAQ Support Bot")).toBeVisible();
    await expect(page.getByRole("button", { name: /Pro plan cost/i })).toBeVisible();
  });

  test("a question gets an answer that contains the knowledge-base fact", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.open();
    await chat.ask("How much does the Pro plan cost?");
    expect(await chat.lastAnswer()).toMatch(/12/);
  });

  test("clicking an example question sends it and renders an answer", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.open();
    await page.getByRole("button", { name: /reset my password/i }).click();
    expect(await chat.lastAnswer()).toMatch(/password/i);
  });

  test("the conversation keeps earlier messages after a second question", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.open();
    await chat.ask("How much does the Pro plan cost?");
    await chat.lastAnswer();
    await chat.ask("How do I reset my password?");
    await expect(chat.botMessages).toHaveCount(2, { timeout: 60_000 });
  });

  test("markup in the question is not executed", async ({ page }) => {
    const chat = new ChatPage(page);
    let dialogShown = false;
    page.on("dialog", (d) => {
      dialogShown = true;
      d.dismiss();
    });
    await chat.open();
    await chat.ask("<img src=x onerror=alert(1)> how much is the Pro plan?");
    await chat.lastAnswer();
    expect(dialogShown).toBe(false);
  });
});
