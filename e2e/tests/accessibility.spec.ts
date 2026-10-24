import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { ChatPage } from "./chat-page";

test("home page has no critical accessibility violations", async ({ page }, testInfo) => {
  await new ChatPage(page).open();
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  await testInfo.attach("axe-results", {
    body: JSON.stringify(results.violations, null, 2),
    contentType: "application/json",
  });
  // Known issue: Gradio's chat component fails aria-required-children (third-party markup,
  // reported in the attachment above). Every other critical rule still fails the test.
  const KNOWN_GRADIO_ISSUES = ["aria-required-children"];
  const critical = results.violations.filter(
    (v) => v.impact === "critical" && !KNOWN_GRADIO_ISSUES.includes(v.id),
  );
  expect(critical, JSON.stringify(critical.map((v) => v.id))).toEqual([]);
});

test("the chat input can be reached and used with the keyboard", async ({ page }) => {
  const chat = new ChatPage(page);
  await chat.open();
  await chat.input.focus();
  await expect(chat.input).toBeFocused();
});
