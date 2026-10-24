import { expect, Page } from "@playwright/test";

/** Page object for the Gradio chat UI. */
export class ChatPage {
  constructor(private readonly page: Page) {}

  get input() {
    return this.page.getByRole("textbox").first();
  }

  get botMessages() {
    return this.page.locator('[data-testid="bot"]');
  }

  async open() {
    await this.page.goto("/");
    await expect(this.input).toBeVisible();
  }

  async ask(question: string) {
    await this.input.fill(question);
    await this.input.press("Enter");
  }

  /** Waits for the bot's reply and returns its text. */
  async lastAnswer(): Promise<string> {
    await expect(this.botMessages.last()).toHaveText(/\S/);
    return (await this.botMessages.last().innerText()).trim();
  }
}
