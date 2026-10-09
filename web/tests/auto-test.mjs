// Browser regression: npm run build && npm run test:auto
// Uses a local mock API only; no paid endpoint or real credentials are needed.
import puppeteer from "puppeteer-core";
import http from "node:http";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const calls = [];
const counts = {};
let active = 0;
let peak = 0;
const numbers = Array.from(
  { length: 310 },
  (_, i) => ((i * 73) % 355) + 1,
).join(" ");
const mock = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "authorization,content-type");
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  if (req.method === "OPTIONS") {
    res.writeHead(204).end();
    return;
  }
  let raw = "";
  for await (const chunk of req) raw += chunk;
  const data = JSON.parse(raw);
  calls.push({ url: req.url, data, auth: req.headers.authorization });
  active++;
  peak = Math.max(peak, active);
  counts[data.model] = (counts[data.model] || 0) + 1;
  const attempt = counts[data.model];
  await delay(500);
  active--;
  res.setHeader("Content-Type", "application/json");
  if (data.model === "bad-key") {
    res.writeHead(401).end(
      JSON.stringify({
        error: {
          message: "Invalid API key",
          type: "authentication_error",
          code: "invalid_api_key",
        },
      }),
    );
    return;
  }
  const text =
    data.model === "invalid" || (data.model === "chat-model" && attempt === 1)
      ? "too short 1 2 3"
      : numbers;
  const response = req.url.endsWith("/responses")
    ? {
        id: "resp_mock",
        created_at: Math.floor(Date.now() / 1000),
        model: data.model,
        output: [
          {
            id: "msg_mock",
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text, annotations: [] }],
          },
        ],
        usage: { input_tokens: 20, output_tokens: 1000 },
        status: "completed",
      }
    : {
        id: "chatcmpl_mock",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: data.model,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: text },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 20,
          completion_tokens: 1000,
          total_tokens: 1020,
        },
      };
  res.end(JSON.stringify(response));
});
await new Promise((resolve) => mock.listen(0, "127.0.0.1", resolve));
const api = `http://127.0.0.1:${mock.address().port}/v1`;
const reserve = http.createServer();
await new Promise((resolve) => reserve.listen(0, "127.0.0.1", resolve));
const port = reserve.address().port;
await new Promise((resolve) => reserve.close(resolve));
const origin = `http://127.0.0.1:${port}`;
const preview = spawn(process.execPath, [".output/server/index.mjs"], {
  cwd: fileURLToPath(new URL("../", import.meta.url)),
  env: { ...process.env, PORT: String(port), HOST: "127.0.0.1" },
  stdio: "ignore",
});
let browser;
try {
  let ready = false;
  for (let i = 0; i < 100; i++) {
    try {
      ready = (await fetch(origin)).ok;
      if (ready) break;
    } catch {
      /* server starting */
    }
    await delay(100);
  }
  assert(ready, "Preview failed to start; run npm run build first");
  browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 950 });
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    // The mock intentionally returns 401 for one provider.
    if (
      message.type() === "error" &&
      !message.text().includes("Failed to load resource")
    )
      errors.push(message.text());
  });
  await page.goto(origin, { waitUntil: "networkidle0" });
  async function clickText(text, selector = "button") {
    // Allow Reka dialog/select transitions to finish before a real pointer click.
    await delay(250);
    const element = await page.evaluateHandle(
      (text, selector) =>
        [...document.querySelectorAll(selector)].find(
          (el) =>
            el.textContent.trim() === text &&
            el.getBoundingClientRect().height > 0,
        ),
      text,
      selector,
    );
    assert(element.asElement(), `Missing button: ${text}`);
    await element.asElement().click();
    await element.dispose();
    await delay(300);
  }
  const fill = (id, text) =>
    page.$eval(
      id,
      (el, value) => {
        el.value = value;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      },
      text,
    );
  const dialogClosed = () =>
    page.waitForFunction(() => !document.querySelector("[role=dialog]"));

  await page.waitForSelector("textarea");
  await fill("textarea", "manual kept");
  await clickText("自动测试", "[role=tab]");
  await clickText("添加服务商");
  await page.waitForSelector("[role=dialog]");
  await clickText("保存", "[role=dialog] button");
  assert.equal(
    await page.$$eval("[role=dialog] [aria-invalid=true]", (els) => els.length),
    4,
  );
  await fill("#preset-name", "Chat Provider");
  await fill("#preset-base-url", "not-a-url");
  await fill("#preset-api-key", "sk-test-chat");
  await fill("#preset-model", "chat-model");
  await clickText("保存", "[role=dialog] button");
  assert(await page.$("#endpoint-error"));
  await fill("#preset-base-url", api);
  await clickText("保存", "[role=dialog] button");
  await dialogClosed();

  async function add(name, model, protocol = "chat") {
    await clickText("添加服务商");
    await page.waitForSelector("#preset-name");
    await fill("#preset-name", name);
    await fill("#preset-base-url", api);
    await fill("#preset-api-key", `sk-test-${model}`);
    await fill("#preset-model", model);
    if (protocol === "responses") {
      await page.click("[role=dialog] summary");
      await page.click("#preset-protocol");
      await clickText("Responses API", "[role=option]");
      await page.waitForFunction(
        () => !document.querySelector("[role=listbox]"),
      );
    }
    await clickText("保存", "[role=dialog] button");
    await dialogClosed();
  }
  await add("Responses Provider", "responses-model", "responses");
  await add("Bad key", "bad-key");
  await add("Invalid output", "invalid");
  assert.equal(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem("modeltrace.presets.v1")).presets
          .length,
    ),
    4,
  );

  await page.click('button[aria-label="测试 Chat Provider"]');
  await page.waitForFunction(() =>
    document
      .querySelector('[aria-label="服务商测试详情"]')
      .textContent.includes("有效查询 1/3"),
  );
  assert(await page.$('button[aria-label="编辑 Chat Provider"][disabled]'));
  await clickText("手动测试", "[role=tab]");
  assert.equal(await page.$eval("textarea", (el) => el.value), "manual kept");
  await clickText("自动测试", "[role=tab]");
  await page.waitForFunction(() =>
    document
      .querySelector('[aria-label="服务商测试详情"]')
      .textContent.includes("测试完成"),
  );
  assert.equal(calls.length, 3); // 3 challenges total: one short + two valid
  assert.equal(calls[0].url, "/v1/chat/completions");
  assert.equal(calls[0].auth, "Bearer sk-test-chat");
  assert.equal(typeof calls[0].data.messages[0].content, "string");
  assert.match(
    await page.$eval('[aria-label="服务商测试详情"]', (el) => el.textContent),
    /有效查询 2\/3/,
  );
  assert.equal(
    (await page.$$('[aria-label="挑战与模型回答"] details')).length,
    3,
  );
  assert(
    await page.$eval('[aria-label="服务商测试详情"]', (el) => {
      const result = el.querySelector('[aria-label="归因结果"]');
      const info = el.querySelector('[aria-label="服务商信息"]');
      return (
        !!(
          result.compareDocumentPosition(info) &
          Node.DOCUMENT_POSITION_FOLLOWING
        ) &&
        result.nextElementSibling?.getAttribute("data-orientation") ===
          "horizontal"
      );
    }),
    "Results and separator must precede provider information",
  );
  assert(
    !(await page.$eval('[aria-label="服务商测试详情"]', (el) =>
      el.textContent.includes("sk-test-chat"),
    )),
  );

  peak = 0;
  await clickText("一键测全部");
  await page.waitForFunction(() =>
    document.body.textContent.includes("排队中"),
  );
  await page.waitForFunction(
    () =>
      [...document.querySelectorAll("button")].some(
        (button) =>
          button.textContent.trim() === "一键测全部" && !button.disabled,
      ),
    { timeout: 30000 },
  );
  assert(peak <= 2, `Batch exceeded concurrency limit: ${peak}`);
  assert(
    calls.some(
      (call) =>
        call.url === "/v1/responses" && call.data.model === "responses-model",
    ),
  );
  assert.equal(counts["bad-key"], 1); // stop immediately on authentication failure
  assert.equal(counts.invalid, 3); // never generate replacement challenges
  assert.equal(counts["responses-model"], 3);
  await clickText("Bad key", "article strong");
  assert.match(
    await page.$eval('[aria-label="服务商测试详情"]', (el) => el.textContent),
    /Invalid API key/,
  );
  assert.equal(
    await page.$$eval(
      "button",
      (buttons) =>
        buttons.filter((button) => button.textContent.includes("终端方式"))
          .length,
    ),
    0,
  );
  assert.equal(
    (await page.$$('[role=dialog] textarea[aria-label="挑战 1 终端输出"]'))
      .length,
    0,
  );
  await clickText("Invalid output", "article strong");
  assert.match(
    await page.$eval('[aria-label="服务商测试详情"]', (el) => el.textContent),
    /没有获得可分析输出/,
  );
  await clickText("Chat Provider", "article strong");
  await page.click(
    '[aria-label="挑战与模型回答"] details:first-of-type summary',
  );
  assert.equal(
    await page.$eval(
      '[aria-label="挑战与模型回答"] pre',
      (el) => el.textContent,
    ),
    numbers,
  );

  await page.click('button[aria-label="删除 Bad key"]');
  await page.waitForSelector("[role=alertdialog]");
  await clickText("删除", "[role=alertdialog] button");
  await page.waitForFunction(
    () =>
      JSON.parse(localStorage.getItem("modeltrace.presets.v1")).presets
        .length === 3,
  );
  await page.click('button[aria-label="编辑 Responses Provider"]');
  await page.waitForSelector("#preset-name");
  assert.equal(
    await page.$eval("#preset-api-key", (el) => el.type),
    "password",
  );
  await fill("#preset-name", "Renamed responses");
  await clickText("保存", "[role=dialog] button");
  await dialogClosed();
  await page.reload({ waitUntil: "networkidle0" });
  await clickText("自动测试", "[role=tab]");
  assert(await page.$('button[aria-label="编辑 Renamed responses"]'));
  assert.equal(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem("modeltrace.presets.v1")).presets
          .length,
    ),
    3,
  );
  await page.setViewport({ width: 390, height: 844 });
  assert(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    "Mobile horizontal overflow",
  );
  assert.deepEqual(errors, []);
  console.log("PASS", JSON.stringify({ requests: calls.length, peak, counts }));
} finally {
  await browser?.close();
  preview.kill();
  mock.close();
}
