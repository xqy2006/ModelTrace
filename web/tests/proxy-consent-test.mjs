// Run after building with NUXT_PUBLIC_PROXY_URL=http://127.0.0.1:3244/v1
import puppeteer from "puppeteer-core";
import http from "node:http";
import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const numbers = Array.from(
  { length: 310 },
  (_, i) => ((i * 73) % 355) + 1,
).join(" ");
let proxyCalls = 0;
let targetCalls = 0;
const target = http.createServer((req, res) => {
  targetCalls++;
  res.writeHead(403).end();
}); // no CORS
await new Promise((resolve) => target.listen(0, "127.0.0.1", resolve));
const proxy = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "authorization,content-type,x-modeltrace-endpoint",
  );
  res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
  if (req.method === "OPTIONS") {
    res.writeHead(204).end();
    return;
  }
  proxyCalls++;
  assert.equal(req.headers.authorization, "Bearer sk-test");
  assert.equal(
    req.headers["x-modeltrace-endpoint"],
    `http://127.0.0.1:${target.address().port}/v1`,
  );
  res.setHeader("Content-Type", "application/json");
  res.end(
    JSON.stringify({
      id: "mock",
      object: "chat.completion",
      created: 1,
      model: "test-model",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: numbers },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 800, total_tokens: 810 },
    }),
  );
});
await new Promise((resolve) => proxy.listen(3244, "127.0.0.1", resolve));
const server = spawn(process.execPath, [".output/server/index.mjs"], {
  cwd: fileURLToPath(new URL("../", import.meta.url)),
  env: { ...process.env, PORT: "3245" },
  stdio: "ignore",
});
let browser;
try {
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch("http://127.0.0.1:3245")).ok) break;
    } catch {}
    await delay(100);
  }
  browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();
  await page.goto("http://127.0.0.1:3245", { waitUntil: "networkidle0" });
  const click = async (text) => {
    const node = await page.evaluateHandle(
      (text) =>
        [...document.querySelectorAll("button")].find(
          (button) => button.textContent.trim() === text && button.offsetWidth,
        ),
      text,
    );
    assert(node.asElement(), `button not found: ${text}`);
    await node.asElement().click();
    await delay(300);
  };
  const fill = (selector, value) =>
    page.$eval(
      selector,
      (el, v) => {
        el.value = v;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      },
      value,
    );
  await page.$("[role=tab]");
  const tab = await page.evaluateHandle(() =>
    [...document.querySelectorAll("[role=tab]")].find((e) =>
      e.textContent.includes("自动测试"),
    ),
  );
  await tab.asElement().click();
  await delay(300);
  await click("添加服务商");
  await fill("#preset-name", "CORS blocked");
  await fill(
    "#preset-base-url",
    `http://127.0.0.1:${target.address().port}/v1`,
  );
  await fill("#preset-api-key", "sk-test");
  await fill("#preset-model", "test-model");
  await click("保存");
  await delay(350);
  await click("开始测试");
  await page.waitForSelector("[role=alertdialog]", { timeout: 30000 });
  assert.equal(proxyCalls, 0, "Must not proxy before consent");
  assert.equal(targetCalls, 0, "Must ask before ANY API request");
  await page.keyboard.press("Escape");
  await delay(350);
  assert.equal(
    targetCalls + proxyCalls,
    0,
    "Dismissing consent must cancel testing",
  );
  await click("一键测全部");
  await page.waitForSelector("[role=alertdialog]");
  assert.equal(
    targetCalls + proxyCalls,
    0,
    "Batch must also ask before testing",
  );
  await click("仅本次直连");
  await page.waitForFunction(
    () => !document.querySelector("[role=alertdialog]"),
  );
  assert.equal(proxyCalls, 0, "Declining must not proxy");
  assert.equal((await page.$$("[role=dialog]")).length, 0);
  assert.equal(
    await page.$$eval(
      "button",
      (buttons) =>
        buttons.filter((button) => button.textContent.includes("终端方式"))
          .length,
    ),
    0,
  );
  await page.waitForFunction(() =>
    document
      .querySelector('[aria-label="服务商测试详情"]')
      ?.textContent.includes("没有获得可分析输出"),
  );
  assert(targetCalls > 0);
  assert.equal(proxyCalls, 0);
  assert.equal(
    (await page.$$("[role=alertdialog]")).length,
    0,
    "Declined batch must not prompt again after CORS failure",
  );
  await click("通过代理测试");
  await page.waitForSelector("[role=alertdialog]");
  await click("同意并通过代理测试");
  await page.waitForFunction(
    () =>
      document
        .querySelector('[aria-label="服务商测试详情"]')
        ?.textContent.includes("有效查询 3/3"),
    { timeout: 30000 },
  );
  assert.equal(proxyCalls, 3);
  const directCallsAfterDecline = targetCalls;
  assert.equal(
    await page.evaluate(() =>
      localStorage.getItem("modeltrace.proxy-consent.v1"),
    ),
    "http://127.0.0.1:3244/v1",
  );

  // A second provider is also covered by the explicitly granted future-test consent.
  await click("添加服务商");
  await fill("#preset-name", "Second provider");
  await fill(
    "#preset-base-url",
    `http://127.0.0.1:${target.address().port}/v1`,
  );
  await fill("#preset-api-key", "sk-test");
  await fill("#preset-model", "test-model");
  await click("保存");
  await click("一键测全部");
  await page.waitForFunction(() =>
    document.body.textContent.includes("批量测试完成：成功 2 个"),
  );
  assert.equal(proxyCalls, 9);
  assert.equal(
    targetCalls,
    directCallsAfterDecline,
    "Authorized batches must use proxy directly",
  );
  assert.equal((await page.$$("[role=alertdialog]")).length, 0);

  await page.reload({ waitUntil: "networkidle0" });
  await click("自动测试");
  await click("开始测试");
  await page.waitForFunction(() =>
    document
      .querySelector('[aria-label="服务商测试详情"]')
      ?.textContent.includes("测试完成"),
  );
  assert.equal(proxyCalls, 12, "Remember consent across reloads");
  assert.equal((await page.$$("[role=alertdialog]")).length, 0);

  await click("撤销代理授权");
  await click("一键测全部");
  await page.waitForSelector("[role=alertdialog]");
  assert.equal(proxyCalls, 12, "Revocation requires renewed consent");
  await page.keyboard.press("Escape");
  await page.evaluate(() =>
    localStorage.setItem(
      "modeltrace.proxy-consent.v1",
      "https://different-proxy.invalid/v1",
    ),
  );
  await page.reload({ waitUntil: "networkidle0" });
  await click("自动测试");
  await click("开始测试");
  await page.waitForSelector("[role=alertdialog]");
  assert.equal(
    proxyCalls,
    12,
    "Consent to a different proxy must not be reused",
  );
  console.log(
    "PASS preflight single/batch consent, decline, remembered consent, revocation and URL scoping",
  );
} finally {
  await browser?.close();
  server.kill();
  await new Promise((resolve) => proxy.close(resolve));
  await new Promise((resolve) => target.close(resolve));
}
