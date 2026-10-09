// Smoke-test the extracted release with a plain static file server, not Nuxt.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const directory = await mkdtemp(join(tmpdir(), "modeltrace-package-test-"));
execFileSync("tar", [
  "-xzf",
  fileURLToPath(
    new URL("../../dist/modeltrace-static.tar.gz", import.meta.url),
  ),
  "-C",
  directory,
]);
const root = join(directory, "modeltrace-static/site");
const metadata = JSON.parse(
  await readFile(join(directory, "modeltrace-static/build-info.json"), "utf8"),
);
assert.equal(metadata.baseURL, "/");
assert(metadata.proxyURL, "This consent smoke test expects a configured proxy");
const errors = [];
const mime = {
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".woff2": "font/woff2",
};
const server = http.createServer(async (req, res) => {
  const pathname = decodeURIComponent(
    new URL(req.url, "http://localhost").pathname,
  );
  // Browsers may request an optional favicon even though the app declares none.
  if (pathname === "/favicon.ico") return res.writeHead(204).end();
  const file = resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`);
  if (!file.startsWith(root + "/")) return res.writeHead(403).end();
  try {
    let content = await readFile(file);
    if (pathname.endsWith("/data/unified_bank.json")) {
      // Deterministic high-confidence fixture: exercise the real attribution + UI early stop.
      const bank = JSON.parse(content);
      for (const calibration of Object.values(bank.calibration))
        calibration.beta = 1000;
      content = Buffer.from(JSON.stringify(bank));
    }
    const extension = file.slice(file.lastIndexOf("."));
    res.setHeader("Content-Type", mime[extension] || "text/html");
    res.end(content);
  } catch {
    errors.push(`404 ${pathname}`);
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
let browser;
let proxyCalls = 0;
try {
  browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    if (request.url().startsWith(metadata.proxyURL)) {
      const headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers":
          "authorization,content-type,x-modeltrace-endpoint",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
      };
      if (request.method() === "OPTIONS")
        return request.respond({ status: 204, headers });
      proxyCalls++;
      return request.respond({
        status: 200,
        headers,
        contentType: "application/json",
        body: JSON.stringify({
          id: "mock",
          object: "chat.completion",
          created: 1,
          model: "test-model",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: Array.from(
                  { length: 310 },
                  (_, i) => ((i * 73) % 355) + 1,
                ).join(" "),
              },
              finish_reason: "stop",
            },
          ],
        }),
      });
    }
    if (request.url().startsWith("https://api.vendor.com/")) {
      errors.push("Unexpected direct API request");
      return request.abort();
    }
    return request.continue();
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/`, {
    waitUntil: "networkidle0",
  });
  await page.waitForFunction(() =>
    /指纹库 \d+ 模型/.test(document.body.innerText),
  );
  const click = async (text) => {
    const handle = await page.evaluateHandle(
      (text) =>
        [...document.querySelectorAll("button")].find(
          (button) => button.textContent.trim() === text && button.offsetWidth,
        ),
      text,
    );
    assert(handle.asElement(), `Missing button: ${text}`);
    await handle.asElement().click();
    await new Promise((resolve) => setTimeout(resolve, 300));
  };
  await click("自动测试");
  await click("添加服务商");
  for (const [id, value] of Object.entries({
    "preset-name": "Static test",
    "preset-base-url": "https://api.vendor.com/v1",
    "preset-api-key": "mock-key",
    "preset-model": "test-model",
  })) {
    await page.$eval(
      `#${id}`,
      (el, value) => {
        el.value = value;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      },
      value,
    );
  }
  await click("保存");
  await click("开始测试");
  await page.waitForSelector("[role=alertdialog]");
  assert.equal(proxyCalls, 0);
  await click("同意并通过代理测试");
  await page.waitForFunction(() =>
    document
      .querySelector('[aria-label="服务商测试详情"]')
      ?.textContent.includes("成功检验"),
  );
  assert.equal(proxyCalls, 1, "High confidence must stop after first answer");
  assert.equal(
    (await page.$$('[aria-label="挑战与模型回答"] details')).length,
    3,
  );
  assert.equal(
    await page.$$eval(
      '[aria-label="挑战与模型回答"] summary',
      (items) =>
        items.filter((item) => item.textContent.includes("未调用")).length,
    ),
    2,
  );
  assert.equal(
    await page.$eval('[aria-label="调用进度"] [role=progressbar]', (el) =>
      el.getAttribute("aria-valuenow"),
    ),
    "100",
  );
  assert.deepEqual(errors, []);
  console.log(
    "PASS extracted static package, root assets/bank, proxy opt-in and one-answer 99% early stop",
  );
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
  await rm(directory, { recursive: true, force: true });
}
