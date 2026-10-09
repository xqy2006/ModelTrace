import assert from "node:assert/strict";
import http from "node:http";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { build } from "esbuild";
import worker from "../../worker/src/index.mjs";

const exec = promisify(execFile);
// Bundle TypeScript independently of Nuxt so the command/parser can be exercised in Node.
const output = await build({
  entryPoints: ["lib/terminal-commands.ts"],
  bundle: true,
  write: false,
  platform: "node",
  format: "esm",
});
const { terminalCommand, extractTerminalAnswer } = await import(
  `data:text/javascript;base64,${Buffer.from(output.outputFiles[0].contents).toString("base64")}`
);
const numbers = Array.from({ length: 300 }, (_, i) => (i % 355) + 1).join(" ");
const server = http.createServer(async (req, res) => {
  let body = "";
  for await (const chunk of req) body += chunk;
  const json = JSON.parse(body);
  assert.equal(req.headers.authorization, "Bearer test-secret");
  assert.match(JSON.stringify(json), /quote'测试/);
  res.setHeader("content-type", "application/json");
  res.end(
    JSON.stringify(
      req.url.endsWith("/responses")
        ? {
            output: [
              {
                type: "message",
                content: [{ type: "output_text", text: numbers }],
              },
            ],
          }
        : { choices: [{ message: { content: numbers } }] },
    ),
  );
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
try {
  const baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
  for (const apiType of ["chat", "responses"]) {
    for (const format of ["text", "json"]) {
      const command = terminalCommand(
        { baseUrl, apiType, model: "gpt-test", temperature: null },
        "quote'测试 1 到 355 的整数",
        "posix",
        format,
      );
      assert(
        !command.includes("test-secret"),
        "Secret must not be placed in command",
      );
      const { stdout } = await exec("bash", ["-c", command], {
        env: { ...process.env, MODELTRACE_API_KEY: "test-secret" },
        maxBuffer: 1e6,
      });
      assert.equal(extractTerminalAnswer(stdout), numbers);
      const ps = terminalCommand(
        { baseUrl, apiType, model: "gpt-test", temperature: null },
        "quote'测试",
        "powershell",
        format,
      );
      assert.match(ps, /quote''测试/);
      assert.match(ps, /Set-Clipboard/);
      assert(!ps.includes("test-secret"));
    }
  }
  assert.throws(() => extractTerminalAnswer("{invalid"), /JSON/);
  assert.throws(
    () =>
      extractTerminalAnswer(
        JSON.stringify({ error: { message: "unauthorized" } }),
      ),
    /unauthorized/,
  );
} finally {
  await new Promise((resolve) => server.close(resolve));
}

const env = {
  SITE_ORIGIN: "https://app.example",
  RATE_LIMITER: { limit: async () => ({ success: true }) },
};
const prompt = "请直接输出 300 个 1 到 355 的整数，不要解释";
const mk = (body, headers = {}, path = "/v1/chat/completions") =>
  new Request(`https://worker.example${path}`, {
    method: "POST",
    headers: {
      Origin: "https://app.example",
      Authorization: "Bearer test-secret",
      "X-ModelTrace-Endpoint": "https://api.vendor.com/v1",
      "Content-Type": "application/json",
      "CF-Connecting-IP": "203.0.113.7",
      ...headers,
    },
    body: JSON.stringify(body),
  });
const chatBody = {
  model: "gpt-test",
  messages: [{ role: "user", content: prompt }],
  stream: false,
};
const originalFetch = globalThis.fetch;
let called;
try {
  globalThis.fetch = async (url, opts) => {
    called = { url: String(url), opts };
    return new Response(
      JSON.stringify({ choices: [{ message: { content: numbers } }] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };
  const ok = await worker.fetch(mk(chatBody), env);
  assert.equal(ok.status, 200);
  assert.equal(ok.headers.get("access-control-allow-origin"), env.SITE_ORIGIN);
  assert.equal(called.url, "https://api.vendor.com/v1/chat/completions");
  assert.equal(called.opts.headers.Authorization, "Bearer test-secret");
  assert.equal(JSON.parse(called.opts.body).max_tokens, 4096);
  assert.equal(
    (
      await worker.fetch(
        mk(
          {
            model: "gpt-test",
            input: [
              { role: "user", content: [{ type: "input_text", text: prompt }] },
            ],
            stream: false,
          },
          {},
          "/v1/responses",
        ),
        env,
      )
    ).status,
    200,
  );
  assert.equal(called.url, "https://api.vendor.com/v1/responses");
  assert.equal(
    (await worker.fetch(mk(chatBody, { Origin: "https://evil.example" }), env))
      .status,
    403,
  );
  assert.equal(
    (
      await worker.fetch(
        mk(chatBody, { "X-ModelTrace-Endpoint": "https://localhost/v1" }),
        env,
      )
    ).status,
    403,
  );
  assert.equal(
    (
      await worker.fetch(
        mk({
          ...chatBody,
          tools: [{ name: "bad" }],
          messages: [{ role: "system", content: prompt }],
        }),
        env,
      )
    ).status,
    400,
  );
  assert.equal(
    (
      await worker.fetch(mk(chatBody), {
        ...env,
        RATE_LIMITER: { limit: async () => ({ success: false }) },
      })
    ).status,
    429,
  );
  assert.equal(
    (
      await worker.fetch(
        mk(chatBody, { "X-ModelTrace-Endpoint": "https://127.0.0.1/v1" }),
        env,
      )
    ).status,
    403,
  );
  assert.equal(
    (await worker.fetch(mk(chatBody), { ...env, RATE_LIMITER: undefined }))
      .status,
    503,
  );
} finally {
  globalThis.fetch = originalFetch;
}
console.log("PASS terminal command, JSON parsing, proxy restrictions");
