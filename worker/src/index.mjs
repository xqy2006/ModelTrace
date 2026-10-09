// Optional, self-hosted Cloudflare Worker relay.
// Accepts arbitrary public HTTPS OpenAI-compatible endpoints, but only for a tightly
// constrained, single non-streaming ModelTrace challenge. It is not a generic URL proxy.
const MAX_BODY = 12_000;
const MAX_RESPONSE = 1_000_000;

function forbiddenHost(hostname) {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (!/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(host)) return true;
  if (/^\d{1,3}(?:\.\d{1,3}){3}$/.test(host) || host.includes(":")) return true;
  const suffixes = [
    "localhost",
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home",
    ".corp",
    ".test",
    ".invalid",
    ".example",
    ".arpa",
    ".onion",
  ];
  if (suffixes.some((suffix) => host === suffix || host.endsWith(suffix)))
    return true;
  return [
    "nip.io",
    "sslip.io",
    "xip.io",
    "localtest.me",
    "lvh.me",
    "traefik.me",
  ].some((domain) => host === domain || host.endsWith(`.${domain}`));
}

function reply(message, status, headers) {
  return new Response(JSON.stringify({ error: { message } }), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", ...headers },
  });
}

async function readLimited(stream, limit) {
  if (!stream) return "";
  const reader = stream.getReader();
  const chunks = [];
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > limit) throw new Error("内容超过大小限制");
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  }
  const out = new Uint8Array(bytes);
  let cursor = 0;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.length;
  }
  return new TextDecoder().decode(out);
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin");
    // Origin is not authentication; mandatory per-IP rate limiting remains required.
    if (!env.SITE_ORIGIN || !env.RATE_LIMITER) return reply("代理未配置", 503);
    if (origin !== env.SITE_ORIGIN) return reply("不允许的页面来源", 403);
    const cors = {
      "Access-Control-Allow-Origin": env.SITE_ORIGIN,
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers":
        "authorization,content-type,x-modeltrace-endpoint",
      Vary: "Origin",
      "Cache-Control": "no-store",
    };
    if (request.method === "OPTIONS")
      return new Response(null, { status: 204, headers: cors });
    const path = new URL(request.url).pathname;
    if (
      request.method !== "POST" ||
      !["/v1/chat/completions", "/v1/responses"].includes(path)
    )
      return reply("不支持的路径或方法", 405, cors);
    const ip = request.headers.get("CF-Connecting-IP");
    if (!ip) return reply("缺少限流标识", 403, cors);
    const quota = await env.RATE_LIMITER.limit({ key: ip });
    if (!quota.success) return reply("请求过于频繁", 429, cors);

    let base;
    try {
      base = new URL(request.headers.get("X-ModelTrace-Endpoint") || "");
      if (
        base.protocol !== "https:" ||
        forbiddenHost(base.hostname) ||
        base.port ||
        base.username ||
        base.password ||
        base.search ||
        base.hash ||
        !/^\/(?:[a-z0-9_-]+\/)*v1\/?$/i.test(base.pathname)
      ) {
        return reply("服务商地址不是允许的公网 HTTPS API 根地址", 403, cors);
      }
    } catch {
      return reply("无效的服务商地址", 400, cors);
    }
    const auth = request.headers.get("Authorization");
    if (!auth || !/^Bearer [\x21-\x7e]{1,300}$/.test(auth))
      return reply("缺少服务商密钥", 400, cors);
    if (Number(request.headers.get("Content-Length") || 0) > MAX_BODY)
      return reply("请求过大", 413, cors);
    let body;
    try {
      body = JSON.parse(await readLimited(request.body, MAX_BODY));
    } catch {
      return reply("无效或过大的 JSON 请求", 400, cors);
    }
    if (
      !body ||
      typeof body !== "object" ||
      Array.isArray(body) ||
      typeof body.model !== "string" ||
      !/^[\w.:-]{1,128}$/.test(body.model) ||
      body.stream === true ||
      (body.temperature !== undefined &&
        (typeof body.temperature !== "number" ||
          body.temperature < 0 ||
          body.temperature > 2))
    ) {
      return reply("仅允许非流式模型请求", 400, cors);
    }
    const input = body.input;
    const prompt =
      path === "/v1/responses"
        ? typeof input === "string"
          ? input
          : Array.isArray(input) &&
              input.length === 1 &&
              input[0]?.role === "user" &&
              Array.isArray(input[0]?.content) &&
              input[0].content.length === 1 &&
              input[0].content[0]?.type === "input_text"
            ? input[0].content[0].text
            : null
        : Array.isArray(body.messages) &&
            body.messages.length === 1 &&
            body.messages[0]?.role === "user"
          ? body.messages[0].content
          : null;
    if (
      typeof prompt !== "string" ||
      prompt.length < 10 ||
      prompt.length > 4000 ||
      !prompt.includes("1 到 355") ||
      !prompt.includes("整数")
    ) {
      return reply("只允许 ModelTrace 数值挑战", 400, cors);
    }
    // Drop every caller-supplied parameter except the four validated fields.
    const outgoing =
      path === "/v1/responses"
        ? {
            model: body.model,
            input: prompt,
            stream: false,
            max_output_tokens: 4096,
          }
        : {
            model: body.model,
            messages: [{ role: "user", content: prompt }],
            stream: false,
            max_tokens: 4096,
          };
    if (body.temperature !== undefined) outgoing.temperature = body.temperature;
    const url = new URL(path.slice(4), base.href.replace(/\/+$/, "") + "/");
    try {
      const response = await fetch(url, {
        method: "POST",
        redirect: "manual",
        signal: AbortSignal.timeout(180_000),
        headers: {
          Authorization: auth,
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(outgoing),
      });
      if (response.status >= 300 && response.status < 400)
        return reply("服务商重定向已被拒绝", 502, cors);
      if (!response.headers.get("content-type")?.toLowerCase().includes("json"))
        return reply("服务商返回了非 JSON 响应", 502, cors);
      const text = await readLimited(response.body, MAX_RESPONSE);
      return new Response(text, {
        status: response.status,
        headers: { ...cors, "content-type": "application/json; charset=utf-8" },
      });
    } catch {
      return reply("服务商请求失败或响应过大", 502, cors);
    }
  },
};
