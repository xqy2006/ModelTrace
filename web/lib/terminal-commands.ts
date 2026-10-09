import type { EndpointPreset } from "../composables/usePresets";

export type TerminalShell = "posix" | "powershell";
export type OutputFormat = "text" | "json";

const shellQuote = (value: string) => `'${value.replaceAll("'", "'\"'\"'")}'`;
const psQuote = (value: string) => `'${value.replaceAll("'", "''")}'`;

/** Only generated prompts and the URL go into the command; NEVER interpolate the API key. */
export function terminalCommand(
  preset: Pick<EndpointPreset, "baseUrl" | "model" | "apiType" | "temperature">,
  prompt: string,
  shell: TerminalShell,
  format: OutputFormat = "text",
): string {
  const endpoint = `${preset.baseUrl.replace(/\/+$/, "")}/${preset.apiType === "responses" ? "responses" : "chat/completions"}`;
  const body =
    preset.apiType === "responses"
      ? {
          model: preset.model,
          input: prompt,
          stream: false,
          ...(preset.temperature === null
            ? {}
            : { temperature: preset.temperature }),
        }
      : {
          model: preset.model,
          messages: [{ role: "user", content: prompt }],
          stream: false,
          ...(preset.temperature === null
            ? {}
            : { temperature: preset.temperature }),
        };
  const json = JSON.stringify(body);

  if (shell === "powershell") {
    const parse =
      format === "json"
        ? "$answer = $raw"
        : preset.apiType === "responses"
          ? '$j = $raw | ConvertFrom-Json; if ($j.error) { throw $j.error.message }; $answer = (($j.output | Where-Object { $_.type -eq "message" } | ForEach-Object { $_.content } | Where-Object { $_.type -eq "output_text" } | ForEach-Object { $_.text }) -join "`n")'
          : "$j = $raw | ConvertFrom-Json; if ($j.error) { throw $j.error.message }; $answer = $j.choices[0].message.content";
    return [
      'if (-not $env:MODELTRACE_API_KEY) { throw "请先在本终端输入 API Key" }',
      '$p = Join-Path $env:TEMP ("modeltrace-" + [Guid]::NewGuid().ToString("N") + ".json")',
      '$r = Join-Path $env:TEMP ("modeltrace-" + [Guid]::NewGuid().ToString("N") + ".json")',
      "try {",
      `  [IO.File]::WriteAllText($p, ${psQuote(json)}, [Text.UTF8Encoding]::new($false))`,
      `  curl.exe -sS -X POST ${psQuote(endpoint)} -H "Authorization: Bearer $env:MODELTRACE_API_KEY" -H 'Content-Type: application/json' --data-binary "@$p" -o $r`,
      '  if ($LASTEXITCODE -ne 0) { throw "curl 请求失败" }',
      "  $raw = [IO.File]::ReadAllText($r, [Text.Encoding]::UTF8)",
      `  ${parse}`,
      '  if (-not $answer) { throw "没有找到模型回答；可切换为原始 JSON 命令查看错误" }',
      "  $answer | Set-Clipboard; $answer",
      "} finally { Remove-Item -LiteralPath $p, $r -Force -ErrorAction SilentlyContinue }",
    ].join("\n");
  }
  const curl = `curl -sS -X POST ${shellQuote(endpoint)} -H "Authorization: Bearer $MODELTRACE_API_KEY" -H 'Content-Type: application/json' --data-binary ${shellQuote(json)}`;
  // POSIX command requires Python 3 only for parsed output; JSON option needs nothing beyond curl.
  const python =
    preset.apiType === "responses"
      ? 'import sys,json; d=json.load(sys.stdin); e=d.get("error"); e and sys.exit(str(e.get("message",e))); t="\\n".join(c.get("text","") for m in d.get("output",[]) if m.get("type")=="message" for c in m.get("content",[]) if c.get("type")=="output_text"); t or sys.exit("没有找到模型回答"); print(t)'
      : 'import sys,json; d=json.load(sys.stdin); e=d.get("error"); e and sys.exit(str(e.get("message",e))); a=d.get("choices") or []; a or sys.exit("没有找到模型回答"); t=a[0].get("message",{}).get("content",""); t or sys.exit("没有找到模型回答"); print(t)';
  const command = format === "text" ? `${curl} | python3 -c '${python}'` : curl;
  return [
    'if [ -z "$MODELTRACE_API_KEY" ]; then echo "请先在本终端输入 API Key" >&2; else',
    `  answer=$(${command}) && { printf '%s\\n' "$answer"; if command -v pbcopy >/dev/null; then printf '%s' "$answer" | pbcopy 2>/dev/null || true; elif command -v wl-copy >/dev/null; then printf '%s' "$answer" | wl-copy 2>/dev/null || true; elif command -v xclip >/dev/null; then printf '%s' "$answer" | xclip -selection clipboard 2>/dev/null || true; fi; }`,
    "fi",
  ].join("\n");
}

export function terminalKeySetup(shell: TerminalShell): string {
  if (shell === "powershell")
    return '$env:MODELTRACE_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "API Key" -AsSecureString)).Password';
  return 'printf "API Key: "; IFS= read -rs MODELTRACE_API_KEY; printf "\\n"; export MODELTRACE_API_KEY';
}

/** Accept either extracted plain text or non-streaming OpenAI-compatible JSON. */
export function extractTerminalAnswer(value: string): string {
  const raw = value.trim();
  if (!raw) return "";
  if (!raw.startsWith("{") && !raw.startsWith("[")) return raw;
  let data: any;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(
      "JSON 不完整或格式错误；请粘贴完整响应，或使用终端中已提取的回答文本。",
    );
  }
  if (!data || Array.isArray(data) || typeof data !== "object")
    throw new Error("不是有效的 API 响应对象");
  if (data.error)
    throw new Error(
      `服务商返回错误：${String(data.error.message || data.error).slice(0, 180)}`,
    );
  const content = data.choices?.[0]?.message?.content;
  if (typeof content === "string" && content.trim()) return content.trim();
  if (Array.isArray(content)) {
    const text = content
      .filter((part: any) => part && typeof part.text === "string")
      .map((part: any) => part.text)
      .join("\n")
      .trim();
    if (text) return text;
  }
  if (Array.isArray(data.output)) {
    const text = data.output
      .filter((item: any) => item?.type === "message")
      .flatMap((item: any) => item.content || [])
      .filter(
        (item: any) =>
          item?.type === "output_text" && typeof item.text === "string",
      )
      .map((item: any) => item.text)
      .join("\n")
      .trim();
    if (text) return text;
  }
  throw new Error("未找到 Chat Completions 或 Responses 格式的回答文本");
}
