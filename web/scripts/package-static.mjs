import { execFileSync } from "node:child_process";
import {
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(web, "..");
const proxyURL =
  process.env.NUXT_PUBLIC_PROXY_URL ??
  "https://llm-iq-proxy.hanmo5888.workers.dev/v1";
// Nginx 包固定部署在站点根路径，不继承 GitHub Pages 的 /ModelTrace/。
execFileSync("npm", ["run", "generate"], {
  cwd: web,
  env: {
    ...process.env,
    NUXT_APP_BASE_URL: "/",
    NUXT_PUBLIC_PROXY_URL: proxyURL,
  },
  stdio: "inherit",
});
const dist = join(root, "dist");
mkdirSync(dist, { recursive: true });
const staging = mkdtempSync(join(tmpdir(), "modeltrace-static-"));
const archive = join(dist, "modeltrace-static.tar.gz");
try {
  const bundle = join(staging, "modeltrace-static");
  mkdirSync(bundle);
  cpSync(join(web, ".output/public"), join(bundle, "site"), {
    recursive: true,
  });
  cpSync(join(root, "deploy/nginx/nginx.conf"), join(bundle, "nginx.conf"));
  cpSync(join(root, "deploy/nginx/README.md"), join(bundle, "README.md"));
  cpSync(join(root, "LICENSE"), join(bundle, "LICENSE"));
  writeFileSync(
    join(bundle, "build-info.json"),
    JSON.stringify(
      {
        builtAt: new Date().toISOString(),
        baseURL: "/",
        proxyURL,
        sourceCommit: execFileSync("git", ["rev-parse", "HEAD"], {
          cwd: root,
          encoding: "utf8",
        }).trim(),
        workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], {
          cwd: root,
          encoding: "utf8",
        }).trim(),
      },
      null,
      2,
    ) + "\n",
  );
  execFileSync("tar", ["-czf", archive, "-C", staging, "modeltrace-static"]);
  const hash = createHash("sha256").update(readFileSync(archive)).digest("hex");
  writeFileSync(`${archive}.sha256`, `${hash}  modeltrace-static.tar.gz\n`);
  console.log(`\nStatic Nginx bundle: ${archive}\nSHA-256: ${hash}`);
} finally {
  rmSync(staging, { recursive: true, force: true });
}
