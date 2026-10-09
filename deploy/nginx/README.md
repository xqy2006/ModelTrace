# ModelTrace 纯静态 Nginx 部署包

源码：[Hanmo123/ModelTrace](https://github.com/Hanmo123/ModelTrace)。基于 [xqy2006/ModelTrace](https://github.com/xqy2006/ModelTrace) 继续开发，遵循包内 `LICENSE` 的 MIT 许可证，保留原作者版权声明。

本包不需要 Node.js、Python 或应用后端。Nginx 只提供 `site/` 中的 HTML、JS、CSS 和指纹库；归因计算在浏览器中完成。可选 Worker 负责转发模型请求，不由 Nginx 代理。

## 部署

```bash
# 在压缩包所在目录执行。若有校验文件，先核验完整性：
sha256sum -c modeltrace-static.tar.gz.sha256

tar -xzf modeltrace-static.tar.gz
sudo mkdir -p /var/www/modeltrace
sudo cp -a modeltrace-static/site /var/www/modeltrace/
sudo chmod -R a+rX /var/www/modeltrace/site
sudo cp modeltrace-static/nginx.conf /etc/nginx/conf.d/modeltrace.conf
# 编辑配置：把 server_name _ 改为你的域名，确认没有同域名的默认站点冲突。
sudo nginx -t
sudo systemctl reload nginx
```

Nginx 的 `root` 必须指向包含 `index.html` 的 `/var/www/modeltrace/site`，而不是压缩包外层目录。确认 `/etc/nginx/nginx.conf` 的 `http {}` 中 include 了 `/etc/nginx/conf.d/*.conf`。若部署环境没有 systemd，可使用 `sudo nginx -s reload`。

该包的资源路径为 `/`，请部署到域名根路径，不要挂在 `/ModelTrace/` 子路径。生产环境建议通过证书或入口反向代理启用 HTTPS。若使用 Docker Nginx，可把 `site/` 挂载到 `/var/www/modeltrace/site:ro`，把 `nginx.conf` 挂载到 `/etc/nginx/conf.d/default.conf:ro`。

## Worker 来源配置（重要）

打包时默认内置代理地址：`https://llm-iq-proxy.hanmo5888.workers.dev/v1`，实际值见 `build-info.json`。

**部署静态页面不会自动更新 Worker 的来源限制。** 在 Worker 项目的 `worker/wrangler.toml` 中将 `SITE_ORIGIN` 改为页面的真实 Origin，例如：

```toml
[vars]
SITE_ORIGIN = "https://modeltrace.your-domain.com"
```

Origin 包含协议、域名和非默认端口，不含路径或末尾 `/`。HTTP 和 HTTPS 是不同来源。修改后从源码仓库执行：

```bash
npx wrangler@latest deploy --config worker/wrangler.toml
```

保留 `RATE_LIMITER` binding；无需配置任何服务商 API Key。当前 Worker 只支持一个精确来源。配置错误时代理预检返回 403。未获授权时不会向 Worker 发送密钥；用户在测试前同意后，本浏览器后续单个/批量测试将直接使用代理，可在页面撤销授权。代理地址变更需要重新同意。

API Key 由用户在浏览器填写，仅保存在该浏览器 localStorage。不要把密钥写进压缩包或 Nginx/Worker 配置；只信任你认可的代理运营者。

## 重新打包

在源码目录执行（构建需要 Node.js 22、npm 和 tar，部署服务器不需要）：

```bash
cd web
npm ci
npm run package:static
# 可覆盖构建时的代理 URL；显式设为空字符串则禁用代理：
# NUXT_PUBLIC_PROXY_URL=https://your-worker.workers.dev/v1 npm run package:static
# NUXT_PUBLIC_PROXY_URL='' npm run package:static
```

产物为仓库根目录 `dist/modeltrace-static.tar.gz` 及 SHA-256 校验文件。Nginx 运行时设置环境变量无法更改已生成的代理 URL，需重新构建。
