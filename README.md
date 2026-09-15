# ModelTrace

ModelTrace 是一个本地运行的主动模型归因工具。它通过三条独立的长整数生成挑战提取输出指纹，在统一候选库中自动判断模型家族和具体版本。

## 运行

```powershell
python -m pip install -r requirements.txt
python start.py
```

页面地址为 `http://127.0.0.1:7860/`。

## GitHub Pages

`static/index.html` 是不依赖后端的手动测试版本，归因计算和指纹库读取都在浏览器本地完成。仓库附带的 GitHub Actions 会将 `static/` 部署到 GitHub Pages。

## 使用

- **手动测试**：复制三条挑战，分别发送给同一个待测模型，再粘贴每次完整输出。
- **API 自动测试**：选择服务提供方，填写所需信息。程序会自动尝试 OpenAI Chat Completions 与 Anthropic Messages 格式，以三份有效回答为目标完成归因。
- **指纹库管理**：可以新建指纹库，或通过 API 为现有指纹库添加模型指纹。

自定义提供方需要填写 Base URL、API Key 和模型名。OrcaRouter 提供方不需要填写这些内容，见下节。

### OrcaRouter 提供方

服务提供方选择 **OrcaRouter - API** 或 **OrcaRouter - Auth** 后，模型名不再手动输入，而是从 OrcaRouter 的实时模型目录中按当前能力筛选出的下拉列表选择。两种接入方式获得的都是同一把 `sk-orca-…` 密钥，归因请求同样发往 `https://api.orcarouter.ai/v1`。

| 接入方式 | 提供方 | 凭据来源 |
| --- | --- | --- |
| API Key | OrcaRouter - API | 粘贴已有 `sk-orca-…` 密钥，或用 `ORCAROUTER_API_KEY` 环境变量 |
| 账户授权 | OrcaRouter - Auth | OAuth 2.0 + PKCE（S256）浏览器授权，无需 client secret |

授权流程使用本机回环地址接收回调（Flow A），因此不需要预先注册回调地址；若在同意页面选择“显示授权码”，把授权码粘贴回对话框即可。授权入口固定为 `https://www.orcarouter.ai/auth`，兑换固定为 `https://www.orcarouter.ai/api/v1/auth/keys`，模型目录与推理固定为 `https://api.orcarouter.ai/v1`。

凭据保存在 `~/.modeltrace/orcarouter.json`（权限 0600，可用 `MODELTRACE_CREDENTIAL_FILE` 覆盖），也可以用 `ORCA_BASE_URL` 统一覆盖自建部署地址，或用 `ORCA_AUTH_BASE_URL`、`ORCA_API_BASE_URL` 分别覆盖认证与推理地址。可以在界面上清除已保存凭据，或在 [OrcaRouter 已授权应用](https://www.orcarouter.ai/console/authorized-apps) 一键撤销。

PKCE 换回的是长期有效的普通 API Key，不是 refresh token：程序会一直复用它，直到被撤销，不会主动刷新，也不会每次启动重新授权。若推理请求返回 `401`，说明该凭据已被撤销，界面会标记需要重新连接并提示重新授权，而不是重试或伪造刷新。

模型目录来自 `GET {接口地址}/models`，请求由后端携带凭据发出，浏览器只会拿到模型 ID、上下文长度、输入模态等最小元数据。目录不可用时不会退化成自由输入，而是显示最近一次成功获取的列表或一小份已验证的备用列表，并在界面上标注为降级状态。

自定义提供方需要填写 Base URL、API Key 和模型名。API Key 只用于当前页面发起请求，不写入磁盘。
自动采集会在对应的 `*_reference.jsonl` 中保存实际 user prompt、base prompt、system prompt 和 user prefix；拟合后的 `*_bank.json` 与 `unified_bank.json` 只保存统计指纹和校准参数。

## 测试

```powershell
python -m unittest discover -s tests -t .
node --test tests/ui-orcarouter.test.mjs
```

设置 `ORCAROUTER_API_KEY` 后，`tests/test_orcarouter_live.py` 会额外对 `https://api.orcarouter.ai/v1` 发起一次真实请求并核对实时模型目录；未设置时该文件自动跳过。界面证据不随仓库分发：`python tests/capture_gui_evidence.py` 会现场启动应用，用 Playwright 驱动真实界面，把截图、`manifest.json` 与 sha256 写入已被 `.gitignore` 忽略的 `orca-evidence/`。

## 指纹方法

### Codex 任务内监测插件

[ModelTrace Guard](codex-plugin/modeltrace-guard/README.md) 按工具调用次数，在后台从原 Codex 任务的冻结快照分别 fork 进行普通检测和异常复测，测后清理临时分支。正常结果静默保存，主任务无需等待检测。用户可调整工具间隔、复测次数（默认 3）、有效期和十种语言，不设每轮或任务累计探针上限。网页仪表盘提供完整历史分页和实时提醒；首次不一致要求智能体告知并暂停工作，额外复测全部不一致时停止原任务并拦截后续受支持的工作工具。探针由 Codex 自身生成并消耗对应推理额度，评分使用插件附带的 ModelTrace 指纹库，页面刷新不消耗模型额度。

安装后在 Codex CLI 的 `/hooks` 中审阅并信任插件 hooks，再在目标任务中开启监测。安装、配置和结果解释详见[插件使用说明](codex-plugin/modeltrace-guard/README.md)。

### 核心算法

所有模型使用同一个全局特征空间、同一套特征与权重：

```text
0.75 × 去除环境方向后的 Hellinger 模型中心相似度
+ 0.25 × 有序块数字序列特征
```

建库时使用共享环境的平均偏移估计主要干扰方向。归因时先从数字分布和有序特征中投影掉这些方向，再同时与全部模型中心比较。三份回答分别评分后取平均，并使用与查询数量对应的校准温度转换为概率。

具体模型概率由一次全局 softmax 得到，家族概率是该家族下各模型概率之和：

```text
P(具体模型) = softmax(β × 全局模型得分)
P(模型家族) = Σ P(该家族中的具体模型)
```

结果是当前候选库内、均匀先验下的闭集概率。未收录模型仍会被归到最相似的现有候选。

## 项目结构

```text
app.py              Web 接口
orcarouter.py       OrcaRouter 提供方：凭据、OAuth 2.0 + PKCE、模型目录
fingerprint.py      指纹提取与归因
bank_builder.py     指纹库构建与概率校准
challenge_suite.py  自动建库挑战
enrollment.py       API 调用与指纹采集
rebuild_unified_bank.py  重建统一全局库
data/               参考数据与指纹库
static/             页面资源
templates/          页面模板
tests/              OrcaRouter 提供方与 Web 接口测试
```

## 指纹库说明

项目中现有指纹库共包含两个模型家族、13 个模型：

```
gpt-5.4
gpt-5.5
gpt-5.6-luna
gpt-5.6-terra
gpt-5.6-sol
gpt-6-astra
claude-haiku-4-5-20251001
claude-sonnet-4-6
claude-sonnet-5
claude-opus-4-6
claude-opus-4-7
claude-opus-4-8
claude-opus-5
```

GPT 采集自官方订阅 Codex，Claude 采集自 [OAIPro](https://api.oaipro.com/)。

## 声明

测试结果仅供参考，并非判断模型的决定性证据

注意该项目为归因工具，只对指纹库内的模型尽可能去判断属于哪种模型，若待测模型不在指纹库中，得到任何结果都是有可能的

目前已知因系统提示词严重影响模型偏好，在Claude Code中得到的测试结果存在较大偏差，建议不要在Claude Code中测试

## 致谢

感谢 [hanlinwenyuan/hlwy-ai-checker](https://github.com/hanlinwenyuan/hlwy-ai-checker)。该项目较早将语言模型的随机数字生成偏差用于第三方 API 渠道一致性检查，为 ModelTrace 提供了重要参考。



[LINUX DO](https://linux.do/)
