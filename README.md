# ModelTrace

ModelTrace 是一个本地运行的主动模型归因工具。它通过三条独立的长整数生成挑战提取输出指纹，在统一候选库中自动判断模型家族和具体版本。

## 运行

```powershell
python -m pip install -r requirements.txt
python start.py
```

页面地址为 `http://127.0.0.1:7860/`。

## 终端 CLI

CLI 的归因计算与 API 调用仅使用 Python 标准库；Codex 模式需要本机安装 `codex`。

```bash
# Codex 模式，默认串行，三份有效回答，单次请求最长 180 秒
python3 modeltrace_cli.py -m claude-opus-5-5 --timeout 180

# 纯 Python API 模式；key 推荐通过环境变量传入
export MODELTRACE_BASE_URL='https://your-api.example/v1'
export MODELTRACE_API_KEY='your-key'
python3 modeltrace_cli.py --backend api --api-format responses -m claude-opus-5-5 --json

# URL / key 也可使用 --base-url / --api-key（别名 --url / --key）
# Codex 和 API 均可追加 --parallel 3 开启三路并行，默认同时请求数为 1
# --api-format 支持 chat、responses、anthropic，默认 chat
python3 modeltrace_cli.py --backend api --api-format chat -m your-model --save-outputs outputs.jsonl

# 指定远端库时，每次从此 URL 下载，不采用旧的本地缓存
python3 modeltrace_cli.py -m your-model --bank-url https://raw.githubusercontent.com/xqy2006/ModelTrace/main/data/unified_bank.json
```

默认每次运行均下载官方远端指纹库，不优先采用 checkout 或旧缓存。`--bank-file` 可指定离线库，与 `--bank-url` 互斥。远端库按 URL 分别缓存，显式 URL 下载失败会报错，不会静默回退到旧库。

Codex 在临时目录中运行，以 `--json` 审计全部事件；发现工具调用、未知项目、损坏事件或不完整结束时拒绝该回答，超时后终止进程树。审计能发现已发生的工具调用，不等于从根本上禁用 Codex 工具。API 模式不提供或执行工具，拒绝返回工具调用或截断回答；禁止 HTTP 重定向以避免转发凭据。`--json` 的 stdout 仅包含结果 JSON，进度和性能表写入 stderr。

`-n` 是最大尝试次数，`--target` 是目标有效回答数，`-c` 控制并发；如果 `-n` 小于目标则提升到目标次数。`--save-outputs` 保存逐轮原始回答与性能，不保存 API key；未达到目标时报告实际有效数量。上游缺少输出 token usage 时明确显示 TPS 不可用；部分回答缺少 usage 时，仅使用有有效 usage 的回答及其耗时计算单请求 TPS，吞吐为已知 token 吞吐，不估算 token。闭集归因不能证明服务端真实模型身份，也无法观察服务端未披露的工具使用。

CLI 回归测试：`python3 -m unittest -v test_modeltrace_cli`（评分一致性测试需要 requirements 中的 NumPy）。

## GitHub Pages

`static/index.html` 是不依赖后端的手动测试版本，归因计算和指纹库读取都在浏览器本地完成。仓库附带的 GitHub Actions 会将 `static/` 部署到 GitHub Pages。

## 使用

- **手动测试**：复制三条挑战，分别发送给同一个待测模型，再粘贴每次完整输出。
- **API 自动测试**：填写 Base URL、API Key 和模型名。程序会自动尝试 OpenAI Chat Completions 与 Anthropic Messages 格式，以三份有效回答为目标完成归因。
- **指纹库管理**：可以新建指纹库，或通过 API 为现有指纹库添加模型指纹。

API Key 只用于当前页面发起请求，不写入磁盘。
自动采集会在对应的 `*_reference.jsonl` 中保存实际 user prompt、base prompt、system prompt 和 user prefix；拟合后的 `*_bank.json` 与 `unified_bank.json` 只保存统计指纹和校准参数。

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
fingerprint.py      指纹提取与归因
bank_builder.py     指纹库构建与概率校准
challenge_suite.py  自动建库挑战
enrollment.py       API 调用与指纹采集
rebuild_unified_bank.py  重建统一全局库
data/               参考数据与指纹库
static/             页面资源
templates/          页面模板
```

## 指纹库说明

项目中现有指纹库共包含两个模型家族、18 个模型：

```
gpt-5.4
gpt-5.5
gpt-5.6-luna
gpt-5.6-terra
gpt-5.6-sol
gpt-6-astra
gpt-6-sol
gpt-6-luna
claude-haiku-4-5-20251001
claude-haiku-5-5
claude-sonnet-4-6
claude-sonnet-5
claude-sonnet-5-5
claude-opus-4-6
claude-opus-4-7
claude-opus-4-8
claude-opus-5
claude-opus-5-5
```

GPT 采集自官方订阅 Codex，Claude 采集自 [OAIPro](https://api.oaipro.com/)。

## 声明

测试结果仅供参考，并非判断模型的决定性证据

注意该项目为归因工具，只对指纹库内的模型尽可能去判断属于哪种模型，若待测模型不在指纹库中，得到任何结果都是有可能的

目前已知因系统提示词严重影响模型偏好，在Claude Code中得到的测试结果存在较大偏差，建议不要在Claude Code中测试

## 致谢

感谢 [hanlinwenyuan/hlwy-ai-checker](https://github.com/hanlinwenyuan/hlwy-ai-checker)。该项目较早将语言模型的随机数字生成偏差用于第三方 API 渠道一致性检查，为 ModelTrace 提供了重要参考。



[LINUX DO](https://linux.do/)
