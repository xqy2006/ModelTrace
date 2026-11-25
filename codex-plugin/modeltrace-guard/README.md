# ModelTrace Guard

作者：xqy2006。

按工具调用次数，在后台使用原 Codex 任务的冻结历史快照创建临时 fork 进行指纹检查。正常结果静默保存，主任务无需等待探针，也不会收到探针数字数组；异常时提醒并进入复测。评分使用插件附带的 ModelTrace 指纹库。探针由 Codex 自身生成，使用其已配置的账户、模型与厂商，消耗对应推理额度，不需要另填 API Key。

## 从 GitHub 安装

在终端运行：

```powershell
codex plugin marketplace add xqy2006/ModelTrace
codex plugin add modeltrace-guard@modeltrace
```

也可先下载源码，在 ModelTrace 仓库根运行：

```powershell
codex plugin marketplace add .
codex plugin add modeltrace-guard@modeltrace
```

市场入口为仓库根的 `.agents/plugins/marketplace.json`，插件位于 `codex-plugin/modeltrace-guard/`，清单为该目录下的 `.codex-plugin/plugin.json`。

需要 Node.js 18+，以及支持原生快照 fork、临时任务、任务删除和异步 hooks 的 Codex App Server。可用 `node scripts/guard.mjs doctor --fork true` 检查兼容性。若系统有多个运行时，将 `MODELTRACE_CODEX_PATH` 设置为 Codex 桌面应用配套可执行文件的绝对路径。

### 加载与信任

1. 安装后打开一个新 Codex 任务加载插件。
2. 在终端运行兼容的 `codex`，进入其交互输入框后输入 `/hooks`。这不是 PowerShell 命令。
3. 找到 ModelTrace Guard，审阅并信任所需 hooks，包括同步 `PreToolUse` 工具拦截，以及异步 `SessionStart`、`UserPromptSubmit`、`PostToolUse`。插件更新后，如果 hook 定义发生变化，需要重新审阅。
4. 在目标任务中请求开启。探针执行器会检查拦截 hook 是否已启用并受信任。

## 开启、调整、停止

对目标任务说：

> 使用 $modeltrace-guard 为本任务开启监测，每隔 16–32 次工作工具调用检测，异常后复测 3 次，使用中文和英语。

> 把复测次数改成 5 次。

> 停止当前任务的持续检测。

命令示例（在插件目录运行，`CODEX_THREAD_ID` 应指向目标任务）：

```powershell
node scripts/guard.mjs doctor
node scripts/guard.mjs doctor --fork true
node scripts/guard.mjs start --tool-min 16 --tool-max 32 --retry-count 3 --languages zh,en
node scripts/guard.mjs configure --retry-count 5
node scripts/guard.mjs status
node scripts/guard.mjs dashboard
node scripts/guard.mjs stop
```

`doctor --fork true` 检查原生分支、后台 hooks 和清理能力，不发起推理。`start` 将首次检测排入队列后立即返回，由后台 hook 自动执行，不需要智能体再运行 `probe` 或轮询状态。外部终端手动开启时，目标任务的下一个已加载且受信任的 hook 会接手检测；不会主动唤醒空闲任务。

可选 `--expected <模型标签>` 指定比较对象，`--expected auto` 跟随 Codex 元数据。缺少预期标签与预期标签未收录分开显示；历史样本保留采样时的标签。

| 设置 | 默认与含义 |
| --- | --- |
| `--tool-min N --tool-max M` | 16–32 次观察到的工作工具完成；N=M 固定间隔 |
| `--retry-count N` | 首次不一致后额外复测，默认 3，允许 1–100 |
| `--pending-seconds N` | 从发出探针到完成评分的时限，默认 180 秒，不是检测间隔 |
| `--languages zh,en,...` | 默认中英；支持 zh/en/ja/ko/fr/de/es/pt/ru/ar，单项固定语言 |

没有每轮或任务累计探针数量上限。普通检查按工具间隔进行，空闲不启动新采样，已开始的后台检查可在本轮回答结束后完成。同一任务最多一个在途探针；多个工具同时完成不会重复发起。修改设置不清空历史；当前复测组保持创建时的配置。

升级保留已有任务保存的间隔。要调整已开启任务，可在仪表盘保存 16–32，或对该任务说“把探测间隔改成 16–32 次工具调用”。

仪表盘按任务区分监测状态，不按工作目录合并。首次后台快照会从 Codex 任务元数据读取真实标题，用户自定义名称优先；标题暂不可用时显示“未命名任务（工作目录）”及时间。列表同时保留已停止任务的历史记录，并标明开关状态。打开页面或选择任务不会开启探测。

## 后台运行

`SessionStart`、`UserPromptSubmit`、`PostToolUse` 使用 Codex 原生 `async: true` 命令 hook。后台进程负责 fork、生成、评分和清理；正常检测不向主会话注入指令或状态提醒，也不阻止工具调用和最终回答。`status` 与网页仪表盘可按需查看结果及后台运行状态。

异常结果完成评分后立即保存。后台 hook 在 Codex 的下一个安全边界将提醒交给智能体；同步 `PreToolUse` 也会检查尚未告知的异常。当前请求或工具已经运行时不会被强行终止；任务已经空闲时不会被自动唤醒，提醒会保留到下一轮。网页连接正常时可独立收到实时提醒。参见 [Codex 后台 hook 说明](https://learn.chatgpt.com/docs/hooks#run-hooks-in-the-background)。

停止监测、用户中断、上下文压缩或任务结束会使对应后台探针失效，其迟到结果不会计入检测。监测需要任务 hooks，不依赖网页服务常驻。

采样分叉会明确区分自身与原任务：它只回答本次数字请求，不接手原任务的 `/goal` 或插件管理工作，也不改变原任务的目标。输出可以是完整 JSON 整数数组，或仅包裹该数组的单个 JSON／无语言标记代码块；不会从混杂正文中抽取数字，不拼接多个数组，也不补齐或修改样本。

检测失败时，`status` 的 `forkHealth.diagnostic` 和历史中的 `fork_probe_failed` 事件会记录失败阶段、类型及可用的字符数／数组长度；仪表盘也会显示最近的失败原因。诊断不保存失败回复原文、数字内容或任务上下文。无效样本记为检测缺口，不参与模型不一致计数；后续排队不代表检测已成功。

## 同一快照、独立 fork、测后清理

1. 由 Codex 原生 `thread/fork` 创建一个持久化临时基准；它不运行模型、不接受后续工作。
2. 固定基准的轮次边界；首次探针与 N 次复测分别对它调用 `thread/fork`，且每次设置 `ephemeral: true` 和相同 `lastTurnId`，并固定原任务的工作目录。执行器只允许显式请求的一轮探针，发现额外自动续轮时关闭临时进程；原任务的目标和历史消息不变。
3. 子 fork 只输出一个数组，受信任 hook 拦截其工具调用。后台执行器读取结果、关闭临时 Codex 进程，再离线评分。正常结果只保存在本地，异常时原任务收到候选、时间与通知要求。
4. 普通检查结束或整组复测结束后，独立清理进程核对所属任务与轮次，并检查所有来源类型、包括已归档的持久化后代，通过 `thread/delete` 删除基准。临时基准可能短暂出现在任务列表中，请勿把它用于工作。被使用或产生其他持久化后代的基准会保留并报告清理阻塞，避免连带删除用户内容。

每次复测都读取首次检测使用的快照，不继承其他探针的数字，也不包含告警后主任务增加的内容。原任务的已有消息保持不变。

快照按 Codex 原生 fork 机制继承已保存的上下文及其压缩结果；插件不会将历史消息全文重新拼入探针提示词。快照校验仅读取轮次元数据，`sha256Scope: turn_boundaries_v1` 表示校验值对应轮次边界，不是消息全文。尚未落盘的内容不在快照中；运行中的轮次可能带有 Codex 的中断标记。

使用 ChatGPT 登录及默认 Codex 服务时，探针沿用原任务的缓存会话，临时任务和轮次 ID 保持独立。插件只在该探针的本机连接上调整缓存会话请求头；上下文、压缩后的请求正文、工具定义、模型和思考程度均保持不变，连接在检测完成或取消后关闭。样本的 `fork.cacheScope: source_session` 表示已沿用原任务的缓存会话，`fork.usage.cachedInputTokens` 是 Codex 返回的实际缓存计数，`null` 表示未提供计数。

API Key 登录、自定义服务地址、代理和自定义 CA 配置继续使用 Codex 原生连接，对应 `fork.cacheScope: native_fork`。缓存过期、上下文压缩或前缀配置变化后，首次请求仍可能需要重新建立缓存。

模型、厂商、推理强度与工作目录直接读取 Codex 的任务元数据，不扫描本地历史文件。接口缺少所需配置时报告检测未能启动，不回退到读取历史或使用全局默认模型。

## 提醒与停止

首次候选不一致立即写入提醒，要求智能体先告诉用户“预期 X，指纹第一候选 Y”，再确认已告知。之后后台连续完成用户设定的 N 次复测，不等待普通间隔，也不递归再开启一组。复测期间原任务暂停工作；智能体可执行返回的 `wait --confirmation <ID>` 等待进度，每次最多 30 秒。该命令只读取进度，不执行探针。复测不一致时仍需先告知并确认，匹配结果则自动推进下一次复测。所有 N 次均不一致才锁定任务；其中任一次相符则本组不触发停止，但异常记录保留。缺失、无效、超时或中断不计为不一致。

任务锁定后 `PreToolUse` 拒绝后续受支持的工作工具，只允许直接运行本插件的状态、告知确认、停止和用户授权恢复等管理命令。智能体仍须主动说明全部候选并停止工作。它不取消已经运行的命令；托管工具及不经此 hook 的专用路径也不属于完整拦截范围。只有用户明确要求恢复后，智能体才能执行 `resume --halt <当前停止ID>`。停止监测、确认提醒或换模型均不会自动解除锁定。

网页通过带鉴权的实时事件流接收提醒，不等待智能体确认；点击“开启桌面提醒”并授权后可发浏览器系统通知。需要保持网页连接，浏览器/系统可能限制通知；未授权时保留网页提醒与智能体告知渠道。完整本地网址携带访问凭证，请勿公开。

## 重启、存储与清理状态

监测设置保存在稳定数据目录。重新打开原任务时，启动/恢复 hook 会触发自检，并在观察到工作工具调用后验证运行状态。网页服务和监测相互独立：使用 `dashboard` 命令打开网页，关闭网页不会停止任务监测。

数据默认在 `~/.codex/modeltrace-guard`，可用 `MODELTRACE_GUARD_DATA` 覆盖。状态文件保留近期记录，完整样本、事件及已确认提醒保存在 `_history` 中，可从网页“完整历史”翻页查看。数据格式升级时保留已有历史。状态锁仅在确认所属进程已退出后自动回收；无法确认的锁会报告错误，需检查相关进程状态。

`_fork_cleanup/<临时基准ID>.json` 保存清理状态：`active` 等待本组结束、`pending` 待删除、`deleted` 已删除、`blocked` 需检查。启动/恢复任务会扫描遗留清理记录；也可运行 `node scripts/fork-cleanup.mjs <数据目录>`。默认跳过已阻塞项；检查原因后可追加 `--retry-blocked` 重新核验。清理仅针对所属任务、轮次及后代检查均通过的临时基准；失败时保留错误与目标信息，供排查使用。

## 指纹、校准与验证

指纹库来自 ModelTrace 仓库的 `data/unified_bank.json`，评分器来自 `static/fingerprint-core.js`，主题来自 `static/styles.css`。它们随插件打包，来源与校验和见 `assets/provenance.json`，运行时不自动下载或更新。

检测结果是参考库内的指纹相似性比较，反映 fork 请求的输出特征，不是主任务此前请求的后端身份认证。长上下文、语言和输出通道可能改变数字分布；共享上下文的复测结果也可能相关。同上下文与多语言条件下的误报率未经过独立校准。此外，`gpt-6.1-sol` 与 `gpt-6-astra` 输出分布非常接近，两者之间的分辨结果可能不准确。

提供离线评估入口：`node scripts/evaluate-calibration.mjs <独立验证集.jsonl>`。验证集需标记 `split: holdout`、`mode: ephemeral_fork`、受控来源的真实模型、预期模型、候选、语言、快照 ID、样本 ID、bank hash 和复测序号；脚本按快照整组计算观察到的误报/停工率，不假定“三次误报概率等于单次概率的三次方”。不提供数据时会明确返回尚无独立验证记录，不自动采集、不拟合阈值或修改库。

## 源码构建与测试

在 ModelTrace 仓库根目录运行：

```powershell
node codex-plugin/build-plugin.mjs
node --test codex-plugin/modeltrace-guard/tests/*.test.mjs
node codex-plugin/verify-parity.mjs
```

自动化测试覆盖协议、后台并发与静默输出、取消与迟到结果、同快照复测、清理边界、状态迁移、分页、告警和同步工具拦截；`verify-parity.mjs` 使用当前打包模型的已有参考样本检查评分实现的一致性。模型归因准确率需使用独立验证集评估。
