# ModelTrace

ModelTrace 是一个本地运行的主动模型归因工具。它通过三条独立的长整数生成挑战提取多视角输出指纹，在统一候选库中自动判断模型家族和具体版本，并对未收录的库外模型自动拒绝归因。

## 运行

### Python 后端版

```powershell
python -m pip install -r requirements.txt
python start.py
```

页面地址为 `http://127.0.0.1:7860/`。

### 纯静态浏览器本地版（GitHub Pages）

`static/index.html` 是不依赖后端的手动测试版本，特征提取、归因计算和指纹库读取都在浏览器本地完成：

```powershell
python -m http.server 8080 -d static
```

## 使用

- **手动测试**：复制三条挑战，分别发送给同一个待测模型，再粘贴每次完整输出。
- **API 自动测试**：填写 Base URL、API Key、模型名及并发数（默认 `3`）。程序会自动尝试 OpenAI Chat Completions 与 Anthropic Messages 格式，以三份有效回答为目标并发完成归因。
- **指纹库管理**：可以新建指纹库，或通过 API 为现有指纹库添加模型指纹。

API Key 只用于当前页面发起请求，不写入磁盘。

## 指纹库覆盖范围（2 大家族 · 22 个模型）

```text
gpt-5.4
gpt-5.5
gpt-5.6-sol
gpt-5.6-terra
gpt-5.6-luna
gpt-6-astra
gpt-6-sol
gpt-6-luna
gpt-6.1-sol
gpt-6.1-terra
gpt-6.2-astra
gpt-6.2-sol
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

## 项目结构

```text
app.py                  Web 接口
v6_portable.py          双精度多视角特征提取与开集归因推理引擎
v6_builder.py           指纹库训练、校准与紧凑量化构建器
fingerprint.py          指纹归因入口与挑战生成
bank_builder.py         单库构建入口
challenge_suite.py      自动建库挑战套件
enrollment.py           API 调用与指纹采集
rebuild_unified_bank.py 重建统一全局库
data/                   参考数据与指纹库
static/                 纯静态页面与浏览器本地推理核心 (fingerprint-core.js)
templates/              页面模板
```

## 声明

测试结果仅供参考，并非判断模型的决定性证据。

若待测模型不在指纹库中或置信度不足，系统将拒绝归因。`gpt-6.1-sol` 与 `gpt-6-astra` 的输出分布非常接近，两者之间的分辨结果可能不准确（在高推理强度下尤为明显），请谨慎参考。Claude Code 的系统提示词会影响模型偏好，测试结果存在较大偏差，建议不要在 Claude Code 中测试。

## 致谢

感谢 [hanlinwenyuan/hlwy-ai-checker](https://github.com/hanlinwenyuan/hlwy-ai-checker)。该项目较早将语言模型的随机数字生成偏差用于第三方 API 渠道一致性检查，为 ModelTrace 提供了重要参考。

[LINUX DO](https://linux.do/)
