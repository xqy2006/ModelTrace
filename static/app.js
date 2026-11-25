const state = {
  challenges: [],
  bankId: window.DEFAULT_BANK_ID,
  bank: window.BANK_SUMMARIES[window.DEFAULT_BANK_ID],
  unified: window.UNIFIED_SUMMARY,
};

const byId = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function optionalNumber(id) {
  const value = byId(id).value.trim();
  return value === "" ? null : Number(value);
}

function setMessage(element, text, type = "error") {
  element.textContent = text;
  element.className = `message ${type}`;
  element.hidden = !text;
}

function activateWorkspace(name) {
  document.querySelectorAll(".workspace").forEach((item) => item.classList.toggle("active", item.id === `workspace-${name}`));
  document.querySelectorAll("[data-workspace]").forEach((item) => item.classList.toggle("active", item.dataset.workspace === name));
}

function activateMode(group, name) {
  document.querySelectorAll(`[data-${group}-mode]`).forEach((item) => item.classList.toggle("active", item.dataset[`${group}Mode`] === name));
  document.querySelectorAll(`#workspace-${group === "test" ? "test" : "library"} .mode-panel`).forEach((item) => {
    item.classList.toggle("active", item.id === `${group}-${name}` || item.id === `library-${name}`);
  });
}

async function loadChallenges() {
  byId("regenerate").disabled = true;
  byId("result").hidden = true;
  setMessage(byId("test-message"), "");
  const response = await fetch("/api/challenges");
  state.challenges = (await response.json()).challenges;
  renderChallenges();
  byId("regenerate").disabled = false;
}

function renderChallenges() {
  byId("challenge-list").innerHTML = state.challenges.map((challenge, index) => `
    <article class="challenge-item">
      <div class="challenge-header">
        <strong>挑战 ${index + 1}</strong>
        <span>${challenge.expected_count} 个数字</span>
        <button type="button" data-copy="${index}">复制提示词</button>
      </div>
      <div class="challenge-columns">
        <div><label>发送给待测模型</label><pre>${escapeHtml(challenge.prompt)}</pre></div>
        <div><label for="output-${index}">粘贴完整输出</label><textarea id="output-${index}" spellcheck="false" placeholder="保留文字、标点、代码块和完整数字序列"></textarea></div>
      </div>
    </article>
  `).join("");
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      await navigator.clipboard.writeText(state.challenges[Number(button.dataset.copy)].prompt);
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = "复制提示词"; }, 1000);
    });
  });
}

function renderResult(payload) {
  const isAccept = payload.decision === "accept";
  const headlineModel = isAccept ? payload.prediction_name : "未匹配（拒绝归因）";
  const diagnostics = payload.diagnostics.map((item, index) => `
    <span class="diagnostic ${item.accepted ? "accepted" : "rejected"}">挑战 ${index + 1}: ${item.parsed_numbers} 个数字 · ${item.accepted ? "计入" : "忽略"}</span>
  `).join("");

  const entries = [
    ...payload.results.map((item, index) => ({ kind: "known", item, isTopKnown: index === 0 })),
    { kind: "unknown", probability: Number(payload.unknown_probability || 0) },
  ].sort((a, b) => {
    const pa = a.kind === "known" ? a.item.probability : a.probability;
    const pb = b.kind === "known" ? b.item.probability : b.probability;
    return pb - pa;
  });

  const rows = entries.map((entry, index) => {
    if (entry.kind === "unknown") {
      const pU = entry.probability;
      return `
        <tr class="unknown-row">
          <td>${index + 1}</td><td><strong>未知 / 库外模型</strong></td><td>—</td>
          <td><div class="probability-cell unknown-bar"><span><i style="width:${Math.min(pU * 100, 100)}%"></i></span><strong>${percent(pU)}</strong></div></td>
          <td>—</td>
        </tr>
      `;
    }
    const { item, isTopKnown } = entry;
    return `
      <tr class="${isTopKnown && isAccept ? "winner" : ""}">
        <td>${index + 1}</td><td><strong>${escapeHtml(item.display_name)}</strong></td><td>${escapeHtml(item.family_name)}</td>
        <td><div class="probability-cell"><span><i style="width:${Math.min(item.probability * 100, 100)}%"></i></span><strong>${percent(item.probability)}</strong></div></td>
        <td>${percent(item.profile_similarity)}</td>
      </tr>
    `;
  }).join("");

  const apiNote = payload.api_test
    ? `<span>API 获得 ${payload.api_test.received}/${payload.api_test.requested} 份有效回答，实际尝试 ${payload.api_test.attempted}/${payload.api_test.max_attempts}${payload.api_test.errors.length ? `，${payload.api_test.errors.length} 次未采用` : ""}</span>`
    : "";
  byId("result").innerHTML = `
    <div class="result-summary">
      <div><span>最可能模型</span><strong>${escapeHtml(headlineModel)}</strong></div>
      <div><span>统一库概率</span><strong>${percent(payload.probability)}</strong></div>
      <div><span>自动识别家族</span><strong>${escapeHtml(isAccept ? `${payload.family_prediction_name} · ${percent(payload.family_probability)}` : "—")}</strong></div>
      <div><span>有效查询</span><strong>${payload.used_outputs}/3</strong></div>
    </div>
    <div class="diagnostics">${diagnostics}</div>
    <div class="table-wrap"><table><thead><tr><th>排序</th><th>候选模型</th><th>家族</th><th>归因概率</th><th>分布相似度</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${apiNote ? `<div class="result-note">${apiNote}</div>` : ""}
    <div class="result-guidance" role="note" aria-label="结果说明">
      <p>本工具仅对指纹库内的模型进行归因；若待测模型不在指纹库中或置信度不足，将直接拒绝归因。</p>
      <p>gpt-6.1-sol 与 gpt-6-astra 输出分布非常接近，两者之间的分辨结果可能不准确。</p>
      <p>Claude Code 的系统提示词会影响模型偏好，测试结果存在较大偏差，建议不要在 Claude Code 中测试。</p>
    </div>
  `;
  byId("result").hidden = false;
  byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyzeManual() {
  const button = byId("analyze");
  button.disabled = true;
  setMessage(byId("test-message"), "正在计算……", "working");
  const outputs = state.challenges.map((challenge, index) => ({
    text: byId(`output-${index}`).value,
    expected_count: challenge.expected_count,
  }));
  const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ outputs }) });
  const payload = await response.json();
  if (response.ok) {
    setMessage(byId("test-message"), "");
    renderResult(payload);
  } else {
    setMessage(byId("test-message"), payload.error || "无法完成归因。", "error");
    byId("result").hidden = true;
  }
  button.disabled = false;
}

function renderApiProgress(states, status) {
  const valid = states.filter((st) => st === "done").length;
  const attempted = states.filter((st) => ["done", "invalid", "error"].includes(st)).length;
  const target = 3;
  byId("api-test-progress").hidden = false;
  byId("api-progress-status").textContent = status;
  byId("api-progress-count").textContent = `有效 ${valid}/${target} · 已尝试 ${attempted}/${states.length}`;
  byId("api-progress-fill").style.width = `${(valid / target) * 100}%`;
  byId("api-progress-steps").innerHTML = states.map((st, index) => {
    const labels = { pending: "等待", working: "请求中", done: "有效", invalid: "数字不足", error: "接口失败", skipped: "无需调用" };
    return `<span class="progress-step ${st}"><b>${index + 1}</b>挑战 ${index + 1} · ${labels[st]}</span>`;
  }).join("");
}

async function testViaApi(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  byId("result").hidden = true;
  setMessage(byId("test-message"), "");

  try {
    const challengeResponse = await fetch("/api/challenges");
    const firstBatch = (await challengeResponse.json()).challenges;
    const retryResponse = await fetch("/api/challenges");
    const challenges = firstBatch.concat((await retryResponse.json()).challenges);
    const states = challenges.map(() => "pending");
    const outputs = [];
    const errors = [];
    const target = 3;
    const rawConcurrency = Number.parseInt(byId("test-concurrency").value, 10);
    const concurrency = Number.isFinite(rawConcurrency)
      ? Math.min(Math.max(rawConcurrency, 1), target)
      : 3;
    const configuration = {
      base_url: byId("test-api-base").value,
      api_key: byId("test-api-key").value,
      api_model: byId("test-api-model").value,
      temperature: optionalNumber("test-temperature"),
    };
    renderApiProgress(states, `已生成独立挑战，准备以 ${concurrency} 并发调用模型`);

    let nextIndex = 0;
    let inFlight = 0;
    await new Promise((resolve) => {
      const updateWorkingStatus = () => {
        const running = states
          .map((st, idx) => (st === "working" ? idx + 1 : null))
          .filter((idx) => idx !== null);
        if (running.length) {
          const list = running.map((idx) => `#${idx}`).join("、");
          renderApiProgress(
            states,
            `正在并发请求（${running.length}/${concurrency}：挑战 ${list}），已有 ${outputs.length}/${target} 份有效回答……`,
          );
        } else {
          renderApiProgress(states, `当前已有 ${outputs.length}/${target} 份有效回答`);
        }
      };

      const launchMore = () => {
        while (
          nextIndex < challenges.length
          && inFlight < concurrency
          && outputs.length + inFlight < target
        ) {
          const index = nextIndex;
          nextIndex += 1;
          inFlight += 1;
          states[index] = "working";

          fetch("/api/test/probe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...configuration,
              prompt: challenges[index].prompt,
              expected_count: challenges[index].expected_count,
            }),
          })
            .then(async (response) => {
              const payload = await response.json();
              if (!response.ok) throw new Error(payload.error || "接口请求失败");
              if (payload.accepted) {
                outputs.push({ text: payload.text, expected_count: challenges[index].expected_count });
                states[index] = "done";
              } else {
                errors.push(`尝试 ${index + 1}: 有效数字 ${payload.parsed_numbers}/${payload.minimum_numbers}`);
                states[index] = "invalid";
              }
            })
            .catch((error) => {
              errors.push(`尝试 ${index + 1}: ${error.message}`);
              states[index] = "error";
            })
            .finally(() => {
              inFlight -= 1;
              launchMore();
              if (inFlight === 0) {
                resolve();
              } else {
                updateWorkingStatus();
              }
            });
        }
        updateWorkingStatus();
        if (inFlight === 0) resolve();
      };

      launchMore();
    });

    if (outputs.length === target) {
      states.forEach((st, index) => { if (st === "pending") states[index] = "skipped"; });
    }

    if (!outputs.length) {
      renderApiProgress(states, "六次尝试后仍没有可用回答");
      setMessage(byId("test-message"), `没有获得可分析输出。${errors[0] || ""}`, "error");
      return;
    }

    renderApiProgress(states, "模型回答已收齐，正在计算归因概率……");
    const analysisResponse = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outputs }),
    });
    const result = await analysisResponse.json();
    if (analysisResponse.ok) {
      const attempted = states.filter((st) => ["done", "invalid", "error"].includes(st)).length;
      result.api_test = {
        requested: target,
        attempted,
        max_attempts: challenges.length,
        received: outputs.length,
        concurrency,
        errors,
      };
      renderApiProgress(states, `测试完成：${outputs.length}/${target} 份有效回答进入归因`);
      renderResult(result);
    } else {
      setMessage(byId("test-message"), result.error || "API 自动测试失败。", "error");
    }
  } catch (error) {
    setMessage(byId("test-message"), error.message || "API 自动测试失败。", "error");
  } finally {
    button.disabled = false;
  }
}

function updateUnifiedSummary(summary) {
  state.unified = summary;
  byId("topbar-bank-count").textContent = `${summary.model_count} 个候选模型`;
  byId("active-bank-badge").textContent = `${summary.model_count} 个候选模型`;
}

function renderInventory() {
  byId("selected-bank-name").textContent = state.bank.label;
  byId("model-options").innerHTML = state.bank.models.map((model) => `<option value="${escapeHtml(model.id)}"></option>`).join("");
  byId("bank-inventory").innerHTML = state.bank.models.length
    ? state.bank.models.map((model) => `<span class="fingerprint-item">${escapeHtml(model.display_name)}</span>`).join("")
    : `<span class="empty-inventory">暂无指纹</span>`;
}

async function refreshBank() {
  const response = await fetch(`/api/bank?bank_id=${encodeURIComponent(state.bankId)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "无法读取指纹库");
  state.bank = payload;
  renderInventory();
}

async function selectBank(bankId) {
  state.bankId = bankId;
  await refreshBank();
}

async function enrollAutomatically(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  const requested = Number(byId("sample-count").value);
  const started = Date.now();
  const progressTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    setMessage(byId("enrollment-message"), `正在自动识别协议并采集 ${requested} 份回答 · 已等待 ${seconds} 秒`, "working");
  }, 1000);
  setMessage(byId("enrollment-message"), `正在自动识别协议并采集 ${requested} 份回答`, "working");
  let response;
  try {
    response = await fetch("/api/enroll/auto", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: byId("api-base").value,
        api_key: byId("api-key").value,
        api_model: byId("api-model").value,
        bank_id: state.bankId,
        model_label: byId("auto-model").value,
        sample_count: requested,
        temperature: optionalNumber("temperature"),
      }),
    });
  } catch (error) {
    window.clearInterval(progressTimer);
    setMessage(byId("enrollment-message"), error.message, "error");
    button.disabled = false;
    return;
  }
  window.clearInterval(progressTimer);
  const payload = await response.json();
  if (response.ok) {
    state.bank = payload.bank;
    updateUnifiedSummary(payload.unified);
    renderInventory();
    setMessage(byId("enrollment-message"), `采集完成：收到 ${payload.received}/${payload.requested} 份，${payload.accepted} 份进入指纹库，${payload.rejected} 份无效，${payload.errors.length} 次接口错误。`, "success");
  } else {
    setMessage(byId("enrollment-message"), payload.error || "自动采集失败。", "error");
  }
  button.disabled = false;
}

function renderBankOptions(summaries, selected) {
  byId("bank-select").innerHTML = Object.entries(summaries)
    .map(([bankId, bank]) => `<option value="${escapeHtml(bankId)}"${bankId === selected ? " selected" : ""}>${escapeHtml(bank.label)}</option>`)
    .join("");
}

async function createBank(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  const response = await fetch("/api/banks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: byId("new-bank-name").value }),
  });
  const payload = await response.json();
  if (response.ok) {
    window.BANK_SUMMARIES = payload.banks;
    state.bankId = payload.bank.id;
    state.bank = payload.bank;
    updateUnifiedSummary(payload.unified);
    renderBankOptions(payload.banks, state.bankId);
    renderInventory();
    byId("new-bank-name").value = "";
    byId("create-bank-form").hidden = true;
    setMessage(byId("enrollment-message"), `已创建 ${payload.bank.label}`, "success");
  } else {
    setMessage(byId("enrollment-message"), payload.error || "创建失败。", "error");
  }
  button.disabled = false;
}

document.querySelectorAll("[data-workspace]").forEach((button) => button.addEventListener("click", () => activateWorkspace(button.dataset.workspace)));
document.querySelectorAll("[data-test-mode]").forEach((button) => button.addEventListener("click", () => activateMode("test", button.dataset.testMode)));
byId("bank-select").addEventListener("change", (event) => selectBank(event.target.value));
byId("regenerate").addEventListener("click", loadChallenges);
byId("analyze").addEventListener("click", analyzeManual);
byId("api-test-form").addEventListener("submit", testViaApi);
byId("auto-enrollment").addEventListener("submit", enrollAutomatically);
byId("show-create-bank").addEventListener("click", () => { byId("create-bank-form").hidden = !byId("create-bank-form").hidden; });
byId("create-bank-form").addEventListener("submit", createBank);

renderInventory();
loadChallenges();
