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
  return `${(value * 100).toFixed(1)}%`;
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

function buildResultHtml(payload) {
  const diagnostics = payload.diagnostics.map((item, index) => `
    <span class="diagnostic ${item.accepted ? "accepted" : "rejected"}">挑战 ${index + 1}: ${item.parsed_numbers} 个数字 · ${item.accepted ? "计入" : "忽略"}</span>
  `).join("");
  const rows = payload.results.map((item, index) => `
    <tr class="${index === 0 ? "winner" : ""}">
      <td>${index + 1}</td><td><strong>${escapeHtml(item.display_name)}</strong></td><td>${escapeHtml(item.family_name)}</td>
      <td><div class="probability-cell"><span><i style="width:${item.probability * 100}%"></i></span><strong>${percent(item.probability)}</strong></div></td>
      <td>${percent(item.profile_similarity)}</td>
    </tr>
  `).join("");
  const apiNote = payload.api_test
    ? `<span>API 获得 ${payload.api_test.received}/${payload.api_test.requested} 份有效回答，实际尝试 ${payload.api_test.attempted}/${payload.api_test.max_attempts}${payload.api_test.errors.length ? `，${payload.api_test.errors.length} 次未采用` : ""}</span>`
    : "";
  return `
    <div class="result-summary">
      <div><span>最可能模型</span><strong>${escapeHtml(payload.prediction_name)}</strong></div>
      <div><span>统一库概率</span><strong>${percent(payload.probability)}</strong></div>
      <div><span>自动识别家族</span><strong>${escapeHtml(payload.family_prediction_name)} · ${percent(payload.family_probability)}</strong></div>
      <div><span>有效查询</span><strong>${payload.used_outputs}/3</strong></div>
    </div>
    <div class="diagnostics">${diagnostics}</div>
    <div class="table-wrap"><table><thead><tr><th>排序</th><th>候选模型</th><th>家族</th><th>归因概率</th><th>分布相似度</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${apiNote ? `<div class="result-note">${apiNote}</div>` : ""}
    <div class="result-guidance" role="note" aria-label="结果说明">
      <p>本工具仅对指纹库内的模型进行归因；若待测模型不在指纹库中，得到任何结果都有可能。</p>
      <p>Claude Code 的系统提示词会影响模型偏好，测试结果存在较大偏差，建议不要在 Claude Code 中测试。</p>
    </div>
  `;
}

function renderResult(payload) {
  byId("result").innerHTML = buildResultHtml(payload);
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

const CACHE_KEY = "modeltrace.apiCells.v1";
const apiCells = [];
let cellSeq = 0;

function addCell(data = {}) {
  cellSeq += 1;
  const cell = { id: cellSeq, result: data.result || null, running: false, el: null };
  const el = document.createElement("article");
  el.className = "api-cell";
  el.innerHTML = `
    <div class="api-cell-head">
      <input class="cell-name" placeholder="名称（必须唯一）" aria-label="测试名称">
      <label class="cell-enable"><input type="checkbox" class="cell-active" checked><span>参与测试</span></label>
      <button type="button" class="button secondary cell-detail" disabled>测试详情</button>
      <button type="button" class="button secondary cell-remove" aria-label="删除">×</button>
    </div>
    <div class="form-grid two">
      <label><span>Base URL</span><input class="cell-base" placeholder="https://example.com/v1"></label>
      <label><span>接口模型名</span><input class="cell-model" placeholder="模型名"></label>
      <label><span>API Key</span><span class="key-wrap"><input class="cell-key" type="password" autocomplete="off"><button type="button" class="key-eye" aria-label="显示/隐藏 API Key" title="显示/隐藏">👁</button></span></label>
      <label><span>温度（可选）</span><input class="cell-temp" type="number" min="0" max="2" step="0.1" placeholder="接口默认"></label>
    </div>
    <div class="cell-progress progress-panel" hidden>
      <div class="progress-heading"><strong class="cell-status"></strong><span class="cell-count"></span></div>
      <div class="progress-track"><i class="cell-fill"></i></div>
      <div class="progress-steps cell-steps"></div>
    </div>
    <div class="cell-summary" hidden></div>`;
  cell.el = el;
  const q = (selector) => el.querySelector(selector);
  q(".cell-name").value = data.name || "";
  q(".cell-base").value = data.base || "";
  q(".cell-model").value = data.model || "";
  q(".cell-key").value = data.key || "";
  q(".cell-temp").value = data.temp ?? "";
  q(".cell-active").checked = data.active !== false;
  q(".key-eye").addEventListener("click", (e) => {
    e.preventDefault();
    const input = q(".cell-key");
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    e.currentTarget.classList.toggle("on", show);
  });
  q(".cell-name").addEventListener("input", markDuplicateNames);
  q(".cell-remove").addEventListener("click", () => {
    if (cell.running) return;
    apiCells.splice(apiCells.indexOf(cell), 1);
    el.remove();
    markDuplicateNames();
  });
  q(".cell-detail").addEventListener("click", () => openDetail(cell));
  byId("api-cells").appendChild(el);
  apiCells.push(cell);
  if (!q(".cell-name").value) q(".cell-name").value = uniqueDefaultName();
  if (cell.result) showCellSummary(cell);
  markDuplicateNames();
  return cell;
}

function uniqueDefaultName() {
  const used = new Set(apiCells.map((c) => c.el.querySelector(".cell-name").value.trim()));
  let n = 1;
  while (used.has(`测试 ${n}`)) n += 1;
  return `测试 ${n}`;
}

function cellName(cell) { return cell.el.querySelector(".cell-name").value.trim(); }

function markDuplicateNames() {
  const counts = {};
  apiCells.forEach((c) => { const n = cellName(c); counts[n] = (counts[n] || 0) + 1; });
  apiCells.forEach((c) => {
    const n = cellName(c);
    c.el.querySelector(".cell-name").classList.toggle("invalid", !n || counts[n] > 1);
  });
}

function showCellSummary(cell) {
  const summary = cell.el.querySelector(".cell-summary");
  const result = cell.result;
  summary.innerHTML = `<span>最可能模型</span><strong>${escapeHtml(result.prediction_name)}</strong><em>${percent(result.probability)}</em>`;
  summary.hidden = false;
  cell.el.querySelector(".cell-detail").disabled = false;
}

function openDetail(cell) {
  byId("api-detail-title").textContent = `${cellName(cell)} · 测试详情`;
  byId("api-detail-body").innerHTML = buildResultHtml(cell.result);
  byId("api-detail-modal").hidden = false;
}

function renderCellProgress(cell, states, status) {
  const q = (selector) => cell.el.querySelector(selector);
  const valid = states.filter((s) => s === "done").length;
  const attempted = states.filter((s) => ["done", "invalid", "error"].includes(s)).length;
  q(".cell-progress").hidden = false;
  q(".cell-status").textContent = status;
  q(".cell-count").textContent = `有效 ${valid}/3 · 已尝试 ${attempted}/${states.length}`;
  q(".cell-fill").style.width = `${(valid / 3) * 100}%`;
  const labels = { pending: "等待", working: "请求中", done: "有效", invalid: "数字不足", error: "接口失败", skipped: "无需调用" };
  q(".cell-steps").innerHTML = states.map((s, i) => `<span class="progress-step ${s}"><b>${i + 1}</b>挑战 ${i + 1} · ${labels[s]}</span>`).join("");
}

async function runCell(cell) {
  const q = (selector) => cell.el.querySelector(selector);
  const target = 3;
  cell.running = true;
  cell.result = null;
  q(".cell-summary").hidden = true;
  q(".cell-detail").disabled = true;
  try {
    const tempText = q(".cell-temp").value.trim();
    const configuration = {
      base_url: q(".cell-base").value,
      api_key: q(".cell-key").value,
      api_model: q(".cell-model").value,
      temperature: tempText === "" ? null : Number(tempText),
    };
    const first = (await (await fetch("/api/challenges")).json()).challenges;
    const second = (await (await fetch("/api/challenges")).json()).challenges;
    const challenges = first.concat(second);
    const states = challenges.map(() => "pending");
    const outputs = [];
    const errors = [];
    renderCellProgress(cell, states, "已生成独立挑战，准备调用模型");
    for (let i = 0; i < challenges.length && outputs.length < target; i += 1) {
      states[i] = "working";
      renderCellProgress(cell, states, `正在进行第 ${i + 1} 次尝试……`);
      try {
        const response = await fetch("/api/test/probe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...configuration, prompt: challenges[i].prompt, expected_count: challenges[i].expected_count }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "接口请求失败");
        if (payload.accepted) {
          outputs.push({ text: payload.text, expected_count: challenges[i].expected_count });
          states[i] = "done";
        } else {
          errors.push(`尝试 ${i + 1}: 有效数字 ${payload.parsed_numbers}/${payload.minimum_numbers}`);
          states[i] = "invalid";
        }
      } catch (error) {
        errors.push(`尝试 ${i + 1}: ${error.message}`);
        states[i] = "error";
      }
      renderCellProgress(cell, states, `当前已有 ${outputs.length}/${target} 份有效回答`);
    }
    if (outputs.length === target) states.forEach((s, i) => { if (s === "pending") states[i] = "skipped"; });
    if (!outputs.length) {
      renderCellProgress(cell, states, `没有获得可分析输出。${errors[0] || ""}`);
      return;
    }
    renderCellProgress(cell, states, "正在计算归因概率……");
    const response = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ outputs }),
    });
    const result = await response.json();
    if (!response.ok) {
      renderCellProgress(cell, states, result.error || "归因失败");
      return;
    }
    const attempted = states.filter((s) => ["done", "invalid", "error"].includes(s)).length;
    result.api_test = { requested: target, attempted, max_attempts: challenges.length, received: outputs.length, errors };
    cell.result = result;
    renderCellProgress(cell, states, `测试完成：${outputs.length}/${target} 份有效回答进入归因`);
    showCellSummary(cell);
  } catch (error) {
    renderCellProgress(cell, [], `测试失败：${error.message}`);
  } finally {
    cell.running = false;
  }
}

async function testAllCells() {
  setMessage(byId("test-message"), "");
  markDuplicateNames();
  const names = apiCells.map(cellName);
  const active = apiCells.filter((c) => c.el.querySelector(".cell-active").checked);
  if (!active.length) return setMessage(byId("test-message"), "没有勾选「参与测试」的单元。", "error");
  if (names.some((n) => !n) || new Set(names).size !== names.length) {
    return setMessage(byId("test-message"), "每个测试单元必须有名称，且名称不能重复。", "error");
  }
  const incomplete = active.find((c) => ["base", "model", "key"].some((k) => !c.el.querySelector(`.cell-${k}`).value.trim()));
  if (incomplete) return setMessage(byId("test-message"), `「${cellName(incomplete)}」的 Base URL、模型名和 API Key 不能为空。`, "error");
  const button = byId("api-start");
  button.disabled = true;
  await Promise.all(active.map(runCell));
  button.disabled = false;
}

function saveCells() {
  const data = apiCells.map((c) => {
    const q = (selector) => c.el.querySelector(selector).value;
    return { name: q(".cell-name"), base: q(".cell-base"), model: q(".cell-model"), key: q(".cell-key"), temp: q(".cell-temp"), active: c.el.querySelector(".cell-active").checked, result: c.result };
  });
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(data));
    setMessage(byId("test-message"), `已缓存 ${data.length} 个测试单元到浏览器本地（含 API Key）。`, "success");
  } catch (error) {
    setMessage(byId("test-message"), `缓存失败：${error.message}`, "error");
  }
}

function clearCache() {
  localStorage.removeItem(CACHE_KEY);
  setMessage(byId("test-message"), "已清理前端缓存。", "success");
}

function loadCells() {
  let data = [];
  try { data = JSON.parse(localStorage.getItem(CACHE_KEY) || "[]"); } catch { data = []; }
  (data.length ? data : [{}]).forEach((item) => addCell(item));
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
byId("api-add-cell").addEventListener("click", () => addCell());
byId("api-start").addEventListener("click", testAllCells);
byId("api-cache-save").addEventListener("click", saveCells);
byId("api-cache-clear").addEventListener("click", clearCache);
byId("api-detail-close").addEventListener("click", () => { byId("api-detail-modal").hidden = true; });
byId("api-detail-modal").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.hidden = true; });
loadCells();
byId("auto-enrollment").addEventListener("submit", enrollAutomatically);
byId("show-create-bank").addEventListener("click", () => { byId("create-bank-form").hidden = !byId("create-bank-form").hidden; });
byId("create-bank-form").addEventListener("submit", createBank);

renderInventory();
loadChallenges();
