const state = {
  challenges: [],
  bankId: window.DEFAULT_BANK_ID,
  bank: window.BANK_SUMMARIES[window.DEFAULT_BANK_ID],
  unified: window.UNIFIED_SUMMARY,
  providers: [],
  providerForms: [],
};

const byId = (id) => document.getElementById(id);
const OrcaProvider = window.OrcaRouterProvider;
const OrcaConnect = window.OrcaRouterConnect;

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

/* ------------------------------------------------------------------ *
 * OrcaRouter provider + model selection
 * ------------------------------------------------------------------ */

function renderProviderOptions(select, selected) {
  select.innerHTML = state.providers
    .map((provider) => `<option value="${escapeHtml(provider.id)}"${provider.id === selected ? " selected" : ""}>${escapeHtml(provider.label)}</option>`)
    .join("");
}

function selectedModel(form) {
  if (!OrcaProvider.isOrcaRouter(form.providerId)) return form.customModel.value.trim();
  return form.combobox ? form.combobox.value : "";
}

function clearSelectedModel(form, message) {
  if (form.combobox) {
    form.combobox.value = "";
    form.combobox.render();
  }
  form.customModel.value = "";
  if (message) form.modelMessage(message);
}

function setFormVisibility(form) {
  const orcarouter = OrcaProvider.isOrcaRouter(form.providerId);
  form.root.querySelectorAll("[data-provider-field]").forEach((field) => {
    field.hidden = false;
  });
  form.root.querySelectorAll("[data-provider-field=base_url], [data-provider-field=api_key]").forEach((field) => {
    field.hidden = orcarouter;
  });
  form.root.querySelectorAll("[data-model-field=custom]").forEach((field) => { field.hidden = orcarouter; });
  form.root.querySelectorAll("[data-model-field=orcarouter]").forEach((field) => { field.hidden = !orcarouter; });
  const credential = form.root.querySelector(".orca-inline-credential");
  if (credential) credential.hidden = !orcarouter;
  // The generic base_url / API Key pair must not keep `required` while hidden:
  // a hidden required field silently blocks the form in most browsers.
  form.root.querySelectorAll("input").forEach((input) => {
    if (input.dataset.originalRequired === undefined) {
      input.dataset.originalRequired = input.required ? "1" : "0";
    }
    const field = input.closest("[data-provider-field]");
    input.required = input.dataset.originalRequired === "1" && !(field && field.hidden);
  });
  const orcarouterOnly = form.root.querySelector("[data-orca-submit]");
  if (orcarouterOnly) orcarouterOnly.disabled = false;
}

function renderModelOptions(form) {
  if (!form.combobox) return;
  const capability = form.capability;
  const modality = form.modality ? form.modality() : null;
  if (!OrcaProvider.catalogMatches(capability, modality)) {
    // Never render a mismatch as an empty selector: the catalog for another
    // capability is not evidence that this entry point has no models.
    form.combobox.setStatus("正在读取 OrcaRouter 模型目录……", {});
    return;
  }
  const catalog = OrcaProvider.state.catalog;
  form.combobox.setOptions(
    OrcaProvider.compatibleOptions(capability, modality),
    OrcaProvider.catalogStatusLabel(catalog),
    {
      degraded: Boolean(catalog && catalog.degraded),
      error: (catalog && catalog.error) || "",
      source: (catalog && catalog.source) || "none",
    }
  );
}

async function reloadCatalog(form, options) {
  const config = options || {};
  const modality = form.modality ? form.modality() : null;
  const catalog = await OrcaProvider.loadCatalog({
    providerId: form.providerId,
    capability: form.capability,
    modality: modality,
    refresh: Boolean(config.refresh),
  });
  // A model that is no longer offered must be dropped instead of being kept as
  // a silently wrong value.
  const current = form.combobox ? form.combobox.value : "";
  if (current && !OrcaProvider.isCompatible(current, form.capability, modality)) {
    clearSelectedModel(form, `已选模型 ${current} 不再满足当前入口的能力要求，请重新选择`);
  }
  renderModelOptions(form);
  return catalog;
}

async function applyProvider(form, options) {
  const config = options || {};
  form.providerId = form.providerSelect.value;
  setFormVisibility(form);
  if (!OrcaProvider.isOrcaRouter(form.providerId)) {
    renderModelOptions(form);
    return;
  }
  if (config.teardownConnect !== false && OrcaConnect) OrcaConnect.teardown();
  await reloadCatalog(form, { refresh: Boolean(config.refresh) });
  form.syncModelToList();
}

function createModelCombobox(container, form) {
  const fieldKey = container.dataset.orcaModelSelect;
  container.innerHTML = `
    <button type="button" class="orca-select-trigger" id="${fieldKey}-api-model-trigger" aria-haspopup="listbox" aria-expanded="false" data-orca-model-trigger>
      <span data-orca-model-label>请选择模型</span><span class="orca-select-caret" aria-hidden="true">▾</span>
    </button>
    <div class="orca-select-panel" role="listbox" hidden data-orca-model-panel>
      <input type="search" class="orca-select-search" placeholder="搜索模型" aria-label="搜索模型" data-orca-model-search>
      <p class="orca-select-status" data-orca-model-status></p>
      <ul class="orca-select-options" data-orca-model-options></ul>
    </div>
  `;
  const trigger = container.querySelector("[data-orca-model-trigger]");
  const panel = container.querySelector("[data-orca-model-panel]");
  const search = container.querySelector("[data-orca-model-search]");
  const list = container.querySelector("[data-orca-model-options]");
  const label = container.querySelector("[data-orca-model-label]");
  const status = container.querySelector("[data-orca-model-status]");
  let options = [];
  let meta = { source: "none" };

  const combobox = {
    get value() { return container.dataset.value || ""; },
    set value(next) {
      container.dataset.value = next || "";
      const match = options.find((model) => model.id === next);
      label.textContent = match ? match.name || match.id : (next || "请选择模型");
    },
    get open() { return !panel.hidden; },
    openPanel() {
      panel.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      search.value = "";
      renderList("");
      search.focus();
    },
    closePanel() {
      panel.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
    },
    setStatus(statusLabel, extra) {
      meta = extra || meta;
      status.textContent = statusLabel || "";
      status.className = `orca-select-status${meta.degraded ? " degraded" : ""}`;
    },
    setOptions(next, statusLabel, extra) {
      options = next || [];
      combobox.setStatus(statusLabel, extra);
      if (combobox.value && !options.some((model) => model.id === combobox.value)) {
        combobox.value = "";
      }
      combobox.render();
      if (combobox.open) renderList(search.value);
    },
    render() {
      if (!combobox.value) label.textContent = options.length ? "请选择模型" : "无可用模型";
    },
    options() { return options; },
    meta() { return meta; },
    element: container,
    trigger,
    panel,
  };

  function renderList(query) {
    const needle = (query || "").trim().toLowerCase();
    const visible = needle
      ? options.filter((model) => `${model.id} ${model.name || ""}`.toLowerCase().includes(needle))
      : options;
    list.innerHTML = visible.length
      ? visible.map((model) => {
        const bits = [model.owned_by, model.context_length ? `${model.context_length} ctx` : ""].filter(Boolean).join(" · ");
        return `<li role="option" data-model-option="${escapeHtml(model.id)}" aria-selected="${model.id === combobox.value}">
          <strong>${escapeHtml(model.id)}</strong><small>${escapeHtml(bits)}</small>
          ${model.input_modalities && model.input_modalities.length ? `<em>${escapeHtml(model.input_modalities.join("/"))}</em>` : ""}
        </li>`;
      }).join("")
      : `<li class="orca-select-empty">没有匹配的模型</li>`;
    list.dataset.count = String(visible.length);
    list.querySelectorAll("[data-model-option]").forEach((item) => {
      item.addEventListener("click", () => {
        combobox.value = item.dataset.modelOption;
        combobox.closePanel();
        form.syncModelToList();
      });
    });
  }

  trigger.addEventListener("click", () => {
    if (combobox.open) combobox.closePanel();
    else combobox.openPanel();
  });
  search.addEventListener("input", () => renderList(search.value));
  container.addEventListener("keydown", (event) => {
    if (event.key === "Escape") combobox.closePanel();
  });
  document.addEventListener("click", (event) => {
    if (!container.contains(event.target)) combobox.closePanel();
  });
  return combobox;
}

function createCustomCombobox() {
  return {
    get value() { return ""; },
    set value(_next) {},
    render() {},
    setStatus() {},
    setOptions() {},
    options() { return []; },
    meta() { return { source: "custom" }; },
    closePanel() {},
    element: null,
    trigger: null,
    panel: null,
  };
}

function initProviderForm(config) {
  const form = {
    providerSelect: byId(config.selectId),
    root: byId(config.rootId),
    customModel: byId(config.modelInputId),
    capability: config.capability || "chat",
    modality: config.modality || null,
    modelMessage: config.onModelMessage || (() => {}),
    syncModelToList: () => {},
    providerId: "custom",
    combobox: null,
  };
  const container = form.root.querySelector(config.comboboxSelector);
  form.combobox = container ? createModelCombobox(container, form) : createCustomCombobox();
  form.syncModelToList = config.syncModelToList || (() => {});
  renderProviderOptions(form.providerSelect, config.defaultProvider || "custom");
  form.providerId = form.providerSelect.value;
  form.providerSelect.addEventListener("change", () => {
    // Switching provider or authentication method must release the login lock.
    applyProvider(form, { teardownConnect: true });
    form.syncModelToList();
  });
  state.providerForms.push(form);
  return form;
}

function requestContext(form) {
  const providerId = form.providerId;
  const model = selectedModel(form);
  if (OrcaProvider.isOrcaRouter(providerId)) {
    return { provider_id: providerId, api_model: model, base_url: "" };
  }
  return {
    provider_id: providerId,
    api_model: model,
    base_url: (form.root.querySelector("#test-api-base, #api-base") || {}).value || "",
    api_key: (form.root.querySelector("#test-api-key, #api-key") || {}).value || "",
  };
}

/* ------------------------------------------------------------------ *
 * credential status + connect dialog
 * ------------------------------------------------------------------ */

function credentialSummary(credential) {
  if (!credential || !credential.configured) {
    return credential && credential.needs_reauth ? "已失效，请重新连接" : "未配置";
  }
  const origin = credential.source === "env" ? "环境变量" : credential.source === "api_key" ? "API Key" : "账户授权";
  return `${origin} · ${credential.masked || "已保存"}`;
}

function renderCredentialState(credential) {
  const text = credentialSummary(credential);
  ["test-orca-credential-state", "enroll-orca-credential-state", "orca-credential-state"].forEach((id) => {
    const element = byId(id);
    if (element) element.textContent = text;
  });
  window.ORCA_CREDENTIAL = credential;
}

async function refreshCredentialState() {
  const credential = await OrcaProvider.refreshCredential();
  renderCredentialState(credential);
  return credential;
}

window.onOrcaRouterCredentialChanged = (credential) => {
  renderCredentialState(credential);
  state.providerForms.forEach((form) => {
    if (OrcaProvider.isOrcaRouter(form.providerId)) reloadCatalog(form, { refresh: true });
  });
};

function openConnectDialog() {
  const dialog = byId("orca-connect-dialog");
  if (!dialog) return;
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
  refreshCredentialState();
}

function closeConnectDialog() {
  // Closing the dialog is a terminal path: release the login lock.
  if (OrcaConnect) OrcaConnect.teardown();
  const dialog = byId("orca-connect-dialog");
  if (!dialog) return;
  if (typeof dialog.close === "function" && dialog.open) dialog.close();
  else dialog.removeAttribute("open");
}

function bindConnectDialog() {
  document.querySelectorAll("[data-orca-open-connect]").forEach((button) => {
    button.addEventListener("click", openConnectDialog);
  });
  document.querySelectorAll("[data-orca-close-connect]").forEach((button) => {
    button.addEventListener("click", closeConnectDialog);
  });
  const start = document.querySelector("[data-orca-start-login]");
  if (start) start.addEventListener("click", () => OrcaConnect.startLogin({ openBrowser: true }));
  const cancel = document.querySelector("[data-orca-cancel-login]");
  if (cancel) cancel.addEventListener("click", () => OrcaConnect.teardown());
  const submitCode = document.querySelector("[data-orca-submit-code]");
  if (submitCode) {
    submitCode.addEventListener("click", () => {
      const input = byId("orca-auth-code");
      if (input && input.value.trim()) OrcaConnect.submitCode(input.value.trim());
    });
  }
  const saveKey = document.querySelector("[data-orca-save-key]");
  if (saveKey) {
    saveKey.addEventListener("click", () => {
      const input = byId("orca-api-key");
      if (input && input.value.trim()) OrcaConnect.saveApiKey(input.value.trim());
    });
  }
  const clearKey = document.querySelector("[data-orca-clear-key]");
  if (clearKey) clearKey.addEventListener("click", () => OrcaConnect.clearCredential());
}

/* ------------------------------------------------------------------ *
 * manual test
 * ------------------------------------------------------------------ */

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
  byId("result").innerHTML = `
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
  const valid = states.filter((state) => state === "done").length;
  const attempted = states.filter((state) => ["done", "invalid", "error"].includes(state)).length;
  const target = 3;
  byId("api-test-progress").hidden = false;
  byId("api-progress-status").textContent = status;
  byId("api-progress-count").textContent = `有效 ${valid}/${target} · 已尝试 ${attempted}/${states.length}`;
  byId("api-progress-fill").style.width = `${(valid / target) * 100}%`;
  byId("api-progress-steps").innerHTML = states.map((state, index) => {
    const labels = { pending: "等待", working: "请求中", done: "有效", invalid: "数字不足", error: "接口失败", skipped: "无需调用" };
    return `<span class="progress-step ${state}"><b>${index + 1}</b>挑战 ${index + 1} · ${labels[state]}</span>`;
  }).join("");
}

async function testViaApi(event) {
  event.preventDefault();
  const form = state.providerForms.find((item) => item.root.id === "api-test-form");
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  byId("result").hidden = true;
  setMessage(byId("test-message"), "");

  if (OrcaProvider.isOrcaRouter(form.providerId) && !selectedModel(form)) {
    setMessage(byId("test-message"), "请先从 OrcaRouter 模型目录中选择一个模型。", "error");
    button.disabled = false;
    return;
  }

  const challengeResponse = await fetch("/api/challenges");
  const firstBatch = (await challengeResponse.json()).challenges;
  const retryResponse = await fetch("/api/challenges");
  const challenges = firstBatch.concat((await retryResponse.json()).challenges);
  const states = challenges.map(() => "pending");
  const outputs = [];
  const errors = [];
  const target = 3;
  const configuration = {
    ...requestContext(form),
    temperature: optionalNumber("test-temperature"),
  };
  renderApiProgress(states, "已生成独立挑战，准备调用模型");

  for (let index = 0; index < challenges.length && outputs.length < target; index += 1) {
    states[index] = "working";
    renderApiProgress(states, `正在进行第 ${index + 1} 次尝试，等待模型完整输出……`);
    try {
      const response = await fetch("/api/test/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...configuration,
          prompt: challenges[index].prompt,
          expected_count: challenges[index].expected_count,
        }),
      });
      const payload = await response.json();
      if (payload.credential) renderCredentialState(payload.credential);
      if (!response.ok) throw new Error(payload.error || "接口请求失败");
      if (payload.accepted) {
        outputs.push({ text: payload.text, expected_count: challenges[index].expected_count });
        states[index] = "done";
      } else {
        errors.push(`尝试 ${index + 1}: 有效数字 ${payload.parsed_numbers}/${payload.minimum_numbers}`);
        states[index] = "invalid";
      }
    } catch (error) {
      errors.push(`尝试 ${index + 1}: ${error.message}`);
      states[index] = "error";
    }
    renderApiProgress(states, `当前已有 ${outputs.length}/${target} 份有效回答`);
  }

  if (outputs.length === target) {
    states.forEach((state, index) => { if (state === "pending") states[index] = "skipped"; });
  }

  if (!outputs.length) {
    renderApiProgress(states, "六次尝试后仍没有可用回答");
    setMessage(byId("test-message"), `没有获得可分析输出。${errors[0] || ""}`, "error");
    button.disabled = false;
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
    const attempted = states.filter((state) => ["done", "invalid", "error"].includes(state)).length;
    result.api_test = { requested: target, attempted, max_attempts: challenges.length, received: outputs.length, errors };
    renderApiProgress(states, `测试完成：${outputs.length}/${target} 份有效回答进入归因`);
    renderResult(result);
  } else {
    setMessage(byId("test-message"), result.error || "API 自动测试失败。", "error");
  }
  button.disabled = false;
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
  const form = state.providerForms.find((item) => item.root.id === "auto-enrollment");
  const button = event.currentTarget.querySelector("button[type=submit]");
  if (OrcaProvider.isOrcaRouter(form.providerId) && !selectedModel(form)) {
    setMessage(byId("enrollment-message"), "请先从 OrcaRouter 模型目录中选择一个模型。", "error");
    return;
  }
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
        ...requestContext(form),
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
  if (payload.credential) renderCredentialState(payload.credential);
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

async function bootstrap() {
  state.providers = await OrcaProvider.loadProviders();
  const testForm = initProviderForm({
    selectId: "test-provider",
    rootId: "api-test-form",
    modelInputId: "test-api-model",
    comboboxSelector: '[data-orca-model-select="test"]',
    capability: "chat",
    modality: () => null,
    onModelMessage: (message) => setMessage(byId("test-message"), message, "working"),
    syncModelToList: () => {},
  });
  const enrollForm = initProviderForm({
    selectId: "enroll-provider",
    rootId: "auto-enrollment",
    modelInputId: "api-model",
    comboboxSelector: '[data-orca-model-select="enroll"]',
    capability: "chat",
    modality: () => null,
    onModelMessage: (message) => setMessage(byId("enrollment-message"), message, "working"),
    syncModelToList: () => {},
  });
  // Keep the two entry points consistent: choosing a provider on one form
  // selects it on the other, and both re-derive their model options.
  testForm.providerSelect.addEventListener("change", () => {
    enrollForm.providerSelect.value = testForm.providerId;
    applyProvider(enrollForm, { teardownConnect: false });
  });
  enrollForm.providerSelect.addEventListener("change", () => {
    testForm.providerSelect.value = enrollForm.providerId;
    applyProvider(testForm, { teardownConnect: false });
  });

  bindConnectDialog();
  OrcaConnect.bindPageLifecycle();
  await refreshCredentialState();
  await applyProvider(testForm, { teardownConnect: false });
  await applyProvider(enrollForm, { teardownConnect: false });
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
bootstrap();
