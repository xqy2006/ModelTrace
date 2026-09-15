/*
 * OrcaRouter connect dialog: API Key and OAuth 2.0 + PKCE side by side.
 *
 * Two independent entries, one credential type: pasting an sk-orca-... key and
 * authorizing in a browser both end up as the same backend credential, and the
 * inference path never learns which one was used.
 *
 * Login-state safety rules implemented here:
 *  - a monotonically increasing attempt id, so a late response from a previous
 *    login can never overwrite a newer one;
 *  - every terminal path (success, denial, exchange error, timeout, cancel,
 *    switching provider, closing the dialog, unmount, reload, pagehide) clears
 *    the busy flag and the authorization hint and cancels the server attempt;
 *  - pagehide/beforeunload clear the flags synchronously, because the page may
 *    be restored from the back-forward cache without remounting, and the
 *    generation-guarded cleanup would then leave the dialog permanently busy.
 */
(function (global) {
  "use strict";

  const POLL_INTERVAL_MS = 1200;
  const POLL_TIMEOUT_MS = 600000;

  const connectState = {
    attempt: 0,
    generation: 0,
    polling: false,
    pollTimer: null,
    busy: false,
    hint: "",
    status: "idle",
    lastError: "",
  };

  const byId = (id) => document.getElementById(id);

  function isBusy() {
    return connectState.busy;
  }

  function reset(message, status) {
    connectState.generation += 1;
    connectState.busy = false;
    connectState.hint = "";
    connectState.status = status || "idle";
    connectState.lastError = status === "error" ? message || "" : "";
    if (connectState.pollTimer !== null) {
      global.clearInterval(connectState.pollTimer);
      connectState.pollTimer = null;
    }
    connectState.polling = false;
    connectState.attempt = 0;
    render(message, connectState.status);
  }

  function cancelServerAttempt(attempt, options) {
    // The attempt is passed in explicitly: teardown clears the state first, so
    // reading it back from connectState here would silently skip the cancel.
    if (!attempt) return;
    // keepalive lets the cancellation outlive the page when it is being unloaded.
    fetch("/api/orcarouter/connect/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attempt: attempt }),
      keepalive: Boolean(options && options.keepalive),
    }).catch(() => {});
  }

  function teardown(options) {
    // Synchronous release: pagehide/bfcache restore must never inherit busy state.
    const attempt = connectState.attempt;
    connectState.generation += 1;
    connectState.busy = false;
    connectState.hint = "";
    connectState.status = "idle";
    connectState.polling = false;
    if (connectState.pollTimer !== null) {
      global.clearInterval(connectState.pollTimer);
      connectState.pollTimer = null;
    }
    connectState.attempt = 0;
    if (attempt) cancelServerAttempt(attempt, { keepalive: Boolean(options && options.keepalive) });
    render("", "idle");
  }

  function render(message, status) {
    const box = byId("orca-connect-status");
    if (box) {
      const text = message || connectState.hint || "";
      box.textContent = text;
      box.className = `message ${status === "error" ? "error" : status === "success" ? "success" : "working"}`;
      box.hidden = !text;
    }
    const urlBox = byId("orca-connect-url");
    if (urlBox) {
      const show = connectState.status === "pending" && connectState.url;
      urlBox.hidden = !show;
      if (show) urlBox.querySelector("input").value = connectState.url;
    }
    const codeRow = byId("orca-connect-code-row");
    if (codeRow) codeRow.hidden = connectState.status !== "pending";
    document.querySelectorAll("[data-orca-connect-action]").forEach((button) => {
      button.disabled = connectState.busy;
    });
  }

  function setHint(text) {
    connectState.hint = text || "";
    render(text, connectState.status === "idle" ? "working" : connectState.status);
  }

  async function startLogin(options) {
    const config = options || {};
    teardown();
    const generation = ++connectState.generation;
    connectState.busy = true;
    connectState.status = "pending";
    render("正在向 OrcaRouter 申请授权码……", "working");
    let payload;
    try {
      const response = await fetch("/api/orcarouter/connect/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ open_browser: Boolean(config.openBrowser) }),
      });
      payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法发起 OrcaRouter 授权");
    } catch (error) {
      if (generation !== connectState.generation) return null;
      reset(error.message, "error");
      return null;
    }
    // Ignore a completion that belongs to a superseded login attempt.
    if (generation !== connectState.generation) return null;
    connectState.attempt = payload.attempt;
    connectState.url = payload.authorize_url;
    setHint("已打开授权页面。若浏览器没有自动打开，请复制下面的链接；也可以选择“显示授权码”后把授权码粘贴回这里。");
    poll(generation);
    return payload;
  }

  function poll(generation) {
    if (connectState.pollTimer !== null) global.clearInterval(connectState.pollTimer);
    const startedAt = Date.now();
    connectState.polling = true;
    connectState.pollTimer = global.setInterval(async () => {
      if (generation !== connectState.generation) {
        global.clearInterval(connectState.pollTimer);
        connectState.pollTimer = null;
        return;
      }
      if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
        reset("授权等待超时（10 分钟），请重新连接", "error");
        return;
      }
      let payload;
      try {
        const response = await fetch(`/api/orcarouter/connect/status?attempt=${connectState.attempt}`);
        payload = await response.json();
        if (!response.ok) {
          if (generation !== connectState.generation) return;
          reset(payload.error || "授权状态查询失败", "error");
          return;
        }
      } catch (error) {
        return; // transient network error: keep polling until the deadline
      }
      if (generation !== connectState.generation) return;
      if (payload.status === "pending") return;
      if (payload.status === "authorized") {
        finishSuccess(payload, generation);
        return;
      }
      const messages = {
        denied: "授权被拒绝，OrcaRouter 凭据未更改",
        expired: "授权等待超时（10 分钟），请重新连接",
        cancelled: "已取消授权",
        error: payload.message || "授权失败",
      };
      reset(messages[payload.status] || payload.message || "授权失败", payload.status === "denied" ? "error" : "error");
    }, POLL_INTERVAL_MS);
  }

  function finishSuccess(payload, generation) {
    if (generation !== connectState.generation) return;
    connectState.generation += 1;
    connectState.busy = false;
    connectState.hint = "";
    connectState.status = "success";
    connectState.polling = false;
    if (connectState.pollTimer !== null) {
      global.clearInterval(connectState.pollTimer);
      connectState.pollTimer = null;
    }
    const credential = payload.credential || {};
    let note = `已连接 OrcaRouter（${credential.masked || "凭据已保存"}）`;
    if (credential.scope && credential.scope !== "api") {
      note += `；注意：授权范围为 ${credential.scope}，不是 api`;
    }
    render(note, "success");
    if (typeof global.onOrcaRouterCredentialChanged === "function") {
      global.onOrcaRouterCredentialChanged(credential);
    }
  }

  async function submitCode(code) {
    if (!connectState.attempt) return null;
    const generation = connectState.generation;
    setHint("正在用授权码换取 OrcaRouter 凭据……");
    const response = await fetch("/api/orcarouter/connect/code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attempt: connectState.attempt, code: code }),
    });
    const payload = await response.json();
    if (generation !== connectState.generation) return null;
    if (!response.ok) {
      render(payload.error || "授权码兑换失败", "error");
      return null;
    }
    if (payload.status === "authorized") {
      finishSuccess(payload, generation);
      return payload;
    }
    render(payload.message || "授权码兑换失败", "error");
    return null;
  }

  async function saveApiKey(key) {
    const response = await fetch("/api/orcarouter/credential", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key }),
    });
    const payload = await response.json();
    if (!response.ok) {
      render(payload.error || "API Key 保存失败", "error");
      return null;
    }
    const masked = (payload.credential && payload.credential.masked) || "";
    render(`${payload.message || "API Key 已保存（仅保存在本机）"}${masked ? `（${masked}）` : ""}`, "success");
    if (typeof global.onOrcaRouterCredentialChanged === "function") {
      global.onOrcaRouterCredentialChanged(payload.credential);
    }
    return payload.credential;
  }

  async function clearCredential() {
    const response = await fetch("/api/orcarouter/credential", { method: "DELETE" });
    const payload = await response.json();
    render(payload.message || "已清除本机保存的 OrcaRouter 凭据", "success");
    if (typeof global.onOrcaRouterCredentialChanged === "function") {
      global.onOrcaRouterCredentialChanged(payload.credential);
    }
    return payload.credential;
  }

  function bindPageLifecycle() {
    // Back-forward cache: the page can be restored without remounting, so the
    // busy flag and hint must be cleared here rather than in a guarded finally.
    global.addEventListener("pagehide", () => teardown({ keepalive: true }));
    global.addEventListener("beforeunload", () => teardown({ keepalive: true }));
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") teardown({ keepalive: true });
    });
  }

  global.OrcaRouterConnect = {
    startLogin,
    submitCode,
    saveApiKey,
    clearCredential,
    teardown,
    isBusy,
    state: connectState,
    bindPageLifecycle,
  };
})(window);
