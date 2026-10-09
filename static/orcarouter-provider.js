/*
 * OrcaRouter provider + model discovery for the ModelTrace web UI.
 *
 * The browser never holds an OrcaRouter key: every catalog request goes through
 * the local backend, which owns the credential and returns minimal model
 * metadata (id/name/context/modalities). Model options are always produced by
 * the backend capability filter, so the dropdown can never offer a model the
 * selected entry point cannot use, and it never falls back to free text.
 */
(function (global) {
  "use strict";

  const ORCAROUTER_PROVIDERS = ["orcarouter", "orcarouter-oauth"];
  const CATALOG_TIMEOUT_MS = 12000;

  const providerState = {
    providers: [],
    catalog: null, // { models, source, degraded, error, capability, modality }
    pending: 0,
    generation: 0,
  };

  // One in-flight discovery per (provider, capability, modality). Both entry
  // points ask for the chat catalog at startup, and without this they would
  // race each other into an empty dropdown.
  const inflight = new Map();

  const isOrcaRouter = (providerId) => ORCAROUTER_PROVIDERS.includes(providerId);

  async function jsonFetch(url, options) {
    const controller = new AbortController();
    const timer = global.setTimeout(() => controller.abort(), CATALOG_TIMEOUT_MS);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      const payload = await response.json().catch(() => ({}));
      return { ok: response.ok, status: response.status, payload };
    } finally {
      global.clearTimeout(timer);
    }
  }

  async function loadProviders() {
    const { ok, payload } = await jsonFetch("/api/providers");
    if (ok) {
      providerState.providers = payload.providers || [];
    }
    return providerState.providers;
  }

  async function refreshCredential() {
    const { payload } = await jsonFetch("/api/orcarouter/credential");
    return payload;
  }

  function catalogKey(providerId, capability, modality) {
    return `${providerId}|${capability}|${modality || ""}`;
  }

  async function loadCatalog(options) {
    const config = options || {};
    const capability = config.capability || "chat";
    const modality = config.modality || null;
    const providerId = config.providerId || "orcarouter";
    const key = catalogKey(providerId, capability, modality);
    if (!config.refresh && inflight.has(key)) return inflight.get(key);
    if (config.onState) config.onState("loading", null);
    const pending = performLoad(providerId, capability, modality, config).finally(() => {
      inflight.delete(key);
    });
    inflight.set(key, pending);
    return pending;
  }

  async function performLoad(providerId, capability, modality, config) {
    const generation = ++providerState.generation;
    providerState.pending += 1;
    let result;
    try {
      result = await jsonFetch("/api/orcarouter/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider_id: providerId,
          capability: capability,
          modality: modality,
          refresh: Boolean(config.refresh),
          api_key: config.apiKey || undefined,
        }),
      });
    } catch (error) {
      result = { ok: false, payload: { error: `模型目录请求失败：${error.message}` } };
    }
    providerState.pending -= 1;
    // A stale response must not overwrite a newer provider/capability state.
    if (generation !== providerState.generation) return providerState.catalog;

    if (!result.ok) {
      providerState.catalog = {
        models: [],
        source: "none",
        degraded: true,
        error: result.payload.error || "无法读取 OrcaRouter 模型目录",
        capability: capability,
        modality: modality,
      };
    } else {
      providerState.catalog = { ...result.payload, capability: capability, modality: modality };
    }
    if (config.onState) {
      config.onState(providerState.catalog.models.length ? "ready" : "empty", providerState.catalog);
    }
    return providerState.catalog;
  }

  /** True when the cached catalog is the one this filter needs. */
  function catalogMatches(capability, modality) {
    const catalog = providerState.catalog;
    if (!catalog) return false;
    return catalog.capability === (capability || "chat")
      && (catalog.modality || null) === (modality || null);
  }

  function catalogStatusLabel(catalog) {
    if (!catalog) return "";
    if (catalog.source === "live") return `实时目录 · ${catalog.models.length} 个可用模型`;
    if (catalog.source === "cache") {
      return `目录不可用，正在使用上次成功获取的模型列表 · ${catalog.models.length} 个模型`;
    }
    if (catalog.source === "seed") {
      return `目录不可用，正在使用已验证的备用列表 · ${catalog.models.length} 个模型`;
    }
    return catalog.error || "模型目录不可用";
  }

  /** Options for the selector: always backend-filtered, never free text. */
  function compatibleOptions(capability, modality) {
    if (!catalogMatches(capability, modality)) return [];
    return providerState.catalog.models.slice();
  }

  function isCompatible(modelId, capability, modality) {
    return compatibleOptions(capability, modality).some((model) => model.id === modelId);
  }

  global.OrcaRouterProvider = {
    ORCAROUTER_PROVIDERS,
    isOrcaRouter,
    loadProviders,
    loadCatalog,
    refreshCredential,
    catalogStatusLabel,
    catalogMatches,
    compatibleOptions,
    isCompatible,
    state: providerState,
  };
})(window);
