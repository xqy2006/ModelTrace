// Browser-side tests for the OrcaRouter provider UI and connect dialog.
//
// The two static modules are loaded into a vm sandbox that provides a minimal
// window/document/fetch/timer surface, so the real shipped code runs and the
// assertions are about what the UI would actually do: which options reach the
// model selector, what a degraded catalog looks like, and whether the login
// lock is released on every terminal path (including `pagehide`).
//
//   node --test tests/ui-orcarouter.test.mjs

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(here);

const PROVIDER_SOURCE = await readFile(path.join(repo, 'static', 'orcarouter-provider.js'), 'utf8');
const CONNECT_SOURCE = await readFile(path.join(repo, 'static', 'orcarouter-connect.js'), 'utf8');

const LIVE_CATALOG = {
  models: [
    { id: 'vendor/text-only', name: 'Text Only', input_modalities: ['text'], supported_endpoint_types: ['openai'] },
    { id: 'vendor/vision', name: 'Vision', input_modalities: ['text', 'image'], supported_endpoint_types: ['openai'] },
  ],
  source: 'live',
  degraded: false,
  error: null,
  capability: 'chat',
  modality: null,
};

function makeElement(id, extra = {}) {
  return {
    id,
    textContent: '',
    className: '',
    hidden: false,
    value: '',
    disabled: false,
    querySelector: () => ({ value: '' }),
    ...extra,
  };
}

/** Minimal DOM + timer + fetch harness for the two static modules. */
function createHarness(options = {}) {
  const elements = new Map();
  ['orca-connect-status', 'orca-connect-url', 'orca-connect-code-row', 'orca-credential-state'].forEach((id) => {
    elements.set(id, makeElement(id));
  });
  const windowListeners = new Map();
  const timers = new Map();
  let timerId = 0;
  const requests = [];
  const responses = [...(options.responses || [])];

  const document = {
    visibilityState: 'visible',
    getElementById: (id) => elements.get(id) || null,
    querySelectorAll: () => [],
    addEventListener: () => {},
  };

  const sandbox = {
    console,
    document,
    AbortController,
    URL,
    JSON,
    Object,
    Array,
    Promise,
    Error,
    String,
    Number,
    Boolean,
    Math,
    setTimeout,
    clearTimeout,
    setInterval: (handler) => {
      timerId += 1;
      timers.set(timerId, handler);
      return timerId;
    },
    clearInterval: (id) => timers.delete(id),
    fetch: async (url, init) => {
      requests.push({ url, init, body: init && init.body ? JSON.parse(init.body) : null });
      // Cancellations are fire-and-forget side effects; they must not consume a
      // queued response meant for the next start/status call.
      const next = url.endsWith('/connect/cancel')
        ? { ok: true, status: 200, payload: { cancelled: true } }
        : (responses.length ? responses.shift() : { ok: true, status: 200, payload: {} });
      const resolved = typeof next === 'function' ? await next(url, init) : next;
      if (resolved instanceof Error) throw resolved;
      return {
        ok: resolved.ok !== false,
        status: resolved.status || 200,
        json: async () => resolved.payload || {},
      };
    },
    addEventListener: (type, handler) => {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(handler);
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(PROVIDER_SOURCE, sandbox, { filename: 'orcarouter-provider.js' });
  vm.runInContext(CONNECT_SOURCE, sandbox, { filename: 'orcarouter-connect.js' });

  return {
    sandbox,
    provider: sandbox.OrcaRouterProvider,
    connect: sandbox.OrcaRouterConnect,
    elements,
    requests,
    dispatchWindow: (type, event = {}) => {
      (windowListeners.get(type) || []).forEach((handler) => handler(event));
    },
    tick: () => {
      const pending = [...timers.values()];
      return Promise.all(pending.map((handler) => handler()));
    },
    pendingTimers: () => timers.size,
  };
}

/* ------------------------------------------------------------------ *
 * model catalog -> selector options
 * ------------------------------------------------------------------ */

test('selector options come from the API and never from free text', async () => {
  const harness = createHarness({ responses: [{ payload: LIVE_CATALOG }] });
  await harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat' });

  assert.equal(harness.requests.length, 1);
  assert.equal(harness.requests[0].url, '/api/orcarouter/models');
  assert.deepEqual(harness.requests[0].body, {
    provider_id: 'orcarouter',
    capability: 'chat',
    modality: null,
    refresh: false,
  });
  const options = harness.provider.compatibleOptions('chat', null);
  assert.deepEqual(Array.from(options).map((model) => model.id), ['vendor/text-only', 'vendor/vision']);
  assert.equal(harness.provider.isCompatible('vendor/text-only', 'chat', null), true);
});

test('the two entry points share one catalog request instead of racing', async () => {
  const harness = createHarness({ responses: [{ payload: LIVE_CATALOG }] });
  const [first, second] = await Promise.all([
    harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat' }),
    harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat' }),
  ]);
  assert.equal(harness.requests.length, 1);
  assert.deepEqual(Array.from(first.models).map((m) => m.id), Array.from(second.models).map((m) => m.id));
});

test('a multimodal entry point receives only models declaring that input modality', async () => {
  const harness = createHarness({
    responses: [
      {
        payload: {
          models: [{ id: 'vendor/vision', name: 'Vision', input_modalities: ['text', 'image'] }],
          source: 'live',
          degraded: false,
          capability: 'chat',
          modality: 'image',
        },
      },
    ],
  });
  await harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat', modality: 'image' });
  assert.equal(harness.requests[0].body.modality, 'image');
  assert.deepEqual(
    Array.from(harness.provider.compatibleOptions('chat', 'image')).map((model) => model.id),
    ['vendor/vision'],
  );
  // The previously selected text-only model no longer satisfies the entry point,
  // so it must be reported as incompatible (the selector clears it).
  assert.equal(harness.provider.isCompatible('vendor/text-only', 'chat', 'image'), false);
  // And the text list is not reused for the multimodal selector.
  assert.deepEqual(Array.from(harness.provider.compatibleOptions('chat', null)), []);
  assert.equal(harness.provider.catalogMatches('chat', null), false);
  assert.equal(harness.provider.catalogMatches('chat', 'image'), true);
});

test('a catalog outage surfaces a labelled fallback, never free text', async () => {
  const harness = createHarness({
    responses: [
      { payload: { error: 'OrcaRouter 接口不可用' }, ok: false, status: 400 },
    ],
  });
  await harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat' });
  const catalog = harness.provider.state.catalog;
  assert.equal(catalog.degraded, true);
  assert.equal(catalog.source, 'none');
  assert.deepEqual(Array.from(catalog.models), []);
  assert.match(harness.provider.catalogStatusLabel(catalog), /不可用/);
});

test('a degraded catalog is labelled and only exposes its own verified list', async () => {
  const harness = createHarness({
    responses: [
      {
        payload: {
          models: [{ id: 'openai/gpt-5.5', name: 'OpenAI: GPT-5.5', input_modalities: [] }],
          source: 'seed',
          degraded: true,
          error: '无法连接模型目录：timed out',
          capability: 'chat',
          modality: null,
        },
      },
    ],
  });
  await harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat' });
  const catalog = harness.provider.state.catalog;
  assert.equal(catalog.source, 'seed');
  assert.match(harness.provider.catalogStatusLabel(catalog), /已验证的备用列表/);
  assert.deepEqual(
    Array.from(harness.provider.compatibleOptions('chat', null)).map((model) => model.id),
    ['openai/gpt-5.5'],
  );
});

test('refresh re-requests instead of reusing the in-flight catalog', async () => {
  const harness = createHarness({ responses: [{ payload: LIVE_CATALOG }, { payload: LIVE_CATALOG }] });
  await harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat' });
  await harness.provider.loadCatalog({ providerId: 'orcarouter', capability: 'chat', refresh: true });
  assert.equal(harness.requests.length, 2);
  assert.equal(harness.requests[1].body.refresh, true);
});

/* ------------------------------------------------------------------ *
 * connect dialog lifecycle
 * ------------------------------------------------------------------ */

test('pagehide clears busy state and a second login starts without remounting', async () => {
  const harness = createHarness({
    responses: [
      { payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } },
      { payload: { attempt: 2, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=b' } },
    ],
  });
  harness.connect.bindPageLifecycle();
  await harness.connect.startLogin({ openBrowser: true });
  assert.equal(harness.connect.isBusy(), true);
  assert.equal(harness.connect.state.attempt, 1);
  assert.equal(harness.elements.get('orca-connect-url').hidden, false);

  harness.dispatchWindow('pagehide');
  assert.equal(harness.connect.isBusy(), false, 'pagehide must release the busy flag synchronously');
  assert.equal(harness.connect.state.hint, '');
  assert.equal(harness.connect.state.status, 'idle');
  assert.equal(harness.elements.get('orca-connect-url').hidden, true);
  const keepaliveCancel = harness.requests.find(
    (request) => request.url === '/api/orcarouter/connect/cancel',
  );
  assert.ok(keepaliveCancel, 'pagehide must cancel the server attempt');
  assert.equal(keepaliveCancel.init.keepalive, true);

  // No remount: the same live objects start a second login.
  await harness.connect.startLogin({ openBrowser: true });
  assert.equal(harness.connect.isBusy(), true);
  assert.equal(harness.connect.state.attempt, 2);
  assert.equal(harness.requests.filter((r) => r.url === '/api/orcarouter/connect/start').length, 2);
});

test('a late response from a superseded attempt cannot overwrite the new login', async () => {
  let releaseStatus;
  const harness = createHarness({
    responses: [
      { payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } },
      () => new Promise((resolve) => { releaseStatus = () => resolve({ payload: { attempt: 1, status: 'authorized', credential: { masked: 'sk-orca-…1111' } } }); }),
      { payload: { attempt: 2, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=b' } },
    ],
  });
  await harness.connect.startLogin({ openBrowser: true });
  const stalePoll = harness.tick();
  harness.connect.teardown();
  await harness.connect.startLogin({ openBrowser: true });
  releaseStatus();
  await stalePoll;
  assert.equal(harness.connect.state.attempt, 2);
  assert.notEqual(harness.connect.state.status, 'success');
  assert.equal(harness.connect.state.credential, undefined);
});

test('denial and cancel both end the wait with an actionable message', async () => {
  const harness = createHarness({
    responses: [
      { payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } },
      { payload: { attempt: 1, status: 'denied', message: '授权被拒绝（access_denied）' } },
    ],
  });
  await harness.connect.startLogin({ openBrowser: true });
  await harness.tick();
  assert.equal(harness.connect.isBusy(), false);
  assert.equal(harness.elements.get('orca-connect-status').hidden, false);
  assert.match(harness.elements.get('orca-connect-status').textContent, /拒绝/);
  assert.equal(harness.pendingTimers(), 0, 'polling must stop after a terminal status');

  const second = createHarness({
    responses: [{ payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } }],
  });
  await second.connect.startLogin({ openBrowser: true });
  second.connect.teardown();
  assert.equal(second.connect.isBusy(), false);
  assert.equal(second.pendingTimers(), 0);
  assert.ok(second.requests.some((request) => request.url === '/api/orcarouter/connect/cancel'));
});

test('a successful exchange stores the credential and only shows it masked', async () => {
  const RAW_KEY = 'sk-orca-uitest-0000000000000000000000000000';
  const harness = createHarness({
    responses: [
      { payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } },
      {
        payload: {
          attempt: 1,
          status: 'authorized',
          credential: { masked: 'sk-orca-…0000', scope: 'api', source: 'pkce' },
        },
      },
    ],
  });
  await harness.connect.startLogin({ openBrowser: true });
  await harness.connect.submitCode('fake-code');
  assert.equal(harness.connect.state.status, 'success');
  assert.equal(harness.connect.isBusy(), false);
  assert.equal(harness.pendingTimers(), 0);
  const status = harness.elements.get('orca-connect-status');
  assert.match(status.textContent, /sk-orca-…0000/);
  assert.doesNotMatch(status.textContent, new RegExp(RAW_KEY));
  const exchange = harness.requests.find((request) => request.url === '/api/orcarouter/connect/code');
  assert.deepEqual(exchange.body, { attempt: 1, code: 'fake-code' });
});

test('granted scope is reported when it is narrower than requested', async () => {
  const harness = createHarness({
    responses: [
      { payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } },
      { payload: { attempt: 1, status: 'authorized', credential: { masked: 'sk-orca-…0000', scope: 'connector' } } },
    ],
  });
  await harness.connect.startLogin({ openBrowser: true });
  await harness.connect.submitCode('fake-code');
  assert.match(harness.elements.get('orca-connect-status').textContent, /connector/);
});

test('the API-key adapter saves through the credential endpoint without echoing the key', async () => {
  const RAW_KEY = 'sk-orca-uitest-1111111111111111111111111111';
  const harness = createHarness({
    responses: [{ payload: { credential: { masked: 'sk-orca-…1111', source: 'api_key' } } }],
  });
  await harness.connect.saveApiKey(RAW_KEY);
  const request = harness.requests[0];
  assert.equal(request.url, '/api/orcarouter/credential');
  assert.equal(request.init.method, 'POST');
  assert.equal(request.body.api_key, RAW_KEY);
  assert.doesNotMatch(harness.elements.get('orca-connect-status').textContent, new RegExp(RAW_KEY));
  assert.match(harness.elements.get('orca-connect-status').textContent, /sk-orca-…1111/);
});

test('switching provider or unmounting releases the login lock', async () => {
  const harness = createHarness({
    responses: [{ payload: { attempt: 1, status: 'pending', authorize_url: 'https://www.orcarouter.ai/auth?state=a' } }],
  });
  await harness.connect.startLogin({ openBrowser: true });
  harness.connect.teardown();
  assert.equal(harness.connect.isBusy(), false);
  assert.equal(harness.connect.state.attempt, 0);
  // An explicit cancel with no attempt in flight must not fire a request.
  const before = harness.requests.length;
  harness.connect.teardown();
  assert.equal(harness.requests.length, before);
});
