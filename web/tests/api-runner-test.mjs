// Exercise the real runner with deterministic SDK / attribution responses.
import assert from "node:assert/strict";
import { build } from "esbuild";

const output = await build({
  entryPoints: ["composables/useApiTest.ts"],
  bundle: true,
  write: false,
  platform: "node",
  format: "esm",
  plugins: [
    {
      name: "runner-fixtures",
      setup(build) {
        const modules = {
          ai: `export async function generateText(options) {
          const s = globalThis.runnerScenario;
          s.calls.push(options.model);
          s.active++; s.peak = Math.max(s.peak, s.active);
          await new Promise(resolve => setTimeout(resolve, 5));
          s.active--;
          return { text: s.invalid ? '1 2 3' : Array(310).fill('42').join(' ') };
        }`,
          "@ai-sdk/openai": `export function createOpenAI(config) {
          return { chat: id => ({id, config}), responses: id => ({id, config}) };
        }`,
          "@/lib/challenge": `export function generateChallenges(count) {
          return Array.from({length: count}, (_, id) => ({id, prompt: 'numbers', expected_count: 300}));
        }`,
          "@/lib/fingerprint": `export function parseNumbers(text) { return text.split(' '); }
          export function analyzeGlobalOutputs() {
            const s = globalThis.runnerScenario;
            return {probability: s.probabilities[Math.min(s.analyzed++, s.probabilities.length - 1)]};
          }`,
        };
        build.onResolve(
          { filter: /^(ai|@ai-sdk\/openai|@\/lib\/(challenge|fingerprint))$/ },
          (args) => ({ path: args.path, namespace: "fixture" }),
        );
        build.onLoad({ filter: /.*/, namespace: "fixture" }, (args) => ({
          contents: modules[args.path],
          loader: "js",
        }));
      },
    },
  ],
});
let states = new Map();
// Object.assign avoids declaring Nuxt auto-import globals in the app's TS scope.
Object.assign(globalThis, {
  useState: (key, init) => {
    if (!states.has(key)) states.set(key, { value: init() });
    return states.get(key);
  },
  reactive: (value) => value,
  useBank: () => ({ bank: { value: {} } }),
});
const { useApiTest } = await import(
  `data:text/javascript;base64,${Buffer.from(output.outputFiles[0].contents).toString("base64")}`
);
const preset = {
  id: "one",
  name: "one",
  model: "test",
  apiType: "chat",
  baseUrl: "https://api.vendor.com/v1",
  apiKey: "test-only",
};
function reset(probabilities, invalid = false) {
  states = new Map();
  globalThis.runnerScenario = {
    probabilities,
    invalid,
    calls: [],
    analyzed: 0,
    active: 0,
    peak: 0,
  };
  return useApiTest();
}
for (const [probabilities, expectedCalls] of [
  [[0.99], 1],
  [[1], 1],
  [[0.8, 0.99], 2],
  [[0.989999], 3], // Rounded display may be 99%; raw probability is below threshold.
  [[0.4, 0.7, 0.99], 3],
]) {
  const runner = reset(probabilities);
  const result = await runner.runPreset(preset);
  assert.equal(runnerScenario.calls.length, expectedCalls);
  assert.equal(result.challenges.length, 3);
  assert.equal(result.validCount, expectedCalls);
  assert.equal(result.earlyStopped, expectedCalls < 3);
  assert.equal(result.status, "success");
  assert.equal(result.transport, "direct");
  assert.deepEqual(
    result.steps,
    Array.from({ length: 3 }, (_, i) =>
      i < expectedCalls ? "done" : "skipped",
    ),
  );
  if (result.result.probability >= 0.99)
    assert.match(result.message, /成功检验/);
}
const invalid = await reset([1], true).runPreset(preset);
assert.equal(runnerScenario.calls.length, 3);
assert.equal(invalid.status, "failed");
assert.equal(invalid.result, null);
const batch = reset([0.99]);
await batch.runBatch(
  Array.from({ length: 4 }, (_, i) => ({ ...preset, id: String(i) })),
  "https://proxy.vendor.com/v1",
);
assert.equal(runnerScenario.calls.length, 4);
assert.equal(runnerScenario.peak, 2);
for (const call of runnerScenario.calls) {
  assert.equal(call.config.baseURL, "https://proxy.vendor.com/v1");
  assert.equal(call.config.headers["X-ModelTrace-Endpoint"], preset.baseUrl);
}
assert(
  Object.values(batch.runStates.value).every(
    (state) => state.transport === "proxy" && state.earlyStopped,
  ),
);
assert.equal(batch.batchRunning.value, false);
assert.deepEqual(batch.queuedIds.value, []);
console.log(
  "PASS three-challenge limit, exact 99% early stop, invalid outputs and proxy batch concurrency",
);
