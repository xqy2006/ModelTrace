import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import {
  DEFAULTS, classifySample, dataDirectory, isDue, issue, newState, readState, schedule,
  recentComparable, sessionPath, setTurn, validateConfig, validateNumbers, withState,
} from '../scripts/state.mjs';
import { ROOT, challengeContext, handleHook, isDirectCall, loadArtifacts, summarize } from '../scripts/guard.mjs';
import { run } from './fixture-runner.mjs';
import { LANGUAGES, localizedPrompt, forkPrompt } from '../scripts/prompts.mjs';

const minimum = (lo) => lo;
const session = 'modeltrace-test-task';
const event = (hook_event_name, extra = {}) => ({ session_id: session, hook_event_name, model: 'gpt-5.4', turn_id: 'turn-1', ...extra });
const syntheticNumbers = (count = 300) => JSON.stringify(Array.from({ length: count }, (_, i) => (i * 37 + 11) % 355 + 1));
// Synthetic sequences exercise transport and scoring only; these are NOT model accuracy tests.
async function fixture(t) {
  const directory = await mkdtemp(path.join(tmpdir(), 'modeltrace-test-'));
  t.after(async () => {
    assert.equal(path.dirname(path.resolve(directory)), path.resolve(tmpdir()));
    assert.ok(path.basename(directory).startsWith('modeltrace-test-'));
    await rm(directory, { recursive: true, force: true });
  });
  return directory;
}
async function active(directory, now = Date.now()) {
  await handleHook(event('SessionStart'), directory, now, minimum);
  await withState(directory, session, (s) => {
    s.enabled = true; s.startedAt = now; s.epoch = 1; s.forceProbe = false;
    s.config = validateConfig({ mode: 'tools', toolMin: 2, toolMax: 2, languages: ['en'] });
    schedule(s, now, minimum);
  });
}
const flags = (directory) => ['--session', session, '--data-dir', directory];

test('frequency is tools-only and retry count defaults to 3 with configurable bounds', () => {
  assert.equal(validateConfig({}).mode, 'tools');
  assert.equal(validateConfig({}).retryCount, 3);
  for (const retryCount of [1, 3, 5, 100]) assert.equal(validateConfig({ retryCount }).retryCount, retryCount);
  for (const patch of [{ mode: 'time' }, { mode: 'either' }, { mode: 'both' }, { mode: 'background' }, { toolMin: 0 }, { secondsMin: -1 }, { toolMin: 4, toolMax: 2 }, { secondsMin: 1000, secondsMax: 3 }, { retryCount: 0 }, { retryCount: 101 }, { retryCount: 2.5 }, { retryCount: '3' }, { maxPerTurn: 50, maxPerSession: 40 }, { maxPerSession: 1001 }, { toolMin: NaN }, { toolMax: 2.5 }, { rogue: 1 }, { languages: [] }, { languages: ['xx'] }, { languages: ['en', 'en'] }]) assert.throws(() => validateConfig(patch));
});

test('new tasks default to 16–32 work tools without overwriting saved custom intervals', async (t) => {
  assert.equal(DEFAULTS.toolMin, 16); assert.equal(DEFAULTS.toolMax, 32);
  assert.equal(validateConfig({}).toolMin, 16); assert.equal(validateConfig({}).toolMax, 32);
  const directory = await fixture(t);
  const started = await run(['start', ...flags(directory)], {});
  assert.equal(started.frequency.toolMin, 16); assert.equal(started.frequency.toolMax, 32);
  assert.equal(started.probesIssued, 1);
  for (const [toolMin, toolMax] of [[8, 16], [30, 60], [40, 40]]) {
    await run(['configure', ...flags(directory), '--tool-min', String(toolMin), '--tool-max', String(toolMax)], {});
    const configured = await run(['configure', ...flags(directory), '--retry-count', '5'], {});
    assert.equal(configured.frequency.toolMin, toolMin); assert.equal(configured.frequency.toolMax, toolMax);
    assert.equal(configured.probesIssued, 1, 'configuration does not issue another probe');
  }
});

test('never-enabled legacy hook records pick up new defaults but restarted tasks keep saved intervals', async (t) => {
  const dir = await fixture(t);
  await withState(dir, session, (state) => { state.config.toolMin = 8; state.config.toolMax = 16; });
  const first = await run(['start', ...flags(dir)], {});
  assert.equal(first.frequency.toolMin, 16); assert.equal(first.frequency.toolMax, 32);
  await run(['configure', ...flags(dir), '--tool-min', '8', '--tool-max', '16'], {});
  await run(['stop', ...flags(dir)], {});
  const restarted = await run(['start', ...flags(dir)], {});
  assert.equal(restarted.frequency.toolMin, 8); assert.equal(restarted.frequency.toolMax, 16);
});

test('only work-tool thresholds schedule probes; elapsed time cannot make a checkpoint due', () => {
  const state = newState(session, 1000);
  schedule(state, 1000, minimum);
  assert.equal(state.nextTools, DEFAULTS.toolMin);
  assert.equal(state.nextAt, null);
  assert.equal(isDue(state, 1000000000), false);
  state.workTools = state.nextTools;
  assert.equal(isDue(state, 1001), true);
  schedule(state, 1000, (lo, hi) => hi - 1);
  assert.equal(state.nextTools, state.workTools + DEFAULTS.toolMax);
  assert.equal(state.nextAt, null);
  assert.equal(state.config.pendingSeconds, 180);
});

test('literal parser rejects signs, decimal, exponent, range errors, expressions and prose', () => {
  const good = JSON.parse(syntheticNumbers());
  assert.equal(validateNumbers(JSON.stringify(good), 300).length, 300);
  for (const value of ['-2', '0', '356', '1.5', '3e1', 'null', 'true', '1+1', '"12"']) assert.throws(() => validateNumbers(`[${value},${good.join(',')}]`, 300));
  for (const text of ['numbers: ' + syntheticNumbers(), '[1,2,3]', syntheticNumbers(500), '[01,' + good.join(',') + ']']) assert.throws(() => validateNumbers(text, 300));
  assert.equal(validateNumbers(syntheticNumbers(200), 300).length, 200); // retain natural counting error, not repair
});

test('one difference remains evidence and repeated matching alternatives escalate', () => {
  const result = { results: [{ model: 'other', probability: 0.9 }, { model: 'expected', probability: 0.04 }] };
  const first = classifySample(result, 'expected');
  assert.equal(first.outcome, 'difference_signal');
  assert.equal(classifySample(result, 'expected', [first]).outcome, 'repeated_difference');
  assert.equal(classifySample(result, 'unlisted', [first]).outcome, 'unknown_expected_model');
  assert.equal(classifySample({ results: [{ model: 'expected', probability: 0.7 }] }, 'expected').outcome, 'compatible');
  assert.equal(classifySample({ results: [{ model: 'other', probability: 0.4 }, { model: 'expected', probability: 0.3 }] }, 'expected').outcome, 'inconclusive');
  assert.equal(classifySample(result, 'expected', [first, { prediction: 'third' }, { prediction: 'third' }]).outcome, 'difference_signal');
});

test('confirmation never combines different languages, old epochs or distant hits', () => {
  const state = newState(session); state.epoch = 3;
  state.samples = [{ epoch: 2, language: 'en' }, { epoch: 3, language: 'en' }, { epoch: 3, language: 'ja' }, { epoch: 3, language: 'zh' }];
  assert.deepEqual(recentComparable(state, { language: 'en' }), []);
  assert.equal(recentComparable(state, { language: 'zh' }).length, 1);
  state.samples.push({ epoch: 3, language: 'en' });
  assert.equal(recentComparable(state, { language: 'en' }).length, 1);
});

test('first turn binding preserves the initial checkpoint count', () => {
  const state = newState(session); state.turnIssued = 1;
  setTurn(state, 'first-turn'); assert.equal(state.turnIssued, 1);
  setTurn(state, 'second-turn'); assert.equal(state.turnIssued, 0);
});

test('ten localized prompts preserve count/range/submission, can be sampled individually', () => {
  assert.equal(LANGUAGES.length, 10);
  for (const language of LANGUAGES) {
    const state = newState(session, 1000);
    state.enabled = true; state.forceProbe = true;
    state.config = validateConfig({ languages: [language] });
    issue(state, 1000, minimum);
    assert.equal(state.pending.language, language);
    const context = challengeContext(state, '/tmp/test');
    assert.ok(forkPrompt(language, state.pending.count).includes('292') && forkPrompt(language, state.pending.count).includes('355'));
    assert.ok(context.includes(state.pending.id));
    assert.ok(context.includes('background hook'));
    assert.ok(!context.includes('probe --session'));
    assert.ok(!context.includes('--numbers'));
  }
  assert.throws(() => localizedPrompt('xx', 300, 'command'));
});

test('inactive hook is silent, records heartbeat, never probes', async (t) => {
  const dir = await fixture(t);
  assert.deepEqual(await handleHook(event('PostToolUse', { tool_name: 'Bash', tool_input: { cmd: 'sensitive task text' } }), dir), {});
  const state = await readState(dir, session);
  assert.equal(state.issued, 0);
  assert.ok(state.hooksSeen);
  assert.ok(!JSON.stringify(state).includes('sensitive task text'));
});

test('due hooks queue a same-task checkpoint without a foreground command or context injection', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await handleHook(event('PostToolUse', { tool_use_id: 'a' }), dir, 1001, minimum);
  const output = await handleHook(event('PostToolUse', { tool_use_id: 'b' }), dir, 1002, minimum);
  assert.deepEqual(output, {});
  const state = await readState(dir, session);
  assert.equal(state.issued, 1); assert.equal(state.session, session); assert.ok(state.pending);
});

test('parallel completions issue one checkpoint and replayed event is deduplicated', async (t) => {
  const dir = await fixture(t);
  await active(dir);
  const outputs = await Promise.all(['a', 'b', 'c'].map((id) => handleHook(event('PostToolUse', { tool_use_id: id }), dir)));
  assert.ok(outputs.every((output) => Object.keys(output).length === 0));
  await handleHook(event('PostToolUse', { tool_use_id: 'c' }), dir);
  const state = await readState(dir, session);
  assert.equal(state.workTools, 3);
  assert.equal(state.issued, 1);
});

test('monitor own commands do not count toward frequency or trigger recursive probes', async (t) => {
  const dir = await fixture(t);
  await active(dir);
  for (const command of ['start', 'submit', 'configure', 'status', 'stop', 'doctor', 'models']) {
    assert.deepEqual(await handleHook(event('PostToolUse', { tool_name: 'Bash', tool_input: { cmd: `node 'path with spaces/guard.mjs' ${command}` } }), dir), {});
  }
  assert.equal((await readState(dir, session)).workTools, 0);
});

test('ignored and expired checkpoints become gaps, with cooldown instead of storm', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await withState(dir, session, (s) => { s.forceProbe = true; issue(s, 1000, minimum); });
  const output = await handleHook(event('PostToolUse'), dir, 200000, minimum);
  assert.ok(output.systemMessage);
  const state = await readState(dir, session);
  assert.equal(state.missed, 1); assert.equal(state.pending, null); assert.equal(state.issued, 1);
});

test('a batch of parallel tool completions is not mistaken for ignored probes', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await withState(dir, session, (s) => { s.forceProbe = true; issue(s, 1000, minimum); });
  await Promise.all(Array.from({ length: 10 }, (_, i) => handleHook(event('PostToolUse', { tool_use_id: `parallel-${i}` }), dir, 1001 + i, minimum)));
  const state = await readState(dir, session);
  assert.equal(state.missed, 0); assert.equal(state.issued, 1); assert.ok(state.pending);
});

test('model changes segment comparisons and preserve previous sample history', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await withState(dir, session, (s) => { s.samples.push({ epoch: 1, outcome: 'difference_signal', prediction: 'old' }); s.forceProbe = true; issue(s, 1000, minimum); });
  await handleHook(event('PostToolUse', { model: 'o4-mini' }), dir, 1002, minimum);
  const state = await readState(dir, session);
  assert.equal(state.epoch, 2); assert.equal(state.expected, 'o4-mini');
  assert.equal(state.missed, 1); assert.equal(state.samples.length, 1);
  assert.equal(state.pending.epoch, 2);
});

test('late background events count tools without rolling back a newer turn or model', async (t) => {
  const dir = await fixture(t); await active(dir, 1000);
  await handleHook(event('UserPromptSubmit', { turn_id: 'turn-2', model: 'new-model' }), dir, 1001, minimum);
  const current = await readState(dir, session);
  await handleHook(event('PostToolUse', { turn_id: 'turn-1', model: 'gpt-5.4', tool_use_id: 'late-tool' }), dir, 1002, minimum);
  await handleHook(event('SessionStart', { turn_id: 'turn-1', source: 'resume' }), dir, 1003, minimum);
  const after = await readState(dir, session);
  assert.equal(after.turn, 'turn-2'); assert.equal(after.model, 'new-model'); assert.equal(after.expected, 'new-model');
  assert.equal(after.workTools, current.workTools + 1); assert.equal(after.epoch, current.epoch);
  assert.deepEqual(after.pending, current.pending); assert.equal(after.missed, current.missed);
});

test('explicit expected model is not silently replaced by model field', async (t) => {
  const dir = await fixture(t);
  await active(dir);
  await run(['configure', ...flags(dir), '--expected', 'gpt-5.4'], {});
  await handleHook(event('PostToolUse', { model: 'gpt-4o' }), dir);
  assert.equal((await readState(dir, session)).expected, 'gpt-5.4');
});

test('compaction drops pending, suspends probes, then starts a new epoch', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await withState(dir, session, (s) => { s.forceProbe = true; issue(s, 1000, minimum); });
  await handleHook(event('PreCompact'), dir, 1001, minimum);
  assert.deepEqual(await handleHook(event('PostToolUse'), dir, 1002, minimum), {});
  const output = await handleHook(event('SessionStart', { source: 'compact' }), dir, 1003, minimum);
  assert.deepEqual(output, {});
  const state = await readState(dir, session);
  assert.equal(state.epoch, 2); assert.equal(state.missed, 1); assert.equal(state.compacting, false);
});

test('normal pending probes never block Stop, get abandoned by it, or force extra probes', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await withState(dir, session, (s) => { s.forceProbe = true; issue(s, 1000, minimum); });
  const pending = (await readState(dir, session)).pending;
  const first = await handleHook(event('Stop'), dir, 1001, minimum);
  assert.deepEqual(first, {});
  assert.deepEqual(await handleHook(event('Stop', { stop_hook_active: true }), dir, 1002, minimum), {});
  assert.deepEqual(await handleHook(event('Stop'), dir, 1003, minimum), {});
  const state = await readState(dir, session);
  assert.equal(state.missed, 0); assert.equal(state.issued, 1); assert.deepEqual(state.pending, pending);
});

test('ordinary probes keep running past retired turn and total caps without resetting counts', async (t) => {
  const dir = await fixture(t);
  await active(dir, 1000);
  await withState(dir, session, (s) => { s.issued = 1000000; s.turnIssued = 1000; s.forceProbe = true; });
  await handleHook(event('PostToolUse'), dir, 1001, minimum);
  const first = await readState(dir, session);
  assert.equal(first.issued, 1000001); assert.equal(first.turnIssued, 1001); assert.ok(first.pending);
  await withState(dir, session, (s) => { s.pending = null; s.forceProbe = true; });
  await handleHook(event('UserPromptSubmit', { turn_id: 'turn-2' }), dir, 1002, minimum);
  assert.equal((await readState(dir, session)).issued, 1000002);
  await withState(dir, session, (s) => { s.pending = null; s.forceProbe = true; });
  await handleHook(event('UserPromptSubmit', { turn_id: 'turn-3' }), dir, 1003, minimum);
  const state = await readState(dir, session);
  assert.equal(state.issued, 1000003); assert.equal(state.turnIssued, 1);
  assert.equal(state.events.filter((e) => e.type === 'budget_paused').length, 0);
});

test('configure changes frequency and language mid-task without erasing evidence or counts', async (t) => {
  const dir = await fixture(t);
  await active(dir);
  await withState(dir, session, (s) => { s.issued = 3; s.samples.push({ outcome: 'difference_signal' }); });
  const result = await run(['configure', ...flags(dir), '--tool-min', '5', '--tool-max', '5', '--retry-count', '5', '--languages', 'ja,fr'], {});
  assert.equal(result.frequency.mode, 'tools');
  assert.equal(result.frequency.toolMin, 5); assert.equal(result.frequency.retryCount, 5);
  assert.deepEqual(result.frequency.languages, ['ja', 'fr']);
  assert.equal(result.probesIssued, 3); assert.equal(result.differenceSignals, 1);
  assert.equal(result.epoch, 1);
});

test('start/submit/status/stop round trip stores only probe evidence, replay rejected', async (t) => {
  const dir = await fixture(t);
  await handleHook(event('SessionStart'), dir);
  const started = await run(['start', ...flags(dir), '--languages', 'en'], {});
  assert.ok(started.challengeContext);
  const accepted = await run(['submit', ...flags(dir), '--challenge', started.pending.id, '--numbers', syntheticNumbers()], {});
  assert.equal(accepted.accepted, true);
  assert.equal(accepted.sample.language, 'en');
  assert.equal(accepted.sample.languageAndContextCalibrated, false);
  assert.equal(accepted.sample.numbers, undefined);
  await assert.rejects(() => run(['submit', ...flags(dir), '--challenge', started.pending.id, '--numbers', syntheticNumbers()], {}));
  const status = await run(['status', ...flags(dir)], {});
  assert.equal(status.probesAccepted, 1); assert.equal(status.pending, null);
  const state = await readState(dir, session);
  assert.equal(state.samples[0].numbers.length, 300);
  const stopped = await run(['stop', ...flags(dir)], {});
  assert.equal(stopped.enabled, false); assert.equal(stopped.probesAccepted, 1);
  const afterStop = await handleHook(event('PostToolUse'), dir);
  if (state.alerts.some((alert) => !alert.acknowledgedAt)) {
    assert.match(afterStop.hookSpecificOutput.additionalContext, /NOTIFY THE USER NOW/);
  } else {
    assert.deepEqual(afterStop, {});
  }
});

test('invalid sample cannot become a compatible result or consume pending silently', async (t) => {
  const dir = await fixture(t);
  const start = await run(['start', ...flags(dir)], {});
  await assert.rejects(() => run(['submit', ...flags(dir), '--challenge', start.pending.id, '--numbers', '[0,1]'], {}));
  const state = await readState(dir, session);
  assert.equal(state.samples.length, 0); assert.equal(state.pending.id, start.pending.id);
});

test('late submissions are recorded as missed and cannot count as successful coverage', async (t) => {
  const dir = await fixture(t);
  const start = await run(['start', ...flags(dir)], {});
  await withState(dir, session, (s) => { s.pending.expiresAt = 1; });
  const result = await run(['submit', ...flags(dir), '--challenge', start.pending.id, '--numbers', syntheticNumbers()], {});
  assert.equal(result.accepted, false);
  const state = await readState(dir, session);
  assert.equal(state.samples.length, 0); assert.equal(state.missed, 1);
});

test('reference bank changes cannot silently mix comparison segments', async (t) => {
  const dir = await fixture(t);
  const start = await run(['start', ...flags(dir)], {});
  await withState(dir, session, (s) => { s.bankSha256 = 'old-bank'; });
  const result = await run(['submit', ...flags(dir), '--challenge', start.pending.id, '--numbers', syntheticNumbers()], {});
  assert.equal(result.accepted, false);
  const state = await readState(dir, session);
  assert.equal(state.epoch, start.epoch + 1); assert.equal(state.samples.length, 0);
});

test('interruption cancels pending without restarting and status retains history', async (t) => {
  const dir = await fixture(t);
  await run(['start', ...flags(dir)], {});
  assert.deepEqual(await handleHook(event('Interrupt'), dir), {});
  const state = await readState(dir, session);
  assert.equal(state.pending, null); assert.equal(state.missed, 1);
  assert.equal(state.issued, 1);
});

test('stale state lock fails visibly without stealing a possibly live writer lock', async (t) => {
  const dir = await fixture(t);
  await writeFile(`${sessionPath(dir, session)}.lock`, '');
  await assert.rejects(() => withState(dir, session, () => {}), /State busy/);
  assert.equal(await readFile(`${sessionPath(dir, session)}.lock`, 'utf8'), '');
});

test('wrong task, wrong challenge, subagent and corrupt state fail safely', async (t) => {
  const dir = await fixture(t);
  await assert.rejects(() => run(['start', ...flags(dir)], { CODEX_THREAD_ID: 'other-task' }));
  const started = await run(['start', ...flags(dir)], {});
  await assert.rejects(() => run(['submit', ...flags(dir), '--challenge', 'fake', '--numbers', syntheticNumbers()], {}));
  assert.deepEqual(await handleHook(event('PostToolUse', { agent_id: 'agent' }), dir), {});
  assert.equal((await readState(dir, session)).pending.id, started.pending.id);
  await writeFile(sessionPath(dir, 'broken'), 'not json');
  await assert.rejects(() => run(['start', '--session', 'broken', '--data-dir', dir], {}));
  assert.equal(await readFile(sessionPath(dir, 'broken'), 'utf8'), 'not json');
});

test('checksum bank and scorer are valid and packaged GPT/o-series families are normalized', async () => {
  const { bank, metadata } = await loadArtifacts();
  assert.equal(bank.models.length, metadata.modelCount);
  assert.ok(bank.models.length > 0);
  assert.equal(new Set(bank.models.map((model) => model.id)).size, bank.models.length);
  const gpt = bank.models.filter((model) => /^(?:gpt-|o\d)/.test(model.id));
  assert.ok(gpt.length > 0);
  for (const model of gpt) assert.equal(model.family, 'gpt', model.id);
});

test('stable data path is shared between hooks and task shell even with PLUGIN_DATA differences', () => {
  assert.equal(dataDirectory(null, {}), dataDirectory(null, { PLUGIN_DATA: 'some-version-specific-dir' }));
  assert.equal(dataDirectory('/explicit', { MODELTRACE_GUARD_DATA: '/env' }), path.resolve('/explicit'));
});

test('CLI rejects typos instead of silently ignoring user frequency settings', async () => {
  for (const args of [['start', '--frequency', '1'], ['start', '--max-per-turn', '8'], ['start', '--max-per-session', '40'], ['start', '--mode'], ['start', '--mode', 'tools', '--mode', 'time'], ['unknown']]) await assert.rejects(() => run(args, {}));
  const help = await run(['help'], {});
  assert.equal(help.defaults.maxPerTurn, undefined); assert.equal(help.defaults.maxPerSession, undefined);
  assert.equal(help.defaults.pendingSeconds, 180);
});

test('expiry setting controls new submission deadlines, never changes the live checkpoint or frequency', async (t) => {
  const dir = await fixture(t);
  const start = await run(['start', ...flags(dir)], {});
  assert.equal(start.pending.expiresAt - start.pending.at, 180000);
  const updated = await run(['configure', ...flags(dir), '--pending-seconds', '600'], {});
  assert.equal(updated.pending.expiresAt, start.pending.expiresAt);
  assert.equal(updated.pending.id, start.pending.id);
  await run(['submit', ...flags(dir), '--challenge', start.pending.id, '--numbers', syntheticNumbers()], {});
  await withState(dir, session, (s) => { s.forceProbe = true; });
  await handleHook(event('PostToolUse'), dir);
  const state = await readState(dir, session);
  assert.equal(state.pending.expiresAt - state.pending.at, 600000);
  assert.equal(state.config.mode, 'tools'); assert.equal(state.config.toolMin, start.frequency.toolMin); assert.equal(state.config.toolMax, start.frequency.toolMax);
});

test('expired pending shown as coverage gap even with no further hooks', () => {
  const state = newState(session, 1000); state.enabled = true; state.forceProbe = true;
  issue(state, 1000, minimum);
  assert.equal(summarize(state, '/tmp', 1000000).status, 'coverage_gap');
});

test('an old heartbeat cannot verify monitoring after a new start', () => {
  const state = newState(session, 1000);
  state.enabled = true; state.enabledAt = 2000; state.hooksSeen = 100; state.lastHookAt = 1500;
  assert.equal(summarize(state, '/tmp', 3000).status, 'hooks_unverified');
  assert.equal(summarize(state, '/tmp', 3000).hookObserved, false);
});

test('exact bundled hook launcher works in native shell with spaced paths', async (t) => {
  const dir = await fixture(t);
  const config = JSON.parse(await readFile(path.join(ROOT, 'hooks', 'hooks.json')));
  const command = config.hooks.PostToolUse[0].hooks[0].command;
  const spacedRoot = path.join(dir, 'plugin source with spaces');
  await cp(ROOT, spacedRoot, { recursive: true });
  const env = { ...process.env, PLUGIN_ROOT: spacedRoot, MODELTRACE_GUARD_DATA: dir, CODEX_THREAD_ID: session };
  const result = spawnSync(command, { shell: true, input: JSON.stringify(event('SessionStart')), encoding: 'utf8', env, timeout: 10000 });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), {});
  assert.equal((await readState(dir, session)).hooksSeen, 1);
  if (process.platform === 'win32') {
    const powershell = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], { input: JSON.stringify(event('UserPromptSubmit')), encoding: 'utf8', env, timeout: 10000 });
    assert.equal(powershell.status, 0, powershell.stderr);
    assert.deepEqual(JSON.parse(powershell.stdout), {});
    assert.equal((await readState(dir, session)).hooksSeen, 2);
  }
});

test('modified bundled assets fail integrity checks instead of silently scoring', async (t) => {
  const dir = await fixture(t);
  const copy = path.join(dir, 'modified-plugin');
  await cp(ROOT, copy, { recursive: true });
  await writeFile(path.join(copy, 'assets', 'unified_bank.json'), '{}');
  const result = spawnSync(process.execPath, [path.join(copy, 'scripts', 'guard.mjs'), 'doctor'], { encoding: 'utf8', timeout: 10000 });
  assert.equal(result.status, 1);
  assert.match(JSON.parse(result.stderr).error, /checksum mismatch/);
});

test('hook launcher reports malformed input without preventing actual work', async (t) => {
  const dir = await fixture(t);
  const result = spawnSync(process.execPath, [path.join(ROOT, 'scripts', 'guard.mjs'), 'hook', '--data-dir', dir], { input: 'not json', encoding: 'utf8', timeout: 10000 });
  assert.equal(result.status, 0);
  assert.ok(JSON.parse(result.stdout).systemMessage.includes('coverage is incomplete'));
});

test('isDirectCall correctly identifies direct invocations across symlinks', async (t) => {
  const dir = await fixture(t);
  const guardScript = path.join(ROOT, 'scripts', 'guard.mjs');
  const symlinkScript = path.join(dir, 'symlink-guard.mjs');
  const symlinkChain = path.join(dir, 'symlink-chain.mjs');
  try {
    await symlink(guardScript, symlinkScript);
    await symlink(symlinkScript, symlinkChain);
  } catch (err) {
    if (process.platform === 'win32' && err.code === 'EPERM') return;
    throw err;
  }

  const guardUrl = new URL('../scripts/guard.mjs', import.meta.url);
  assert.equal(isDirectCall(guardUrl.href, null), false);
  assert.equal(isDirectCall(null, guardScript), false);
  assert.equal(isDirectCall(undefined, guardScript), false);
  assert.equal(isDirectCall(guardUrl.href, guardScript), true);
  assert.equal(isDirectCall(guardUrl, symlinkScript), true);
  assert.equal(isDirectCall(guardScript, symlinkScript), true);
  assert.equal(isDirectCall(guardUrl.href, symlinkChain), true);
  assert.equal(isDirectCall(guardUrl.href, path.join(dir, 'nonexistent.mjs')), false);
});

test('CLI invocations via symlinked script execute properly', async (t) => {
  const dir = await fixture(t);
  const guardScript = path.join(ROOT, 'scripts', 'guard.mjs');
  const symlinkScript = path.join(dir, 'symlink-guard.mjs');
  const symlinkChain = path.join(dir, 'symlink-chain.mjs');
  try {
    await symlink(guardScript, symlinkScript);
    await symlink(symlinkScript, symlinkChain);
  } catch (err) {
    if (process.platform === 'win32' && err.code === 'EPERM') return;
    throw err;
  }

  const directResult = spawnSync(process.execPath, [symlinkScript, 'doctor'], { encoding: 'utf8', timeout: 10000 });
  assert.equal(directResult.status, 0, directResult.stderr);
  assert.equal(JSON.parse(directResult.stdout).assetsVerified, true);

  const chainResult = spawnSync(process.execPath, [symlinkChain, 'doctor'], { encoding: 'utf8', timeout: 10000 });
  assert.equal(chainResult.status, 0, chainResult.stderr);
  assert.equal(JSON.parse(chainResult.stdout).assetsVerified, true);

  const preserveResult = spawnSync(process.execPath, ['--preserve-symlinks', symlinkScript, 'doctor'], { encoding: 'utf8', timeout: 10000 });
  assert.equal(preserveResult.status, 0, preserveResult.stderr);
  assert.equal(JSON.parse(preserveResult.stdout).assetsVerified, true);
});

