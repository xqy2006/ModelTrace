import test from "node:test";
import assert from "node:assert/strict";
import { generateChallenges, normalizeChallengeOptions } from "../static/challenge-browser.js";

test("default packs are standard Chinese with distinct lengths", () => {
  const challenges = generateChallenges(3);
  assert.equal(challenges.length, 3);
  assert.equal(new Set(challenges.map((item) => item.expected_count)).size, 3);
  for (const item of challenges) {
    assert.equal(item.style, "standard");
    assert.equal(item.language, "zh");
    assert.match(item.prompt, new RegExp(String(item.expected_count)));
    assert.match(item.prompt, /Python、代码执行器/);
  }
});

test("compact Chinese omits the tool laundry list and asks to stop", () => {
  const challenges = generateChallenges(3, { style: "compact", language: "zh" });
  for (const item of challenges) {
    assert.match(item.prompt, /^随机生成 /);
    assert.match(item.prompt, /立即停止/);
    assert.equal(item.prompt.includes("Python、代码执行器"), false);
    assert.ok(item.prompt.length < 120);
  }
});

test("compact English names the count twice and stops after it", () => {
  const [item] = generateChallenges(1, { style: "compact", language: "en" });
  assert.match(item.prompt, new RegExp(`Randomly choose ${item.expected_count} integers from 1 to 355`));
  assert.match(item.prompt, new RegExp(`Stop immediately after ${item.expected_count} integers`));
});

test("unknown options throw", () => {
  assert.throws(() => generateChallenges(3, { style: "telegraph" }), /风格/);
  assert.throws(() => normalizeChallengeOptions("standard", "ja"), /语言/);
});
