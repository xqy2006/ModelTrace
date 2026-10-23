import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { AMBIGUOUS_MARGIN, CLOSE_MARGIN, CLOSE_TOP, analyzeGlobalOutputs, closenessReport } from "../static/fingerprint-core.js";

const ranked = (...pairs) => pairs.map(([model, probability, family]) => ({
  model,
  display_name: model,
  probability,
  family,
  family_name: family?.toUpperCase(),
}));

test("decisive requires both margin and top-mass", () => {
  const report = closenessReport(ranked(["gpt-6-sol", 0.82, "gpt"], ["gpt-5.5", 0.11, "gpt"]), {
    idKey: "model",
    nameKey: "display_name",
    familyKey: "family",
  });
  assert.equal(report.verdict, "decisive");
  assert.ok(Math.abs(report.margin - 0.71) < 1e-12);
  assert.equal(report.runner_up, "gpt-5.5");
  assert.equal(report.same_family_top_two, true);
});

test("a half-probability lead is close rather than a hard call", () => {
  const report = closenessReport(ranked(["gpt-6-sol", 0.52, "gpt"], ["gpt-5.5", 0.20, "gpt"]), {
    idKey: "model",
    nameKey: "display_name",
    familyKey: "family",
  });
  assert.equal(report.verdict, "close");
  assert.ok(report.margin > AMBIGUOUS_MARGIN);
});

test("near ties are ambiguous and list both contenders", () => {
  const report = closenessReport(
    ranked(["gpt-6-sol", 0.41, "gpt"], ["gpt-5.5", 0.39, "gpt"], ["claude-sonnet-5", 0.12, "claude"]),
    { idKey: "model", nameKey: "display_name", familyKey: "family" },
  );
  assert.equal(report.verdict, "ambiguous");
  assert.deepEqual(report.contenders.map((item) => item.id), ["gpt-6-sol", "gpt-5.5"]);
});

test("saved GPT-5.5 references stay a decisive closed-set hit", async () => {
  const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
  const bank = JSON.parse(await readFile(path.join(root, "data/unified_bank.json"), "utf8"));
  const rows = (await readFile(path.join(root, "data/gpt_reference.jsonl"), "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line))
    .filter((row) => row.strict_valid && row.model_id === "gpt-5.5")
    .slice(0, 3);
  assert.equal(rows.length, 3);
  const result = analyzeGlobalOutputs(
    rows.map((row) => ({ text: row.text, expected_count: row.requested_count })),
    bank,
  );
  assert.equal(result.prediction, "gpt-5.5");
  assert.equal(result.verdict, "decisive");
  assert.ok(result.margin >= CLOSE_MARGIN);
  assert.ok(result.probability >= CLOSE_TOP);
  assert.equal(result.family_verdict, "decisive");
  assert.equal(result.contenders[0].id, "gpt-5.5");
});
