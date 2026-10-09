"""Offline reproduction against the recorded d4131b3 baseline. No network or bank writes."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from bank_builder import build_bank, calibration_records, read_rows
from challenge_suite import fingerprint_suite
from fingerprint import analyze_outputs, count_numbers, parse_numbers

MODEL = "gpt-6.1-sol"
sha = lambda data: hashlib.sha256(data).hexdigest()
read = lambda path: json.loads(path.read_text(encoding="utf-8"))
metadata = read(HERE / "collection.json")
expected = read(HERE / "validation.json")
for name, digest in metadata["base_files_sha256"].items():
    assert sha((ROOT / name).read_bytes()) == digest, f"baseline changed: {name}"

datasets = {}
for group, name in [("enrollment", "enrollment.jsonl"), ("holdout", "holdout.jsonl"), ("failed_attempts", "failed-attempts.jsonl")]:
    data = (HERE / name).read_bytes()
    assert sha(data) == metadata[group + "_sha256"], name
    datasets[group] = [json.loads(line) for line in data.splitlines()]
enrollment, holdout, failures = (datasets[name] for name in ["enrollment", "holdout", "failed_attempts"])
assert [len(enrollment), len(holdout), len(failures)] == [36, 9, 1]
assert {row["challenge_id"] for row in enrollment} == {f"query-{number:02d}" for number in range(1, 37)}
assert {row["challenge_id"] for row in holdout} == {f"query-{number:02d}" for number in range(10, 19)}
tasks = {task["challenge_id"]: task for task in fingerprint_suite()}
all_valid = enrollment + holdout
for row in all_valid:
    task = tasks[row["challenge_id"]]
    prompt = task["user_prefix"] + "\n\nFinal task:\n" + task["prompt"] if task["user_prefix"] else task["prompt"]
    assert row["prompt"] == prompt and row["base_prompt"] == task["prompt"]
    assert row["system_prompt"] == task["system"] and row["user_prefix"] == task["user_prefix"]
    assert row["requested_count"] == task["expected_count"] and row["condition_id"] == task["condition"]
    assert row["model_id"] == MODEL and row["provider"] == "codex"
    assert row["cli_version"] == "0.159.1" and row["reasoning_effort"] == "low"
    assert row["response_model"] is None and row["identity_independently_attested"] is False
    assert row["harness_system_prompt"] is True and row["system_transport"] == "developer_instructions"
    numbers = parse_numbers(row["text"])
    assert row["parsed_count"] == len(numbers)
    assert row["strict_threshold"] == max(80, math.ceil(row["requested_count"] * 0.55))
    assert row["strict_valid"] and len(numbers) >= row["strict_threshold"]
    assert sha(row["text"].encode("utf-8")) == row["response_sha256"]
assert len({row["context_id_sha256"] for row in all_valid}) == 45
assert len({row["context_id_sha256"] for row in all_valid + failures}) == 46
assert len(parse_numbers(failures[0]["text"])) == 4 and failures[0]["strict_valid"] is False
retry = next(row for row in enrollment if row["challenge_id"] == "query-31")
assert retry["attempt"] == 2 and retry["retry_reason"] == "too_short"
assert retry["first_attempt_response_sha256"] == failures[0]["response_sha256"]
assert sha(failures[0]["text"].encode()) == failures[0]["response_sha256"]

def family(rows, identifier, name):
    return [{**row, "family_id": identifier, "family_name": name} for row in rows]

gpt = family(read_rows(ROOT / "data/gpt_reference.jsonl"), "gpt", "GPT")
claude = family(read_rows(ROOT / "data/claude_reference.jsonl"), "claude", "Claude")
new_rows = family([{**row, "numbers": parse_numbers(row["text"]), "counts": count_numbers(parse_numbers(row["text"]))} for row in enrollment], "gpt", "GPT")
baseline_rows, candidate_rows = gpt + claude, gpt + new_rows + claude
baseline = read(ROOT / "data/unified_bank.json")
candidate = build_bank(candidate_rows)
assert [len(baseline["models"]), len(candidate["models"])] == [17, 18]

def cv(rows, ids):
    result = {}
    for count in [1, 2, 3]:
        totals = defaultdict(lambda: {"correct": 0, "n": 0})
        for scores, truth in calibration_records(rows, ids, count):
            totals[ids[truth]]["n"] += 1
            totals[ids[truth]]["correct"] += max(range(len(scores)), key=scores.__getitem__) == truth
        result[str(count)] = dict(totals)
    return result

before = cv(baseline_rows, [row["id"] for row in baseline["models"]])
after = cv(candidate_rows, [row["id"] for row in candidate["models"]])
assert before == expected["before_cv"], "baseline CV differs from the recorded run"
assert after == expected["after_cv"], "candidate CV differs from the recorded run"

def predict(rows, bank):
    value = analyze_outputs([{"text": row["text"], "expected_count": row["requested_count"]} for row in rows], bank)
    return {"prediction": value["prediction"], "used_outputs": value["used_outputs"]}

groups = defaultdict(list)
for row in holdout:
    groups[row["condition_id"]].append(row)
held = {
    "singles": [{"challenge": row["challenge_id"], "before": predict([row], baseline), "after": predict([row], candidate)} for row in holdout],
    "groups": [{"condition": condition, "before": predict(rows, baseline), "after": predict(rows, candidate)} for condition, rows in groups.items()],
}
assert held == expected["holdout"], "held-out predictions differ from the recorded run"
for name, digest in metadata["base_files_sha256"].items():
    assert sha((ROOT / name).read_bytes()) == digest, f"active source was modified: {name}"
print(json.dumps({"integrity": "PASS", "enrollment": 36, "holdout": 9, "retained_failure": 1,
                  "new_model_cv": {count: values[MODEL] for count, values in after.items()},
                  "holdout_single_top1": sum(row["after"]["prediction"] == MODEL for row in held["singles"]),
                  "holdout_group_top1": sum(row["after"]["prediction"] == MODEL for row in held["groups"]),
                  "astra_three_reply_cv_before": before["3"]["gpt-6-astra"],
                  "astra_three_reply_cv_after": after["3"]["gpt-6-astra"],
                  "active_banks_unchanged": True}, ensure_ascii=False))
