from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from fingerprint import (  # noqa: E402
    ALPHA,
    DIMENSION,
    count_numbers,
    parse_numbers,
)
from v6_builder import fit_v6_portable_bank  # noqa: E402


DEFAULT_RESPONSES = PROJECT / "data" / "gpt_reference.jsonl"
DEFAULT_OUTPUT = PROJECT / "data" / "gpt_bank.json"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        if not row.get("strict_valid", True):
            continue
        numbers = parse_numbers(row["text"])
        if len(numbers) < 20:
            continue
        output.append({**row, "numbers": numbers, "counts": count_numbers(numbers)})
    return output


def build_bank(rows: list[dict], fallback_calibration: dict | None = None) -> dict:
    model_ids = list(dict.fromkeys(row["source"] for row in rows))
    model_entries = []
    for model_id in model_ids:
        selected = [row for row in rows if row["source"] == model_id]
        pooled = [sum(row["counts"][index] for row in selected) for index in range(DIMENSION)]
        family_id = selected[0].get("family_id") or selected[0].get("bank_id")
        family_name = selected[0].get("family_name")
        model_entries.append(
            {
                "id": model_id,
                "display_name": model_id,
                "family": family_id,
                "family_name": family_name,
                "response_count": len(selected),
                "valid_number_count": sum(len(row["numbers"]) for row in selected),
                "conditions": dict(Counter(row.get("condition_id", "default") for row in selected)),
                "counts": pooled,
            }
        )

    providers = sorted({row.get("provider") for row in rows if row.get("provider")})
    source_scope = "Reference outputs enrolled for the listed model labels (open-set U calibrated)."

    if len(model_ids) >= 2:
        v6_portable_dict, calibration = fit_v6_portable_bank(rows, model_ids)
        return {
            "schema": "absolute-dominance-v6-bank",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "method": {
                "name": "Absolute-Dominance v6",
                "range": [1, 355],
                "alpha": ALPHA,
                "parser": "longest digit run; alphabetic separators split runs; retain values 1..355",
                "response_score": (
                    "windowed 325-token cdf225/lex1065/PL + AR-Fisher + 5 dual-debiased Cauchy-Woodbury pair views "
                    "+ 6-channel heteroscedastic guard-ring vMF + landmark multimodal KRR + sparse ET/Block-4 sibling experts"
                ),
                "aggregation": "K-adaptive log-Bayes-factor + median/variance-discounted pairwise tournament",
                "probability": "joint posterior over [Known_1..M, U] with absolute-probability and dominance selective abstention",
            },
            "recommended_queries": 3,
            "minimum_valid_numbers": 80,
            "source_scope": source_scope,
            "providers": providers,
            "models": model_entries,
            "v6_portable": v6_portable_dict,
            "calibration": calibration,
        }

    # Graceful fallback when a custom sub-bank temporarily has < 2 models
    default_cal = fallback_calibration or {
        k: {"beta": 1.0, "cv_accuracy": None, "cv_correct": 0, "cv_samples": 0, "cv_nll": None, "fallback": True}
        for k in ("1", "2", "3")
    }
    return {
        "schema": "single-model-subbank-v6",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": "Single-Model Subbank (Unified Bank uses Absolute-Dominance v6)",
            "range": [1, 355],
            "alpha": ALPHA,
        },
        "recommended_queries": 3,
        "minimum_valid_numbers": 80,
        "source_scope": source_scope,
        "providers": providers,
        "models": model_entries,
        "calibration": default_cal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fallback = None
    if args.output.exists():
        fallback = json.loads(args.output.read_text(encoding="utf-8")).get("calibration")
    bank = build_bank(read_rows(args.responses), fallback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    summary = {
        "output": str(args.output),
        "schema": bank["schema"],
        "method": bank["method"]["name"],
        "model_count": len(bank["models"]),
        "total_responses": sum(model["response_count"] for model in bank["models"]),
        "calibration": bank["calibration"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
